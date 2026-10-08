"""Native v8 session boundary for ``server/``.

Ports the ``ctypes`` binding from ``version/v8/scripts/ck_serve_v8.py`` so the
HTTP layer can own one loaded CKE runtime without importing the scripts tree.

Duck-typed surface used by the live app factory (``server/live.py``):

  session.generate(system, user, *, max_tokens, temperature, top_p,
                   on_token, flags=0, stop_on_text=(), stop_at_eos=False) -> dict
  session.count_tokens(text) -> int
  session.cancel() -> None
  session.close() -> None

``SessionV8.open`` is the only constructor that touches the native library
(``build/libck_session_v8.so``, built via ``make ck-session-v8``). Most HTTP
tests inject a fake session; the batch binding test exercises this ctypes path
against a compiled fake generated model.

``enable_batch2`` is an opt-in token/logit boundary for a future coordinated
worker. It does not admit concurrent HTTP requests or perform sampling.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "build"
SESSION_LIB_PATH = BUILD_DIR / "libck_session_v8.so"

CK_SESSION_REQUEST_RAW_PROMPT = 1 << 0

_STOP_REASON_NAMES = {
    0: "none",
    1: "eos",
    2: "token_limit",
    3: "cancelled",
    4: "callback",
    5: "runtime_error",
}

_SESSION_STATUS_NAMES: dict[int, str] = {
    0: "ok",
    -1: "invalid_argument",
    -2: "abi",
    -3: "load",
    -4: "init",
    -5: "capability",
    -6: "busy",
    -7: "runtime",
    -8: "buffer_too_small",
}


class SessionError(RuntimeError):
    """Base exception for native ck_session_v8 errors."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.status_name = _SESSION_STATUS_NAMES.get(status, f"unknown({status})")
        super().__init__(message)


class SessionBusyError(SessionError):
    """Native session already has an active generate request (status -6)."""


_SESSION_STATUS_EXCEPTIONS: dict[int, type[SessionError]] = {
    -6: SessionBusyError,
}


class _Config(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("model_library_path", ctypes.c_char_p),
        ("weights_path", ctypes.c_char_p),
        ("manifest_path", ctypes.c_char_p),
        ("context_length", ctypes.c_int32),
        ("num_threads", ctypes.c_int32),
        ("required_capabilities", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _GenerateRequest(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("system_text", ctypes.c_char_p),
        ("user_text", ctypes.c_char_p),
        ("max_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("flags", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _GenerateResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("prompt_tokens", ctypes.c_int32),
        ("generated_tokens", ctypes.c_int32),
        ("stop_reason", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
        ("prefill_time_ms", ctypes.c_double),
        ("decode_time_ms", ctypes.c_double),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _ModelDescriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint64), ("artifact_role", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32), ("context_length", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("encoder_memory_tokens", ctypes.c_int32),
        ("encoder_memory_dim", ctypes.c_int32),
        ("primary_input_tokens", ctypes.c_int32),
        ("primary_input_dim", ctypes.c_int32),
        ("reserved", ctypes.c_uint64 * 8),
    ]


_BATCH_ABI = {
    "ck_session_v8_get_model_descriptor": (
        [ctypes.c_void_p, ctypes.POINTER(_ModelDescriptor), ctypes.c_size_t],
        ctypes.c_int),
    "ck_session_v8_batch2_enable": (
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)],
        ctypes.c_int),
    "ck_session_v8_batch2_prefill": (
        [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32),
         ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_uint64)], ctypes.c_int),
    "ck_session_v8_batch2_step": (
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
         ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
         ctypes.POINTER(ctypes.c_uint32)], ctypes.c_int),
    "ck_session_v8_batch2_reset_slot": (
        [ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
    "ck_session_v8_batch2_position": (
        [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)],
        ctypes.c_int),
    "ck_session_v8_batch2_request_cancel": (
        [ctypes.c_void_p, ctypes.c_uint64], None),
}


def _configure_batch_abi(lib: Any) -> None:
    try:
        for name, (argtypes, restype) in _BATCH_ABI.items():
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    except AttributeError as exc:
        raise SessionError(-5, "loaded session library lacks the batch2 ABI") from exc


_TOKEN_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.c_int32,
)


def _configure_abi(lib: Any, name: str) -> None:
    fn = getattr(lib, name)
    if name == "ck_session_v8_open":
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    elif name == "ck_session_v8_encode":
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
    elif name == "ck_session_v8_generate":
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_GenerateResult),
        ]
    elif name in ("ck_session_v8_cancel", "ck_session_v8_close"):
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p]
    elif name == "ck_session_v8_last_error":
        fn.restype = ctypes.c_char_p
        fn.argtypes = [ctypes.c_void_p]


def _last_error(lib: Any, session: Any) -> str:
    try:
        value = lib.ck_session_v8_last_error(session)
        return value.decode("utf-8", "replace") if value else "Unknown error occured"
    except Exception as e:
        return f"Unknown error occured during session. Error -> {e}"


def _configure_lib(lib: Any) -> None:
    for name in (
        "ck_session_v8_open",
        "ck_session_v8_encode",
        "ck_session_v8_generate",
        "ck_session_v8_cancel",
        "ck_session_v8_close",
        "ck_session_v8_last_error",
    ):
        _configure_abi(lib, name)


def _raise_native_error(lib: Any, session: Any, operation: str, status: int) -> None:
    if status == 0:
        return
    label = _SESSION_STATUS_NAMES.get(status, f"status={status}")
    detail = _last_error(lib, session)
    message = f"{operation} failed ({label})"
    if detail:
        message += f": {detail}"
    raise _SESSION_STATUS_EXCEPTIONS.get(status, SessionError)(status, message)


class Batch2SessionV8:
    """Serialized native two-slot operations; cancellation may run concurrently.

    This is a token/logit boundary. Logit arrays are reused and valid only
    until the next operation that writes that slot; copy them to retain a
    snapshot. The caller owns per-request sampling, stopping, output delivery,
    and worker lifetime. It must join in-flight cancellation calls before
    closing the parent session.
    """

    def __init__(self, parent: "SessionV8", vocab_size: int,
                 required_bytes: int, native_required_bytes: int):
        self.parent = parent
        self.vocab_size = vocab_size
        self.required_bytes = required_bytes
        self.native_required_bytes = native_required_bytes
        self._logits = ((ctypes.c_float * vocab_size)(),
                        (ctypes.c_float * vocab_size)())
        self._outputs = (ctypes.POINTER(ctypes.c_float) * 2)(*self._logits)

    def _call(self, name: str, *args: Any) -> None:
        with self.parent._lock:
            if self.parent.session is None:
                raise SessionError(-6, "batch2 session is closed")
            status = getattr(self.parent.lib, name)(self.parent.session, *args)
            _raise_native_error(self.parent.lib, self.parent.session, name, status)

    def _validate_tokens(self, tokens: Sequence[int], count: int) -> None:
        if len(tokens) != count or any(
            type(token) is not int or token < 0 or token >= self.vocab_size
            for token in tokens
        ):
            raise ValueError("batch2 token IDs must be integers in the loaded vocabulary")

    def prefill(self, slot: int, tokens: Sequence[int]) -> tuple[int, Any]:
        if type(slot) is not int or slot not in (0, 1) or not tokens:
            raise ValueError("prefill requires slot 0 or 1 and nonempty tokens")
        if len(tokens) > 0x7FFFFFFF:
            raise ValueError("prefill token count exceeds int32 capacity")
        self._validate_tokens(tokens, len(tokens))
        token_values = (ctypes.c_int32 * len(tokens))(*tokens)
        ticket = ctypes.c_uint64()
        self._call("ck_session_v8_batch2_prefill", slot, token_values,
                   len(tokens), self._logits[slot], ctypes.byref(ticket))
        return ticket.value, self._logits[slot]

    def step(self, tokens: tuple[int, int]) -> tuple[int, tuple[Any, Any]]:
        if len(tokens) != 2:
            raise ValueError("batch2 step requires exactly two token IDs")
        self._validate_tokens(tokens, 2)
        inputs = (ctypes.c_int32 * 2)(*tokens)
        advanced = ctypes.c_uint32()
        self._call("ck_session_v8_batch2_step", inputs, self._outputs,
                   ctypes.byref(advanced))
        return advanced.value, self._logits

    def reset_slot(self, slot: int) -> None:
        if type(slot) is not int or slot not in (0, 1):
            raise ValueError("slot must be 0 or 1")
        self._call("ck_session_v8_batch2_reset_slot", slot)

    def position(self, slot: int) -> int:
        if type(slot) is not int or slot not in (0, 1):
            raise ValueError("slot must be 0 or 1")
        position = ctypes.c_int32()
        self._call("ck_session_v8_batch2_position", slot, ctypes.byref(position))
        return position.value

    def request_cancel(self, ticket: int) -> None:
        if type(ticket) is not int or not 0 < ticket <= (1 << 63) - 1:
            raise ValueError("ticket must be a positive batch2 ticket")
        # Native cancellation is atomic and may run during a generated step.
        # Session close remains the caller's responsibility after worker join.
        if self.parent.session is None:
            raise SessionError(-6, "batch2 session is closed")
        self.parent.lib.ck_session_v8_batch2_request_cancel(
            self.parent.session, ticket)


def _detect_threads() -> int:
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except OSError:
        pass
    return max(1, os.cpu_count() or 1)


def _resolve_num_threads(explicit: int | None = None) -> int:
    if explicit is not None:
        if explicit < 1:
            raise ValueError("num_threads must be a positive integer")
        return explicit
    configured = os.environ.get("CK_NUM_THREADS")
    if configured is not None:
        try:
            value = int(configured)
        except ValueError as exc:
            raise ValueError("CK_NUM_THREADS must be a positive integer") from exc
        if value < 1:
            raise ValueError("CK_NUM_THREADS must be a positive integer")
        return value
    return _detect_threads()


def truncate_stop_markers(text: str, markers: Sequence[str]) -> str:
    active = [str(m) for m in (markers or ()) if str(m)]
    if not active or not text:
        return text
    lowered = text.lower()
    hits = [lowered.find(str(m).lower()) for m in active]
    hits = [i for i in hits if i != -1]
    if not hits:
        return text
    return text[: min(hits)]


# Backward-compat alias (mirrors the ck_serve_v8.py helper name).
_truncate_stop_markers = truncate_stop_markers


def _longest_marker_prefix_len(text: str, markers: Sequence[str]) -> int:
    if not markers or not text:
        return 0
    low = text.lower()
    best = 0
    for raw in markers:
        m = str(raw).lower()
        if not m:
            continue
        upper = min(len(m) - 1, len(low))
        for k in range(1, upper + 1):
            if low.endswith(m[:k]) and k > best:
                best = k
    return best


def stop_reason_name(value: Any) -> str:
    try:
        return _STOP_REASON_NAMES.get(int(value), "unknown")
    except (TypeError, ValueError):
        return "unknown"


# Legacy name (mirrors the ck_serve_v8.py helper).
_stop_reason_name = stop_reason_name


class SessionV8:
    """ctypes binding for the session ABI (one process-local model load)."""

    CK_ABI_VERSION = 1
    CK_OK = 0

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self.lib: Any = None
        self.session: Any = None
        self._batch2: Batch2SessionV8 | None = None

    @classmethod
    def open(
        cls,
        work_dir: Path,
        *,
        context_length: int | None = None,
        num_threads: int | None = None,
    ) -> "SessionV8":
        work_dir = work_dir.resolve()
        for name in ("libmodel.so", "weights.bump", "weights_manifest.map"):
            if not (work_dir / name).is_file():
                raise FileNotFoundError(f"missing runtime artifact {name} in {work_dir}")
        if not SESSION_LIB_PATH.is_file():
            raise RuntimeError(
                f"missing native session library {SESSION_LIB_PATH}; run `make ck-session-v8`"
            )

        lib = ctypes.CDLL(str(SESSION_LIB_PATH))
        _configure_lib(lib)

        cfg = _Config(
            struct_size=ctypes.sizeof(_Config),
            abi_version=cls.CK_ABI_VERSION,
            model_library_path=str(work_dir / "libmodel.so").encode(),
            weights_path=str(work_dir / "weights.bump").encode(),
            manifest_path=str(work_dir / "weights_manifest.map").encode(),
            context_length=int(context_length or 2048),
            num_threads=_resolve_num_threads(num_threads),
        )
        session = ctypes.c_void_p()
        status = lib.ck_session_v8_open(ctypes.byref(cfg), ctypes.byref(session))
        if status != cls.CK_OK or not session:
            msg = f"ck_session_v8_open failed ({_SESSION_STATUS_NAMES.get(status, f'status={status}')})"
            native_msg = _last_error(lib, session)
            if native_msg:
                msg += f": {native_msg}"
            exc_cls = _SESSION_STATUS_EXCEPTIONS.get(status, SessionError)
            raise exc_cls(status, msg)

        self = cls.__new__(cls)
        import threading

        self._lock = threading.Lock()
        self.lib = lib
        self.session = session
        self._batch2 = None
        return self

    def count_tokens(self, text: str) -> int:
        with self._lock:
            if self.session is None:
                raise SessionError(-6, "session is closed")
            count = self.lib.ck_session_v8_encode(self.session, (text or "").encode(), None, 0)
            if count < 0:
                msg = "ck_session_v8_encode failed while validating request capacity"
                native_msg = _last_error(self.lib, self.session)
                if native_msg:
                    msg += f": {native_msg}"
                exc_cls = _SESSION_STATUS_EXCEPTIONS.get(count, SessionError)
                raise exc_cls(count, msg)
            return int(count)

    def encode_ids(self, text: str) -> list[int]:
        """Return exact native token IDs for a fully rendered prompt."""
        payload = (text or "").encode()
        with self._lock:
            if self.session is None:
                raise SessionError(-6, "session is closed")
            count = self.lib.ck_session_v8_encode(self.session, payload, None, 0)
            if count < 0:
                _raise_native_error(self.lib, self.session, "ck_session_v8_encode", count)
            if count == 0:
                return []
            output = (ctypes.c_int32 * count)()
            written = self.lib.ck_session_v8_encode(self.session, payload, output, count)
            if written < 0:
                _raise_native_error(self.lib, self.session, "ck_session_v8_encode", written)
            if written != count:
                raise SessionError(-7, "native tokenizer count changed during encoding")
            return list(output)

    def enable_batch2(self, max_extra_bytes: int) -> Batch2SessionV8:
        """Reserve two native slots and reusable host logits within one budget.

        Enabling batch mode is irreversible for this loaded session and
        excludes whole-request generate. No HTTP route enables it yet.
        """
        if type(max_extra_bytes) is not int or not 0 <= max_extra_bytes <= ctypes.c_size_t(-1).value:
            raise ValueError("max_extra_bytes must fit size_t and be nonnegative")
        with self._lock:
            if self.session is None:
                raise SessionError(-6, "session is closed")
            if self._batch2 is not None:
                if max_extra_bytes < self._batch2.required_bytes:
                    raise SessionError(-8, f"batch2 requires {self._batch2.required_bytes} extra bytes; budget is {max_extra_bytes}")
                return self._batch2
            _configure_batch_abi(self.lib)
            descriptor = _ModelDescriptor()
            status = self.lib.ck_session_v8_get_model_descriptor(
                self.session, ctypes.byref(descriptor), ctypes.sizeof(descriptor))
            _raise_native_error(self.lib, self.session, "ck_session_v8_get_model_descriptor", status)
            if descriptor.vocab_size <= 0:
                raise SessionError(-5, "batch2 requires a positive loaded vocabulary size")
            needed = ctypes.c_size_t()
            status = self.lib.ck_session_v8_batch2_enable(
                self.session, 0, ctypes.byref(needed))
            if status == 0:
                raise SessionError(-6, "batch2 is already enabled outside this host owner")
            if status != -8:
                _raise_native_error(self.lib, self.session, "ck_session_v8_batch2_enable", status)
            host_bytes = 2 * descriptor.vocab_size * ctypes.sizeof(ctypes.c_float)
            total_bytes = needed.value + host_bytes
            if total_bytes > max_extra_bytes:
                raise SessionError(-8, f"batch2 requires {total_bytes} extra bytes "
                                   f"({needed.value} native + {host_bytes} logits); "
                                   f"budget is {max_extra_bytes}")
            # Allocate host outputs before enabling native mode, so a Python
            # allocation failure cannot strand an enabled session.
            batch = Batch2SessionV8(self, descriptor.vocab_size,
                                    total_bytes, needed.value)
            status = self.lib.ck_session_v8_batch2_enable(
                self.session, max_extra_bytes - host_bytes, ctypes.byref(needed))
            _raise_native_error(self.lib, self.session, "ck_session_v8_batch2_enable", status)
            self._batch2 = batch
            return self._batch2

    def generate(
        self,
        system: str | None,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        on_token: Callable[[int, str], int],
        flags: int = 0,
        stop_on_text: Sequence[str] = (),
        stop_at_eos: bool = False,
    ) -> dict[str, Any]:
        request = _GenerateRequest(
            struct_size=ctypes.sizeof(_GenerateRequest),
            abi_version=self.CK_ABI_VERSION,
            system_text=(system or "").encode(),
            user_text=(user or "").encode(),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            flags=int(flags),
        )
        result = _GenerateResult()
        result.struct_size = ctypes.sizeof(_GenerateResult)
        result.abi_version = self.CK_ABI_VERSION

        stop_markers = [str(m) for m in (stop_on_text or ()) if str(m)]
        if stop_at_eos:
            stop_markers.append("<eos>")

        emitted: list[str] = []
        visible_len = 0
        callback_stopped = False
        last_token_id = -1

        @_TOKEN_CALLBACK
        def callback(_user_data, token_id, text_bytes, text_len, _sequence_index):
            nonlocal callback_stopped, last_token_id, visible_len
            last_token_id = int(token_id)
            text = ctypes.string_at(text_bytes, text_len).decode("utf-8", "replace")
            emitted.append(text)
            joined = "".join(emitted)
            truncated = truncate_stop_markers(joined, stop_markers) if stop_markers else joined
            hit = len(truncated) < len(joined)
            if hit:
                delta = truncated[visible_len:]
            else:
                hold = _longest_marker_prefix_len(joined, stop_markers) if stop_markers else 0
                hold = min(hold, len(joined) - visible_len)
                target = len(joined) - hold
                delta = joined[visible_len:target] if target > visible_len else ""
            rc = 0
            if delta:
                try:
                    rc = int(on_token(int(token_id), delta) or 0)
                except Exception:
                    rc = 0
            else:
                try:
                    rc = int(on_token(int(token_id), "") or 0)
                except Exception:
                    rc = 0
                if rc == 0 and hit:
                    return 1
            visible_len += len(delta)
            if rc != 0:
                callback_stopped = True
                return rc
            if hit:
                return 1
            return 0

        status = self.lib.ck_session_v8_generate(
            self.session,
            ctypes.byref(request),
            callback,
            None,
            ctypes.byref(result),
        )
        if status == self.CK_OK and not callback_stopped:
            final_text = truncate_stop_markers("".join(emitted), stop_markers)
            pending = final_text[visible_len:]
            if pending:
                try:
                    on_token(last_token_id, pending)
                except Exception:
                    pass
        if status != self.CK_OK:
            msg = f"ck_session_v8_generate failed ({_SESSION_STATUS_NAMES.get(status, f'status={status}')})"
            native_msg = _last_error(self.lib, self.session)
            if native_msg:
                msg += f": {native_msg}"
            exc_cls = _SESSION_STATUS_EXCEPTIONS.get(status, SessionError)
            raise exc_cls(status, msg)
        return {
            "prompt_tokens": int(result.prompt_tokens),
            "generated_tokens": int(result.generated_tokens),
            "stop_reason": int(result.stop_reason),
            "prefill_time_ms": float(result.prefill_time_ms),
            "decode_time_ms": float(result.decode_time_ms),
        }

    def cancel(self) -> None:
        self.lib.ck_session_v8_cancel(self.session)

    def close(self) -> None:
        with self._lock:
            if self.session is not None:
                self.lib.ck_session_v8_close(self.session)
                self.session = None
