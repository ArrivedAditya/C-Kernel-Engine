"""Two-request token scheduler over the opt-in native batch2 session.

Inputs are already rendered and tokenized. The caller owns HTTP admission,
output transport, and worker lifetime. Prefill is synchronous and cannot yet
be interrupted through the native batch2 ABI; cancellation is effective at
decode-step boundaries. No route enables this scheduler by default.
"""

from __future__ import annotations

import ctypes
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

from .session_v8 import Batch2SessionV8, SessionError


@dataclass(frozen=True)
class TokenEvent:
    ticket: int
    slot: int
    token_id: int


@dataclass(frozen=True)
class CompletionEvent:
    ticket: int
    slot: int
    reason: str
    generated_tokens: int


@dataclass(frozen=True)
class BatchTick:
    tokens: tuple[TokenEvent, ...]
    completed: tuple[CompletionEvent, ...]
    advanced_mask: int | None


@dataclass
class _Request:
    slot: int
    ticket: int
    logits: Any
    max_tokens: int
    stop_ids: frozenset[int]
    temperature: float
    top_p: float
    rng: random.Random
    generated: int = 0
    cancel_requested: threading.Event = field(default_factory=threading.Event)


class Batch2RequestLoop:
    """One serialized worker with two independent request policies."""

    def __init__(self, batch: Batch2SessionV8):
        self.batch = batch
        self._operations = threading.Lock()
        self._state = threading.Lock()
        self._slots: dict[int, _Request] = {}
        self._tickets: dict[int, _Request] = {}
        self._poisoned = False
        sampler = getattr(batch.parent.lib, "ck_sample_top_p_v8", None)
        if sampler is None:
            raise SessionError(-5, "loaded session library lacks the explicit sampler ABI")
        sampler.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                            ctypes.c_float, ctypes.c_float, ctypes.c_float]
        sampler.restype = ctypes.c_int
        self._sample = sampler

    def submit(self, slot: int, prompt_ids: Sequence[int], *, max_tokens: int,
               stop_ids: Sequence[int] = (), temperature: float = 0.0,
               top_p: float = 1.0, seed: int = 0) -> int:
        if type(slot) is not int or slot not in (0, 1):
            raise ValueError("slot must be 0 or 1")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if not math.isfinite(temperature) or temperature < 0 or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("invalid sampling parameters")
        if any(type(token) is not int or token < 0 or token >= self.batch.vocab_size
               for token in stop_ids):
            raise ValueError("stop IDs must belong to the loaded vocabulary")
        context = self.batch.parent.context_length
        # Prefill predicts the first output token; only later outputs need
        # another decode position in the native cache.
        if context is not None and len(prompt_ids) + max_tokens - 1 > context:
            raise ValueError("prompt plus output budget exceeds loaded context")
        with self._operations:
            with self._state:
                if self._poisoned:
                    raise SessionError(-7, "batch request loop needs recovery")
                if slot in self._slots:
                    raise SessionError(-6, "batch slot already owns a request")
            try:
                ticket, logits = self.batch.prefill(slot, prompt_ids)
            except SessionError:
                self._fail_all_after_native_error()
                raise
            request = _Request(slot, ticket, logits, max_tokens,
                               frozenset(stop_ids), float(temperature),
                               float(top_p), random.Random(seed))
            with self._state:
                self._slots[slot] = request
                self._tickets[ticket] = request
            return ticket

    def request_cancel(self, ticket: int) -> bool:
        """May run concurrently with native decode; stale tickets are inert."""
        with self._state:
            request = self._tickets.get(ticket)
            if request is None:
                return False
            request.cancel_requested.set()
        self.batch.request_cancel(ticket)
        return True

    def _retire(self, request: _Request, reason: str) -> CompletionEvent:
        self.batch.reset_slot(request.slot)
        with self._state:
            self._slots.pop(request.slot, None)
            self._tickets.pop(request.ticket, None)
        return CompletionEvent(request.ticket, request.slot, reason,
                               request.generated)

    def _fail_all_after_native_error(self) -> tuple[CompletionEvent, ...]:
        with self._state:
            active = tuple(self._slots.values())
            self._poisoned = True
        recovered = True
        for slot in (0, 1):
            try:
                self.batch.reset_slot(slot)
            except SessionError:
                recovered = False
        with self._state:
            self._slots.clear()
            self._tickets.clear()
            self._poisoned = not recovered
        return tuple(CompletionEvent(r.ticket, r.slot, "runtime_error",
                                     r.generated) for r in active)

    def advance(self) -> BatchTick:
        """Sample one token per active request, then decode continuing rows."""
        with self._operations:
            with self._state:
                if self._poisoned:
                    raise SessionError(-7, "batch request loop needs recovery")
                active = tuple(self._slots[i] for i in sorted(self._slots))
            emitted: list[TokenEvent] = []
            completed: list[CompletionEvent] = []
            decode_inputs = [0, 0]
            pending_mask = 0
            try:
                for request in active:
                    if request.cancel_requested.is_set():
                        completed.append(self._retire(request, "cancelled"))
                        continue
                    # The sampler ABI takes float32; a double just below one
                    # can round to 1.0 and violate its [0, 1) contract.
                    draw = min(request.rng.random(), 1.0 - 2.0 ** -24) if request.temperature > 0 else 0.0
                    token = int(self._sample(request.logits, self.batch.vocab_size,
                                             request.temperature, request.top_p,
                                             draw))
                    if token < 0 or token >= self.batch.vocab_size:
                        raise SessionError(-7, "native sampler returned an invalid token")
                    if token in request.stop_ids:
                        completed.append(self._retire(request, "eos"))
                        continue
                    request.generated += 1
                    emitted.append(TokenEvent(request.ticket, request.slot, token))
                    if request.generated >= request.max_tokens:
                        completed.append(self._retire(request, "token_limit"))
                    else:
                        decode_inputs[request.slot] = token
                        pending_mask |= 1 << request.slot
                advanced = 0
                if pending_mask:
                    advanced, _ = self.batch.step(tuple(decode_inputs))
                    if advanced & ~pending_mask:
                        raise SessionError(-7, "native step advanced an unscheduled slot")
                    if advanced != pending_mask:
                        with self._state:
                            missing = [self._slots[i] for i in (0, 1)
                                       if (pending_mask & (1 << i)) and not (advanced & (1 << i))]
                        if any(not r.cancel_requested.is_set() for r in missing):
                            raise SessionError(-7, "native step omitted an active slot")
                        for request in missing:
                            completed.append(self._retire(request, "cancelled"))
                with self._state:
                    newly_cancelled = tuple(r for r in self._slots.values()
                                            if r.cancel_requested.is_set())
                for request in newly_cancelled:
                    completed.append(self._retire(request, "cancelled"))
                return BatchTick(tuple(emitted), tuple(completed), advanced)
            except SessionError:
                completed.extend(self._fail_all_after_native_error())
                # A failing native step may have advanced neither, one, or
                # both slots. The ABI explicitly leaves advancement unknown.
                return BatchTick(tuple(emitted), tuple(completed), None)

    def active_tickets(self) -> tuple[int, ...]:
        with self._state:
            return tuple(r.ticket for _, r in sorted(self._slots.items()))

    def recover(self) -> None:
        """Retry both native resets after a failed cleanup; never reuse early."""
        with self._operations:
            with self._state:
                if self._slots:
                    raise SessionError(-6, "cannot recover while requests are active")
            failed = False
            for slot in (0, 1):
                try:
                    self.batch.reset_slot(slot)
                except SessionError:
                    failed = True
            with self._state:
                self._poisoned = failed
            if failed:
                raise SessionError(-7, "batch request loop recovery did not reset both slots")
