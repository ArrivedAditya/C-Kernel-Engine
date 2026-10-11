#!/usr/bin/env python3
"""Compare one generated mixed row pair with serialized decode replay.

This local, artifact-specific check does not certify arbitrary prefill chunks,
variable batch size, HTTP concurrency, or the model's ordinary prefill entry.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from server.serving_bundle import verified_loaded_symbol_backing
from server.session_v8 import SESSION_LIB_PATH, SessionV8

CAP_BATCH_TWO_ROWS = 1 << 17


def _finite_logit_bytes(values: ctypes.Array, label: str) -> bytes:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise AssertionError(f"{label} has nonfinite logit at index {index}")
    return bytes(values)


def _require_shared_projections(capabilities: int, groups: int | None) -> int:
    if not capabilities & CAP_BATCH_TWO_ROWS:
        raise AssertionError("loaded model lacks the two-row batch capability")
    if groups is None or groups <= 0:
        raise AssertionError("loaded model has no shared projection groups")
    return groups


def _cpu_identity() -> dict:
    model = platform.processor() or platform.machine()
    flags = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
            if line.startswith(("flags", "Features")):
                flags.update(line.partition(":")[2].split())
            if model and flags:
                break
    except OSError:
        pass
    relevant = ("avx", "avx2", "fma", "avx512f", "avx512_vnni", "asimd", "sve")
    return {"model": model, "machine": platform.machine(),
            "isa_flags": [flag for flag in relevant if flag in flags]}


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(part)
    return sha.hexdigest()


def certify(bundle: Path, *, threads: int = 4, context_length: int = 32) -> dict:
    bundle = bundle.resolve()
    if threads <= 0 or context_length < 16:
        raise ValueError("certification requires positive threads and at least 16 context tokens")
    session = SessionV8.open(bundle, context_length=context_length, num_threads=threads)
    try:
        batch = session.enable_batch2(1 << 30)
        model = ctypes.CDLL(str(bundle / "libmodel.so"))
        model.ck_model_sequence_state_requirements.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        model.ck_model_sequence_state_requirements.restype = ctypes.c_int
        model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
        model.ck_model_get_named_activation_ptr.restype = ctypes.c_size_t
        model.ck_model_generated_source_sha256.restype = ctypes.c_char_p
        model.ck_model_get_capabilities.restype = ctypes.c_uint64
        model.ck_model_get_context_window.restype = ctypes.c_int
        if not hasattr(model, "ck_model_batch_decode_projection_groups"):
            raise AssertionError("loaded model lacks shared-projection count")
        model.ck_model_batch_decode_projection_groups.restype = ctypes.c_int
        capabilities = int(model.ck_model_get_capabilities())
        projection_groups = _require_shared_projections(
            capabilities, int(model.ck_model_batch_decode_projection_groups()))
        kv_size = ctypes.c_size_t()
        kv_alignment = ctypes.c_size_t()
        if model.ck_model_sequence_state_requirements(
                ctypes.byref(kv_size), ctypes.byref(kv_alignment)) != 0 or not kv_size.value:
            raise AssertionError("loaded model lacks sequence-state extent")

        loaded = {}
        for name, symbol in (("libmodel.so", "ck_model_decode_batch2"),
                             ("libckernel_engine.so", "gemm_nt_q5_1_q8_1_m2")):
            identity = verified_loaded_symbol_backing(model, symbol, bundle / name)
            if identity["sha256"] != _digest(bundle / name):
                raise AssertionError(f"loaded {name} differs from its bundle file")
            loaded[name] = identity["sha256"]
        session_identity = verified_loaded_symbol_backing(
            session.lib, "ck_session_v8_batch2_mixed_step", SESSION_LIB_PATH)
        source = (bundle / "model_v8.c").read_bytes()
        marker = b'\n\nCK_EXPORT const char *ck_model_generated_source_sha256(void) {\n    return "'
        prefix, found, tail = source.rpartition(marker)
        source_sha256 = model.ck_model_generated_source_sha256()
        if (not found or not source_sha256 or
                tail != source_sha256 + b'";\n}\n' or
                hashlib.sha256(prefix).hexdigest().encode() != source_sha256):
            raise AssertionError("loaded generated source identity mismatch")

        def kv(slot: int) -> tuple[int, str]:
            position = batch.position(slot)  # Activates this sequence's KV arena.
            pointer = model.ck_model_get_named_activation_ptr(b"kv_cache")
            if not pointer:
                raise AssertionError("active KV arena is unavailable")
            data = ctypes.string_at(pointer, kv_size.value)
            return position, hashlib.sha256(data).hexdigest()

        def run(mixed: bool, prefill_slot: int, *, decode_prompt: tuple[int, ...],
                prompt: tuple[int, ...], decode_tokens: tuple[int, ...],
                continuation: tuple[int, int]) -> tuple[tuple, tuple, tuple]:
            decode_slot = prefill_slot ^ 1
            if len(decode_tokens) != len(prompt) - 1:
                raise AssertionError("fixture must pair every remaining prompt token")
            decode_ticket = batch.begin_prefill(decode_slot, len(decode_prompt))
            batch.append_prefill(decode_slot, decode_ticket, decode_prompt)
            prompt_ticket = batch.begin_prefill(prefill_slot, len(prompt))
            batch.append_prefill(prefill_slot, prompt_ticket, prompt[:1])
            tickets = [0, 0]
            tickets[decode_slot], tickets[prefill_slot] = decode_ticket, prompt_ticket
            steps = []
            for index, (decode_token, prompt_token) in enumerate(
                    zip(decode_tokens, prompt[1:], strict=True), start=2):
                tokens = [0, 0]
                tokens[decode_slot], tokens[prefill_slot] = decode_token, prompt_token
                if mixed:
                    mask, consumed, _ = batch.mixed_step(
                        tuple(tokens), tuple(tickets), prefill_slot)
                    if (mask, consumed) != (3, index):
                        raise AssertionError("mixed execution did not advance both rows")
                else:
                    decode_only = tokens.copy()
                    decode_only[prefill_slot] = 0
                    if batch.step(tuple(decode_only))[0] != 1 << decode_slot:
                        raise AssertionError("serialized decode advanced wrong slot")
                    if batch.append_prefill(prefill_slot, prompt_ticket, [prompt_token]) != index:
                        raise AssertionError("serialized prompt replay did not advance")
                steps.append(tuple(_finite_logit_bytes(batch.staged_logits(slot),
                                                       f"{'mixed' if mixed else 'reference'} step {index} slot {slot}")
                                   for slot in (0, 1)))
            continued = [0, 0]
            continued[decode_slot], continued[prefill_slot] = continuation
            if batch.step(tuple(continued))[0] != 3:
                raise AssertionError("post-prefill continuation did not advance both slots")
            steps.append(tuple(_finite_logit_bytes(batch.staged_logits(slot),
                                                   f"{'mixed' if mixed else 'reference'} continuation slot {slot}")
                               for slot in (0, 1)))
            result = (tuple(steps), kv(0), kv(1))
            batch.reset_slot(0)
            batch.reset_slot(1)
            return result

        cases = {}
        fixtures = {
            "unequal": ((100, 101, 102, 103, 104, 105, 106, 107, 108),
                        (200, 201, 202, 203, 204, 205, 206),
                        (109, 110, 111, 112, 113, 114), (115, 207)),
            "near_capacity": ((100,) * (context_length - 4),
                              (200, 201, 202, 203), (105, 106, 107), (108, 204)),
        }
        for name, (decode_prompt, prompt, decode_tokens, continuation) in fixtures.items():
            for prefill_slot in (0, 1):
                parameters = {"decode_prompt": decode_prompt, "prompt": prompt,
                              "decode_tokens": decode_tokens, "continuation": continuation}
                reference = run(False, prefill_slot, **parameters)
                actual = run(True, prefill_slot, **parameters)
                if actual != reference:
                    raise AssertionError(f"mixed/serialized full logits, complete KV, or positions differ in {name} slot {prefill_slot}")
                cases[f"{name}/prefill_slot_{prefill_slot}"] = {
                    "mixed_steps": len(decode_tokens),
                    "continuation_steps": 1,
                    "full_finite_logit_bytes_equal_at_every_step": True,
                    "complete_kv_sha256_equal": True,
                    "positions": [actual[1][0], actual[2][0]],
                }
        return {
            "status": "pass",
            "scope": "one decode row plus one staged prompt row; two slots; generated decode-replay parity only",
            "bundle": str(bundle),
            "loaded_sha256": loaded,
            "session_library_sha256": session_identity["sha256"],
            "generated_source_sha256": source_sha256.decode(),
            "weights_sha256": _digest(bundle / "weights.bump"),
            "capabilities": capabilities,
            "shared_projection_groups": projection_groups,
            "session_context_length": session.context_length,
            "generated_context_length": int(model.ck_model_get_context_window()),
            "threads": threads,
            "cpu": _cpu_identity(),
            "kv_bytes_per_sequence": kv_size.value,
            "cases_by_prefill_slot": cases,
        }
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = certify(args.bundle, threads=args.threads,
                     context_length=args.context_length)
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
