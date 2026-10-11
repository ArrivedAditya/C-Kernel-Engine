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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from server.serving_bundle import verified_loaded_symbol_backing
from server.session_v8 import SESSION_LIB_PATH, SessionV8


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(part)
    return sha.hexdigest()


def certify(bundle: Path, *, threads: int = 4) -> dict:
    bundle = bundle.resolve()
    session = SessionV8.open(bundle, context_length=32, num_threads=threads)
    try:
        batch = session.enable_batch2(1 << 30)
        model = ctypes.CDLL(str(bundle / "libmodel.so"))
        model.ck_model_sequence_state_requirements.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        model.ck_model_sequence_state_requirements.restype = ctypes.c_int
        model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
        model.ck_model_get_named_activation_ptr.restype = ctypes.c_size_t
        model.ck_model_generated_source_sha256.restype = ctypes.c_char_p
        if hasattr(model, "ck_model_batch_decode_projection_groups"):
            model.ck_model_batch_decode_projection_groups.restype = ctypes.c_int
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

        def run(mixed: bool, prefill_slot: int) -> tuple[bytes, bytes, tuple, tuple]:
            decode_slot = prefill_slot ^ 1
            decode_ticket = batch.begin_prefill(decode_slot, 3)
            batch.append_prefill(decode_slot, decode_ticket, [100, 101, 102])
            prompt_ticket = batch.begin_prefill(prefill_slot, 2)
            batch.append_prefill(prefill_slot, prompt_ticket, [200])
            tokens = [0, 0]
            tickets = [0, 0]
            tokens[decode_slot], tokens[prefill_slot] = 103, 201
            tickets[decode_slot], tickets[prefill_slot] = decode_ticket, prompt_ticket
            if mixed:
                mask, consumed, _ = batch.mixed_step(
                    tuple(tokens), tuple(tickets), prefill_slot)
                if (mask, consumed) != (3, 2):
                    raise AssertionError("mixed execution did not advance both rows")
            else:
                decode_only = tokens.copy()
                decode_only[prefill_slot] = 0
                if batch.step(tuple(decode_only))[0] != 1 << decode_slot:
                    raise AssertionError("serialized decode advanced wrong slot")
                if batch.append_prefill(prefill_slot, prompt_ticket, [201]) != 2:
                    raise AssertionError("serialized prompt replay did not complete")
            result = (bytes(batch.staged_logits(0)), bytes(batch.staged_logits(1)),
                      kv(0), kv(1))
            batch.reset_slot(0)
            batch.reset_slot(1)
            return result

        cases = {}
        for prefill_slot in (0, 1):
            reference = run(False, prefill_slot)
            actual = run(True, prefill_slot)
            if any(a != b for a, b in zip(reference, actual)):
                raise AssertionError(f"mixed/serialized logits or complete KV differ for slot {prefill_slot}")
            cases[str(prefill_slot)] = {
                "full_logit_bytes_equal": True,
                "complete_kv_sha256_equal": True,
                "positions": [actual[2][0], actual[3][0]],
            }
        return {
            "status": "pass",
            "scope": "one decode row plus one staged prompt row; two slots; generated decode replay",
            "bundle": str(bundle),
            "loaded_sha256": loaded,
            "session_library_sha256": session_identity["sha256"],
            "generated_source_sha256": source_sha256.decode(),
            "weights_sha256": _digest(bundle / "weights.bump"),
            "shared_projection_groups": (
                model.ck_model_batch_decode_projection_groups()
                if hasattr(model, "ck_model_batch_decode_projection_groups") else None),
            "cases_by_prefill_slot": cases,
        }
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = certify(args.bundle, threads=args.threads)
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
