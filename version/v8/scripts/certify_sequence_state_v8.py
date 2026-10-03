#!/usr/bin/env python3
"""Compare two serialized generated-model sequences with isolated execution.

This certifies state switching, not batched arithmetic or concurrent calls.
Run against a freshly generated, KV-only text decoder bundle.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path


SEQUENCE_SWITCH_CAPABILITY = 1 << 16
TOKENS_A = (100, 101, 102, 103)
TOKENS_B = (200, 201)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def certify(run: Path) -> dict:
    library = run / "libmodel.so"
    weights = run / "weights.bump"
    layout = run / "layout_decode.json"
    config = run / "config.json"
    for path in (library, weights, layout, config):
        if not path.is_file():
            raise ValueError(f"missing generated bundle asset: {path}")

    model = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    model.ck_model_get_capabilities.argtypes = []
    model.ck_model_get_capabilities.restype = ctypes.c_uint64
    caps = int(model.ck_model_get_capabilities())
    if not caps & SEQUENCE_SWITCH_CAPABILITY:
        raise ValueError("generated model does not declare sequence-state switching")
    model.ck_model_init.argtypes = [ctypes.c_char_p]
    model.ck_model_init.restype = ctypes.c_int
    model.ck_model_free.argtypes = []
    model.ck_model_get_vocab_size.argtypes = []
    model.ck_model_get_vocab_size.restype = ctypes.c_int
    model.ck_model_get_context_window.argtypes = []
    model.ck_model_get_context_window.restype = ctypes.c_int
    model.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    model.ck_model_decode.restype = ctypes.c_int
    model.ck_model_kv_cache_reset.argtypes = []
    model.ck_model_sequence_state_default.argtypes = []
    model.ck_model_sequence_state_default.restype = ctypes.c_uint64
    model.ck_model_sequence_state_requirements.argtypes = [
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
    ]
    model.ck_model_sequence_state_requirements.restype = ctypes.c_int
    model.ck_model_sequence_state_create.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)
    ]
    model.ck_model_sequence_state_create.restype = ctypes.c_int
    model.ck_model_sequence_state_activate.argtypes = [ctypes.c_uint64]
    model.ck_model_sequence_state_activate.restype = ctypes.c_int
    model.ck_model_sequence_state_destroy.argtypes = [ctypes.c_uint64]
    model.ck_model_sequence_state_destroy.restype = ctypes.c_int
    model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
    model.ck_model_get_named_activation_ptr.restype = ctypes.c_size_t
    model.ck_model_get_named_activation_runtime_offset.argtypes = [ctypes.c_char_p]
    model.ck_model_get_named_activation_runtime_offset.restype = ctypes.c_ssize_t

    if model.ck_model_init(str(weights).encode()) != 0:
        raise RuntimeError("generated model initialization failed")
    try:
        vocab = int(model.ck_model_get_vocab_size())
        context = int(model.ck_model_get_context_window())
        if vocab <= max(*TOKENS_A, *TOKENS_B):
            raise ValueError("vocabulary is too small for the pinned token fixture")
        if context < max(len(TOKENS_A), len(TOKENS_B)):
            raise ValueError("compiled context is too short for the pinned token fixture")
        output = (ctypes.c_float * vocab)()

        def decode(token: int) -> str:
            if model.ck_model_decode(token, output) != 0:
                raise RuntimeError(f"generated decode failed for token {token}")
            if any(not math.isfinite(value) for value in output):
                raise AssertionError(f"nonfinite logits after token {token}")
            raw = ctypes.string_at(output, ctypes.sizeof(output))
            return hashlib.sha256(raw).hexdigest()

        isolated: dict[str, list[str]] = {}
        for name, tokens in (("a", TOKENS_A), ("b", TOKENS_B)):
            model.ck_model_kv_cache_reset()
            isolated[name] = [decode(token) for token in tokens]

        default = model.ck_model_sequence_state_default()
        if not default:
            raise RuntimeError("default sequence handle is unavailable")
        default_kv = model.ck_model_get_named_activation_ptr(b"kv_cache")
        if not default_kv or model.ck_model_get_named_activation_runtime_offset(b"kv_cache") < 0:
            raise AssertionError("default KV activation lookup is inconsistent")
        kv_bytes, kv_alignment = ctypes.c_size_t(), ctypes.c_size_t()
        if model.ck_model_sequence_state_requirements(
            ctypes.byref(kv_bytes), ctypes.byref(kv_alignment)
        ) != 0 or kv_alignment.value != 64:
            raise RuntimeError("invalid generated sequence-arena requirements")

        def arena() -> tuple[ctypes.Array, ctypes.c_void_p]:
            storage = ctypes.create_string_buffer(kv_bytes.value + kv_alignment.value - 1)
            address = ctypes.addressof(storage)
            aligned = (address + kv_alignment.value - 1) & ~(kv_alignment.value - 1)
            return storage, ctypes.c_void_p(aligned)

        arena_a, arena_a_ptr = arena()
        arena_b, arena_b_ptr = arena()
        a, b, c = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
        if model.ck_model_sequence_state_create(arena_a_ptr, kv_bytes, ctypes.byref(a)) != 0:
            raise RuntimeError("failed to create sequence A")
        if model.ck_model_sequence_state_create(arena_b_ptr, kv_bytes, ctypes.byref(b)) != 0:
            raise RuntimeError("failed to create sequence B")
        try:
            schedule = (
                ("a", a, TOKENS_A[0], 0),
                ("b", b, TOKENS_B[0], 0),
                ("a", a, TOKENS_A[1], 1),
                ("b", b, TOKENS_B[1], 1),
                ("a", a, TOKENS_A[2], 2),
            )
            for name, handle, token, index in schedule:
                if model.ck_model_sequence_state_activate(handle) != 0:
                    raise RuntimeError(f"failed to activate sequence {name}")
                if (
                    model.ck_model_get_named_activation_ptr(b"kv_cache") == default_kv
                    or model.ck_model_get_named_activation_runtime_offset(b"kv_cache") != -1
                ):
                    raise AssertionError("non-default KV activation lookup is inconsistent")
                if decode(token) != isolated[name][index]:
                    raise AssertionError(f"sequence {name} diverged at token {index}")

            if model.ck_model_sequence_state_activate(b) != 0:
                raise RuntimeError("failed to activate sequence B for reset")
            model.ck_model_kv_cache_reset()
            if model.ck_model_sequence_state_activate(a) != 0:
                raise RuntimeError("resetting B corrupted the A handle")
            if decode(TOKENS_A[3]) != isolated["a"][3]:
                raise AssertionError("resetting B changed A's continuation")
            if model.ck_model_sequence_state_activate(b) != 0:
                raise RuntimeError("failed to reactivate reset sequence B")
            if decode(TOKENS_B[0]) != isolated["b"][0]:
                raise AssertionError("reset sequence B retained its old state")

            if model.ck_model_sequence_state_activate(default) != 0:
                raise RuntimeError("failed to restore the default sequence")
            retired_b = b.value
            if model.ck_model_sequence_state_destroy(b) != 0:
                raise RuntimeError("failed to retire sequence B")
            if model.ck_model_sequence_state_activate(retired_b) != -1:
                raise AssertionError("retired handle was accepted")
            b = ctypes.c_uint64()
            if model.ck_model_sequence_state_create(arena_b_ptr, kv_bytes, ctypes.byref(c)) != 0:
                raise RuntimeError("retired sequence slot could not be reused")
            if model.ck_model_sequence_state_activate(c) != 0:
                raise RuntimeError("failed to activate reused slot")
            if model.ck_model_sequence_state_activate(retired_b) != -1:
                raise AssertionError("reused slot accepted stale handle")
            if decode(TOKENS_B[0]) != isolated["b"][0]:
                raise AssertionError("reused slot retained state from its prior owner")
        finally:
            model.ck_model_sequence_state_activate(default)
            for handle in (a, b, c):
                if handle.value:
                    model.ck_model_sequence_state_destroy(handle)
    finally:
        model.ck_model_free()

    if model.ck_model_init(str(weights).encode()) != 0:
        raise RuntimeError("generated model reload failed")
    try:
        if model.ck_model_sequence_state_activate(default) != -1:
            raise AssertionError("prior model-load handle was accepted")
        if model.ck_model_sequence_state_default() == default:
            raise AssertionError("model reload reused the default handle")
    finally:
        model.ck_model_free()

    return {
        "schema": "cke.sequence-state-serialized-v1",
        "status": "pass",
        "scope": "serialized_kv_only_not_batched",
        "compiled_context_length": context,
        "tokens_a": list(TOKENS_A),
        "tokens_b": list(TOKENS_B),
        "exact_logit_rows": len(TOKENS_A) + len(TOKENS_B) + 2,
        "library_sha256": _sha256(library),
        "weights_sha256": _sha256(weights),
        "layout_sha256": _sha256(layout),
        "config_sha256": _sha256(config),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = certify(args.run.resolve())
    body = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(body, encoding="utf-8")
    print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
