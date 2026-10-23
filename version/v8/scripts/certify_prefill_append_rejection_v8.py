#!/usr/bin/env python3
"""Prove a grouped-row external append rejects before changing generated state.

Run separately from full/chunked equivalence: this checks the unsupported
boundary and reset/reuse, while compare_chunked_prefill_v8.py checks accepted
aligned appends against uninterrupted logits and KV.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

from compare_chunked_prefill_v8 import (
    _cpu_model, _identity, _kv_only_state_contract, _loaded_identity,
    _sha256, _source_revision, _tokens,
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _state(lib: ctypes.CDLL, bundle: Path, layers: int, vocab: int) -> dict:
    position = int(lib.ck_model_get_active_tokens())
    stride = int(lib.ck_model_get_logits_stride())
    if position <= 0 or (stride and stride < vocab):
        raise RuntimeError("invalid generated logit extent")
    logits = lib.ck_model_get_logits()
    if not logits:
        raise RuntimeError("missing generated logits")
    offset = (position - 1) * stride if stride else 0
    logits_bytes = ctypes.string_at(
        ctypes.cast(logits, ctypes.c_void_p).value + offset * 4, vocab * 4)
    kv = []
    with tempfile.TemporaryDirectory() as directory:
        for layer in range(layers):
            path = Path(directory) / f"kv-{layer}.bin"
            if lib.ck_model_debug_export_kv(str(path).encode(), layer) != 0:
                raise RuntimeError(f"KV export failed at layer {layer}")
            kv.append(_sha256(path))
    return {"position": position, "logits_sha256": _digest(logits_bytes),
            "kv_sha256": kv}


def certify(bundle: Path, tokens_path: Path, split_at: int) -> dict:
    tokens = _tokens(tokens_path)
    if not 0 < split_at < len(tokens):
        raise ValueError("split position must be inside the prompt")
    _kv_only_state_contract(bundle)
    identity = _identity(bundle)
    lib = ctypes.CDLL(str(bundle / "libmodel.so"), mode=ctypes.RTLD_GLOBAL)
    loaded = _loaded_identity(lib, bundle)
    if any(loaded[name] != identity[name] for name in loaded):
        raise RuntimeError("loaded library identity differs from bundle")
    int_ptr = ctypes.POINTER(ctypes.c_int32)
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [int_ptr, ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_prefill_append_row_group.restype = ctypes.c_int
    lib.ck_model_get_active_tokens.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_get_logits_stride.restype = ctypes.c_int
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_debug_export_kv.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.ck_model_debug_export_kv.restype = ctypes.c_int
    lib.ck_model_free.restype = None
    group = int(lib.ck_model_prefill_append_row_group())
    if group <= 1 or split_at % group == 0:
        raise ValueError("split must violate the generated grouped-row boundary")
    if lib.ck_model_init(str(bundle / "weights.bump").encode()) != 0:
        raise RuntimeError("generated model initialization failed")
    try:
        vocab = int(lib.ck_model_get_vocab_size())
        layers = int(json.loads((bundle / "config.json").read_text())["num_hidden_layers"])

        def embed(values: list[int]) -> int:
            return int(lib.ck_model_embed_tokens(
                (ctypes.c_int32 * len(values))(*values), len(values)))

        if embed(tokens[:split_at]) != 0:
            raise RuntimeError("initial prefill failed")
        before = _state(lib, bundle, layers, vocab)
        rejection = embed(tokens[split_at:])
        after = _state(lib, bundle, layers, vocab)
        state_unchanged = before == after
        lib.ck_model_kv_cache_reset()
        aligned = split_at - split_at % group
        retry = [embed(tokens[:aligned]), embed(tokens[aligned:])]
        reused_position = int(lib.ck_model_get_active_tokens())
        passed = (rejection == -3 and state_unchanged and retry == [0, 0]
                  and reused_position == len(tokens))
        return {
            "schema": "cke.prefill_append_rejection.v1",
            "status": "pass" if passed else "fail",
            "source_revision": _source_revision(),
            "bundle_identity": identity, "loaded_libraries": loaded,
            "token_sha256": _sha256(tokens_path), "token_count": len(tokens),
            "split_at": split_at, "row_group": group,
            "rejection_status": rejection, "state_unchanged": state_unchanged,
            "before": before, "after": after,
            "aligned_retry_split": aligned, "aligned_retry_statuses": retry,
            "reused_position": reused_position,
            "machine": platform.machine(), "cpu_model": _cpu_model(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "ck_num_threads": os.environ.get("CK_NUM_THREADS"),
        }
    finally:
        lib.ck_model_free()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--split-at", type=int, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    report = certify(args.bundle.resolve(), args.tokens.resolve(), args.split_at)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "row_group": report["row_group"],
                      "rejection_status": report["rejection_status"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
