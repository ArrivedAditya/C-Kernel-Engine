#!/usr/bin/env python3
"""Capture full logits and logical KV at a prefill/chunk boundary.

Run once for a full-prefill bundle and once for a chunked bundle built from
the same weights. This is an artifact-backed diagnostic, not a parity verdict.
"""

import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(lib: c.CDLL, bundle: Path, output: Path, stage: str) -> dict:
    vocab = lib.ck_model_get_vocab_size()
    stride = lib.ck_model_get_logits_stride()
    active = lib.ck_model_get_active_tokens()
    pointer = lib.ck_model_get_logits()
    if not pointer or vocab <= 0:
        raise RuntimeError("generated model did not return logits")
    if stride > 0 and active > 0:
        raw = np.ctypeslib.as_array(pointer, shape=(active * stride,))
        values = raw[(active - 1) * stride:(active - 1) * stride + vocab]
    else:
        values = np.ctypeslib.as_array(pointer, shape=(vocab,))
    logits = values.astype("<f4", copy=True)
    if not np.isfinite(logits).all():
        raise RuntimeError("nonfinite generated logits")
    (output / f"{stage}.f32").write_bytes(logits.tobytes())

    layers = int(json.loads((bundle / "config.json").read_text())["num_hidden_layers"])
    kv_hashes = []
    with tempfile.TemporaryDirectory() as scratch:
        for layer in range(layers):
            path = Path(scratch) / f"kv-{layer}.bin"
            result = lib.ck_model_debug_export_kv(str(path).encode(), layer)
            if result:
                raise RuntimeError(f"logical KV export failed at layer {layer}: {result}")
            kv_hashes.append(sha256(path))
    return {
        "top": int(np.argmax(logits)),
        "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
        "kv_sha256": kv_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.length < 1:
        parser.error("--length must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    lib = c.CDLL(str(args.bundle / "libmodel.so"), mode=c.RTLD_GLOBAL)
    lib.ck_model_init.argtypes = [c.c_char_p]
    lib.ck_model_init.restype = c.c_int
    lib.ck_model_embed_tokens.argtypes = [c.POINTER(c.c_int32), c.c_int]
    lib.ck_model_embed_tokens.restype = c.c_int
    lib.ck_model_forward.argtypes = [c.POINTER(c.c_float)]
    lib.ck_model_forward.restype = c.c_int
    lib.ck_model_decode.argtypes = [c.c_int32, c.POINTER(c.c_float)]
    lib.ck_model_decode.restype = c.c_int
    lib.ck_model_get_logits.restype = c.POINTER(c.c_float)
    lib.ck_model_get_vocab_size.restype = c.c_int
    lib.ck_model_get_logits_stride.restype = c.c_int
    lib.ck_model_get_active_tokens.restype = c.c_int
    lib.ck_model_debug_export_kv.argtypes = [c.c_char_p, c.c_int]
    lib.ck_model_debug_export_kv.restype = c.c_int
    lib.ck_model_free.restype = None
    if lib.ck_model_init(str(args.bundle / "weights.bump").encode()):
        raise RuntimeError("generated model initialization failed")
    try:
        token_ids = [9707 + position % 17 for position in range(args.length)]
        tokens = (c.c_int32 * args.length)(*token_ids)
        if lib.ck_model_embed_tokens(tokens, args.length):
            raise RuntimeError("embedding failed")
        if lib.ck_model_forward(None):
            raise RuntimeError("prefill failed")
        prefill = capture(lib, args.bundle, args.output, "prefill")
        if lib.ck_model_decode(9707 + args.length % 17, None):
            raise RuntimeError("one-token continuation failed")
        continuation = capture(lib, args.bundle, args.output, "continuation")
    finally:
        lib.ck_model_free()
    report = {
        "schema": "cke.issue833.boundary_capture.v1",
        "bundle": str(args.bundle),
        "length": args.length,
        "token_recipe": "9707 + position % 17",
        "model_sha256": sha256(args.bundle / "libmodel.so"),
        "engine_sha256": sha256(args.bundle / "libckernel_engine.so"),
        "prefill": prefill,
        "continuation": continuation,
        "status": "DIAGNOSTIC_NOT_CERTIFIED",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"length": args.length, "prefill_top": prefill["top"],
                      "continuation_top": continuation["top"]}))


if __name__ == "__main__":
    main()
