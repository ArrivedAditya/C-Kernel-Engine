#!/usr/bin/env python3
from __future__ import annotations

"""
Export CK hidden-vector snapshots for an explicit token prefix.

The generated model writes snapshots only when CK_DEBUG_EXPORT_HIDDEN points to
an output directory. This script sets that environment variable, replays tokens
through sequential decode, and leaves raw float32 files in the output directory.
It is tokenizer-free so it can be paired with the deterministic logit parity
runner.
"""

import argparse
import ctypes
import os
from pathlib import Path

from compare_first_token_logits_v8 import discover_ck_model_dir, parse_tokens_csv  # type: ignore


def export_ck_hidden(model_dir: Path, tokens: list[int], out_dir: Path) -> None:
    model_dir = discover_ck_model_dir(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CK_DEBUG_EXPORT_HIDDEN"] = str(out_dir)

    lib_path = model_dir / "libmodel.so"
    lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    if hasattr(lib, "ck_model_kv_cache_reset"):
        lib.ck_model_kv_cache_reset.argtypes = []
        lib.ck_model_kv_cache_reset.restype = None
    if hasattr(lib, "ck_model_free"):
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

    init_candidates = [model_dir / "weights.bump", model_dir]
    init_ok = False
    init_errors: list[str] = []
    for init_dir in init_candidates:
        rc = lib.ck_model_init(str(init_dir.resolve()).encode("utf-8"))
        if rc == 0:
            init_ok = True
            break
        init_errors.append(f"{init_dir}:rc={rc}")
    if not init_ok:
        raise RuntimeError(f"ck_model_init failed ({', '.join(init_errors)})")

    try:
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        for tok in tokens:
            rc = lib.ck_model_decode(ctypes.c_int32(int(tok)), None)
            if rc != 0:
                raise RuntimeError(f"ck_model_decode failed rc={rc} token={tok}")
    finally:
        if hasattr(lib, "ck_model_free"):
            lib.ck_model_free()


def main() -> int:
    ap = argparse.ArgumentParser(description="Export CK hidden snapshots for explicit token IDs")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--tokens", required=True, help="comma-separated token IDs")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    export_ck_hidden(
        model_dir=args.model_dir,
        tokens=parse_tokens_csv(args.tokens),
        out_dir=args.out_dir,
    )
    files = sorted(args.out_dir.glob("*.f32"))
    print(f"exported={len(files)} out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
