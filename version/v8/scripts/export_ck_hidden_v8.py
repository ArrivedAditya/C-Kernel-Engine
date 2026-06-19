#!/usr/bin/env python3
from __future__ import annotations

"""
Export CK hidden-vector snapshots for an explicit token prefix.

The generated model writes snapshots only when CK_DEBUG_EXPORT_HIDDEN points to
an output directory. This script sets that environment variable, replays tokens
through either sequential decode or batched prefill, and leaves raw float32
files in the output directory. It is tokenizer-free so it can be paired with
the deterministic logit parity runner.
"""

import argparse
import ctypes
import os
from pathlib import Path

from compare_ck_prefill_decode_logits_v8 import _extract_logits, _init_model  # type: ignore
from compare_first_token_logits_v8 import discover_ck_model_dir, parse_tokens_csv  # type: ignore


def export_ck_hidden(
    model_dir: Path,
    tokens: list[int],
    out_dir: Path,
    mode: str = "decode",
    prompt_length: int | None = None,
) -> None:
    model_dir = discover_ck_model_dir(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.f32"):
        old.unlink()
    os.environ["CK_DEBUG_EXPORT_HIDDEN"] = str(out_dir)

    lib_path = model_dir / "libmodel.so"
    lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    if hasattr(lib, "ck_model_embed_tokens"):
        lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        lib.ck_model_embed_tokens.restype = ctypes.c_int
    if hasattr(lib, "ck_model_forward"):
        lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_forward.restype = ctypes.c_int
    if hasattr(lib, "ck_model_kv_cache_reset"):
        lib.ck_model_kv_cache_reset.argtypes = []
        lib.ck_model_kv_cache_reset.restype = None
    if hasattr(lib, "ck_model_get_vocab_size"):
        lib.ck_model_get_vocab_size.argtypes = []
        lib.ck_model_get_vocab_size.restype = ctypes.c_int
    if hasattr(lib, "ck_model_free"):
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

    _init_model(lib, model_dir)

    try:
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        if mode == "decode":
            for tok in tokens:
                rc = lib.ck_model_decode(ctypes.c_int32(int(tok)), None)
                if rc != 0:
                    raise RuntimeError(f"ck_model_decode failed rc={rc} token={tok}")
        elif mode == "prefill":
            if not hasattr(lib, "ck_model_embed_tokens") or not hasattr(lib, "ck_model_forward"):
                raise RuntimeError("ck_model_embed_tokens/ck_model_forward unavailable")
            arr = (ctypes.c_int32 * len(tokens))(*[int(t) for t in tokens])
            rc = lib.ck_model_embed_tokens(arr, len(tokens))
            if rc != 0:
                raise RuntimeError(f"ck_model_embed_tokens failed rc={rc}")
            rc = lib.ck_model_forward(None)
            if rc != 0:
                raise RuntimeError(f"ck_model_forward failed rc={rc}")
        elif mode == "prefill-decode":
            if not hasattr(lib, "ck_model_embed_tokens") or not hasattr(lib, "ck_model_forward"):
                raise RuntimeError("ck_model_embed_tokens/ck_model_forward unavailable")
            if prompt_length is None:
                raise RuntimeError("--prompt-length is required for --mode prefill-decode")
            prompt_length = int(prompt_length)
            if prompt_length <= 0 or prompt_length > len(tokens):
                raise RuntimeError(f"invalid prompt_length={prompt_length} for {len(tokens)} tokens")
            arr = (ctypes.c_int32 * prompt_length)(*[int(t) for t in tokens[:prompt_length]])
            rc = lib.ck_model_embed_tokens(arr, prompt_length)
            if rc != 0:
                raise RuntimeError(f"ck_model_embed_tokens failed rc={rc}")
            rc = lib.ck_model_forward(None)
            if rc != 0:
                raise RuntimeError(f"ck_model_forward failed rc={rc}")
            for tok in tokens[prompt_length:]:
                rc = lib.ck_model_decode(ctypes.c_int32(int(tok)), None)
                if rc != 0:
                    raise RuntimeError(f"ck_model_decode failed rc={rc} token={tok}")
        else:
            raise ValueError(f"unknown mode: {mode}")

        if hasattr(lib, "ck_model_get_vocab_size"):
            vocab = int(lib.ck_model_get_vocab_size())
            if vocab > 0:
                logits, _stride, _active = _extract_logits(lib, vocab, len(tokens))
                token_indices = []
                for path in out_dir.glob("tok_*.f32"):
                    name = path.name
                    if name.startswith("tok_") and len(name) >= 8:
                        try:
                            token_indices.append(int(name[4:8]))
                        except ValueError:
                            pass
                ck_token_index = max(token_indices) if token_indices else 0
                logits.astype("float32", copy=False).tofile(out_dir / f"tok_{ck_token_index:04d}_layer_-01_logits.f32")
    finally:
        if hasattr(lib, "ck_model_free"):
            lib.ck_model_free()


def main() -> int:
    ap = argparse.ArgumentParser(description="Export CK hidden snapshots for explicit token IDs")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--tokens", required=True, help="comma-separated token IDs")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=("decode", "prefill", "prefill-decode"), default="decode")
    ap.add_argument("--prompt-length", type=int, help="number of initial tokens to prefill for --mode prefill-decode")
    args = ap.parse_args()

    export_ck_hidden(
        model_dir=args.model_dir,
        tokens=parse_tokens_csv(args.tokens),
        out_dir=args.out_dir,
        mode=args.mode,
        prompt_length=args.prompt_length,
    )
    files = sorted(args.out_dir.glob("*.f32"))
    print(f"exported={len(files)} out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
