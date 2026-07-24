#!/usr/bin/env python3
"""Exact PyTorch oracle for the Qwen3-VL BF16 embedding provider."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(
    os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so"))
)
KERNEL = LIB.embedding_forward_bf16_fp32
I32_P = ctypes.POINTER(ctypes.c_int32)
U16_P = ctypes.POINTER(ctypes.c_uint16)
FLOAT_P = ctypes.POINTER(ctypes.c_float)
KERNEL.argtypes = [
    I32_P, ctypes.c_int, ctypes.c_int, U16_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
KERNEL.restype = None


def main() -> int:
    rng = np.random.default_rng(20260723)
    vocab, tokens, context = 17, 5, 8
    embed_dim, aligned_dim = 4096, 4160
    weights_t = torch.from_numpy(
        rng.standard_normal((vocab, aligned_dim), dtype=np.float32)
    ).to(torch.bfloat16)
    weights = weights_t.view(torch.uint16).numpy()
    positions = rng.standard_normal((context, aligned_dim), dtype=np.float32)
    token_ids = np.array([3, 16, -1, vocab, 7], dtype=np.int32)

    for add_pos in (0, 1):
        actual = np.full((context, aligned_dim), np.nan, dtype=np.float32)
        KERNEL(
            token_ids.ctypes.data_as(I32_P),
            tokens,
            vocab,
            weights.ctypes.data_as(U16_P),
            positions.ctypes.data_as(FLOAT_P),
            actual.ctypes.data_as(FLOAT_P),
            embed_dim,
            aligned_dim,
            context,
            add_pos,
        )
        safe_ids = torch.tensor([3, 16, 0, 0, 7], dtype=torch.long)
        expected = np.zeros_like(actual)
        selected = weights_t.index_select(0, safe_ids).float().numpy()
        expected[:tokens, :embed_dim] = selected[:, :embed_dim]
        if add_pos:
            expected[:tokens, :embed_dim] += positions[:tokens, :embed_dim]
        if not np.array_equal(actual, expected):
            mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
            index = int(mismatch[0])
            raise AssertionError(
                f"BF16 embedding add_pos={add_pos}: "
                f"{mismatch.size}/{actual.size} FP32 values differ; first={index}"
            )
        print(
            f"add_pos={add_pos}: {actual.size}/{actual.size} "
            "byte-exact FP32 outputs"
        )

    print("Qwen3-VL BF16 embedding parity: 2/2 exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
