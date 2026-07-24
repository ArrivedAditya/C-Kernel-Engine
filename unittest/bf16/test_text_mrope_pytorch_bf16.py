#!/usr/bin/env python3
"""Exact PyTorch BF16 parity for text M-RoPE storage boundaries."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so")))
KERNEL = LIB.mrope_qk_text_imrope_bf16_pytorch_storage
KERNEL.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)] + [
    ctypes.c_int
] * 12 + [ctypes.c_float] * 6
KERNEL.restype = None


def reference(source: torch.Tensor, position: int, base: float) -> torch.Tensor:
    head_dim = int(source.shape[-1])
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    frequencies = inv_freq * float(position)
    embedding = torch.cat((frequencies, frequencies))
    cosine = embedding.cos().to(torch.bfloat16)
    sine = embedding.sin().to(torch.bfloat16)
    value = source.to(torch.bfloat16)
    rotated = torch.cat((-value[..., head_dim // 2 :], value[..., : head_dim // 2]), dim=-1)
    return ((value * cosine) + (rotated * sine)).float()


def run_case(position: int, tokens: int, seed: int) -> None:
    heads = 32
    kv_heads = 8
    head_dim = 128
    base = 5_000_000.0
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(heads, tokens, head_dim, generator=generator, dtype=torch.bfloat16)
    k = torch.randn(kv_heads, tokens, head_dim, generator=generator, dtype=torch.bfloat16)
    expected_q = torch.stack([reference(q[:, tok], position + tok, base) for tok in range(tokens)], dim=1).numpy()
    expected_k = torch.stack([reference(k[:, tok], position + tok, base) for tok in range(tokens)], dim=1).numpy()
    actual_q = q.float().numpy().copy()
    actual_k = k.float().numpy().copy()

    KERNEL(
        actual_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        actual_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        heads, kv_heads, tokens, head_dim, head_dim, position, head_dim,
        24, 20, 20, 0, 262144,
        base, 1.0, 0.0, 1.0, 0.0, 0.0,
    )
    if not np.array_equal(actual_q, expected_q) or not np.array_equal(actual_k, expected_k):
        raise AssertionError(
            f"PyTorch BF16 text M-RoPE mismatch position={position} tokens={tokens} "
            f"q_max={float(np.max(np.abs(actual_q - expected_q)))} "
            f"k_max={float(np.max(np.abs(actual_k - expected_k)))}"
        )


if __name__ == "__main__":
    run_case(0, 4, 7401)
    run_case(40, 14, 7402)
    run_case(106, 1, 7403)
    print("PyTorch BF16 text M-RoPE parity: 3/3 exact")
