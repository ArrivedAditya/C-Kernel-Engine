#!/usr/bin/env python3
"""Exact PyTorch BF16 parity for positions-aware interleaved text M-RoPE."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so")))
KERNEL = LIB.mrope_qk_text_imrope_positions_bf16_pytorch_storage
KERNEL.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int32),
] + [ctypes.c_int] * 11 + [ctypes.c_float] * 6
KERNEL.restype = None


def reference(source: torch.Tensor, positions: torch.Tensor, sections: list[int], base: float) -> torch.Tensor:
    head_dim = int(source.shape[-1])
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = inv_freq[None, None, :, None] @ positions[:, None, None, :].float()
    freqs = freqs.transpose(2, 3)
    merged = freqs[0].clone()
    for axis in (1, 2):
        merged[..., slice(axis, sections[axis] * 3, 3)] = freqs[
            axis, ..., slice(axis, sections[axis] * 3, 3)
        ]
    embedding = torch.cat((merged, merged), dim=-1)
    cosine = embedding.cos().to(torch.bfloat16)
    sine = embedding.sin().to(torch.bfloat16)
    value = source.to(torch.bfloat16)
    rotated = torch.cat((-value[..., head_dim // 2 :], value[..., : head_dim // 2]), dim=-1)
    return ((value * cosine) + (rotated * sine)).float()


def run_case(tokens: int, seed: int) -> None:
    heads = 4
    kv_heads = 2
    head_dim = 128
    sections = [24, 20, 20]
    base = 5_000_000.0
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(heads, tokens, head_dim, generator=generator, dtype=torch.bfloat16)
    k = torch.randn(kv_heads, tokens, head_dim, generator=generator, dtype=torch.bfloat16)
    positions = torch.arange(tokens, dtype=torch.int32).repeat(3, 1)
    if tokens >= 8:
        positions[1, 2:6] = torch.tensor([40, 40, 41, 41], dtype=torch.int32)
        positions[2, 2:6] = torch.tensor([50, 51, 50, 51], dtype=torch.int32)
    positions4 = torch.cat((positions, torch.zeros(1, tokens, dtype=torch.int32)), dim=0)
    expected_q = reference(q, positions, sections, base).numpy()
    expected_k = reference(k, positions, sections, base).numpy()
    actual_q = q.float().numpy().copy()
    actual_k = k.float().numpy().copy()
    positions_np = np.ascontiguousarray(positions4.numpy())

    KERNEL(
        actual_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        actual_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        positions_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        heads, kv_heads, tokens, head_dim, head_dim, head_dim,
        sections[0], sections[1], sections[2], 0, 262144,
        base, 1.0, 0.0, 1.0, 0.0, 0.0,
    )
    if not np.array_equal(actual_q, expected_q) or not np.array_equal(actual_k, expected_k):
        raise AssertionError(
            f"positions-aware PyTorch BF16 M-RoPE mismatch tokens={tokens} "
            f"q_max={float(np.max(np.abs(actual_q - expected_q)))} "
            f"k_max={float(np.max(np.abs(actual_k - expected_k)))}"
        )


if __name__ == "__main__":
    run_case(8, 7501)
    run_case(1026, 7502)
    print("PyTorch BF16 positions-aware text M-RoPE parity: 2/2 exact")
