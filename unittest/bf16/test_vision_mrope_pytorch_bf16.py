#!/usr/bin/env python3
"""PyTorch/MKL-exact BF16 Qwen3-VL vision M-RoPE parity."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so")))
try:
    KERNEL = LIB.mrope_qk_vision_bf16_pytorch_storage
except AttributeError as exc:
    print("SKIP: PyTorch-exact vision M-RoPE requires a USE_MKL engine")
    raise SystemExit(0) from exc

KERNEL.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int32),
] + [ctypes.c_int] * 11 + [ctypes.c_float] * 6
KERNEL.restype = None


def merge_major_positions(grid_h: int, grid_w: int, merge: int) -> np.ndarray:
    positions: list[tuple[int, int]] = []
    for block_y in range(grid_h // merge):
        for block_x in range(grid_w // merge):
            for dy in range(merge):
                for dx in range(merge):
                    positions.append((block_y * merge + dy, block_x * merge + dx))
    return np.ascontiguousarray(np.asarray(positions, dtype=np.int32).T)


def reference(x: torch.Tensor, positions: np.ndarray, grid_h: int, grid_w: int) -> torch.Tensor:
    head_dim = x.shape[-1]
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    table = torch.outer(torch.arange(max(grid_h, grid_w), dtype=torch.float32), inv_freq)
    frequencies = table[torch.from_numpy(positions.T.astype(np.int64))].flatten(1)
    embedding = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(0)
    source = x.float()
    rotated = torch.cat((-source[..., head_dim // 2 :], source[..., : head_dim // 2]), dim=-1)
    return ((source * embedding.cos()) + (rotated * embedding.sin())).to(torch.bfloat16).float()


def run_case(grid_h: int, grid_w: int, heads: int, head_dim: int, seed: int) -> None:
    tokens = grid_h * grid_w
    positions = merge_major_positions(grid_h, grid_w, 2)
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(heads, tokens, head_dim, generator=generator, dtype=torch.bfloat16)
    k = torch.randn(heads, tokens, head_dim, generator=generator, dtype=torch.bfloat16)
    expected_q = reference(q, positions, grid_h, grid_w).numpy()
    expected_k = reference(k, positions, grid_h, grid_w).numpy()
    actual_q = q.float().numpy().copy()
    actual_k = k.float().numpy().copy()
    axis_pairs = head_dim // 4

    KERNEL(
        actual_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        actual_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        heads, heads, tokens, head_dim, head_dim, head_dim,
        axis_pairs, axis_pairs, 0, 0, 32768,
        10000.0, 1.0, 0.0, 1.0, 32.0, 1.0,
    )

    if not np.array_equal(actual_q, expected_q) or not np.array_equal(actual_k, expected_k):
        q_diff = np.abs(actual_q - expected_q)
        k_diff = np.abs(actual_k - expected_k)
        raise AssertionError(
            f"PyTorch BF16 vision M-RoPE mismatch grid={grid_h}x{grid_w} "
            f"q_max={float(q_diff.max())} k_max={float(k_diff.max())}"
        )


if __name__ == "__main__":
    run_case(4, 6, 2, 72, 7021)
    run_case(56, 72, 2, 72, 7022)
    print("PyTorch/MKL BF16 vision M-RoPE parity: 2/2 exact")
