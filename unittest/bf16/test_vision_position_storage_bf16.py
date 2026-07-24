#!/usr/bin/env python3
"""Exact BF16 storage-boundary parity for tiled 2D position interpolation."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so")))
LIB.position_embeddings_add_tiled_2d_align_corners_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
LIB.position_embeddings_add_tiled_2d_align_corners_bf16.restype = None


def tile_order_indices(grid_h: int, grid_w: int, merge_size: int) -> torch.Tensor:
    values = []
    for tile_y in range(0, grid_h, merge_size):
        for tile_x in range(0, grid_w, merge_size):
            for dy in range(merge_size):
                for dx in range(merge_size):
                    values.append((tile_y + dy) * grid_w + tile_x + dx)
    return torch.tensor(values, dtype=torch.long)


def reference(x: torch.Tensor, table: torch.Tensor, grid_h: int, grid_w: int, merge_size: int) -> torch.Tensor:
    source = int(round(table.shape[0] ** 0.5))
    ys = torch.linspace(0, source - 1, grid_h, dtype=torch.float32)
    xs = torch.linspace(0, source - 1, grid_w, dtype=torch.float32)
    y0 = ys.to(torch.long); x0 = xs.to(torch.long)
    y1 = (y0 + 1).clamp(max=source - 1); x1 = (x0 + 1).clamp(max=source - 1)
    dy = ys - y0; dx = xs - x0
    weights = [
        ((1 - dy)[:, None] * (1 - dx)[None, :]).to(torch.bfloat16),
        ((1 - dy)[:, None] * dx[None, :]).to(torch.bfloat16),
        (dy[:, None] * (1 - dx)[None, :]).to(torch.bfloat16),
        (dy[:, None] * dx[None, :]).to(torch.bfloat16),
    ]
    table = table.to(torch.bfloat16).view(source, source, -1)
    terms = [
        table[y0[:, None], x0[None, :]] * weights[0][..., None],
        table[y0[:, None], x1[None, :]] * weights[1][..., None],
        table[y1[:, None], x0[None, :]] * weights[2][..., None],
        table[y1[:, None], x1[None, :]] * weights[3][..., None],
    ]
    interpolated = (terms[0] + terms[1] + terms[2] + terms[3]).reshape(grid_h * grid_w, -1)
    ordered = interpolated.index_select(0, tile_order_indices(grid_h, grid_w, merge_size))
    return (x.to(torch.bfloat16) + ordered).to(torch.float32)


def run_case(grid_h: int, grid_w: int, source: int, dim: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(grid_h * grid_w, dim, generator=generator, dtype=torch.float32)
    table = torch.randn(source * source, dim, generator=generator, dtype=torch.float32).to(torch.bfloat16).float()
    expected = reference(x, table, grid_h, grid_w, 2).numpy()
    actual = x.numpy().copy()
    LIB.position_embeddings_add_tiled_2d_align_corners_bf16(
        actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        table.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grid_h, grid_w, dim, 2, source,
    )
    if not np.array_equal(actual, expected):
        diff = np.abs(actual - expected)
        index = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(
            f"BF16 position storage mismatch shape={grid_h}x{grid_w} dim={dim} "
            f"max_abs={float(diff[index])} index={index} got={actual[index]} ref={expected[index]}"
        )


if __name__ == "__main__":
    run_case(6, 10, 4, 8, 4411)
    run_case(28, 36, 32, 16, 4412)
    run_case(36, 28, 32, 72, 4413)
    run_case(56, 72, 32, 1152, 4414)
    print("BF16 tiled position storage parity: 4/4 exact")
