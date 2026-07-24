#!/usr/bin/env python3
"""Exact PyTorch oracle for Qwen3-VL BF16 image patch projection."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(
    os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so"))
)
KERNEL = LIB.patch_projection_image_bf16_pytorch_onednn_conv3d_storage
FLOAT_P = ctypes.POINTER(ctypes.c_float)
U16_P = ctypes.POINTER(ctypes.c_uint16)
KERNEL.argtypes = [
    FLOAT_P, U16_P, U16_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
KERNEL.restype = None


def merge_major_indices(grid_h: int, grid_w: int, merge: int) -> torch.Tensor:
    indices: list[int] = []
    for tile_y in range(0, grid_h, merge):
        for tile_x in range(0, grid_w, merge):
            for dy in range(merge):
                for dx in range(merge):
                    indices.append((tile_y + dy) * grid_w + tile_x + dx)
    return torch.tensor(indices, dtype=torch.long)


def run_case(grid_h: int, grid_w: int, out_channels: int, seed: int) -> None:
    channels, temporal, patch, merge = 3, 2, 14, 2
    image_h, image_w = grid_h * patch, grid_w * patch
    rng = np.random.default_rng(seed)
    image = rng.standard_normal((channels, image_h, image_w), dtype=np.float32)
    w0_t = torch.from_numpy(
        rng.standard_normal(
            (out_channels, channels, patch, patch), dtype=np.float32
        )
    ).to(torch.bfloat16)
    w1_t = torch.from_numpy(
        rng.standard_normal(
            (out_channels, channels, patch, patch), dtype=np.float32
        )
    ).to(torch.bfloat16)
    bias = rng.standard_normal(out_channels, dtype=np.float32)
    output = np.empty((grid_h * grid_w, out_channels), dtype=np.float32)

    w0 = w0_t.view(torch.uint16).numpy()
    w1 = w1_t.view(torch.uint16).numpy()
    KERNEL(
        image.ctypes.data_as(FLOAT_P),
        w0.ctypes.data_as(U16_P),
        w1.ctypes.data_as(U16_P),
        bias.ctypes.data_as(FLOAT_P),
        output.ctypes.data_as(FLOAT_P),
        channels, image_h, image_w, patch, out_channels, merge,
    )

    image_t = torch.from_numpy(image).to(torch.bfloat16).unsqueeze(0)
    unfolded = F.unfold(image_t, kernel_size=patch, stride=patch)
    row_major = (
        unfolded.reshape(1, channels, patch, patch, grid_h * grid_w)
        .squeeze(0)
        .permute(3, 0, 1, 2)
    )
    source = row_major.unsqueeze(2).expand(
        -1, -1, temporal, -1, -1
    ).contiguous()
    weight = torch.stack((w0_t, w1_t), dim=2)
    expected = F.conv3d(
        source,
        weight,
        torch.from_numpy(bias).to(torch.bfloat16),
    ).reshape(grid_h * grid_w, out_channels)
    expected = expected.index_select(
        0, merge_major_indices(grid_h, grid_w, merge)
    ).float().numpy()

    if not np.array_equal(output, expected):
        mismatch = np.flatnonzero(output.view(np.uint32) != expected.view(np.uint32))
        index = int(mismatch[0])
        diff = np.abs(output - expected)
        raise AssertionError(
            f"patch projection grid={grid_h}x{grid_w} N={out_channels}: "
            f"{mismatch.size}/{output.size} differ; first={index} "
            f"max_abs={float(diff.max())}"
        )
    print(
        f"grid={grid_h}x{grid_w} N={out_channels}: "
        f"{output.size}/{output.size} byte-exact"
    )


def main() -> int:
    torch.set_num_threads(int(os.environ.get("CK_NUM_THREADS", "20")))
    run_case(4, 6, 32, 20260723)
    run_case(8, 8, 1152, 20260724)
    if os.environ.get("CK_BF16_FULL_SHAPES") == "1":
        run_case(56, 72, 1152, 20260725)
    print("Qwen3-VL BF16 patch projection parity: exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
