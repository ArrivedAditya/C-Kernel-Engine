#!/usr/bin/env python3
"""PyTorch oracle for tanh GELU with BF16 storage boundaries."""
from __future__ import annotations
import ctypes
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
KERNEL = LIB.gelu_pytorch_tanh_bf16_storage
KERNEL.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]

def main() -> int:
    rng = np.random.default_rng(17)
    for shape in [(7,), (4, 72), (3, 4304)]:
        source = rng.standard_normal(shape, dtype=np.float32) * 3.0
        source = torch.from_numpy(source).to(torch.bfloat16).float().numpy()
        actual = source.copy()
        KERNEL(actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), actual.size)
        expected = torch.nn.functional.gelu(
            torch.from_numpy(source).to(torch.bfloat16), approximate="tanh"
        ).float().numpy()
        diff = np.abs(actual - expected)
        max_abs = float(diff.max(initial=0.0))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        if max_abs > 0.03125 or rmse > 0.001:
            raise AssertionError(f"shape={shape} max_abs={max_abs} rmse={rmse}")
        print(f"shape={shape} max_abs={max_abs:.9g} rmse={rmse:.9g}")
    print("BF16 PyTorch-tanh GELU storage parity: 3/3")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
