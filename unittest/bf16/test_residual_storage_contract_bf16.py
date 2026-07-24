#!/usr/bin/env python3
"""Exact PyTorch oracle for float-buffer BF16 residual addition."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(
    os.environ.get("CK_ENGINE_SO")
    or os.environ.get("CK_ENGINE_LIB")
    or ROOT / "build" / "libckernel_engine.so"
)
LIB = ctypes.CDLL(str(LIB_PATH))
KERNEL = LIB.ck_residual_add_token_major_bf16_storage
FLOAT_P = ctypes.POINTER(ctypes.c_float)
KERNEL.argtypes = [FLOAT_P, FLOAT_P, FLOAT_P, ctypes.c_int, ctypes.c_int]
KERNEL.restype = None


def main() -> int:
    rng = np.random.default_rng(7)
    cases = ((3, 8), (4, 72), (2, 1152), (1008, 4096))
    for tokens, dim in cases:
        a_t = torch.from_numpy(rng.standard_normal((tokens, dim), dtype=np.float32)).to(torch.bfloat16)
        b_t = torch.from_numpy(rng.standard_normal((tokens, dim), dtype=np.float32)).to(torch.bfloat16)
        a = a_t.float().numpy()
        b = b_t.float().numpy()
        actual = np.empty_like(a)
        KERNEL(a.ctypes.data_as(FLOAT_P), b.ctypes.data_as(FLOAT_P),
               actual.ctypes.data_as(FLOAT_P), tokens, dim)
        expected = (a_t + b_t).float().numpy()
        if not np.array_equal(actual, expected):
            diff = np.abs(actual - expected)
            raise AssertionError(
                f"BF16 residual mismatch T={tokens} D={dim}: max={diff.max()}"
            )
        in_place = a.copy()
        KERNEL(in_place.ctypes.data_as(FLOAT_P), b.ctypes.data_as(FLOAT_P),
               in_place.ctypes.data_as(FLOAT_P), tokens, dim)
        if not np.array_equal(in_place, expected):
            diff = np.abs(in_place - expected)
            raise AssertionError(
                f"BF16 in-place residual mismatch T={tokens} D={dim}: max={diff.max()}"
            )
        print(f"T={tokens} D={dim} exact (out-of-place and in-place)")
    print(f"BF16 residual storage parity: {len(cases)}/{len(cases)} exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
