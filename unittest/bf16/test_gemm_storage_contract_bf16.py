#!/usr/bin/env python3
"""PyTorch oracle for BF16 GEMM with BF16 output storage."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
KERNEL = LIB.gemm_nt_bf16_bf16_storage
FLOAT_P = ctypes.POINTER(ctypes.c_float)
UINT16_P = ctypes.POINTER(ctypes.c_uint16)
KERNEL.argtypes = [
    FLOAT_P, ctypes.c_void_p, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
KERNEL.restype = None


def bf16_bits(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).view(torch.uint16).numpy()


def bf16_values(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


def run_case_detailed(m: int, n: int, k: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = bf16_values(rng.standard_normal((m, k), dtype=np.float32))
    b_values = bf16_values(rng.standard_normal((n, k), dtype=np.float32))
    b = bf16_bits(b_values)
    bias = bf16_values(rng.standard_normal(n, dtype=np.float32))
    actual = np.empty((m, n), dtype=np.float32)
    KERNEL(
        a.ctypes.data_as(FLOAT_P),
        ctypes.c_void_p(b.ctypes.data),
        bias.ctypes.data_as(FLOAT_P),
        actual.ctypes.data_as(FLOAT_P),
        m, n, k,
    )
    expected = torch.nn.functional.linear(
        torch.from_numpy(a).to(torch.bfloat16),
        torch.from_numpy(b_values).to(torch.bfloat16),
        torch.from_numpy(bias).to(torch.bfloat16),
    ).float().numpy()
    diff = np.abs(actual - expected)
    return {
        "max_abs": float(diff.max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "exact_ratio": float(np.count_nonzero(diff == 0.0) / diff.size),
        "different_outputs": int(np.count_nonzero(diff)),
        "output_count": int(diff.size),
    }


def run_case(m: int, n: int, k: int, seed: int) -> tuple[float, float]:
    metrics = run_case_detailed(m, n, k, seed)
    return metrics["max_abs"], metrics["rmse"]


def main() -> int:
    cases = [(2, 24, 16, 1), (3, 144, 72, 2), (1, 3456, 1152, 3)]
    for m, n, k, seed in cases:
        max_abs, rmse = run_case(m, n, k, seed)
        if max_abs > 0.5 or rmse > 0.03:
            raise AssertionError(
                f"BF16 GEMM storage mismatch M={m} N={n} K={k}: "
                f"max_abs={max_abs:.9g} rmse={rmse:.9g}"
            )
        print(f"M={m} N={n} K={k} max_abs={max_abs:.9g} rmse={rmse:.9g}")
    print(f"BF16 GEMM output storage parity: {len(cases)}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
