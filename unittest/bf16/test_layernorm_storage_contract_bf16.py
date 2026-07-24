#!/usr/bin/env python3
"""PyTorch oracle for float-buffer LayerNorm with BF16 storage boundaries."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(os.environ.get("CK_ENGINE_SO", ROOT / "build" / "libckernel_engine.so")).resolve()
LIB = ctypes.CDLL(str(LIB_PATH))
KERNEL = LIB.layernorm_naive_serial_bf16_storage
PYTORCH_KERNEL = LIB.layernorm_pytorch_welford_bf16_storage
FLOAT_P = ctypes.POINTER(ctypes.c_float)
ARGTYPES = [
    FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
KERNEL.argtypes = ARGTYPES
KERNEL.restype = None
PYTORCH_KERNEL.argtypes = ARGTYPES
PYTORCH_KERNEL.restype = None


def bf16_values(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


def run_case(tokens: int, dim: int, eps: float, seed: int, *, kernel=KERNEL) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = bf16_values(rng.standard_normal((tokens, dim), dtype=np.float32))
    gamma = bf16_values(rng.standard_normal(dim, dtype=np.float32))
    beta = bf16_values(rng.standard_normal(dim, dtype=np.float32))
    actual = np.empty_like(x)
    mean = np.empty(tokens, dtype=np.float32)
    rstd = np.empty(tokens, dtype=np.float32)
    kernel(
        x.ctypes.data_as(FLOAT_P),
        gamma.ctypes.data_as(FLOAT_P),
        beta.ctypes.data_as(FLOAT_P),
        actual.ctypes.data_as(FLOAT_P),
        mean.ctypes.data_as(FLOAT_P),
        rstd.ctypes.data_as(FLOAT_P),
        tokens,
        dim,
        eps,
    )
    expected = torch.nn.functional.layer_norm(
        torch.from_numpy(x).to(torch.bfloat16),
        (dim,),
        torch.from_numpy(gamma).to(torch.bfloat16),
        torch.from_numpy(beta).to(torch.bfloat16),
        eps,
    ).float().numpy()
    diff = np.abs(actual - expected)
    return float(diff.max(initial=0.0)), float(np.sqrt(np.mean(diff * diff)))


def main() -> int:
    cases = [(3, 8, 1e-5, 1), (4, 72, 1e-6, 2), (2, 1152, 1e-6, 3)]
    for tokens, dim, eps, seed in cases:
        max_abs, rmse = run_case(tokens, dim, eps, seed)
        if max_abs > 0.03125 or rmse > 0.003:
            raise AssertionError(
                f"LayerNorm BF16 storage mismatch T={tokens} D={dim}: "
                f"max_abs={max_abs:.9g} rmse={rmse:.9g}"
            )
        print(f"T={tokens} D={dim} max_abs={max_abs:.9g} rmse={rmse:.9g}")
    for seed in (3, 107, 211):
        max_abs, rmse = run_case(4032, 1152, 1e-6, seed, kernel=PYTORCH_KERNEL)
        if max_abs != 0.0 or rmse != 0.0:
            raise AssertionError(
                "PyTorch Welford LayerNorm is not byte-exact at the production shape: "
                f"seed={seed} max_abs={max_abs:.9g} rmse={rmse:.9g}"
            )
        print(
            f"T=4032 D=1152 seed={seed} pytorch_welford "
            f"max_abs={max_abs:.9g} rmse={rmse:.9g}"
        )
    print(f"BF16 LayerNorm storage contract parity: {len(cases)}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
