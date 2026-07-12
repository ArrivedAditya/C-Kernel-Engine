#!/usr/bin/env python3
"""PyTorch eager oracle for full attention with BF16 storage boundaries."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
KERNEL = LIB.attention_forward_full_head_major_gqa_flash_strided_bf16_storage
FLOAT_P = ctypes.POINTER(ctypes.c_float)
KERNEL.argtypes = [
    FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
KERNEL.restype = None


def bf16_values(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


def run_case(heads: int, tokens: int, dim: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    q = bf16_values(rng.standard_normal((heads, tokens, dim), dtype=np.float32))
    k = bf16_values(rng.standard_normal((heads, tokens, dim), dtype=np.float32))
    v = bf16_values(rng.standard_normal((heads, tokens, dim), dtype=np.float32))
    actual = np.empty_like(q)
    KERNEL(
        q.ctypes.data_as(FLOAT_P), k.ctypes.data_as(FLOAT_P),
        v.ctypes.data_as(FLOAT_P), actual.ctypes.data_as(FLOAT_P),
        heads, heads, tokens, dim, dim, tokens,
    )
    tq = torch.from_numpy(q).to(torch.bfloat16)
    tk = torch.from_numpy(k).to(torch.bfloat16)
    tv = torch.from_numpy(v).to(torch.bfloat16)
    scores = torch.matmul(tq, tk.transpose(-2, -1)) * (1.0 / math.sqrt(dim))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    expected = torch.matmul(probs, tv).float().numpy()
    diff = np.abs(actual - expected)
    return float(diff.max(initial=0.0)), float(np.sqrt(np.mean(diff * diff)))


def main() -> int:
    cases = [(2, 7, 8, 1), (2, 17, 72, 2), (4, 33, 72, 3)]
    for heads, tokens, dim, seed in cases:
        max_abs, rmse = run_case(heads, tokens, dim, seed)
        if max_abs > 0.03125 or rmse > 0.004:
            raise AssertionError(
                f"BF16 attention mismatch H={heads} T={tokens} D={dim}: "
                f"max_abs={max_abs:.9g} rmse={rmse:.9g}"
            )
        print(f"H={heads} T={tokens} D={dim} max_abs={max_abs:.9g} rmse={rmse:.9g}")
    print(f"BF16 full-attention storage parity: {len(cases)}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
