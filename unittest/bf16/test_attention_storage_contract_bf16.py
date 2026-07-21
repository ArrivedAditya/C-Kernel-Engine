#!/usr/bin/env python3
"""PyTorch CPU SDPA oracle for full attention with BF16 storage boundaries."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
KERNEL = LIB.attention_forward_full_head_major_gqa_sdpa_bf16_storage
FLOAT_P = ctypes.POINTER(ctypes.c_float)
KERNEL.argtypes = [
    FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
KERNEL.restype = None


def bf16_values(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


def run_case_detailed(
    heads: int,
    kv_heads: int,
    tokens: int,
    dim: int,
    aligned_dim: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    q = bf16_values(rng.standard_normal((heads, tokens, dim), dtype=np.float32))
    k = bf16_values(rng.standard_normal((kv_heads, tokens, dim), dtype=np.float32))
    v = bf16_values(rng.standard_normal((kv_heads, tokens, dim), dtype=np.float32))
    q_padded = np.zeros((heads, tokens, aligned_dim), dtype=np.float32)
    k_padded = np.zeros((kv_heads, tokens, aligned_dim), dtype=np.float32)
    v_padded = np.zeros((kv_heads, tokens, aligned_dim), dtype=np.float32)
    actual_padded = np.full_like(q_padded, np.nan)
    q_padded[..., :dim] = q
    k_padded[..., :dim] = k
    v_padded[..., :dim] = v
    KERNEL(
        q_padded.ctypes.data_as(FLOAT_P), k_padded.ctypes.data_as(FLOAT_P),
        v_padded.ctypes.data_as(FLOAT_P), actual_padded.ctypes.data_as(FLOAT_P),
        heads, kv_heads, tokens, dim, aligned_dim, tokens,
    )
    tq = torch.from_numpy(q).to(torch.bfloat16)
    tk = torch.from_numpy(k).to(torch.bfloat16)
    tv = torch.from_numpy(v).to(torch.bfloat16)
    expected = torch.nn.functional.scaled_dot_product_attention(
        tq,
        tk,
        tv,
        enable_gqa=heads != kv_heads,
    ).float().numpy()
    actual = actual_padded[..., :dim]
    diff = np.abs(actual - expected)
    padding = actual_padded[..., dim:]
    padding_max = float(np.abs(padding).max(initial=0.0))
    different_outputs = int(np.count_nonzero(actual != expected))
    output_count = int(actual.size)
    return {
        "max_abs": float(diff.max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "padding_max": padding_max,
        "exact_ratio": float((output_count - different_outputs) / output_count),
        "different_outputs": different_outputs,
        "output_count": output_count,
    }


def run_case(
    heads: int,
    kv_heads: int,
    tokens: int,
    dim: int,
    aligned_dim: int,
    seed: int,
) -> tuple[float, float, float]:
    metrics = run_case_detailed(
        heads, kv_heads, tokens, dim, aligned_dim, seed
    )
    return metrics["max_abs"], metrics["rmse"], metrics["padding_max"]


def main() -> int:
    cases = [
        (2, 2, 7, 8, 8, 1),
        (2, 2, 17, 72, 72, 2),
        (4, 4, 33, 72, 72, 3),
        (4, 2, 19, 72, 80, 4),
    ]
    for heads, kv_heads, tokens, dim, aligned_dim, seed in cases:
        metrics = run_case_detailed(
            heads, kv_heads, tokens, dim, aligned_dim, seed
        )
        if (
            metrics["max_abs"] > 0.03125
            or metrics["rmse"] > 0.004
            or metrics["padding_max"] != 0.0
        ):
            raise AssertionError(
                f"BF16 attention mismatch H={heads} KV={kv_heads} T={tokens} "
                f"D={dim} A={aligned_dim}: max_abs={metrics['max_abs']:.9g} "
                f"rmse={metrics['rmse']:.9g} "
                f"padding_max={metrics['padding_max']:.9g}"
            )
        print(
            f"H={heads} KV={kv_heads} T={tokens} D={dim} A={aligned_dim} "
            f"max_abs={metrics['max_abs']:.9g} rmse={metrics['rmse']:.9g} "
            f"exact={metrics['exact_ratio']:.9%} "
            f"different={metrics['different_outputs']}/{metrics['output_count']} "
            f"padding_max={metrics['padding_max']:.9g}"
        )
    print(f"BF16 full-attention storage parity: {len(cases)}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
