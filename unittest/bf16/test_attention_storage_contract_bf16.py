#!/usr/bin/env python3
"""PyTorch CPU SDPA oracle for full attention with BF16 storage boundaries."""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(os.environ.get("CK_ENGINE_SO", ROOT / "build" / "libckernel_engine.so")).resolve()
LIB = ctypes.CDLL(str(LIB_PATH))
KERNEL = LIB.attention_forward_full_head_major_gqa_sdpa_bf16_storage
PYTORCH_FLASH_KERNEL = (
    LIB.attention_forward_full_head_major_gqa_pytorch_cpu_flash_bf16_storage
)
FLOAT_P = ctypes.POINTER(ctypes.c_float)
KERNEL.argtypes = [
    FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
KERNEL.restype = None
PYTORCH_FLASH_KERNEL.argtypes = KERNEL.argtypes
PYTORCH_FLASH_KERNEL.restype = None
SCALE = LIB.ck_attention_pytorch_sdpa_scale_f32
SCALE.argtypes = [ctypes.c_int]
SCALE.restype = ctypes.c_float
SET_THREADS = LIB.ck_set_num_threads
SET_THREADS.argtypes = [ctypes.c_int]
SET_THREADS.restype = None


def bf16_values(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


def assert_exact_provider_hard_faults_instead_of_falling_back() -> None:
    probe = subprocess.run(
        [sys.executable, __file__, "--probe-exact-provider-hard-fault"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0 or "no numerically different fallback" not in probe.stderr:
        raise AssertionError(
            "PyTorch-exact attention provider did not hard-fault on an "
            f"unsupported shape: rc={probe.returncode} stderr={probe.stderr!r}"
        )


def run_exact_provider_hard_fault_probe() -> None:
    values = np.zeros((1, 7, 8), dtype=np.float32)
    PYTORCH_FLASH_KERNEL(
        values.ctypes.data_as(FLOAT_P), values.ctypes.data_as(FLOAT_P),
        values.ctypes.data_as(FLOAT_P), values.ctypes.data_as(FLOAT_P),
        1, 1, 7, 8, 8, 7,
    )


def run_case_detailed(
    heads: int,
    kv_heads: int,
    tokens: int,
    dim: int,
    aligned_dim: int,
    seed: int,
    threads=None,
    kernel=KERNEL,
) -> dict[str, float | int]:
    if threads is not None:
        SET_THREADS(threads)
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
    kernel(
        q_padded.ctypes.data_as(FLOAT_P), k_padded.ctypes.data_as(FLOAT_P),
        v_padded.ctypes.data_as(FLOAT_P), actual_padded.ctypes.data_as(FLOAT_P),
        heads, kv_heads, tokens, dim, aligned_dim, tokens,
    )
    tq = torch.from_numpy(q).to(torch.bfloat16).unsqueeze(0)
    tk = torch.from_numpy(k).to(torch.bfloat16).unsqueeze(0)
    tv = torch.from_numpy(v).to(torch.bfloat16).unsqueeze(0)
    if heads != kv_heads:
        repeats = heads // kv_heads
        tk = tk.repeat_interleave(repeats, dim=1)
        tv = tv.repeat_interleave(repeats, dim=1)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected = torch.nn.functional.scaled_dot_product_attention(
            tq, tk, tv
        )[0].float().numpy()
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
    assert_exact_provider_hard_faults_instead_of_falling_back()
    scale_bits = struct.unpack("<I", struct.pack("<f", SCALE(72)))[0]
    if scale_bits != 0x3DF15BEF:
        raise AssertionError(
            f"PyTorch SDPA D=72 scale bits mismatch: 0x{scale_bits:08x}"
        )
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
    if sys.argv[1:] == ["--probe-exact-provider-hard-fault"]:
        run_exact_provider_hard_fault_probe()
        raise SystemExit(0)
    raise SystemExit(main())
