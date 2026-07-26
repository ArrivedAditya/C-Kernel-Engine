#!/usr/bin/env python3
"""Numerical and memory-layout contract for independently strided RMSNorm."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def reference(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    logical = x[:, : gamma.size]
    variance = np.mean(logical * logical, axis=1, dtype=np.float32)
    rstd = np.float32(1.0) / np.sqrt(variance + np.float32(eps))
    return (logical * rstd[:, None] * gamma[None, :]).astype(np.float32)


def main() -> int:
    lib = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
    fn = lib.rmsnorm_forward_strided_f32
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]

    rows, logical_width, input_stride, output_stride = 7, 512, 576, 512
    rng = np.random.default_rng(20260726)
    source = rng.normal(0.0, 0.3, size=(rows, input_stride)).astype(np.float32)
    gamma = rng.normal(1.0, 0.1, size=logical_width).astype(np.float32)
    expected = reference(source, gamma, 1.0e-5)

    guard = np.full(rows * output_stride + 32, np.float32(12345.0), dtype=np.float32)
    actual = guard[: rows * output_stride].reshape(rows, output_stride)
    fn(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        None,
        rows,
        logical_width,
        input_stride,
        output_stride,
        ctypes.c_float(1.0e-5),
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)
    if not np.all(guard[rows * output_stride :] == np.float32(12345.0)):
        raise AssertionError("strided RMSNorm wrote beyond its compact output")
    print(
        "rmsnorm_strided_7x512_input576_output512 "
        f"max_diff={float(np.max(np.abs(actual - expected))):.9g} guard=PASS [PASS]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
