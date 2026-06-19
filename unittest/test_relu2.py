#!/usr/bin/env python3
"""PyTorch parity test for relu2 activation."""

from __future__ import annotations

import argparse
import ctypes
import time
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"[SKIP] torch not available: {exc}")
    sys.exit(0)


ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libckernel_engine.so"
if not LIB_PATH.exists():  # pragma: no cover
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)

LIB = ctypes.CDLL(str(LIB_PATH))
fptr = ctypes.POINTER(ctypes.c_float)
LIB.relu2_forward.argtypes = [fptr, fptr, ctypes.c_size_t]
LIB.relu2_forward.restype = None
LIB.relu2_backward.argtypes = [fptr, fptr, fptr, ctypes.c_size_t]
LIB.relu2_backward.restype = None


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _time_us(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1.0e6 / max(1, iterations)


def run_benchmark() -> None:
    torch.set_num_threads(1)
    shapes = [
        ("tiny", (4096,), 2000),
        ("nemotron_mlp", (13, 1856), 500),
        ("prefill_tile", (512, 1856), 50),
    ]
    rng = np.random.default_rng(101)
    print("kernel                pytorch_us      ck_us       speedup")
    for name, shape, iterations in shapes:
        x = (0.5 * rng.standard_normal(shape)).astype(np.float32)
        d_out = (0.25 * rng.standard_normal(shape)).astype(np.float32)
        ck_out = np.zeros_like(x)
        ck_dx = np.zeros_like(x)
        tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        td = torch.tensor(d_out, dtype=torch.float32)

        def torch_step() -> None:
            if tx.grad is not None:
                tx.grad.zero_()
            out = torch.relu(tx).square()
            out.backward(td)

        def ck_step() -> None:
            LIB.relu2_forward(_as_ptr(x), _as_ptr(ck_out), x.size)
            LIB.relu2_backward(_as_ptr(x), _as_ptr(d_out), _as_ptr(ck_dx), x.size)

        torch_step()
        ck_step()
        torch_us = _time_us(torch_step, iterations)
        ck_us = _time_us(ck_step, iterations)
        speedup = torch_us / max(ck_us, 1e-12)
        print(f"relu2_{name:<12} {torch_us:10.3f} {ck_us:10.3f} {speedup:8.2f}x")


class TestReLU2(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def _run_case(self, shape: tuple[int, ...], seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = (0.5 * rng.standard_normal(shape)).astype(np.float32)
        d_out = (0.25 * rng.standard_normal(shape)).astype(np.float32)
        ck_out = np.zeros_like(x)
        ck_dx = np.zeros_like(x)

        LIB.relu2_forward(_as_ptr(x), _as_ptr(ck_out), x.size)
        LIB.relu2_backward(_as_ptr(x), _as_ptr(d_out), _as_ptr(ck_dx), x.size)

        tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = torch.relu(tx).square()
        out.backward(torch.tensor(d_out, dtype=torch.float32))

        np.testing.assert_allclose(ck_out, out.detach().numpy(), atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dx, tx.grad.detach().numpy(), atol=1e-6, rtol=0.0)

    def test_vector(self) -> None:
        self._run_case((4097,), 3)

    def test_token_major_matrix(self) -> None:
        self._run_case((13, 1856), 11)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    args, remaining = ap.parse_known_args()
    if args.benchmark:
        run_benchmark()
    else:
        sys.argv = [sys.argv[0], *remaining]
        unittest.main(verbosity=2)
