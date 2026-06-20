#!/usr/bin/env python3
"""PyTorch parity test for routed ReLU2 MoE expert MLP."""

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
iptr = ctypes.POINTER(ctypes.c_int)
LIB.moe_relu2_expert_forward_f32.argtypes = [fptr, iptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.moe_relu2_expert_forward_f32.restype = None
LIB.moe_relu2_expert_backward_f32.argtypes = [fptr, fptr, iptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.moe_relu2_expert_backward_f32.restype = None


def _fptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return a.ctypes.data_as(fptr)


def _iptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_int):
    return a.ctypes.data_as(iptr)


def torch_moe(hidden, indices, weights, expert_up, expert_down):
    rows, hidden_dim = hidden.shape
    top_k = indices.shape[1]
    out = torch.zeros_like(hidden)
    for r in range(rows):
        for s in range(top_k):
            e = int(indices[r, s])
            pre = torch.matmul(expert_up[e], hidden[r])
            act = torch.relu(pre).square()
            expert_out = torch.matmul(expert_down[e], act)
            out[r] = out[r] + weights[r, s] * expert_out
    return out


class TestMoERelu2Expert(unittest.TestCase):
    def _run_case(self, rows: int, hidden_dim: int, intermediate_dim: int, n_experts: int, top_k: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        hidden_np = (0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32)
        up_np = (0.15 * rng.standard_normal((n_experts, intermediate_dim, hidden_dim))).astype(np.float32)
        down_np = (0.15 * rng.standard_normal((n_experts, hidden_dim, intermediate_dim))).astype(np.float32)
        idx_np = np.empty((rows, top_k), dtype=np.int32)
        for r in range(rows):
            idx_np[r] = rng.choice(n_experts, size=top_k, replace=False).astype(np.int32)
        w_raw = rng.random((rows, top_k)).astype(np.float32)
        weights_np = np.ascontiguousarray(w_raw / w_raw.sum(axis=1, keepdims=True))
        d_out_np = (0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32)

        ck_out = np.empty_like(hidden_np)
        LIB.moe_relu2_expert_forward_f32(_fptr(hidden_np), _iptr(idx_np), _fptr(weights_np), _fptr(up_np), _fptr(down_np), _fptr(ck_out), rows, hidden_dim, intermediate_dim, n_experts, top_k)

        ck_dh = np.empty_like(hidden_np)
        ck_dw = np.empty_like(weights_np)
        ck_dup = np.empty_like(up_np)
        ck_ddown = np.empty_like(down_np)
        LIB.moe_relu2_expert_backward_f32(_fptr(d_out_np), _fptr(hidden_np), _iptr(idx_np), _fptr(weights_np), _fptr(up_np), _fptr(down_np), _fptr(ck_dh), _fptr(ck_dw), _fptr(ck_dup), _fptr(ck_ddown), rows, hidden_dim, intermediate_dim, n_experts, top_k)

        hidden = torch.tensor(hidden_np, dtype=torch.float32, requires_grad=True)
        up = torch.tensor(up_np, dtype=torch.float32, requires_grad=True)
        down = torch.tensor(down_np, dtype=torch.float32, requires_grad=True)
        weights = torch.tensor(weights_np, dtype=torch.float32, requires_grad=True)
        indices = torch.tensor(idx_np, dtype=torch.long)
        ref = torch_moe(hidden, indices, weights, up, down)
        ref.backward(torch.tensor(d_out_np, dtype=torch.float32))

        np.testing.assert_allclose(ck_out, ref.detach().numpy(), atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dh, hidden.grad.detach().numpy(), atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dw, weights.grad.detach().numpy(), atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dup, up.grad.detach().numpy(), atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(ck_ddown, down.grad.detach().numpy(), atol=2e-6, rtol=0.0)

    def test_small_experts(self) -> None:
        self._run_case(rows=4, hidden_dim=8, intermediate_dim=12, n_experts=5, top_k=3, seed=5)

    def test_nano_shape_tiny_batch(self) -> None:
        self._run_case(rows=3, hidden_dim=16, intermediate_dim=10, n_experts=8, top_k=4, seed=13)


def _time_us(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1.0e6 / max(1, iterations)


def run_benchmark() -> None:
    rows, hidden_dim, intermediate_dim, n_experts, top_k = 16, 64, 48, 16, 4
    rng = np.random.default_rng(41)
    hidden = np.ascontiguousarray((0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))
    up = np.ascontiguousarray((0.12 * rng.standard_normal((n_experts, intermediate_dim, hidden_dim))).astype(np.float32))
    down = np.ascontiguousarray((0.12 * rng.standard_normal((n_experts, hidden_dim, intermediate_dim))).astype(np.float32))
    idx = np.empty((rows, top_k), dtype=np.int32)
    for r in range(rows):
        idx[r] = rng.choice(n_experts, size=top_k, replace=False).astype(np.int32)
    w = rng.random((rows, top_k)).astype(np.float32)
    weights = np.ascontiguousarray(w / w.sum(axis=1, keepdims=True))
    out = np.empty_like(hidden)
    th = torch.tensor(hidden, dtype=torch.float32)
    tu = torch.tensor(up, dtype=torch.float32)
    td = torch.tensor(down, dtype=torch.float32)
    tw = torch.tensor(weights, dtype=torch.float32)
    ti = torch.tensor(idx, dtype=torch.long)

    def ck_step() -> None:
        LIB.moe_relu2_expert_forward_f32(_fptr(hidden), _iptr(idx), _fptr(weights), _fptr(up), _fptr(down), _fptr(out), rows, hidden_dim, intermediate_dim, n_experts, top_k)

    def torch_step() -> None:
        torch_moe(th, ti, tw, tu, td)

    ck_step(); torch_step()
    torch_us = _time_us(torch_step, 100)
    ck_us = _time_us(ck_step, 100)
    print("kernel                    pytorch_us      ck_us       speedup")
    print(f"moe_relu2_expert_forward {torch_us:10.3f} {ck_us:10.3f} {torch_us / max(ck_us, 1e-12):8.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    args, remaining = ap.parse_known_args()
    if args.benchmark:
        run_benchmark()
    else:
        sys.argv = [sys.argv[0], *remaining]
        unittest.main(verbosity=2)
