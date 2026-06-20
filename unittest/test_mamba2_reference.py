#!/usr/bin/env python3
"""PyTorch parity tests for scalar Nemotron-H/Mamba2 reference kernels."""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
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

LIB.mamba2_in_proj_split_f32.argtypes = [fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.mamba2_in_proj_split_f32.restype = None
LIB.mamba2_conv1d_decode_f32.argtypes = [fptr, fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.mamba2_conv1d_decode_f32.restype = None
LIB.mamba2_dt_softplus_f32.argtypes = [fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
LIB.mamba2_dt_softplus_f32.restype = None
LIB.mamba2_selective_state_update_decode_f32.argtypes = [fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.mamba2_selective_state_update_decode_f32.restype = None
LIB.mamba2_rmsnorm_gate_f32.argtypes = [fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
LIB.mamba2_rmsnorm_gate_f32.restype = None


def _fptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return a.ctypes.data_as(fptr)


def _np(x: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(x.detach().cpu().numpy().astype(np.float32))


class TestMamba2Reference(unittest.TestCase):
    def test_in_proj_split_matches_torch(self) -> None:
        rows, d_mlp, inter, conv_dim, heads = 3, 5, 8, 14, 4
        torch.manual_seed(1)
        projected = torch.randn(rows, 2 * d_mlp + inter + conv_dim + heads, dtype=torch.float32)
        _, _, gate_ref, hidden_bc_ref, dt_ref = projected.split([d_mlp, d_mlp, inter, conv_dim, heads], dim=-1)
        projected_np = _np(projected)
        gate = np.empty((rows, inter), dtype=np.float32)
        hidden_bc = np.empty((rows, conv_dim), dtype=np.float32)
        dt = np.empty((rows, heads), dtype=np.float32)
        LIB.mamba2_in_proj_split_f32(_fptr(projected_np), _fptr(gate), _fptr(hidden_bc), _fptr(dt), rows, d_mlp, inter, conv_dim, heads)
        np.testing.assert_allclose(gate, _np(gate_ref), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(hidden_bc, _np(hidden_bc_ref), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(dt, _np(dt_ref), atol=0.0, rtol=0.0)

    def test_conv1d_decode_matches_torch(self) -> None:
        rows, conv_dim, kernel = 2, 11, 4
        torch.manual_seed(2)
        state = torch.randn(rows, conv_dim, kernel, dtype=torch.float32) * 0.2
        x = torch.randn(rows, conv_dim, dtype=torch.float32) * 0.2
        weight = torch.randn(conv_dim, kernel, dtype=torch.float32) * 0.3
        bias = torch.randn(conv_dim, dtype=torch.float32) * 0.1
        state_ref = torch.roll(state, shifts=-1, dims=-1)
        state_ref[:, :, -1] = x
        conv_ref = torch.nn.functional.silu((state_ref * weight.unsqueeze(0)).sum(dim=-1) + bias)
        state_np, x_np, weight_np, bias_np = map(_np, (state, x, weight, bias))
        conv = np.empty((rows, conv_dim), dtype=np.float32)
        state_out = np.empty((rows, conv_dim, kernel), dtype=np.float32)
        LIB.mamba2_conv1d_decode_f32(_fptr(state_np), _fptr(x_np), _fptr(weight_np), _fptr(bias_np), _fptr(conv), _fptr(state_out), rows, conv_dim, kernel)
        np.testing.assert_allclose(state_out, _np(state_ref), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(conv, _np(conv_ref), atol=1e-6, rtol=0.0)

    def test_dt_softplus_matches_torch(self) -> None:
        rows, heads = 4, 7
        torch.manual_seed(3)
        dt = torch.randn(rows, heads, dtype=torch.float32)
        bias = torch.randn(heads, dtype=torch.float32) * 0.2
        ref = torch.nn.functional.softplus(dt + bias).clamp(0.01, 2.0)
        dt_np, bias_np = map(_np, (dt, bias))
        out = np.empty((rows, heads), dtype=np.float32)
        LIB.mamba2_dt_softplus_f32(_fptr(dt_np), _fptr(bias_np), _fptr(out), rows, heads, ctypes.c_float(0.01), ctypes.c_float(2.0))
        np.testing.assert_allclose(out, _np(ref), atol=1e-6, rtol=0.0)

    def test_selective_state_update_decode_matches_torch(self) -> None:
        rows, heads, head_dim, state_dim, groups = 2, 6, 5, 4, 3
        torch.manual_seed(4)
        state = torch.randn(rows, heads, head_dim, state_dim, dtype=torch.float32) * 0.1
        x = torch.randn(rows, heads, head_dim, dtype=torch.float32) * 0.2
        dt = torch.rand(rows, heads, dtype=torch.float32) * 0.3 + 0.01
        a = -(torch.rand(heads, dtype=torch.float32) * 0.5 + 0.1)
        b = torch.randn(rows, groups, state_dim, dtype=torch.float32) * 0.2
        c = torch.randn(rows, groups, state_dim, dtype=torch.float32) * 0.2
        d = torch.randn(heads, dtype=torch.float32) * 0.05
        state_ref = torch.empty_like(state)
        y_ref = torch.empty_like(x)
        for r in range(rows):
            for h in range(heads):
                g = h * groups // heads
                decay = torch.exp(dt[r, h] * a[h])
                for hd in range(head_dim):
                    new_state = state[r, h, hd] * decay + dt[r, h] * b[r, g] * x[r, h, hd]
                    state_ref[r, h, hd] = new_state
                    y_ref[r, h, hd] = (new_state * c[r, g]).sum() + d[h] * x[r, h, hd]
        arrays = list(map(_np, (state, x, dt, a, b, c, d)))
        state_out = np.empty((rows, heads, head_dim, state_dim), dtype=np.float32)
        y = np.empty((rows, heads, head_dim), dtype=np.float32)
        LIB.mamba2_selective_state_update_decode_f32(*map(_fptr, arrays), _fptr(state_out), _fptr(y), rows, heads, head_dim, state_dim, groups)
        np.testing.assert_allclose(state_out, _np(state_ref), atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(y, _np(y_ref), atol=1e-6, rtol=0.0)

    def test_rmsnorm_gate_matches_torch_grouped_after_gate(self) -> None:
        rows, inner_dim, group_size = 3, 24, 6
        torch.manual_seed(5)
        x = torch.randn(rows, inner_dim, dtype=torch.float32) * 0.2
        gate = torch.randn(rows, inner_dim, dtype=torch.float32) * 0.2
        weight = torch.randn(inner_dim, dtype=torch.float32) * 0.2 + 1.0
        gated = x * torch.nn.functional.silu(gate)
        chunks = []
        for start in range(0, inner_dim, group_size):
            chunk = gated[:, start:start + group_size]
            inv = torch.rsqrt(chunk.square().mean(dim=-1, keepdim=True) + 1e-5)
            chunks.append(chunk * inv * weight[start:start + group_size])
        ref = torch.cat(chunks, dim=-1)
        x_np, gate_np, weight_np = map(_np, (x, gate, weight))
        out = np.empty((rows, inner_dim), dtype=np.float32)
        LIB.mamba2_rmsnorm_gate_f32(_fptr(x_np), _fptr(gate_np), _fptr(weight_np), _fptr(out), rows, inner_dim, group_size, ctypes.c_float(1e-5))
        np.testing.assert_allclose(out, _np(ref), atol=1e-6, rtol=0.0)


def _time_us(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1.0e6 / max(1, iterations)


def run_benchmark() -> None:
    rows, heads, head_dim, state_dim, groups = 4, 64, 64, 128, 8
    rng = np.random.default_rng(23)
    state = np.ascontiguousarray((0.02 * rng.standard_normal((rows, heads, head_dim, state_dim))).astype(np.float32))
    x = np.ascontiguousarray((0.02 * rng.standard_normal((rows, heads, head_dim))).astype(np.float32))
    dt = np.ascontiguousarray((0.01 + 0.1 * rng.random((rows, heads))).astype(np.float32))
    a = np.ascontiguousarray((-(0.1 + 0.5 * rng.random(heads))).astype(np.float32))
    b = np.ascontiguousarray((0.02 * rng.standard_normal((rows, groups, state_dim))).astype(np.float32))
    c = np.ascontiguousarray((0.02 * rng.standard_normal((rows, groups, state_dim))).astype(np.float32))
    d = np.ascontiguousarray((0.01 * rng.standard_normal(heads)).astype(np.float32))
    state_out = np.empty_like(state)
    y = np.empty_like(x)

    def ck_step() -> None:
        LIB.mamba2_selective_state_update_decode_f32(
            _fptr(state), _fptr(x), _fptr(dt), _fptr(a), _fptr(b), _fptr(c), _fptr(d),
            _fptr(state_out), _fptr(y), rows, heads, head_dim, state_dim, groups
        )

    ck_step()
    ck_us = _time_us(ck_step, 20)
    elems = rows * heads * head_dim * state_dim
    print("kernel                                  elems        ck_us")
    print(f"mamba2_selective_state_update_decode {elems:10d} {ck_us:10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    args, remaining = ap.parse_known_args()
    if args.benchmark:
        run_benchmark()
    else:
        sys.argv = [sys.argv[0], *remaining]
        unittest.main(verbosity=2)
