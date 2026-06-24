#!/usr/bin/env python3
"""PyTorch parity test for Nemotron-H group-limited MoE router."""

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
LIB.nemotron_group_limited_topk_router_f32.argtypes = [
    fptr, fptr, iptr, fptr,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_float,
]
LIB.nemotron_group_limited_topk_router_f32.restype = None


def _fptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return a.ctypes.data_as(fptr)


def _iptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_int):
    return a.ctypes.data_as(iptr)


def torch_router(scores: torch.Tensor, bias: torch.Tensor, top_k: int, n_group: int, topk_group: int, norm: bool, scale: float):
    rows, n_experts = scores.shape
    choice = scores + bias.unsqueeze(0)
    group_scores = choice.view(rows, n_group, n_experts // n_group).topk(2, dim=-1, largest=True, sorted=True)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, largest=True, sorted=True)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(rows, n_group, n_experts // n_group).reshape(rows, n_experts)
    masked_choice = choice.masked_fill(~score_mask.bool(), 0.0)
    indices = torch.topk(masked_choice, k=top_k, dim=-1, largest=True, sorted=True)[1]
    weights = scores.gather(1, indices)
    if norm:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return indices, weights * scale


def _time_us(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1.0e6 / max(1, iterations)


def run_benchmark() -> None:
    torch.set_num_threads(1)
    rows, n_experts, top_k, n_group, topk_group = 512, 128, 6, 8, 3
    rng = np.random.default_rng(301)
    logits = (0.7 * rng.standard_normal((rows, n_experts))).astype(np.float32)
    scores = np.ascontiguousarray(1.0 / (1.0 + np.exp(-logits)))
    bias = np.ascontiguousarray((0.05 * rng.standard_normal(n_experts)).astype(np.float32))
    ck_idx = np.empty((rows, top_k), dtype=np.int32)
    ck_w = np.empty((rows, top_k), dtype=np.float32)
    ts = torch.tensor(scores, dtype=torch.float32)
    tb = torch.tensor(bias, dtype=torch.float32)

    def torch_step() -> None:
        torch_router(ts, tb, top_k, n_group, topk_group, True, 1.0)

    def ck_step() -> None:
        LIB.nemotron_group_limited_topk_router_f32(
            _fptr(scores), _fptr(bias), _iptr(ck_idx), _fptr(ck_w),
            rows, n_experts, top_k, n_group, topk_group, 1, ctypes.c_float(1.0),
        )

    torch_step()
    ck_step()
    torch_us = _time_us(torch_step, 200)
    ck_us = _time_us(ck_step, 200)
    print("kernel                            pytorch_us      ck_us       speedup")
    print(f"nemotron_group_limited_router {torch_us:12.3f} {ck_us:10.3f} {torch_us / max(ck_us, 1e-12):8.2f}x")


class TestNemotronRouter(unittest.TestCase):
    def _run_case(self, rows: int, n_experts: int, top_k: int, n_group: int, topk_group: int, norm: bool, scale: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        logits = (0.7 * rng.standard_normal((rows, n_experts))).astype(np.float32)
        scores = 1.0 / (1.0 + np.exp(-logits))
        bias = (0.05 * rng.standard_normal(n_experts)).astype(np.float32)
        ck_idx = np.empty((rows, top_k), dtype=np.int32)
        ck_w = np.empty((rows, top_k), dtype=np.float32)
        LIB.nemotron_group_limited_topk_router_f32(
            _fptr(np.ascontiguousarray(scores)),
            _fptr(np.ascontiguousarray(bias)),
            _iptr(ck_idx),
            _fptr(ck_w),
            rows, n_experts, top_k, n_group, topk_group, int(norm), ctypes.c_float(scale),
        )
        ref_idx, ref_w = torch_router(
            torch.tensor(scores, dtype=torch.float32),
            torch.tensor(bias, dtype=torch.float32),
            top_k, n_group, topk_group, norm, scale,
        )
        np.testing.assert_array_equal(ck_idx, ref_idx.numpy().astype(np.int32))
        np.testing.assert_allclose(ck_w, ref_w.numpy(), atol=1e-6, rtol=0.0)

    def test_nano_like_router(self) -> None:
        self._run_case(rows=7, n_experts=128, top_k=6, n_group=8, topk_group=3, norm=True, scale=1.0, seed=23)

    def test_no_norm_scaled_router(self) -> None:
        self._run_case(rows=5, n_experts=16, top_k=4, n_group=4, topk_group=2, norm=False, scale=0.7, seed=31)

    def test_kimi_vl_like_router(self) -> None:
        self._run_case(rows=9, n_experts=64, top_k=6, n_group=1, topk_group=1, norm=True, scale=2.446, seed=41)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    args, remaining = ap.parse_known_args()
    if args.benchmark:
        run_benchmark()
    else:
        sys.argv = [sys.argv[0], *remaining]
        unittest.main(verbosity=2)
