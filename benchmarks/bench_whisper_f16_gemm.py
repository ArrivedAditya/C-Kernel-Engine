#!/usr/bin/env python3
"""Benchmark exact FP16 Whisper GEMM providers at production shapes."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np


SHAPES = {
    "tiny_projection": (1500, 384, 384),
    "base_projection": (1500, 512, 512),
    "base_mlp_up": (1500, 2048, 512),
    "base_mlp_down": (1500, 512, 2048),
    "small_projection": (1500, 768, 768),
}


def _load(path: Path, threads: int) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path.resolve()))
    lib.ck_set_num_threads.argtypes = [ctypes.c_int]
    lib.ck_set_num_threads(threads)
    lib.ck_gemm_nt_f16_simd_lanes.restype = ctypes.c_int
    lib.gemm_nt_f16.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    return lib


def _fixture(shape: tuple[int, int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    m, n, k = shape
    rng = np.random.default_rng(seed)
    activation = np.ascontiguousarray(
        rng.standard_normal((m, k), dtype=np.float32) * np.float32(0.1)
    )
    weights = np.ascontiguousarray(
        (rng.standard_normal((n, k), dtype=np.float32) * np.float32(0.1)).astype(np.float16)
    )
    return activation, weights


def _invoke(
    lib: ctypes.CDLL,
    activation: np.ndarray,
    weights: np.ndarray,
    output: np.ndarray,
    *,
    baseline: bool,
) -> float:
    previous = os.environ.get("CK_DISABLE_F16_GEMM_M4N2")
    try:
        if baseline:
            os.environ["CK_DISABLE_F16_GEMM_M4N2"] = "1"
        else:
            os.environ.pop("CK_DISABLE_F16_GEMM_M4N2", None)
        start = time.perf_counter()
        lib.gemm_nt_f16(
            activation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(weights.ctypes.data),
            None,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            activation.shape[0],
            weights.shape[0],
            activation.shape[1],
        )
        return time.perf_counter() - start
    finally:
        if previous is None:
            os.environ.pop("CK_DISABLE_F16_GEMM_M4N2", None)
        else:
            os.environ["CK_DISABLE_F16_GEMM_M4N2"] = previous


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=Path("build/libckernel_engine.so"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--shape", choices=tuple(SHAPES), action="append")
    parser.add_argument("--mode", choices=("compare", "baseline", "m4n2"), default="compare")
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.threads < 1 or args.repeats < 1:
        parser.error("--threads and --repeats must be positive")

    lib = _load(args.library, args.threads)
    lanes = int(lib.ck_gemm_nt_f16_simd_lanes())
    selected = args.shape or list(SHAPES)
    report: dict[str, object] = {
        "schema": "cke.whisper_f16_gemm_performance",
        "schema_version": 1,
        "status": "pass",
        "simd_lanes": lanes,
        "threads": args.threads,
        "mode": args.mode,
        "cases": [],
    }

    if lanes != 8:
        report["status"] = "skip"
        report["reason"] = "M4xN2 provider is AVX2-specific"
    else:
        for index, name in enumerate(selected):
            shape = SHAPES[name]
            activation, weights = _fixture(shape, 20260801 + index)
            baseline_output = np.empty((shape[0], shape[1]), dtype=np.float32)
            optimized_output = np.empty_like(baseline_output)
            baseline_times: list[float] = []
            optimized_times: list[float] = []

            if args.mode in ("compare", "baseline"):
                _invoke(lib, activation, weights, baseline_output, baseline=True)
            if args.mode in ("compare", "m4n2"):
                _invoke(lib, activation, weights, optimized_output, baseline=False)

            for _ in range(args.repeats):
                if args.mode in ("compare", "baseline"):
                    baseline_times.append(
                        _invoke(lib, activation, weights, baseline_output, baseline=True)
                    )
                if args.mode in ("compare", "m4n2"):
                    optimized_times.append(
                        _invoke(lib, activation, weights, optimized_output, baseline=False)
                    )

            case: dict[str, object] = {
                "name": name,
                "shape": {"tokens": shape[0], "outputs": shape[1], "width": shape[2]},
            }
            if baseline_times:
                case["baseline_median_seconds"] = statistics.median(baseline_times)
            if optimized_times:
                case["m4n2_median_seconds"] = statistics.median(optimized_times)
            if args.mode == "compare":
                exact = bool(np.array_equal(baseline_output, optimized_output))
                speedup = statistics.median(baseline_times) / statistics.median(optimized_times)
                case["bit_exact"] = exact
                case["speedup"] = speedup
                case["passed"] = exact and speedup >= args.min_speedup
                if not case["passed"]:
                    report["status"] = "fail"
            report["cases"].append(case)

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] in ("pass", "skip") else 1


if __name__ == "__main__":
    raise SystemExit(main())
