#!/usr/bin/env python3
"""Thread-scaling and numerical sweep for practical BF16 projection shapes."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THREADS = (1, 4, 8, 16, 24)


def _amx_available() -> bool:
    lib = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
    probe = lib.ck_gemm_bf16_amx_available
    probe.argtypes = []
    probe.restype = ctypes.c_int
    return bool(probe())


def _bf16_bits(values: torch.Tensor) -> np.ndarray:
    return values.contiguous().view(torch.uint16).numpy()


def _worker(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    a = torch.from_numpy(
        rng.standard_normal((args.m, args.k), dtype=np.float32)
    ).to(torch.bfloat16)
    b = torch.from_numpy(
        rng.standard_normal((args.n, args.k), dtype=np.float32)
    ).to(torch.bfloat16)
    bias = torch.from_numpy(
        rng.standard_normal(args.n, dtype=np.float32)
    ).to(torch.bfloat16)

    a_fp32 = a.float().numpy()
    b_bits = _bf16_bits(b)
    bias_fp32 = bias.float().numpy()
    actual = np.empty((args.m, args.n), dtype=np.float32)

    lib = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
    float_p = ctypes.POINTER(ctypes.c_float)
    kernel = getattr(lib, args.kernel)
    kernel.argtypes = [
        float_p, ctypes.c_void_p, float_p, float_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    call_args = (
        a_fp32.ctypes.data_as(float_p),
        ctypes.c_void_p(b_bits.ctypes.data),
        bias_fp32.ctypes.data_as(float_p),
        actual.ctypes.data_as(float_p),
        args.m, args.n, args.k,
    )

    def ck_call() -> None:
        kernel(*call_args)

    def torch_call() -> torch.Tensor:
        return torch.nn.functional.linear(a, b, bias)

    ck_call()
    with torch.inference_mode():
        expected = torch_call()
    ck_times = []
    torch_times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        ck_call()
        ck_times.append((time.perf_counter() - start) * 1000.0)
        with torch.inference_mode():
            start = time.perf_counter()
            expected = torch_call()
            torch_times.append((time.perf_counter() - start) * 1000.0)

    expected_np = expected.float().numpy()
    diff = np.abs(actual - expected_np)
    return {
        "kernel": args.kernel,
        "threads": args.threads,
        "ck_ms": statistics.median(ck_times),
        "pytorch_ms": statistics.median(torch_times),
        "pytorch_speedup_over_ck": statistics.median(ck_times) / statistics.median(torch_times),
        "max_abs": float(diff.max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "exact_ratio": float(np.mean(diff == 0.0)),
        "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
    }


def _run_parent(args: argparse.Namespace) -> dict[str, object]:
    rows = []
    kernels = list(args.kernels)
    skipped = []
    if "gemm_nt_bf16_amx_bf16_storage" in kernels and not _amx_available():
        kernels.remove("gemm_nt_bf16_amx_bf16_storage")
        skipped.append({
            "kernel": "gemm_nt_bf16_amx_bf16_storage",
            "reason": "AMX BF16 is unavailable in the current CPU/library build",
        })
    if not kernels:
        raise RuntimeError("no requested BF16 performance kernel is available")
    for kernel in kernels:
        for threads in args.threads:
            env = os.environ.copy()
            env["CK_NUM_THREADS"] = str(threads)
            env["OMP_NUM_THREADS"] = "1"
            cmd = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--threads-one", str(threads), "--m", str(args.m),
            "--n", str(args.n), "--k", str(args.k),
            "--iterations", str(args.iterations), "--seed", str(args.seed),
                "--kernel", kernel,
            ]
            completed = subprocess.run(
            cmd, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            rows.append(json.loads(lines[-1]))

    fastest_ck = min(rows, key=lambda row: row["ck_ms"])
    fastest_torch = min(rows, key=lambda row: row["pytorch_ms"])
    max_abs = max(row["max_abs"] for row in rows)
    rmse = max(row["rmse"] for row in rows)
    min_exact = min(row["exact_ratio"] for row in rows)
    deterministic = all(
        len({row["output_sha256"] for row in rows if row["kernel"] == kernel}) == 1
        for kernel in kernels
    )
    status = "pass" if (
        max_abs <= args.max_abs
        and rmse <= args.max_rmse
        and min_exact >= args.min_exact_ratio
        and deterministic
    ) else "fail"
    report = {
        "schema": "cke.bf16_performance_sweep",
        "schema_version": 1,
        "status": status,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "iterations": args.iterations,
        "rows": rows,
        "skipped": skipped,
        "thread_deterministic": deterministic,
        "fastest_ck": fastest_ck,
        "fastest_pytorch": fastest_torch,
        "pytorch_speedup_over_fastest_ck": (
            fastest_ck["ck_ms"] / fastest_torch["pytorch_ms"]
        ),
        "limits": {
            "max_abs": args.max_abs,
            "max_rmse": args.max_rmse,
            "min_exact_ratio": args.min_exact_ratio,
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    ratio = report["pytorch_speedup_over_fastest_ck"]
    print(
        f"Fastest CK: {fastest_ck['ck_ms']:.3f} ms at "
        f"{fastest_ck['threads']} threads"
    )
    print(
        f"Fastest PyTorch: {fastest_torch['pytorch_ms']:.3f} ms at "
        f"{fastest_torch['threads']} threads"
    )
    print(f"PyTorch is {ratio:.2f}x faster than fastest CK")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--threads-one", type=int, default=1)
    parser.add_argument("--kernel", default="gemm_nt_bf16_native_bf16_storage")
    parser.add_argument("--kernels", nargs="+", default=("gemm_nt_bf16_native_bf16_storage", "gemm_nt_bf16_amx_bf16_storage"))
    parser.add_argument("--threads", type=int, nargs="+", default=DEFAULT_THREADS)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=3456)
    parser.add_argument("--k", type=int, default=1152)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-abs", type=float, default=0.5)
    parser.add_argument("--max-rmse", type=float, default=1.0e-3)
    parser.add_argument("--min-exact-ratio", type=float, default=0.9997)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.worker:
        args.threads = args.threads_one
        print(json.dumps(_worker(args)))
        return 0
    report = _run_parent(args)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
