#!/usr/bin/env python3
"""Practical-shape BF16 precision matrix for Qwen3-VL vision blocks."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def native_bf16_supported() -> bool:
    override = os.environ.get("CK_NATIVE_BF16_TEST", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "avx512_bf16" in cpuinfo or "amx_bf16" in cpuinfo

def load(name: str):
    path = ROOT / "unittest" / "bf16" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def record(rows, family, shape, metrics, limits, *, min_exact_ratio=None):
    if isinstance(metrics, dict):
        max_abs, rmse = metrics["max_abs"], metrics["rmse"]
        exact_ratio = metrics.get("exact_ratio")
    else:
        max_abs, rmse = metrics
        exact_ratio = None
    ok = max_abs <= limits[0] and rmse <= limits[1]
    if min_exact_ratio is not None:
        ok = ok and exact_ratio is not None and exact_ratio >= min_exact_ratio
    row = {
        "kernel_family": family,
        "shape": shape,
        "max_abs": max_abs,
        "rmse": rmse,
        "max_abs_limit": limits[0],
        "rmse_limit": limits[1],
        "status": "pass" if ok else "fail",
    }
    if exact_ratio is not None:
        row["exact_ratio"] = exact_ratio
        row["different_outputs"] = metrics["different_outputs"]
        row["output_count"] = metrics["output_count"]
        row["min_exact_ratio"] = min_exact_ratio
    rows.append(row)
    if not ok:
        raise AssertionError(
            f"{family} shape={shape}: max_abs={max_abs} rmse={rmse} limits={limits}"
        )

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--full-shapes",
        action="store_true",
        help="Run high-memory production-shape attention coverage.",
    )
    args = parser.parse_args()
    if not native_bf16_supported():
        report = {
            "schema": "cke.bf16_practical_precision",
            "schema_version": 1,
            "model_shape_family": "qwen3_vl_vision",
            "status": "skip",
            "reason": "native AVX512-BF16 or AMX-BF16 is unavailable",
            "rows": [],
            "scope": "Native practical-shape parity; portable leaf contracts run separately.",
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    layernorm = load("test_layernorm_storage_contract_bf16.py")
    gemm = load("test_gemm_storage_contract_bf16.py")
    attention = load("test_attention_storage_contract_bf16.py")
    rows = []

    record(rows, "layernorm", {"tokens": 32, "dim": 1152},
           layernorm.run_case(32, 1152, 1e-6, 101), (0.03125, 0.003))
    record(rows, "qkv_gemm", {"m": 4, "n": 3456, "k": 1152},
           gemm.run_case_detailed(4, 3456, 1152, 102), (0.03125, 3.0e-4),
           min_exact_ratio=0.9997)
    record(rows, "mlp_up_gemm", {"m": 2, "n": 4304, "k": 1152},
           gemm.run_case_detailed(2, 4304, 1152, 103), (0.03125, 3.5e-4),
           min_exact_ratio=0.9997)
    record(rows, "qkv_gemm_native_threadpool", {"m": 4, "n": 3456, "k": 1152},
           gemm.run_case_detailed(4, 3456, 1152, 102, kernel=gemm.NATIVE_KERNEL),
           (0.03125, 3.0e-4), min_exact_ratio=0.9997)
    record(rows, "mlp_up_gemm_native_threadpool", {"m": 2, "n": 4304, "k": 1152},
           gemm.run_case_detailed(2, 4304, 1152, 103, kernel=gemm.NATIVE_KERNEL),
           (0.03125, 3.5e-4), min_exact_ratio=0.9997)
    record(rows, "qkv_gemm_amx_tile32", {"m": 32, "n": 3456, "k": 1152},
           gemm.run_case_detailed(32, 3456, 1152, 104, kernel=gemm.AMX_KERNEL),
           (0.5, 1.0e-3), min_exact_ratio=0.9997)
    record(rows, "vision_attention", {"heads": 2, "tokens": 128, "dim": 72},
           attention.run_case(2, 2, 128, 72, 72, 104)[:2], (0.03125, 0.003))
    record(rows, "vision_attention", {"heads": 2, "tokens": 512, "dim": 72},
           attention.run_case(2, 2, 512, 72, 72, 105)[:2], (0.03125, 0.003))
    if args.full_shapes:
        record(
            rows,
            "layernorm_pytorch_welford_production",
            {"tokens": 4032, "dim": 1152},
            layernorm.run_case(
                4032, 1152, 1e-6, 107, kernel=layernorm.PYTORCH_KERNEL
            ),
            (0.03125, 0.003),
        )
        record(
            rows,
            "vision_attention_production",
            {"heads": 16, "tokens": 4032, "dim": 72},
            attention.run_case_detailed(16, 16, 4032, 72, 72, 106),
            (0.001, 7.0e-5),
            min_exact_ratio=0.58,
        )

    report = {
        "schema": "cke.bf16_practical_precision",
        "schema_version": 1,
        "model_shape_family": "qwen3_vl_vision",
        "status": "pass",
        "rows": rows,
        "scope": (
            "Practical leaf/reduction shapes. The production H=16, T=4032, D=72 "
            "attention row is present only when --full-shapes is requested."
        ),
        "full_shapes": args.full_shapes,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
