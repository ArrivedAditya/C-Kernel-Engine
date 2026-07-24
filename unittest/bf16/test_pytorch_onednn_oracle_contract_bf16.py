#!/usr/bin/env python3
"""Static guardrails for the optional production-shape oneDNN oracle."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research" / "bf16"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    required = (
        RESEARCH / "compare_onednn_pytorch_linear.py",
        RESEARCH / "onednn_pytorch_linear_probe.c",
        RESEARCH / "run_pytorch_onednn_real_shapes.py",
    )
    for path in required:
        require(path.is_file(), f"missing oneDNN oracle source: {path.relative_to(ROOT)}")

    runner_path = required[2]
    spec = importlib.util.spec_from_file_location("bf16_onednn_real_shapes", runner_path)
    require(spec is not None and spec.loader is not None, "cannot load real-shape runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(
        module.CASES == (
            ("qkv", 4032, 3456, 1152),
            ("mlp_up", 4032, 4304, 1152),
            ("mlp_down", 4032, 1152, 4304),
            ("decode_qkv", 1, 4096, 4096),
            ("decode_gate_up", 1, 24576, 4096),
            ("decode_mlp_down", 1, 4096, 12288),
        ),
        "production-shape BF16 matrix changed without updating its contract test",
    )

    comparison_path = required[0]
    comparison = comparison_path.read_text()
    require("8d263e693366ef8db40acc569cc7d8edf644556d" in comparison,
            "exact oneDNN 3.7.1 source identity is not enforced")
    require("mismatch_count" in comparison and "exact_ratio" in comparison,
            "zero-tolerance oracle metrics are missing")
    require("np.random.default_rng" in comparison and "input_sha256" in comparison,
            "thread-independent fixture identity is not enforced")
    require("torch.backends.mkldnn.is_available()" in comparison,
            "PyTorch oneDNN availability is not enforced by the oracle")
    require("output.dtype != torch.bfloat16" in comparison,
            "PyTorch BF16 output storage is not enforced by the oracle")

    comparison_spec = importlib.util.spec_from_file_location(
        "bf16_onednn_comparison", comparison_path
    )
    require(comparison_spec is not None and comparison_spec.loader is not None,
            "cannot load oneDNN comparison module")
    comparison_module = importlib.util.module_from_spec(comparison_spec)
    comparison_spec.loader.exec_module(comparison_module)
    with tempfile.TemporaryDirectory() as temporary:
        workdir = Path(temporary)
        args = argparse.Namespace(workdir=workdir, m=2, n=3, k=2, threads=1)
        files = comparison_module.paths(workdir)
        expected = np.array(
            [[0x3F80, 0x4000, 0x4040], [0x4080, 0x40A0, 0x40C0]],
            dtype=np.uint16,
        )
        expected.tofile(files["pytorch"])
        expected.tofile(files["onednn"])
        exact = comparison_module.compare(args, "onednn")
        require(exact["mismatch_count"] == 0 and exact["exact_ratio"] == 1.0,
                "byte-exact BF16 outputs are not accepted")
        changed = expected.copy()
        changed[1, 2] += 1
        changed.tofile(files["ck"])
        mismatch = comparison_module.compare(args, "ck")
        require(mismatch["mismatch_count"] == 1 and mismatch["exact_ratio"] < 1.0,
                "a one-element BF16 mismatch is not rejected by zero-tolerance comparison")

    makefile = (ROOT / "Makefile").read_text()
    require("test-bf16-pytorch-onednn-oracle-auto" in makefile,
            "automatic BF16 oracle target is not registered")
    require("BF16_ONEDNN_REPORT" in makefile,
            "machine-readable BF16 oracle report is not registered")
    nightly = (ROOT / "scripts" / "nightly_runner.py").read_text()
    require('"target": "test-bf16-pytorch-onednn-oracle-auto"' in nightly,
            "automatic BF16 oracle target is not registered in nightly")
    print("PASS: BF16 PyTorch/oneDNN oracle sources and production-shape contract")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
