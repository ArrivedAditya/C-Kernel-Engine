#!/usr/bin/env python3
"""Run the production-shape PyTorch/oneDNN/CK BF16 linear oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


CASES = (
    ("qkv", 4032, 3456, 1152),
    ("mlp_up", 4032, 4304, 1152),
    ("mlp_down", 4032, 1152, 4304),
    ("decode_qkv", 1, 4096, 4096),
    ("decode_gate_up", 1, 24576, 4096),
    ("decode_mlp_down", 1, 4096, 12288),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--ck-library", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=(1, 16, 20, 24))
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    probe = Path(__file__).with_name("compare_onednn_pytorch_linear.py")
    rows = []
    for threads in args.threads:
        for name, m, n, k in CASES:
            case_dir = args.workdir / f"t{threads}" / name
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(threads)
            command = [
                sys.executable, str(probe),
                "--workdir", str(case_dir),
                "--library", str(args.library),
                "--ck-library", str(args.ck_library),
                "--m", str(m), "--n", str(n), "--k", str(k),
                "--threads", str(threads), "--seed", str(args.seed),
            ]
            subprocess.run(command, check=True, env=env)
            row = json.loads((case_dir / "comparison.json").read_text())
            row["name"] = name
            rows.append(row)
            if not args.keep_artifacts:
                for artifact in case_dir.glob("*.bf16"):
                    artifact.unlink()
    for name, _m, _n, _k in CASES:
        hashes = {
            json.dumps(row["provenance"]["input_sha256"], sort_keys=True)
            for row in rows if row["name"] == name
        }
        if len(hashes) != 1:
            raise RuntimeError(f"{name}: thread sweep did not use byte-identical input tensors")
    report = {
        "status": "pass",
        "threads": args.threads,
        "case_count": len(rows),
        "cases": rows,
        "contract": "bf16_weight_bf16_input_pytorch_onednn_brgemm_bf16_output",
        "acceptance": {"mismatch_count": 0, "tolerance_relaxed": False},
    }
    report_path = args.report or args.workdir / "real_shape_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {len(rows)}/{len(rows)} production-shape BF16 oracle cases")
    print(report_path)


if __name__ == "__main__":
    main()
