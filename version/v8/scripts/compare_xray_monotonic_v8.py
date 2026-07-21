#!/usr/bin/env python3
"""Reject a numerical provider whose X-ray drift regresses at required edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINTS = (
    "vision.frontend.position.output",
    "vision.layer.0.output",
    "vision.layer.8.output",
    "vision.layer.16.output",
    "vision.layer.24.output",
    "vision.layer.26.output",
    "vision.spatial_merge.output",
    "vision.projector.output",
    "vision.prefix.output",
)


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparisons = payload.get("comparisons", [])
    if isinstance(comparisons, list):
        return {
            row["checkpoint_id"]: row
            for row in comparisons
            if isinstance(row, dict) and isinstance(row.get("checkpoint_id"), str)
        }
    if not isinstance(comparisons, dict):
        return {}

    selector_names = {
        "vision_position_embeddings": "vision.frontend.position.output",
        "vision_spatial_merge": "vision.spatial_merge.output",
        "vision_projector_out": "vision.projector.output",
        "vision_output": "vision.prefix.output",
    }
    rows: dict[str, dict[str, Any]] = {}
    for selector, metrics in comparisons.items():
        checkpoint = selector_names.get(selector)
        if checkpoint is None and selector.startswith("layer_out@"):
            checkpoint = f"vision.layer.{selector.split('@', 1)[1]}.output"
        if checkpoint is None or not isinstance(metrics, dict):
            continue
        rows[checkpoint] = {"checkpoint_id": checkpoint, "metrics": metrics}
    return rows


def compare_reports(
    baseline_path: Path,
    candidate_path: Path,
    checkpoints: tuple[str, ...],
    epsilon: float,
) -> dict[str, Any]:
    baseline = _rows(baseline_path)
    candidate = _rows(candidate_path)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for checkpoint in checkpoints:
        if checkpoint not in baseline or checkpoint not in candidate:
            failures.append(f"missing checkpoint: {checkpoint}")
            rows.append({"checkpoint_id": checkpoint, "status": "missing"})
            continue
        base_metrics = baseline[checkpoint].get("metrics", {})
        candidate_metrics = candidate[checkpoint].get("metrics", {})
        base_relative = float(base_metrics["relative_rmse"])
        candidate_relative = float(candidate_metrics["relative_rmse"])
        base_max_abs = float(base_metrics["max_abs"])
        candidate_max_abs = float(candidate_metrics["max_abs"])
        relative_ok = candidate_relative <= base_relative + epsilon
        max_abs_ok = candidate_max_abs <= base_max_abs + epsilon
        status = "pass" if relative_ok and max_abs_ok else "fail"
        if status == "fail":
            failures.append(checkpoint)
        rows.append(
            {
                "checkpoint_id": checkpoint,
                "status": status,
                "baseline_relative_rmse": base_relative,
                "candidate_relative_rmse": candidate_relative,
                "baseline_max_abs": base_max_abs,
                "candidate_max_abs": candidate_max_abs,
            }
        )
    return {
        "schema": "cke.xray_monotonic_provider_gate",
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "epsilon": epsilon,
        "rows": rows,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", dest="checkpoints")
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    checkpoints = tuple(args.checkpoints or DEFAULT_CHECKPOINTS)
    report = compare_reports(args.baseline, args.candidate, checkpoints, args.epsilon)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
