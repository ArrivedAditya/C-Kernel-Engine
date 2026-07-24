#!/usr/bin/env python3
"""Plan the next bounded semantic checkpoints from a parity report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator


V8_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCHEMA = V8_ROOT / "schemas" / "parity_profile.schema.json"


def load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_profile(profile: Dict[str, Any]) -> None:
    errors = list(Draft202012Validator(load(PROFILE_SCHEMA)).iter_errors(profile))
    if errors:
        error = errors[0]
        raise ValueError(f"invalid parity profile at {list(error.absolute_path)}: {error.message}")


def _expand_interval(profile: Dict[str, Any], lower: str, upper: str) -> list[str]:
    interval = f"{lower}->{upper}"
    expansion = list(profile["interval_expansions"].get(interval, []))
    if not expansion and upper.startswith("vision.layer.") and upper.endswith(".output"):
        parts = upper.split(".")
        if len(parts) == 4 and parts[2].isdigit():
            generic = profile["interval_expansions"].get("vision.layer.input->vision.layer.output", [])
            expansion = [name.replace("{layer}", parts[2]) for name in generic]
    return expansion


def plan(
    profile: Dict[str, Any],
    report: Dict[str, Any],
    checkpoint_order: list[str] | None = None,
) -> Dict[str, Any]:
    validate_profile(profile)
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("report.comparisons must be an array")
    results = {str(row.get("checkpoint_id")): str(row.get("status")) for row in comparisons}
    order: List[str] = checkpoint_order or profile["checkpoint_order"]
    first_failure = next((name for name in order if results.get(name) == "fail"), None)
    if first_failure is None:
        ranking = report.get("ranking_divergence")
        if isinstance(ranking, dict):
            measured = [
                row for row in comparisons
                if row.get("checkpoint_id") in order and isinstance(row.get("metrics"), dict)
            ]
            candidates = []
            previous = None
            for row in measured:
                current = float(row["metrics"].get("relative_rmse", 0.0))
                if previous is not None:
                    candidates.append((current - previous[1], previous[0], str(row["checkpoint_id"])))
                previous = (str(row["checkpoint_id"]), current)
            if candidates:
                growth, lower, upper = max(candidates, key=lambda item: item[0])
                if growth <= 0.0:
                    return {
                        "status": "ranking_attributed",
                        "reason": "ranking_failed_without_nonzero_sparse_tensor_growth",
                        "first_failure": None,
                        "passing_lower_bound": order[-1] if order else None,
                        "interval": None,
                        "next_checkpoints": [],
                    }
                expansion = _expand_interval(profile, lower, upper)
                pending = [name for name in expansion if results.get(name) not in {"pass", "fail"}]
                return {
                    "status": "granular_accumulated_drift" if expansion else "ranking_attributed",
                    "reason": "ranking_failed_after_sparse_tensor_thresholds_passed",
                    "first_failure": None,
                    "passing_lower_bound": lower,
                    "interval": f"{lower}->{upper}",
                    "next_checkpoints": pending,
                }
        pending = [name for name in order if results.get(name) not in {"pass", "fail"}]
        return {"status": "sparse", "first_failure": None, "next_checkpoints": pending[:4]}
    index = order.index(first_failure)
    lower = order[index - 1] if index else "<input>"
    interval = f"{lower}->{first_failure}"
    expansion = _expand_interval(profile, lower, first_failure)
    pending = [name for name in expansion if results.get(name) not in {"pass", "fail"}]
    return {
        "status": "granular" if expansion else "attributed_interval",
        "first_failure": first_failure,
        "passing_lower_bound": lower,
        "interval": interval,
        "next_checkpoints": pending,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = plan(load(args.profile), load(args.report))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
