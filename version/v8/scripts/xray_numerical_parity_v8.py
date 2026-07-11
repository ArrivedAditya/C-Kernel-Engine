#!/usr/bin/env python3
"""Canonical checkpoint comparison and bounded numerical divergence diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from jsonschema import Draft202012Validator


V8_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = V8_ROOT / "schemas" / "checkpoint_manifest.schema.json"
PROFILE_SCHEMA = V8_ROOT / "schemas" / "parity_profile.schema.json"
RANKING_SCHEMA = V8_ROOT / "schemas" / "xray_ranking_report.schema.json"
PLANNER_PATH = Path(__file__).resolve().parent / "plan_parity_bisection_v8.py"


class XRayError(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise XRayError(f"expected JSON object: {path}")
    return value


def validate(value: Dict[str, Any], schema_path: Path, context: str) -> None:
    errors = sorted(
        Draft202012Validator(load_json(schema_path)).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise XRayError(f"{context} violates {schema_path.name} at {location}: {error.message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tensor(entry: Dict[str, Any]) -> np.ndarray:
    path = Path(entry["tensor_path"])
    dtype = entry["exported_dtype"]
    if dtype == "fp32":
        values = np.fromfile(path, dtype=np.float32)
    elif dtype == "fp16":
        values = np.fromfile(path, dtype=np.float16).astype(np.float32)
    elif dtype == "bf16":
        raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32)
        values = (raw << 16).view(np.float32)
    else:
        raise XRayError(f"unsupported exported dtype {dtype!r}")
    physical_shape = tuple(int(value) for value in entry["physical_shape"])
    if math.prod(physical_shape) != values.size:
        raise XRayError(
            f"{entry['checkpoint_id']}: file has {values.size} values, "
            f"physical_shape requires {math.prod(physical_shape)}"
        )
    tensor = values.reshape(physical_shape)
    physical_axes = list(entry["physical_axis_names"])
    logical_axes = list(entry["axis_names"])
    if set(physical_axes) != set(logical_axes) or len(physical_axes) != len(logical_axes):
        raise XRayError(
            f"{entry['checkpoint_id']}: physical axes {physical_axes} cannot canonicalize to {logical_axes}"
        )
    permutation = [physical_axes.index(axis) for axis in logical_axes]
    tensor = np.transpose(tensor, axes=permutation) if permutation != list(range(len(permutation))) else tensor
    logical_shape = tuple(int(value) for value in entry["logical_shape"])
    if tensor.shape != logical_shape:
        raise XRayError(
            f"{entry['checkpoint_id']}: canonical shape {tensor.shape} != declared {logical_shape}"
        )
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _index_manifest(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    validate(manifest, MANIFEST_SCHEMA, f"{manifest.get('backend', 'backend')} checkpoint manifest")
    indexed: Dict[str, Dict[str, Any]] = {}
    for entry in manifest["checkpoints"]:
        checkpoint_id = entry["checkpoint_id"]
        if checkpoint_id in indexed:
            raise XRayError(f"duplicate checkpoint ID in manifest: {checkpoint_id}")
        path = Path(entry["tensor_path"])
        if not path.is_file():
            raise XRayError(f"checkpoint tensor does not exist: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise XRayError(f"checkpoint tensor checksum changed: {path}")
        indexed[checkpoint_id] = entry
    return indexed


def _metadata_fault(subject: Dict[str, Any], oracle: Dict[str, Any], required: Iterable[str]) -> tuple[str, str] | None:
    field_map = {"checkpoint_id": "checkpoint_id", "producer": "producer", "logical_layout": "logical_layout", "axis_names": "axis_names", "resolved_contract_id": "resolved_contract_id", "kernel_id": "kernel_id", "function": "function"}
    for requested in required:
        field = field_map.get(requested, requested)
        if subject.get(field) == oracle.get(field):
            continue
        if field == "producer":
            return "CIRCUIT_PRODUCER_MISMATCH", field
        if field in {"logical_layout", "axis_names", "logical_shape"}:
            return "LAYOUT_MISMATCH", field
        if field == "resolved_contract_id":
            checkpoint_id = str(subject.get("checkpoint_id", ""))
            if "rope" in checkpoint_id or "position" in checkpoint_id:
                return "POSITION_CONTRACT_MISMATCH", field
            if ".attention." in checkpoint_id:
                return "REDUCTION_CONTRACT_MISMATCH", field
            return "NUMERICAL_CONTRACT_MISMATCH", field
        if field in {"kernel_id", "function"}:
            return "KERNEL_BINDING_MISMATCH", field
        return "CHECKPOINT_ABI_MISMATCH", field
    if subject["storage_dtype"] != oracle["storage_dtype"]:
        return "STORAGE_CONTRACT_MISMATCH", "storage_dtype"
    return None


REMEDIATIONS = {
    "MISSING_CHECKPOINT": "Add the semantic checkpoint exporter or backend mapping; do not compare a substitute tensor.",
    "CIRCUIT_PRODUCER_MISMATCH": "Fix the circuit producer/consumer edge or stale backend mapping.",
    "LAYOUT_MISMATCH": "Correct named-axis canonicalization or the declared logical tensor layout.",
    "STORAGE_CONTRACT_MISMATCH": "Declare and implement the backend-matched storage/rounding boundary through the circuit and kernel map.",
    "REDUCTION_CONTRACT_MISMATCH": "Select or implement a kernel with the required accumulator, reduction, merge, and threading order.",
    "POSITION_CONTRACT_MISMATCH": "Align RoPE/M-RoPE pairing, width, axes, frequency precision, and rounding contract.",
    "NUMERICAL_CONTRACT_MISMATCH": "Register the measured semantic contract and resolve exactly one compatible kernel.",
    "KERNEL_BINDING_MISMATCH": "Fix resolver/lowering propagation; generated IR must retain the exact kernel ID and function.",
    "DIAGNOSTIC_EXPORT_MAPPING": "Fix exporter extent, dtype, shape, or axis metadata before blaming model math.",
    "KERNEL_IMPLEMENTATION_DIVERGENCE": "Reproduce the checkpoint in the isolated kernel parity test and fix its arithmetic.",
    "NONFINITE_OUTPUT": "Stop at this edge and fix the first NaN/Inf-producing kernel or input contract.",
    "RANKING_DIVERGENCE": "Attribute logits at this teacher-forced position; do not follow independently generated tokens.",
    "STATE_CACHE_DIVERGENCE": "Compare persistent decode with full replay and fix cache/state commit semantics.",
    "MISSING_TOLERANCE_PROFILE": "Add a backend/dtype threshold to the parity profile, not the circuit.",
}


def _metrics(reference: np.ndarray, actual: np.ndarray, axes: list[str]) -> Dict[str, Any]:
    if reference.shape != actual.shape:
        raise XRayError(f"canonical tensor shape mismatch: {reference.shape} != {actual.shape}")
    ref64 = reference.astype(np.float64, copy=False)
    got64 = actual.astype(np.float64, copy=False)
    diff = got64 - ref64
    abs_diff = np.abs(diff)
    flat_index = int(np.argmax(abs_diff)) if diff.size else 0
    coordinate = np.unravel_index(flat_index, diff.shape) if diff.size else tuple(0 for _ in diff.shape)
    denom = float(np.linalg.norm(ref64) * np.linalg.norm(got64))
    rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    ref_rms = float(np.sqrt(np.mean(ref64 * ref64))) if diff.size else 0.0
    return {
        "cosine": float(np.dot(ref64.reshape(-1), got64.reshape(-1)) / denom) if denom else 1.0,
        "rmse": rmse,
        "relative_rmse": rmse / ref_rms if ref_rms else (0.0 if rmse == 0.0 else float("inf")),
        "mean_abs": float(np.mean(abs_diff)) if diff.size else 0.0,
        "max_abs": float(abs_diff.reshape(-1)[flat_index]) if diff.size else 0.0,
        "worst_coordinate": {axis: int(value) for axis, value in zip(axes, coordinate)},
        "finite": bool(np.isfinite(reference).all() and np.isfinite(actual).all()),
    }


def _metric_status(metrics: Dict[str, Any], threshold: Dict[str, Any]) -> tuple[str, list[str]]:
    failures = []
    if threshold["finite_required"] and not metrics["finite"]:
        failures.append("nonfinite")
    if metrics["cosine"] < threshold["cosine_min"]:
        failures.append("cosine")
    if metrics["rmse"] > threshold["rmse_max"]:
        failures.append("rmse")
    if metrics["relative_rmse"] > threshold["relative_rmse_max"]:
        failures.append("relative_rmse")
    if metrics["max_abs"] > threshold["max_abs_max"]:
        failures.append("max_abs")
    return ("fail" if failures else "pass"), failures


def _load_planner():
    spec = importlib.util.spec_from_file_location("plan_parity_bisection_v8", PLANNER_PATH)
    if spec is None or spec.loader is None:
        raise XRayError(f"cannot load parity planner: {PLANNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_manifests(
    subject: Dict[str, Any],
    oracle: Dict[str, Any],
    profile: Dict[str, Any],
    ranking_report: Dict[str, Any] | None = None,
    checkpoint_order: list[str] | None = None,
) -> Dict[str, Any]:
    validate(profile, PROFILE_SCHEMA, "parity profile")
    subject_index = _index_manifest(subject)
    oracle_index = _index_manifest(oracle)
    rows = []
    first_classification = None
    last_passing_checkpoint = None
    unresolved_contracts = []
    active_order = checkpoint_order or profile["checkpoint_order"]
    for checkpoint_id in active_order:
        left = subject_index.get(checkpoint_id)
        right = oracle_index.get(checkpoint_id)
        if left is None or right is None:
            row = {"checkpoint_id": checkpoint_id, "status": "fail", "classification": "MISSING_CHECKPOINT", "subject_present": left is not None, "oracle_present": right is not None}
            rows.append(row)
            first_classification = row
            break
        if left.get("resolved_contract_id") == "unresolved":
            unresolved_contracts.append(checkpoint_id)
        mismatch = _metadata_fault(left, right, profile["required_match_fields"])
        if mismatch is not None:
            classification, field = mismatch
            row = {"checkpoint_id": checkpoint_id, "status": "fail", "classification": classification, "field": field, "subject": left.get(field), "oracle": right.get(field)}
        else:
            try:
                ref = _load_tensor(right)
                got = _load_tensor(left)
                metrics = _metrics(ref, got, left["axis_names"])
            except (XRayError, ValueError) as exc:
                row = {"checkpoint_id": checkpoint_id, "status": "fail", "classification": "DIAGNOSTIC_EXPORT_MAPPING", "detail": str(exc)}
                rows.append(row)
                first_classification = row
                break
            threshold_key = left["storage_dtype"] if left["storage_dtype"] in profile["dtype_thresholds"] else left["exported_dtype"]
            threshold = profile["dtype_thresholds"].get(threshold_key)
            if threshold is None:
                row = {"checkpoint_id": checkpoint_id, "status": "fail", "classification": "MISSING_TOLERANCE_PROFILE", "dtype": threshold_key}
            else:
                status, failed_metrics = _metric_status(metrics, threshold)
                classification = "NONFINITE_OUTPUT" if "nonfinite" in failed_metrics else ("KERNEL_IMPLEMENTATION_DIVERGENCE" if status == "fail" else "MATCH")
                row = {"checkpoint_id": checkpoint_id, "status": status, "classification": classification, "metrics": metrics, "failed_metrics": failed_metrics, "threshold": threshold}
        rows.append(row)
        if row["status"] == "pass":
            last_passing_checkpoint = checkpoint_id
        if row["status"] == "fail" and first_classification is None:
            first_classification = row
            break

    ranking = None
    if first_classification is None and ranking_report is not None:
        validate(ranking_report, RANKING_SCHEMA, "X-ray ranking report")
        checks = ranking_report.get("checks", [])
        first = next((item for item in checks if item.get("status") == "fail"), None)
        if first is not None:
            kind = str(first.get("kind", "ranking"))
            classification = "STATE_CACHE_DIVERGENCE" if kind == "persistent_vs_replay" else "RANKING_DIVERGENCE"
            ranking = {**first, "classification": classification}
            first_classification = ranking

    if first_classification is not None:
        classification = str(first_classification.get("classification", ""))
        first_classification["recommended_action"] = REMEDIATIONS.get(
            classification, "Inspect the first failing semantic edge and its resolved contract metadata."
        )

    planner = _load_planner()
    plan = planner.plan(profile, {"comparisons": rows}, checkpoint_order=active_order)
    return {
        "schema": "cke.xray_numerical_report",
        "schema_version": 1,
        "subject_backend": subject["backend"],
        "oracle_backend": oracle["backend"],
        "status": "fail" if first_classification is not None else "pass",
        "comparisons": rows,
        "first_divergence": first_classification,
        "last_passing_checkpoint": last_passing_checkpoint,
        "unresolved_contract_checkpoints": unresolved_contracts,
        "ranking_divergence": ranking,
        "next_plan": plan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-manifest", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ranking-report", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_manifests(
        load_json(args.subject_manifest), load_json(args.oracle_manifest), load_json(args.profile),
        load_json(args.ranking_report) if args.ranking_report else None,
        args.checkpoint or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    divergence = result.get("first_divergence") or {}
    print(f"status={result['status']}")
    if divergence:
        print(f"fail_at={divergence.get('checkpoint_id', divergence.get('position', 'ranking'))}")
        print(f"class={divergence.get('classification')}")
    print(f"next={','.join(result['next_plan'].get('next_checkpoints', []))}")
    return 1 if result["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
