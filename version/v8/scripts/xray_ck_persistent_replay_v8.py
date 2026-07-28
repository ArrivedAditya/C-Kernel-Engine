#!/usr/bin/env python3
from __future__ import annotations

"""Compare one CK persistent-decode step with CK full replay at the same prefix."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROFILE = (
    Path(__file__).resolve().parent.parent
    / "parity_profiles"
    / "text_ck_persistent_replay_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _operation(
    call_ir: dict[str, Any], layer: int, op_name: str, occurrence: int
) -> dict[str, Any] | None:
    matches = [
        op
        for op in call_ir.get("operations", [])
        if int(op.get("layer", -1)) == layer and str(op.get("op", "")) == op_name
    ]
    return matches[occurrence] if occurrence < len(matches) else None


def _execution(op: dict[str, Any] | None, phase: str) -> dict[str, Any]:
    if op is None:
        return {
            "phase": phase,
            "op_idx": None,
            "kernel_id": None,
            "function": None,
            "resolved_contract_id": None,
        }
    resolved = op.get("resolved_contract") if isinstance(op.get("resolved_contract"), dict) else {}
    execution = op.get("resolved_execution") if isinstance(op.get("resolved_execution"), dict) else {}
    return {
        "phase": phase,
        "layer": int(op.get("layer", -1)),
        "op_idx": int(op["idx"]) if op.get("idx") is not None else None,
        "kernel_id": str(op.get("kernel") or execution.get("kernel_id") or "") or None,
        "function": str(op.get("function") or execution.get("function") or "") or None,
        "resolved_contract_id": str(
            resolved.get("resolved_contract_id")
            or resolved.get("contract_id")
            or execution.get("resolved_contract_id")
            or ""
        ) or None,
    }


def _last_replay_row(
    values: np.ndarray,
    *,
    layout: str,
    tokens: int,
    config: dict[str, Any],
) -> np.ndarray:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if layout == "token_major":
        if values.size % tokens:
            raise ValueError(f"token-major extent {values.size} is not divisible by {tokens}")
        return values.reshape(tokens, -1)[-1].copy()
    heads_key = "num_heads" if layout == "q_head_major" else "num_kv_heads"
    heads = int(config.get(heads_key, 0))
    if heads <= 0 or values.size % (heads * tokens):
        raise ValueError(
            f"{layout} extent {values.size} does not match heads={heads}, tokens={tokens}"
        )
    return values.reshape(heads, tokens, -1)[:, -1, :].copy().reshape(-1)


def _metrics(subject: np.ndarray, oracle: np.ndarray) -> dict[str, Any]:
    if subject.shape != oracle.shape:
        raise ValueError(f"shape mismatch: {subject.shape} != {oracle.shape}")
    delta = subject.astype(np.float64) - oracle.astype(np.float64)
    abs_delta = np.abs(delta)
    rmse = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    oracle_rms = float(np.sqrt(np.mean(oracle.astype(np.float64) ** 2))) if oracle.size else 0.0
    subject_norm = float(np.linalg.norm(subject.astype(np.float64)))
    oracle_norm = float(np.linalg.norm(oracle.astype(np.float64)))
    cosine = (
        float(np.dot(subject.astype(np.float64), oracle.astype(np.float64)) / (subject_norm * oracle_norm))
        if subject_norm > 0.0 and oracle_norm > 0.0
        else 1.0
    )
    exact = int(np.count_nonzero(subject.view(np.uint32) == oracle.view(np.uint32)))
    return {
        "byte_exact": exact == subject.size,
        "exact_elements": exact,
        "total_elements": int(subject.size),
        "exact_ratio": float(exact / subject.size) if subject.size else 1.0,
        "max_abs": float(abs_delta.max()) if abs_delta.size else 0.0,
        "mean_abs": float(abs_delta.mean()) if abs_delta.size else 0.0,
        "rmse": rmse,
        "relative_rmse": float(rmse / oracle_rms) if oracle_rms > 0.0 else rmse,
        "cosine": cosine,
    }


def _same_execution_contract(
    subject: dict[str, Any], oracle: dict[str, Any]
) -> bool:
    """Return true only when both captures identify the same implementation."""
    identity_fields = ("kernel_id", "function", "resolved_contract_id")
    known = False
    for field in identity_fields:
        subject_value = subject.get(field)
        oracle_value = oracle.get(field)
        if subject_value is None or oracle_value is None:
            continue
        known = True
        if subject_value != oracle_value:
            return False
    return known


def compare_captures(
    *,
    persistent_dir: Path,
    replay_dir: Path,
    decode_call_ir: dict[str, Any],
    prefill_call_ir: dict[str, Any],
    layer: int,
    logical_token: int,
    replay_tokens: int,
    profile: dict[str, Any],
) -> dict[str, Any]:
    config = (
        prefill_call_ir.get("config")
        if isinstance(prefill_call_ir.get("config"), dict)
        else {}
    )
    rows: list[dict[str, Any]] = []
    rows_by_checkpoint: dict[str, dict[str, Any]] = {}
    first_observed_divergence = None
    first_causal_divergence = None
    mappings = profile.get("backend_mappings")
    if not isinstance(mappings, dict):
        raise ValueError("parity profile requires backend_mappings")
    for sequence_index, checkpoint_template in enumerate(profile.get("checkpoint_order", [])):
        mapping = mappings.get(checkpoint_template)
        if not isinstance(mapping, dict):
            raise ValueError(f"missing checkpoint mapping for {checkpoint_template}")
        name = str(mapping.get("capture_tensor") or "")
        layout = str(mapping.get("logical_layout") or "")
        op_name = str(mapping.get("operation") or "")
        occurrence = int(mapping.get("occurrence", 0))
        checkpoint_id = str(checkpoint_template).replace("{layer}", str(layer))
        dependencies = [
            str(item).replace("{layer}", str(layer))
            for item in mapping.get("depends_on", [])
        ]
        if not name or not layout or not op_name:
            raise ValueError(
                f"checkpoint mapping {checkpoint_template} requires "
                "capture_tensor, logical_layout, and operation"
            )
        persistent_path = persistent_dir / (
            f"tok_{logical_token:04d}_layer_{layer:03d}_{name}.f32"
        )
        replay_path = replay_dir / f"tok_0000_layer_{layer:03d}_{name}.f32"
        if not persistent_path.is_file() or not replay_path.is_file():
            continue
        persistent = np.fromfile(persistent_path, dtype=np.float32)
        replay = _last_replay_row(
            np.fromfile(replay_path, dtype=np.float32),
            layout=layout,
            tokens=replay_tokens,
            config=config,
        )
        metrics = _metrics(persistent, replay)
        persistent_op = _operation(decode_call_ir, layer, op_name, occurrence)
        replay_op = _operation(prefill_call_ir, layer, op_name, occurrence)
        subject_execution = _execution(persistent_op, "decode")
        oracle_execution = _execution(replay_op, "prefill")
        same_execution = _same_execution_contract(
            subject_execution, oracle_execution
        )
        divergent_dependencies = [
            dependency
            for dependency in dependencies
            if (
                dependency in rows_by_checkpoint
                and rows_by_checkpoint[dependency]["status"] != "pass"
            )
        ]
        if metrics["byte_exact"]:
            classification = "MATCH"
            attribution_status = "not_applicable"
        elif divergent_dependencies:
            classification = "PROPAGATED_PROVIDER_SCHEDULE_DIFFERENCE"
            attribution_status = "non_causal_propagated"
        elif same_execution:
            classification = "PERSISTENT_REPLAY_DIVERGENCE"
            attribution_status = "causal_candidate"
        else:
            classification = "PROVIDER_SCHEDULE_DIFFERENCE"
            attribution_status = "non_causal_mode_change"
        row = {
            "sequence_index": sequence_index,
            "checkpoint_id": checkpoint_id,
            "op_idx": subject_execution["op_idx"],
            "layer": layer,
            "status": "pass" if metrics["byte_exact"] else "fail",
            "classification": classification,
            "attribution_status": attribution_status,
            "metrics": metrics,
            "resolved_execution": subject_execution,
            "subject_execution": subject_execution,
            "oracle_execution": oracle_execution,
            "depends_on": dependencies,
            "divergent_dependencies": divergent_dependencies,
            "subject_tensor": {
                "path": str(persistent_path),
                "sha256": _sha256(persistent_path),
            },
            "oracle_tensor": {
                "path": str(replay_path),
                "sha256": _sha256(replay_path),
                "row_selection": f"last logical token {replay_tokens - 1}",
            },
        }
        rows.append(row)
        rows_by_checkpoint[checkpoint_id] = row
        if first_observed_divergence is None and row["status"] == "fail":
            first_observed_divergence = dict(row)
        if (
            first_causal_divergence is None
            and row["status"] == "fail"
            and same_execution
            and not divergent_dependencies
        ):
            first_causal_divergence = {
                **row,
                "fix_owner": "persistent_decode_state",
                "recommended_action": (
                    "Compare persistent state and cache bytes at this first "
                    "non-exact edge; both captures resolve the same provider."
                ),
            }
    observed_only = first_observed_divergence is not None and first_causal_divergence is None
    return {
        "schema": "cke.xray_numerical_report",
        "schema_version": 1,
        "backend": "ck_persistent_vs_full_replay",
        "subject_backend": "ck_persistent",
        "oracle_backend": "ck_full_replay",
        "run": {
            "phase": "decode",
            "layer": layer,
            "logical_token": logical_token,
            "replay_tokens": replay_tokens,
        },
        "status": (
            "observed"
            if observed_only
            else ("fail" if first_causal_divergence else "pass")
        ),
        "comparisons": rows,
        "first_divergence": first_causal_divergence,
        "first_observed_divergence": first_observed_divergence,
        "last_passing_checkpoint": next(
            (row["checkpoint_id"] for row in reversed(rows) if row["status"] == "pass"),
            None,
        ),
        "acceptance_policy": (
            "byte-exact diagnostic; no tolerance is applied; differences between "
            "distinct provider schedules are observations, not causal attribution"
        ),
        "provenance": {
            "profile": profile.get("name"),
            "persistent_dir": str(persistent_dir),
            "replay_dir": str(replay_dir),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--decode-call-ir", type=Path, required=True)
    parser.add_argument("--prefill-call-ir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--logical-token", type=int, required=True)
    parser.add_argument("--replay-tokens", type=int, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_captures(
        persistent_dir=args.persistent_dir.resolve(),
        replay_dir=args.replay_dir.resolve(),
        decode_call_ir=json.loads(args.decode_call_ir.read_text(encoding="utf-8")),
        prefill_call_ir=json.loads(args.prefill_call_ir.read_text(encoding="utf-8")),
        layer=int(args.layer),
        logical_token=int(args.logical_token),
        replay_tokens=int(args.replay_tokens),
        profile=json.loads(args.profile.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.get("first_divergence"), sort_keys=True))
    return 0 if report["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
