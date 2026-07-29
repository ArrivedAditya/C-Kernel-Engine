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


def _model_layer_count(*call_irs: dict[str, Any]) -> int:
    for call_ir in call_irs:
        config = call_ir.get("config")
        if not isinstance(config, dict):
            continue
        for key in ("num_layers", "block_count", "n_layer"):
            value = config.get(key)
            if value is not None and int(value) > 0:
                return int(value)
    layers = {
        int(op["layer"])
        for call_ir in call_irs
        for op in call_ir.get("operations", [])
        if op.get("layer") is not None and int(op["layer"]) >= 0
    }
    return max(layers) + 1 if layers else 0


def _capture_scope(
    *,
    comparison: str,
    phase: str,
    model_layer_count: int,
    selected_layers: list[int],
    checkpoint_granularity: str,
) -> dict[str, Any]:
    covered = len(set(selected_layers))
    return {
        "comparison": comparison,
        "phase": phase,
        "model_layer_count": model_layer_count,
        "selected_layers": selected_layers,
        "covered_layer_count": covered,
        "coverage_complete": model_layer_count > 0 and covered == model_layer_count,
        "checkpoint_granularity": checkpoint_granularity,
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
    model_layer_count = _model_layer_count(decode_call_ir, prefill_call_ir)
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
        "capture_scope": _capture_scope(
            comparison="CK persistent decode vs CK full replay",
            phase="decoder",
            model_layer_count=model_layer_count,
            selected_layers=[layer],
            checkpoint_granularity="detailed_operator_chain",
        ),
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


def compare_layer_sweep(
    *,
    persistent_dir: Path,
    replay_dir: Path,
    decode_call_ir: dict[str, Any],
    prefill_call_ir: dict[str, Any],
    logical_token: int,
    replay_tokens: int,
    checkpoint: str = "layer_out",
    amplification_ratio: float = 100.0,
) -> dict[str, Any]:
    model_layer_count = _model_layer_count(decode_call_ir, prefill_call_ir)
    if model_layer_count <= 0:
        raise ValueError("cannot determine model layer count from call IR")
    rows: list[dict[str, Any]] = []
    first_observed = None
    first_amplification = None
    previous_rmse = 0.0
    for layer in range(model_layer_count):
        persistent_path = persistent_dir / (
            f"tok_{logical_token:04d}_layer_{layer:03d}_{checkpoint}.f32"
        )
        replay_path = replay_dir / (
            f"tok_0000_layer_{layer:03d}_{checkpoint}.f32"
        )
        if not persistent_path.is_file() or not replay_path.is_file():
            continue
        persistent = np.fromfile(persistent_path, dtype=np.float32)
        replay = _last_replay_row(
            np.fromfile(replay_path, dtype=np.float32),
            layout="token_major",
            tokens=replay_tokens,
            config={},
        )
        metrics = _metrics(persistent, replay)
        rmse = float(metrics["rmse"])
        growth = (
            float(rmse / previous_rmse)
            if previous_rmse > 0.0
            else (float("inf") if rmse > 0.0 else 1.0)
        )
        op_name = "residual_add" if checkpoint == "layer_out" else "residual_save"
        operation = _operation(decode_call_ir, layer, op_name, 0)
        row = {
            "sequence_index": len(rows),
            "checkpoint_id": f"text.layer.{layer}.{checkpoint}",
            "op_idx": int(operation["idx"]) if operation and operation.get("idx") is not None else None,
            "layer": layer,
            "status": "pass" if metrics["byte_exact"] else "observed",
            "classification": "MATCH" if metrics["byte_exact"] else "CROSS_PHASE_SCHEDULE_DRIFT",
            "attribution_status": "coarse_sweep",
            "metrics": metrics,
            "rmse_growth_from_previous_layer": growth,
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
        if first_observed is None and not metrics["byte_exact"]:
            first_observed = dict(row)
        if (
            first_amplification is None
            and previous_rmse > 0.0
            and growth >= amplification_ratio
        ):
            first_amplification = {
                **row,
                "classification": "LAYER_AMPLIFICATION_CANDIDATE",
                "recommended_action": (
                    f"Capture the detailed operator chain for layer {layer}; "
                    f"RMSE grew {growth:.1f}x from the previous layer."
                ),
            }
        previous_rmse = rmse
    selected_layers = [int(row["layer"]) for row in rows]
    return {
        "schema": "cke.xray_numerical_report",
        "schema_version": 1,
        "backend": "ck_persistent_vs_full_replay",
        "subject_backend": "ck_persistent",
        "oracle_backend": "ck_full_replay",
        "run": {
            "phase": "decode",
            "logical_token": logical_token,
            "replay_tokens": replay_tokens,
        },
        "capture_scope": _capture_scope(
            comparison="CK persistent decode vs CK full replay",
            phase="decoder",
            model_layer_count=model_layer_count,
            selected_layers=selected_layers,
            checkpoint_granularity=f"coarse_{checkpoint}_layer_sweep",
        ),
        "status": "observed" if first_observed else "pass",
        "comparisons": rows,
        "first_divergence": None,
        "first_observed_divergence": first_observed,
        "first_layer_amplification": first_amplification,
        "amplification_rule": {
            "metric": "rmse_growth_from_previous_layer",
            "minimum_ratio": amplification_ratio,
            "classification_only": True,
        },
        "acceptance_policy": (
            "coarse cross-phase diagnostic; differences are observations until "
            "a detailed same-provider or external-oracle capture attributes them"
        ),
        "provenance": {
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
    parser.add_argument("--layer", type=int)
    parser.add_argument("--sweep-all-layers", action="store_true")
    parser.add_argument("--sweep-checkpoint", default="layer_out")
    parser.add_argument("--amplification-ratio", type=float, default=100.0)
    parser.add_argument("--logical-token", type=int, required=True)
    parser.add_argument("--replay-tokens", type=int, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sweep_all_layers and args.layer is not None:
        parser.error("--layer and --sweep-all-layers are mutually exclusive")
    decode_call_ir = json.loads(args.decode_call_ir.read_text(encoding="utf-8"))
    prefill_call_ir = json.loads(args.prefill_call_ir.read_text(encoding="utf-8"))
    common = {
        "persistent_dir": args.persistent_dir.resolve(),
        "replay_dir": args.replay_dir.resolve(),
        "decode_call_ir": decode_call_ir,
        "prefill_call_ir": prefill_call_ir,
        "logical_token": int(args.logical_token),
        "replay_tokens": int(args.replay_tokens),
    }
    if args.sweep_all_layers:
        report = compare_layer_sweep(
            **common,
            checkpoint=str(args.sweep_checkpoint),
            amplification_ratio=float(args.amplification_ratio),
        )
    else:
        report = compare_captures(
            **common,
            layer=int(args.layer if args.layer is not None else 0),
            profile=json.loads(args.profile.read_text(encoding="utf-8")),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.get("first_divergence"), sort_keys=True))
    return 0 if report["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
