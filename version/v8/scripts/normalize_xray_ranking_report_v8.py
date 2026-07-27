#!/usr/bin/env python3
"""Normalize mixed-prefill or teacher-forced parity JSON for X-ray."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _trace_top_k(row: dict, backend: str) -> list[dict]:
    top_k = row.get("top_k")
    if not isinstance(top_k, list) or not top_k:
        raise ValueError(f"{backend} trace step {row.get('step')} has no top_k values")
    return top_k


def normalize(source: dict, kind: str, provenance: dict | None = None) -> dict:
    checks = []
    ck_trace = source.get("ck_logit_trace")
    oracle_trace = source.get("torch_logit_trace", source.get("llama_logit_trace"))
    prov = dict(provenance or {})
    subject_backend = prov.get("subject_backend") or source.get("subject_backend", "ck")
    oracle_backend = prov.get("oracle_backend") or source.get("oracle_backend") or (
        "pytorch" if "torch_logit_trace" in source
        else ("llamacpp" if "llama_logit_trace" in source else None)
    )
    run = source.get("run") or {}
    if isinstance(run, dict):
        run_id = prov.get("run_id") or source.get("run_id") or run.get("id") or run.get("run_id")
        phase = prov.get("phase") or source.get("phase") or run.get("phase")
    else:
        run_id = prov.get("run_id") or source.get("run_id")
        phase = prov.get("phase") or source.get("phase")
    model_sha256 = prov.get("model_sha256") or source.get("model_sha256")
    subject_runtime_sha256 = prov.get("subject_runtime_sha256") or (
        (source.get("subject") or {}).get("runtime_sha256")
        if isinstance(source.get("subject"), dict) else None
    )
    subject_generated_model_sha256 = prov.get("subject_generated_model_sha256") or (
        (source.get("subject") or {}).get("generated_model_sha256")
        if isinstance(source.get("subject"), dict) else None
    )
    oracle_runtime_sha256 = prov.get("oracle_runtime_sha256") or (
        (source.get("oracle") or {}).get("runtime_sha256")
        if isinstance(source.get("oracle"), dict) else None
    )
    oracle_fingerprint_sha256 = prov.get("oracle_fingerprint_sha256") or (
        (source.get("oracle") or {}).get("fingerprint_sha256")
        if isinstance(source.get("oracle"), dict) else None
    )
    oracle_components = (
        (source.get("oracle") or {}).get("components")
        if isinstance(source.get("oracle"), dict) else None
    )
    oracle_commit = prov.get("oracle_commit") or (
        (source.get("oracle") or {}).get("commit")
        if isinstance(source.get("oracle"), dict) else None
    )
    oracle_mode = prov.get("oracle_mode") or (
        (source.get("oracle") or {}).get("mode")
        if isinstance(source.get("oracle"), dict) else None
    )
    if isinstance(ck_trace, list) and isinstance(oracle_trace, list):
        if len(ck_trace) != len(oracle_trace):
            raise ValueError(
                f"ranking traces have different lengths: CK={len(ck_trace)} oracle={len(oracle_trace)}"
            )
        for position, (ck_row, oracle_row) in enumerate(zip(ck_trace, oracle_trace)):
            ck_step = int(ck_row.get("step", position))
            oracle_step = int(oracle_row.get("step", position))
            if ck_step != oracle_step:
                raise ValueError(f"ranking trace step mismatch: CK={ck_step} oracle={oracle_step}")
            ck_top_k = _trace_top_k(ck_row, "CK")
            oracle_top_k = _trace_top_k(oracle_row, "oracle")
            ck = int(ck_top_k[0]["token_id"])
            oracle = int(oracle_top_k[0]["token_id"])
            ck_ids = {int(item["token_id"]) for item in ck_top_k}
            oracle_ids = {int(item["token_id"]) for item in oracle_top_k}
            check = {
                "kind": kind,
                "position": ck_step,
                "status": "pass" if ck == oracle else "fail",
                "ck_top1": ck,
                "oracle_top1": oracle,
                "ck_top1_margin": (
                    float(ck_top_k[0]["logit"]) - float(ck_top_k[1]["logit"])
                    if len(ck_top_k) > 1 else None
                ),
                "oracle_top1_margin": (
                    float(oracle_top_k[0]["logit"]) - float(oracle_top_k[1]["logit"])
                    if len(oracle_top_k) > 1 else None
                ),
                "topk_overlap_count": len(ck_ids & oracle_ids),
                "topk": min(len(ck_top_k), len(oracle_top_k)),
            }
            checks.append(check)
    elif isinstance(source.get("steps"), list):
        for row in source["steps"]:
            ck = int(row.get("ck_next", row.get("top1_ck", -1)))
            oracle = int(row.get("llama_next", row.get("torch_next", row.get("top1_llama", -1))))
            checks.append({
                "kind": kind,
                "position": int(row.get("step", len(checks))),
                "status": "pass" if bool(row.get("top1_match", ck == oracle)) else "fail",
                "ck_top1": ck,
                "oracle_top1": oracle,
                "cosine": float(row.get("cosine", 0.0)),
                "rmse": float(row.get("rmse", 0.0)),
                "topk_overlap_count": int(row.get("topk_overlap_count", 0)),
                "topk": int(row.get("top_k", source.get("top_k", 16))),
            })
    else:
        ck = int(source.get("ck_top1", source.get("top1_ck", -1)))
        oracle = int(source.get("torch_top1", source.get("llama_top1", source.get("top1_oracle", -1))))
        checks.append({
            "kind": kind, "position": int(source.get("position", 0)),
            "status": "pass" if ck == oracle else "fail", "ck_top1": ck, "oracle_top1": oracle,
            "cosine": float(source.get("cosine", 0.0)), "rmse": float(source.get("rmse", 0.0)),
            "topk_overlap_count": int(source.get("topk_overlap_count", 0)),
            "topk": int(source.get("top_k", 16)),
        })
    result: dict[str, Any] = {
        "schema": "cke.xray_ranking_report",
        "schema_version": 1,
        "checks": checks,
    }
    if run_id is not None:
        result["run_id"] = str(run_id)
    if phase is not None:
        result["phase"] = str(phase)
    if model_sha256 is not None:
        result["model_sha256"] = str(model_sha256)
    if subject_backend is not None:
        subject: dict[str, Any] = {"backend": str(subject_backend)}
        if subject_runtime_sha256 is not None:
            subject["runtime_sha256"] = str(subject_runtime_sha256)
        if subject_generated_model_sha256 is not None:
            subject["generated_model_sha256"] = str(subject_generated_model_sha256)
        result["subject"] = subject
        result["subject_backend"] = str(subject_backend)
    if oracle_backend is not None:
        oracle: dict[str, Any] = {"backend": str(oracle_backend)}
        if oracle_runtime_sha256 is not None:
            oracle["runtime_sha256"] = str(oracle_runtime_sha256)
        if oracle_fingerprint_sha256 is not None:
            oracle["fingerprint_sha256"] = str(oracle_fingerprint_sha256)
        if oracle_commit is not None:
            oracle["commit"] = str(oracle_commit)
        if oracle_mode is not None:
            oracle["mode"] = str(oracle_mode)
        if isinstance(oracle_components, list) and oracle_components:
            oracle["components"] = oracle_components
        result["oracle"] = oracle
        result["oracle_backend"] = str(oracle_backend)
    if isinstance(run, dict) and run:
        result["run"] = run
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kind", choices=("mixed_prefill", "teacher_forced", "persistent_vs_replay"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    # Provenance flags: correlation in the visualizer fails closed unless the
    # ranking report carries the same complete identity as the parity report.
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--phase", choices=("prefill", "decode", "mixed_prefill", "teacher_forced"), default=None)
    parser.add_argument("--model-sha256", default=None)
    parser.add_argument("--subject-backend", default=None)
    parser.add_argument("--subject-runtime-sha256", default=None)
    parser.add_argument("--subject-generated-model-sha256", default=None)
    parser.add_argument("--oracle-backend", default=None)
    parser.add_argument("--oracle-runtime-sha256", default=None)
    parser.add_argument("--oracle-fingerprint-sha256", default=None)
    parser.add_argument("--oracle-commit", default=None)
    parser.add_argument("--oracle-mode", default=None)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    provenance = {
        "run_id": args.run_id,
        "phase": args.phase,
        "model_sha256": args.model_sha256,
        "subject_backend": args.subject_backend,
        "subject_runtime_sha256": args.subject_runtime_sha256,
        "subject_generated_model_sha256": args.subject_generated_model_sha256,
        "oracle_backend": args.oracle_backend,
        "oracle_runtime_sha256": args.oracle_runtime_sha256,
        "oracle_fingerprint_sha256": args.oracle_fingerprint_sha256,
        "oracle_commit": args.oracle_commit,
        "oracle_mode": args.oracle_mode,
    }
    result = normalize(source, args.kind, provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
