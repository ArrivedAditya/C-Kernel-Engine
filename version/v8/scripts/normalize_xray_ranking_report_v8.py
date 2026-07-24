#!/usr/bin/env python3
"""Normalize mixed-prefill or teacher-forced parity JSON for X-ray."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _trace_top_k(row: dict, backend: str) -> list[dict]:
    top_k = row.get("top_k")
    if not isinstance(top_k, list) or not top_k:
        raise ValueError(f"{backend} trace step {row.get('step')} has no top_k values")
    return top_k


def normalize(source: dict, kind: str) -> dict:
    checks = []
    ck_trace = source.get("ck_logit_trace")
    oracle_trace = source.get("torch_logit_trace", source.get("llama_logit_trace"))
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
    return {"schema": "cke.xray_ranking_report", "schema_version": 1, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kind", choices=("mixed_prefill", "teacher_forced", "persistent_vs_replay"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = normalize(source, args.kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
