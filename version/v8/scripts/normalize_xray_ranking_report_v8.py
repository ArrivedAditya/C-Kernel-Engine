#!/usr/bin/env python3
"""Normalize mixed-prefill or teacher-forced parity JSON for X-ray."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize(source: dict, kind: str) -> dict:
    checks = []
    if isinstance(source.get("steps"), list):
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
