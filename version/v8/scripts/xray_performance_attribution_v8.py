#!/usr/bin/env python3
"""Aggregate CK X-ray timings and account for a CK-versus-llama wall-time gap.

CK_PROFILE provides exact generated-runtime layer/op timings. llama.cpp's
ordinary CLI exposes an end-to-end prompt time, but not compatible per-node
timings, so the report labels that external remainder explicitly instead of
inventing a cross-runtime layer mapping.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _rows(path: Path, mode: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("entries")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain an entries list")
    return [row for row in rows if str(row.get("mode", "")) == mode]


def _group(
    rows: Iterable[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    totals: dict[tuple[Any, ...], float] = defaultdict(float)
    calls: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        totals[key] += float(row.get("time_us", 0.0) or 0.0)
        calls[key] += 1
    result = []
    for key, time_us in sorted(totals.items(), key=lambda item: -item[1]):
        item = {field: value for field, value in zip(fields, key)}
        item.update(time_ms=time_us / 1000.0, calls=calls[key])
        result.append(item)
    return result


def build_report(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    ck_wall_ms: float | None,
    llama_wall_ms: float | None,
) -> dict[str, Any]:
    profiled_ms = sum(float(row.get("time_us", 0.0) or 0.0) for row in rows) / 1000.0
    report: dict[str, Any] = {
        "mode": mode,
        "profiled_ck_ms": profiled_ms,
        "ck_wall_ms": ck_wall_ms,
        "llama_wall_ms": llama_wall_ms,
        "ck_unprofiled_ms": (
            max(0.0, ck_wall_ms - profiled_ms) if ck_wall_ms is not None else None
        ),
        "ck_vs_llama_speed_ratio": (
            ck_wall_ms / llama_wall_ms
            if ck_wall_ms is not None and llama_wall_ms not in (None, 0.0)
            else None
        ),
        "external_gap_ms": (
            ck_wall_ms - llama_wall_ms
            if ck_wall_ms is not None and llama_wall_ms is not None
            else None
        ),
        "by_layer": _group(rows, ("layer",)),
        "by_op": _group(rows, ("op",)),
        "by_kernel_op": _group(rows, ("kernel", "op")),
        "by_layer_op": _group(rows, ("layer", "op")),
        "attribution_contract": {
            "ck": "generated CK_PROFILE layer/op timers",
            "llama": "end-to-end wall time; per-layer attribution unavailable",
            "rule": "Do not present CK layer timers as llama.cpp layer timers.",
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CK per-layer/op performance X-ray with llama wall-time accounting"
    )
    parser.add_argument("--ck-profile", type=Path, required=True)
    parser.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--ck-wall-ms", type=float)
    parser.add_argument("--llama-wall-ms", type=float)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = build_report(
        _rows(args.ck_profile, args.mode),
        mode=args.mode,
        ck_wall_ms=args.ck_wall_ms,
        llama_wall_ms=args.llama_wall_ms,
    )
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"CK profiled={report['profiled_ck_ms']:.3f} ms"
        + (
            f", wall={report['ck_wall_ms']:.3f} ms"
            if report["ck_wall_ms"] is not None
            else ""
        )
        + (
            f", llama wall={report['llama_wall_ms']:.3f} ms"
            if report["llama_wall_ms"] is not None
            else ""
        )
    )
    print("rank  time_ms  calls  layer  op")
    for rank, row in enumerate(report["by_layer_op"][: args.top], 1):
        print(
            f"{rank:4d}  {row['time_ms']:7.3f}  {row['calls']:5d}  "
            f"{str(row['layer']):>5}  {row['op']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
