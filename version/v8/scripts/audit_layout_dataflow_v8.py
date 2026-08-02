#!/usr/bin/env python3
"""Audit physical layout conversions in generated v8 execution IR.

Hardware profilers measure cache and DRAM traffic, but they cannot determine
whether a tensor movement was architecturally necessary.  This tool accounts
for the logical bytes implied by generated layout bridges so those bytes can be
compared with VTune, Advisor, or memory-controller measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "bf16": 2,
    "f16": 2,
    "float16": 2,
    "float32": 4,
    "fp16": 2,
    "fp32": 4,
}

LAYOUT_OPERATIONS = {
    "transpose_qkv_to_head_major": ("query", "num_heads", "tokens"),
    "transpose_kv_to_head_major": ("key_or_value", "num_kv_heads", "tokens"),
    "transpose_attn_out_to_token_major": ("attention_output", "num_heads", "tokens"),
    "transpose_cross_q_to_head_major": ("cross_query", "num_heads", "tokens"),
    "transpose_cross_kv_to_head_major": ("cross_key_or_value", "num_kv_heads", "kv_tokens"),
    "transpose_cross_attn_out_to_token_major": (
        "cross_attention_output",
        "num_heads",
        "tokens",
    ),
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"IR root must be an object: {path}")
    return payload


def _operations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("operations")
    if not isinstance(rows, list):
        raise ValueError("IR must contain an operations array")
    return [row for row in rows if isinstance(row, dict)]


def _dtype(op: dict[str, Any]) -> str:
    for group_name in ("outputs", "activations"):
        group = op.get(group_name)
        if not isinstance(group, dict):
            continue
        for value in group.values():
            if isinstance(value, dict) and value.get("dtype"):
                return str(value["dtype"]).lower()
    return "fp32"


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def audit(
    payload: dict[str, Any],
    *,
    tokens: int,
    kv_tokens: int | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    tokens = _positive_int(tokens, "tokens")
    kv_tokens = tokens if kv_tokens is None else _positive_int(kv_tokens, "kv_tokens")
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    findings: list[dict[str, Any]] = []

    for op in _operations(payload):
        op_name = str(op.get("op", ""))
        spec = LAYOUT_OPERATIONS.get(op_name)
        if spec is None:
            continue
        tensor_role, head_key, token_key = spec
        params = op.get("params") if isinstance(op.get("params"), dict) else {}
        heads = _positive_int(params.get(head_key, config.get(head_key)), head_key)
        head_dim = _positive_int(params.get("head_dim", config.get("head_dim")), "head_dim")
        runtime_tokens = tokens if token_key == "tokens" else kv_tokens
        dtype = _dtype(op)
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported layout-conversion dtype {dtype!r} at op {op.get('idx')}")

        payload_bytes = runtime_tokens * heads * head_dim * DTYPE_BYTES[dtype]
        # Current generated bridges copy to a temporary buffer and then back.
        copy_passes = 2
        copied_bytes = payload_bytes * copy_passes
        logical_read_write_bytes = copied_bytes * 2
        resolved_contract = op.get("resolved_contract_id") or op.get("numerical_contract")
        kernel_id = op.get("kernel_id")
        mapped = bool(resolved_contract and kernel_id)
        parallel = op.get("parallel") if isinstance(op.get("parallel"), dict) else {}
        parallel_enabled = bool(parallel.get("enabled"))
        ownership = parallel.get("ownership")
        if not parallel_enabled:
            false_sharing_status = "not_applicable_serial"
        elif ownership:
            false_sharing_status = "requires_cache_line_range_validation"
        else:
            false_sharing_status = "unresolved_parallel_ownership"
        findings.append(
            {
                "idx": op.get("idx"),
                "layer": op.get("layer"),
                "section": op.get("section"),
                "op": op_name,
                "tensor_role": tensor_role,
                "dtype": dtype,
                "tokens": runtime_tokens,
                "heads": heads,
                "head_dim": head_dim,
                "payload_bytes": payload_bytes,
                "copy_passes": copy_passes,
                "copied_bytes": copied_bytes,
                "logical_read_write_bytes": logical_read_write_bytes,
                "classification": "avoidable_standalone_layout_conversion",
                "kernel_map_resolved": mapped,
                "kernel_id": kernel_id,
                "resolved_contract_id": resolved_contract,
                "parallel_enabled": parallel_enabled,
                "parallel_ownership": ownership,
                "false_sharing_status": false_sharing_status,
            }
        )

    counts = Counter(row["op"] for row in findings)
    total_payload = sum(int(row["payload_bytes"]) for row in findings)
    total_copied = sum(int(row["copied_bytes"]) for row in findings)
    total_rw = sum(int(row["logical_read_write_bytes"]) for row in findings)
    unmapped = sum(not bool(row["kernel_map_resolved"]) for row in findings)
    unresolved_ownership = sum(
        row["false_sharing_status"] == "unresolved_parallel_ownership" for row in findings
    )
    return {
        "schema": "cke.v8.layout_dataflow_audit",
        "schema_version": 1,
        "source": source,
        "model": config.get("model"),
        "runtime_shape": {"tokens": tokens, "kv_tokens": kv_tokens},
        "summary": {
            "layout_conversion_count": len(findings),
            "avoidable_conversion_count": len(findings),
            "unmapped_conversion_count": unmapped,
            "unresolved_parallel_ownership_count": unresolved_ownership,
            "payload_bytes": total_payload,
            "copied_bytes": total_copied,
            "logical_read_write_bytes": total_rw,
            "counts_by_op": dict(sorted(counts.items())),
        },
        "findings": findings,
    }


def _human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"model={report.get('model') or 'unknown'} tokens={report['runtime_shape']['tokens']}")
    print(
        "layout_conversions="
        f"{summary['layout_conversion_count']} avoidable={summary['avoidable_conversion_count']} "
        f"unmapped={summary['unmapped_conversion_count']} "
        f"unresolved_parallel_ownership={summary['unresolved_parallel_ownership_count']}"
    )
    print(
        f"payload={_human_bytes(summary['payload_bytes'])} "
        f"copied={_human_bytes(summary['copied_bytes'])} "
        f"logical_read_write={_human_bytes(summary['logical_read_write_bytes'])}"
    )
    for name, count in summary["counts_by_op"].items():
        print(f"  {name}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True, help="lowered or call-ready v8 IR")
    parser.add_argument("--tokens", type=int, required=True, help="runtime query/prefill token count")
    parser.add_argument("--kv-tokens", type=int, help="cross-attention K/V token count")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-on-avoidable", action="store_true")
    args = parser.parse_args()

    try:
        report = audit(
            _load(args.ir),
            tokens=args.tokens,
            kv_tokens=args.kv_tokens,
            source=str(args.ir.resolve()),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _print_summary(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.fail_on_avoidable and report["summary"]["avoidable_conversion_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
