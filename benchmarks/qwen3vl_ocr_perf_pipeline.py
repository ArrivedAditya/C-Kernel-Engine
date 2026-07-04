#!/usr/bin/env python3
"""Deterministic Qwen3-VL OCR accuracy/performance pipeline.

This wraps ``bench_v8_qwen3vl_ocr.py`` with repeatable defaults, correctness
checks, hotspot aggregation, baseline comparison, and optional VTune command
emission. It is intentionally wall-clock first so it works on OpenShift/no-sudo;
VTune/Advisor artifacts can be attached later without changing the workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "bench_v8_qwen3vl_ocr.py"
DEFAULT_IMAGE = ROOT / "version" / "v8" / "test_assets" / "v8_ocr_clean_text.ppm"
DEFAULT_EXPECT = "CK OCR TEST\nTOTAL 42"



def _qwen3vl_ocr_fast_profile_enabled(env: dict[str, str]) -> bool:
    alias = env.get("CK_QWEN3VL_OCR_FAST", "")
    if alias and alias != "0":
        return True
    return env.get("CK_SPEED_PROFILE", "") in {"qwen3vl_ocr_xeon_avx512", "qwen3vl_ocr_fast", "qwen3vl_ocr"}


def _apply_qwen3vl_ocr_fast_defaults(env: dict[str, str]) -> None:
    if not _qwen3vl_ocr_fast_profile_enabled(env):
        return
    env.setdefault("CK_ENABLE_Q80_FP32_M4N4", "1")
    env.setdefault("CK_ENABLE_Q4K_GATEUP_SWIGLU_X16", "1")
    env.setdefault("CK_Q4K_GATEUP_SWIGLU_X16_THREAD_CAP", "20")
    env.setdefault("CK_Q4K_X16_CHUNK4", "1")
    env.setdefault("CK_ATTENTION_QBLOCK4", "1")
    env.setdefault("CK_ATTENTION_THREAD_CAP", "16")
    env.setdefault("CK_Q4K_PACKED_META_X8_MAX_M", "2048")
    env.setdefault("CK_NUM_THREADS", "20")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMP_DYNAMIC", "FALSE")

def _run(cmd: list[str], *, env: dict[str, str], timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return int(proc.returncode), proc.stdout


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results") or payload.get("rows") or []
    return rows if isinstance(rows, list) else []


def _num(obj: dict[str, Any], key: str) -> float:
    try:
        return float(obj.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _read_encoder_profile(report_path: Path) -> list[dict[str, Any]]:
    csv_path = report_path.parent / "encoder" / "encoder_profile_current.csv"
    if not csv_path.exists():
        csv_path = report_path.parent / "encoder" / "encoder_profile.csv"
    if not csv_path.exists():
        return []
    agg: dict[tuple[str, str], float] = defaultdict(float)
    cnt: Counter[tuple[str, str]] = Counter()
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            op = str(r.get("op") or "")
            kernel = str(r.get("kernel") or "")
            try:
                ms = float(r.get("time_us") or 0.0) / 1000.0
            except ValueError:
                ms = 0.0
            key = (op, kernel)
            agg[key] += ms
            cnt[key] += 1
    return [
        {"op": op, "kernel": kernel, "total_ms": ms, "count": int(cnt[(op, kernel)])}
        for (op, kernel), ms in sorted(agg.items(), key=lambda item: item[1], reverse=True)
    ]


def _aggregate_profile_csv(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    agg: dict[tuple[str, str], float] = defaultdict(float)
    cnt: Counter[tuple[str, str]] = Counter()
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            op = str(r.get("op") or "")
            kernel = str(r.get("kernel") or "")
            try:
                ms = float(r.get("time_us") or 0.0) / 1000.0
            except ValueError:
                ms = 0.0
            key = (op, kernel)
            agg[key] += ms
            cnt[key] += 1
    return [
        {"op": op, "kernel": kernel, "total_ms": ms, "count": int(cnt[(op, kernel)])}
        for (op, kernel), ms in sorted(agg.items(), key=lambda item: item[1], reverse=True)
    ]


def _read_decoder_profile(row: dict[str, Any], report_path: Path) -> list[dict[str, Any]]:
    csv_path = report_path.parent / "decoder" / "decoder_mixed_prefill_profile.csv"
    csv_items = _aggregate_profile_csv(csv_path)
    if csv_items:
        return csv_items
    prof = row.get("decoder_profile") if isinstance(row.get("decoder_profile"), dict) else {}
    by_op = prof.get("by_op") if isinstance(prof, dict) else {}
    if not isinstance(by_op, dict):
        return []
    out = []
    for op, v in by_op.items():
        if not isinstance(v, dict):
            continue
        out.append(
            {
                "op": str(op),
                "kernel": str(v.get("kernel") or ""),
                "total_ms": float(v.get("total_ms") or v.get("ms") or 0.0),
                "count": int(v.get("count") or 0),
            }
        )
    return sorted(out, key=lambda item: float(item["total_ms"]), reverse=True)


def _summarize(payload: dict[str, Any], *, expect_text: str | None) -> dict[str, Any]:
    rows = _rows(payload)
    summaries = []
    for row in rows:
        report_path = Path(str(row.get("report_path") or ""))
        generated = str(row.get("generated_text") or "")
        ok = str(row.get("status") or "") == "ok"
        text_ok = True if expect_text is None else generated.strip() == expect_text.strip()
        enc_top = _read_encoder_profile(report_path) if report_path.exists() else []
        dec_top = _read_decoder_profile(row, report_path)
        summaries.append(
            {
                "image_name": str(row.get("image_name") or ""),
                "status": str(row.get("status") or ""),
                "correct": bool(ok and text_ok),
                "expected_text": expect_text,
                "generated_text": generated,
                "prefix_tokens": int(row.get("prefix_tokens") or 0),
                "total_prefill_tokens": int(row.get("total_prefill_tokens") or 0),
                "encoder_execute_ms": _num(row, "encoder_execute_ms"),
                "decoder_forward_mixed_ms": _num(row, "decoder_forward_mixed_ms"),
                "decoder_generation_ms": _num(row, "decoder_generation_ms"),
                "steady_state_ms": _num(row, "steady_state_ms"),
                "encoder_top_ops": enc_top[:15],
                "decoder_top_ops": dec_top[:15],
                "report_path": str(report_path) if report_path else "",
            }
        )
    return {"results": summaries}


def _compare(cur: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    cur_rows = cur.get("results") or []
    base_rows = {str(r.get("image_name")): r for r in (base.get("results") or [])}
    out = []
    for row in cur_rows:
        name = str(row.get("image_name"))
        b = base_rows.get(name)
        if not b:
            continue
        cur_ms = float(row.get("steady_state_ms") or 0.0)
        base_ms = float(b.get("steady_state_ms") or 0.0)
        enc = float(row.get("encoder_execute_ms") or 0.0) - float(b.get("encoder_execute_ms") or 0.0)
        mixed = float(row.get("decoder_forward_mixed_ms") or 0.0) - float(b.get("decoder_forward_mixed_ms") or 0.0)
        gen = float(row.get("decoder_generation_ms") or 0.0) - float(b.get("decoder_generation_ms") or 0.0)
        out.append(
            {
                "image_name": name,
                "steady_delta_ms": cur_ms - base_ms,
                "steady_speedup": (base_ms / cur_ms) if cur_ms > 0.0 else 0.0,
                "encoder_delta_ms": enc,
                "mixed_delta_ms": mixed,
                "generation_delta_ms": gen,
            }
        )
    return {"comparisons": out}


def _recommend(summary: dict[str, Any]) -> list[str]:
    rows = summary.get("results") or []
    if not rows:
        return ["No successful rows to analyze."]
    first = rows[0]
    enc = float(first.get("encoder_execute_ms") or 0.0)
    mixed = float(first.get("decoder_forward_mixed_ms") or 0.0)
    enc_top = first.get("encoder_top_ops") or []
    dec_top = first.get("decoder_top_ops") or []
    recs = []
    if not first.get("correct"):
        recs.append("Correctness failed: stop performance work and inspect bridge/model output first.")
        return recs
    if enc >= mixed and enc_top:
        top = enc_top[0]
        recs.append(f"Next target: encoder {top.get('op')} / {top.get('kernel')} ({float(top.get('total_ms') or 0.0):.1f} ms).")
    elif dec_top:
        top = dec_top[0]
        recs.append(f"Next target: decoder {top.get('op')} / {top.get('kernel')} ({float(top.get('total_ms') or 0.0):.1f} ms).")
    elif enc_top:
        top = enc_top[0]
        recs.append(f"Decoder profile missing; next measured target: encoder {top.get('op')} / {top.get('kernel')} ({float(top.get('total_ms') or 0.0):.1f} ms).")
    else:
        recs.append("Next target unclear: enable encoder and decoder profiling.")
    recs.append("Reject any change that improves time but changes expected OCR text or layer/logit parity.")
    return recs


def _write_markdown(path: Path, summary: dict[str, Any], comparison: dict[str, Any] | None, recs: list[str]) -> None:
    lines = ["# Qwen3-VL OCR Perf Pipeline", ""]
    lines.append("| Image | Correct | Prefix | Encoder ms | Mixed ms | Gen ms | Steady ms | Text |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in summary.get("results") or []:
        text = str(row.get("generated_text") or "").replace("\n", "\\n")
        if len(text) > 48:
            text = text[:45] + "..."
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("image_name") or ""),
                    "yes" if row.get("correct") else "no",
                    str(row.get("prefix_tokens") or 0),
                    f"{float(row.get('encoder_execute_ms') or 0.0):.1f}",
                    f"{float(row.get('decoder_forward_mixed_ms') or 0.0):.1f}",
                    f"{float(row.get('decoder_generation_ms') or 0.0):.1f}",
                    f"{float(row.get('steady_state_ms') or 0.0):.1f}",
                    f"`{text}`",
                ]
            )
            + " |"
        )
    if comparison and comparison.get("comparisons"):
        lines += ["", "## Baseline Comparison", "", "| Image | Steady delta ms | Speedup | Encoder delta | Mixed delta | Gen delta |", "|---|---:|---:|---:|---:|---:|"]
        for row in comparison["comparisons"]:
            lines.append(
                f"| {row['image_name']} | {row['steady_delta_ms']:.1f} | {row['steady_speedup']:.3f}x | "
                f"{row['encoder_delta_ms']:.1f} | {row['mixed_delta_ms']:.1f} | {row['generation_delta_ms']:.1f} |"
            )
    lines += ["", "## Top Encoder Ops", "", "| Op | Kernel | ms | count |", "|---|---|---:|---:|"]
    for row in (summary.get("results") or [{}])[0].get("encoder_top_ops", [])[:10]:
        lines.append(f"| {row.get('op')} | {row.get('kernel')} | {float(row.get('total_ms') or 0.0):.1f} | {row.get('count')} |")
    lines += ["", "## Recommendations", ""]
    lines += [f"- {r}" for r in recs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _vtune_commands(args: argparse.Namespace) -> list[str]:
    env_prefix = [
        f"CK_ENABLE_Q80_FP32_M4N4={int(bool(args.enable_q80_m4n4))}",
        f"CK_ENABLE_Q4K_GATEUP_SWIGLU_X16={int(bool(args.enable_q4_gateup_x16))}",
        f"CK_Q4K_X16_CHUNK4={int(bool(args.enable_q4_chunk4))}",
        f"CK_ATTENTION_QBLOCK4={int(bool(args.enable_attention_qblock4))}",
        f"CK_NUM_THREADS={args.threads}",
        "OMP_NUM_THREADS=1",
    ]
    base = [
        "vtune -collect hotspots -result-dir build/vtune-qwen3vl-ocr-hotspots --",
        "env",
        *env_prefix,
        f"{sys.executable} -B benchmarks/bench_v8_qwen3vl_ocr.py",
        f"--images {args.image}",
        f"--threads {args.threads}",
        f"--max-tokens {args.max_tokens}",
        f"--context-len {args.context_len}",
        f"--image-min-tokens {args.image_tokens}",
        f"--image-max-tokens {args.image_tokens}",
        "--bridge-runtime decode-staged --bridge-generation-mode incremental-decode --profile-decoder --force-compile",
    ]
    return [" ".join(base)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--image-tokens", type=int, default=1024)
    ap.add_argument("--context-len", type=int, default=1536)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--expected-text", default=DEFAULT_EXPECT)
    ap.add_argument("--allow-any-text", action="store_true", help="Treat status=ok as correct without matching generated text.")
    ap.add_argument("--json-out", type=Path, default=ROOT / "build" / "qwen3vl_ocr_perf_pipeline.json")
    ap.add_argument("--md-out", type=Path, default=ROOT / "build" / "qwen3vl_ocr_perf_pipeline.md")
    ap.add_argument("--raw-json-out", type=Path, default=ROOT / "build" / "qwen3vl_ocr_perf_raw.json")
    ap.add_argument("--analyze-existing", type=Path, default=None, help="Analyze an existing bench_v8_qwen3vl_ocr.py JSON instead of running.")
    ap.add_argument("--baseline", type=Path, default=None, help="Previous pipeline JSON or raw benchmark JSON for comparison.")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--enable-q80-m4n4", action="store_true")
    ap.add_argument("--enable-q4-gateup-x16", action="store_true")
    ap.add_argument("--enable-q4-chunk4", action="store_true")
    ap.add_argument("--enable-attention-qblock4", action="store_true")
    ap.add_argument("--emit-vtune", action="store_true")
    args = ap.parse_args()

    env = os.environ.copy()
    _apply_qwen3vl_ocr_fast_defaults(env)
    env["CK_NUM_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = "1"
    if args.enable_q80_m4n4:
        env["CK_ENABLE_Q80_FP32_M4N4"] = "1"
    if args.enable_q4_gateup_x16:
        env["CK_ENABLE_Q4K_GATEUP_SWIGLU_X16"] = "1"
        env.setdefault("CK_Q4K_GATEUP_SWIGLU_X16_THREAD_CAP", str(args.threads))
    if args.enable_q4_chunk4:
        env["CK_Q4K_X16_CHUNK4"] = "1"
    if args.enable_attention_qblock4:
        env["CK_ATTENTION_QBLOCK4"] = "1"

    if args.analyze_existing is None:
        cmd = [
            sys.executable,
            "-B",
            str(BENCH),
            "--images",
            str(args.image),
            "--threads",
            str(args.threads),
            "--max-tokens",
            str(args.max_tokens),
            "--context-len",
            str(args.context_len),
            "--image-min-tokens",
            str(args.image_tokens),
            "--image-max-tokens",
            str(args.image_tokens),
            "--bridge-runtime",
            "decode-staged",
            "--bridge-generation-mode",
            "incremental-decode",
            "--profile-decoder",
            "--force-compile",
            "--json-out",
            str(args.raw_json_out),
        ]
        rc, out = _run(cmd, env=env, timeout=args.timeout)
        print(out, end="")
        if rc != 0:
            return rc
        raw = _load_json(args.raw_json_out)
    else:
        raw = _load_json(args.analyze_existing)

    summary = _summarize(raw, expect_text=None if args.allow_any_text else args.expected_text)
    comparison = None
    if args.baseline:
        base_payload = _load_json(args.baseline)
        if "results" in base_payload and base_payload["results"] and "encoder_top_ops" in base_payload["results"][0]:
            base_summary = base_payload
        else:
            base_summary = _summarize(base_payload, expect_text=args.expected_text)
        comparison = _compare(summary, base_summary)
    recs = _recommend(summary)
    payload = {
        "config": {
            "threads": int(args.threads),
            "image_tokens": int(args.image_tokens),
            "context_len": int(args.context_len),
            "max_tokens": int(args.max_tokens),
            "enable_q80_m4n4": bool(args.enable_q80_m4n4),
            "enable_q4_gateup_x16": bool(args.enable_q4_gateup_x16),
            "enable_q4_chunk4": bool(args.enable_q4_chunk4),
            "enable_attention_qblock4": bool(args.enable_attention_qblock4),
        },
        **summary,
        "comparison": comparison,
        "recommendations": recs,
        "vtune_commands": _vtune_commands(args) if args.emit_vtune else [],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(args.md_out, summary, comparison, recs)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    for rec in recs:
        print(f"NEXT: {rec}")
    if args.emit_vtune:
        print("\nVTune commands:")
        for cmd in payload["vtune_commands"]:
            print(cmd)
    return 0 if all(r.get("correct") for r in summary.get("results", [])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
