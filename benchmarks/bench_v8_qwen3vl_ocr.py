#!/usr/bin/env python3
"""Wall-clock Qwen3-VL OCR benchmark for CKE v8.

This benchmark runs the existing v8 multimodal bridge path and reads the
structured timing fields from ``bridge_report.json``. It is designed for
OpenShift/no-sudo environments: no perf counters, only wall-clock timings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CK_RUN_V8 = ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"
ASSET_DIR = ROOT / "version" / "v8" / "test_assets"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
REPORT_RE = re.compile(r"Wrote bridge report to\s+([^\s\x1b]+)|done report=([^\s\x1b]+)")

DEFAULT_MODEL = "hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ = "hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
DEFAULT_IMAGES = [
    ASSET_DIR / "v8_ocr_clean_text.ppm",
    ASSET_DIR / "v8_ocr_table.ppm",
    ASSET_DIR / "v8_ocr_receipt.ppm",
    ASSET_DIR / "v8_ocr_paragraph.ppm",
]


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
    return proc.returncode, proc.stdout


def _extract_report_path(output: str) -> Path | None:
    clean = ANSI_RE.sub("", output)
    matches = list(REPORT_RE.finditer(clean))
    if not matches:
        return None
    match = matches[-1]
    value = match.group(1) or match.group(2)
    if not value:
        return None
    return Path(value).expanduser()


def _num(obj: dict[str, Any], key: str) -> float:
    value = obj.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _run_one(
    *,
    model: str,
    mmproj: str,
    image: Path,
    prompt: str,
    threads: int,
    max_tokens: int,
    context_len: int,
    image_min_tokens: int,
    image_max_tokens: int | None,
    force_compile: bool,
    force_convert: bool,
    bridge_runtime: str,
    bridge_generation_mode: str,
    vision_activation_prefs: list[str],
    profile_decoder: bool,
    timeout: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    cmd = [
        sys.executable,
        str(CK_RUN_V8),
        "run",
        model,
        "--mmproj",
        mmproj,
        "--image-path",
        str(image),
        "--image-min-tokens",
        str(image_min_tokens),
        "--prompt",
        prompt,
        "--context-len",
        str(context_len),
        "--thinking-mode",
        "suppressed",
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0.0",
    ]
    if image_max_tokens is not None:
        cmd.extend(["--image-max-tokens", str(image_max_tokens)])
    if bridge_runtime:
        cmd.extend(["--bridge-runtime", bridge_runtime])
    if bridge_generation_mode:
        cmd.extend(["--bridge-generation-mode", bridge_generation_mode])
    if profile_decoder:
        cmd.append("--profile")
    for pref in vision_activation_prefs:
        cmd.extend(["--vision-activation-pref", pref])
    if force_compile:
        cmd.append("--force-compile")
    if force_convert:
        cmd.append("--force-convert")
    rc, out = _run(cmd, env=env, timeout=timeout)
    report_path = _extract_report_path(out)
    row: dict[str, Any] = {
        "image": str(image),
        "image_name": image.name,
        "returncode": int(rc),
        "command": cmd,
        "stdout_tail": "\n".join(ANSI_RE.sub("", out).splitlines()[-80:]),
        "report_path": None if report_path is None else str(report_path),
    }
    if rc != 0 or report_path is None or not report_path.exists():
        row["status"] = "fail"
        return row
    report = json.loads(report_path.read_text(encoding="utf-8"))
    timings = report.get("timings") if isinstance(report.get("timings"), dict) else {}
    decoder_profile = report.get("decoder_profile") if isinstance(report.get("decoder_profile"), dict) else {}
    encoder_execute_ms = _num(timings, "encoder_execute_ms")
    decoder_forward_ms = _num(timings, "decoder_forward_mixed_ms")
    decoder_generation_ms = _num(timings, "decoder_generation_ms")
    steady_state_ms = encoder_execute_ms + decoder_forward_ms + decoder_generation_ms
    generated_tokens = int(report.get("generated_token_count") or timings.get("decoder_generated_tokens") or 0)
    row.update(
        {
            "status": str(report.get("status") or "ok"),
            "generated_text": str(report.get("generated_text") or ""),
            "generated_token_count": generated_tokens,
            "generation_stop_reason": str(report.get("generation_stop_reason") or ""),
            "prefix_tokens": int(report.get("prefix_tokens") or timings.get("prefix_tokens") or 0),
            "total_prefill_tokens": int(report.get("total_prefill_tokens") or timings.get("total_prefill_tokens") or 0),
            "timings": timings,
            "startup_ms": _num(timings, "encoder_prepare_ms") + _num(timings, "decoder_prepare_ms"),
            "steady_state_ms": steady_state_ms,
            "encoder_execute_ms": encoder_execute_ms,
            "decoder_forward_mixed_ms": decoder_forward_ms,
            "decoder_generation_ms": decoder_generation_ms,
            "decoder_generation_tok_s": _num(timings, "decoder_generation_tok_s"),
            "steady_generated_tok_s": (generated_tokens / (steady_state_ms / 1000.0)) if steady_state_ms > 0 and generated_tokens > 0 else 0.0,
            "decoder_profile": decoder_profile,
        }
    )
    return row


def _fmt(value: Any, digits: int = 1) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\n| Image | Status | Prefix | Prefill | Enc ms | Mixed ms | Gen ms | Gen tok/s | Steady tok/s | Text |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        text = str(row.get("generated_text") or "").replace("\n", "\\n")
        if len(text) > 48:
            text = text[:45] + "..."
        print(
            "| "
            + " | ".join(
                [
                    str(row.get("image_name", "")),
                    str(row.get("status", "")),
                    str(row.get("prefix_tokens", "")),
                    str(row.get("total_prefill_tokens", "")),
                    _fmt(row.get("encoder_execute_ms", 0.0)),
                    _fmt(row.get("decoder_forward_mixed_ms", 0.0)),
                    _fmt(row.get("decoder_generation_ms", 0.0)),
                    _fmt(row.get("decoder_generation_tok_s", 0.0), 3),
                    _fmt(row.get("steady_generated_tok_s", 0.0), 3),
                    f"`{text}`",
                ]
            )
            + " |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mmproj", default=DEFAULT_MMPROJ)
    parser.add_argument("--images", nargs="*", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--prompt", default="Read the text in this image. Return only the visible text.")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("CK_NUM_THREADS", "24")))
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--image-min-tokens", type=int, default=128)
    parser.add_argument("--image-max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--bridge-runtime", choices=["prefill", "decode-staged"], default=None, help="Override template bridge runtime policy")
    parser.add_argument("--bridge-generation-mode", choices=["incremental-decode", "mixed-replay"], default=None, help="Override template bridge generation policy")
    parser.add_argument(
        "--vision-activation-pref",
        action="append",
        default=[],
        metavar="OP=PREF",
        help="Forward a vision activation preference override to ck_run_v8.py, for example out_proj=q8_0",
    )
    parser.add_argument(
        "--profile-decoder",
        action="store_true",
        help="Compile the decoder with CK_PROFILE and include mixed-prefill top ops in the JSON report",
    )
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--json-out", type=Path, default=ROOT / "build" / "v8_qwen3vl_ocr_bench.json")
    args = parser.parse_args()

    rows = []
    for image in args.images:
        image = image.resolve()
        if not image.exists():
            rows.append({"image": str(image), "image_name": image.name, "status": "missing"})
            continue
        print(f"== OCR image {image.name} ==", flush=True)
        rows.append(
            _run_one(
                model=str(args.model),
                mmproj=str(args.mmproj),
                image=image,
                prompt=str(args.prompt),
                threads=int(args.threads),
                max_tokens=int(args.max_tokens),
                context_len=int(args.context_len),
                image_min_tokens=int(args.image_min_tokens),
                image_max_tokens=None if args.image_max_tokens is None else int(args.image_max_tokens),
                force_compile=bool(args.force_compile),
                force_convert=bool(args.force_convert),
                bridge_runtime=args.bridge_runtime,
                bridge_generation_mode=args.bridge_generation_mode,
                vision_activation_prefs=list(args.vision_activation_pref or []),
                profile_decoder=bool(args.profile_decoder),
                timeout=int(args.timeout),
            )
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {
            "model": str(args.model),
            "mmproj": str(args.mmproj),
            "images": [str(p) for p in args.images],
            "threads": int(args.threads),
            "max_tokens": int(args.max_tokens),
            "context_len": int(args.context_len),
            "image_min_tokens": int(args.image_min_tokens),
            "image_max_tokens": None if args.image_max_tokens is None else int(args.image_max_tokens),
            "bridge_runtime": args.bridge_runtime,
            "bridge_generation_mode": args.bridge_generation_mode,
            "vision_activation_pref": list(args.vision_activation_pref or []),
            "profile_decoder": bool(args.profile_decoder),
        },
        "results": rows,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_table(rows)
    print(f"\nWrote {args.json_out}")
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
