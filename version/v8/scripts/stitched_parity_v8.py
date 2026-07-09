#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _run_logged(
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    timeout_sec: int = 0,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=None if int(timeout_sec) <= 0 else int(timeout_sec),
        )
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(
            cmd,
            124,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr if isinstance(exc.stderr, str) else "") + f"\nTIMEOUT after {int(timeout_sec)}s\n",
        )
    elapsed = time.time() - started
    with log_path.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.write(f"exit={proc.returncode} elapsed_sec={elapsed:.3f}\n")
        if proc.stdout:
            f.write("\n[stdout]\n")
            f.write(proc.stdout)
            if not proc.stdout.endswith("\n"):
                f.write("\n")
        if proc.stderr:
            f.write("\n[stderr]\n")
            f.write(proc.stderr)
            if not proc.stderr.endswith("\n"):
                f.write("\n")
        f.write("\n")
    return proc


def _layers_for_mode(mode: str, explicit: str | None) -> list[int]:
    if explicit:
        return [int(x.strip()) for x in explicit.split(",") if x.strip()]
    if mode == "fast":
        return [0, 1]
    if mode == "nightly":
        return [0, 1, 4, 8, 16, 31]
    return list(range(32))


def _bridge_command(args: argparse.Namespace, bridge_dir: Path, prefix_f32: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_multimodal_bridge_v8.py"),
        "--decoder-gguf",
        str(args.decoder_gguf),
        "--encoder-gguf",
        str(args.mmproj_gguf),
        "--workdir",
        str(bridge_dir),
        "--prompt",
        str(args.prompt),
        "--chat-template",
        str(args.chat_template),
        "--image-path",
        str(args.image_path),
        "--decoder-context-len",
        str(int(args.ctx_len)),
        "--dump-prefix-f32",
        str(prefix_f32),
        "--report-top-k",
        str(int(args.top_k)),
        "--max-tokens",
        "0",
        "--temperature",
        "0",
        "--no-stream-output",
        "--strict-parity",
    ]
    if args.image_min_tokens is not None:
        cmd.extend(["--image-min-tokens", str(int(args.image_min_tokens))])
    if args.image_max_tokens is not None:
        cmd.extend(["--image-max-tokens", str(int(args.image_max_tokens))])
    return cmd


def _multitoken_command(args: argparse.Namespace, bridge_report: Path, prefix_f32: Path, out_json: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "compare_multimodal_multitoken_logits_v8.py"),
        "--bridge-report",
        str(bridge_report),
        "--prefix-f32",
        str(prefix_f32),
        "--workdir",
        str(args.workdir / "multitoken"),
        "--ctx-len",
        str(int(args.ctx_len)),
        "--threads",
        str(int(args.threads)),
        "--top-k",
        str(int(args.top_k)),
        "--max-new-tokens",
        str(int(args.max_new_tokens)),
        "--append-on-divergence",
        "stop",
        "--json-out",
        str(out_json),
        "--summary",
    ]


def _encoder_numeric_command(args: argparse.Namespace, out_dir: Path, out_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "numeric_parity_qwen3vl_mmproj_v8.py"),
        "--gguf",
        str(args.mmproj_gguf),
        "--output-dir",
        str(out_dir),
        "--image-path",
        str(args.image_path),
        "--threads",
        str(int(args.threads)),
        "--ck-threads",
        str(int(args.threads)),
        "--strict-parity",
        "--report",
        str(out_json),
    ]
    if args.image_min_tokens is not None:
        cmd.extend(["--image-min-tokens", str(int(args.image_min_tokens))])
    if args.image_max_tokens is not None:
        cmd.extend(["--image-max-tokens", str(int(args.image_max_tokens))])
    return cmd


def _granular_command(args: argparse.Namespace, layer: int, out_dir: Path, out_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "activation_parity_qwen3vl_mmproj_v8.py"),
        "--gguf",
        str(args.mmproj_gguf),
        "--output-dir",
        str(out_dir),
        "--image-path",
        str(args.image_path),
        "--threads",
        str(int(args.threads)),
        "--ck-threads",
        str(int(args.threads)),
        "--strict-parity",
        "--llama-dump-layer",
        str(int(layer)),
        "--ck-strict-dump-layer",
        str(int(layer)),
        "--llama-dump-names",
        str(args.granular_dump_names),
        "--report",
        str(out_json),
        "--quiet",
    ]
    if args.granular_ck_stop:
        cmd.extend(["--ck-stop-layer", str(int(layer))])
    if args.image_min_tokens is not None:
        cmd.extend(["--image-min-tokens", str(int(args.image_min_tokens))])
    if args.image_max_tokens is not None:
        cmd.extend(["--image-max-tokens", str(int(args.image_max_tokens))])
    return cmd


def _encoder_numeric_pass(report: dict[str, Any], args: argparse.Namespace) -> bool:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return False
    try:
        cosine = float(metrics.get("cosine", 0.0))
        rmse = float(metrics.get("rmse", float("inf")))
        max_abs = float(metrics.get("max_abs", float("inf")))
    except (TypeError, ValueError):
        return False
    return (
        cosine >= float(args.encoder_cosine_min)
        and rmse <= float(args.encoder_rmse_max)
        and max_abs <= float(args.encoder_max_abs_max)
    )


def _first_granular_issue(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reports:
        report = row.get("report")
        if not isinstance(report, dict):
            continue
        issue = report.get("first_issue")
        if isinstance(issue, dict):
            out = dict(issue)
            out.setdefault("layer", row.get("layer"))
            out["report_path"] = row.get("report_path")
            return out
    return None


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v8 Stitched Parity Report",
        "",
        f"- template: `{report.get('template')}`",
        f"- status: `{report.get('status')}`",
        f"- workdir: `{report.get('workdir')}`",
        "",
    ]
    mismatch = report.get("first_divergence")
    encoder_metrics = report.get("encoder_numeric_metrics")
    if isinstance(encoder_metrics, dict):
        lines.extend(
            [
                "## Encoder Numeric Parity",
                "",
                f"- pass: `{report.get('encoder_numeric_pass')}`",
                f"- cosine: `{encoder_metrics.get('cosine')}`",
                f"- rmse: `{encoder_metrics.get('rmse')}`",
                f"- max abs: `{encoder_metrics.get('max_abs')}`",
                f"- report: `{report.get('encoder_numeric_report')}`",
                "",
            ]
        )
    if isinstance(mismatch, dict):
        lines.extend(
            [
                "## First Multitoken Divergence",
                "",
                f"- step: `{mismatch.get('step')}`",
                f"- CK token: `{mismatch.get('ck_next')}` `{mismatch.get('ck_next_text')}`",
                f"- llama token: `{mismatch.get('llama_next')}` `{mismatch.get('llama_next_text')}`",
                f"- cosine: `{mismatch.get('cosine')}`",
                f"- rmse: `{mismatch.get('rmse')}`",
                f"- top-k overlap: `{mismatch.get('topk_overlap_count')}`",
                "",
            ]
        )
    issue = report.get("first_granular_issue")
    if isinstance(issue, dict):
        lines.extend(
            [
                "## First Granular Issue",
                "",
                f"- layer: `{issue.get('layer')}`",
                f"- op: `{issue.get('op')}`",
                f"- status: `{issue.get('status')}`",
                f"- max abs diff: `{issue.get('max_abs_diff')}`",
                f"- rmse: `{issue.get('rmse')}`",
                f"- report: `{issue.get('report_path')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- bridge report: `{report.get('bridge_report')}`",
            f"- prefix f32: `{report.get('prefix_f32')}`",
            f"- multitoken report: `{report.get('multitoken_report')}`",
            f"- command log: `{report.get('command_log')}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Clean v8 stitched parity runner with automatic granular attribution")
    ap.add_argument("--template", choices=["qwen3vl"], default="qwen3vl")
    ap.add_argument("--mode", choices=["fast", "nightly", "deep"], default="fast")
    ap.add_argument("--decoder-gguf", type=Path, required=True)
    ap.add_argument("--mmproj-gguf", type=Path, required=True)
    ap.add_argument("--image-path", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default="Extract visible form fields as compact JSON.")
    ap.add_argument("--chat-template", type=str, default="qwen3vl")
    ap.add_argument("--workdir", type=Path, default=REPO_ROOT / "build" / "stitched_parity" / "qwen3vl")
    ap.add_argument("--ctx-len", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--image-min-tokens", type=int, default=None)
    ap.add_argument("--image-max-tokens", type=int, default=1024)
    ap.add_argument("--skip-encoder-numeric", action="store_true", help="Skip final vision prefix numeric parity against llama.cpp")
    ap.add_argument("--encoder-cosine-min", type=float, default=0.9999)
    ap.add_argument("--encoder-rmse-max", type=float, default=1.0e-3)
    ap.add_argument("--encoder-max-abs-max", type=float, default=1.0e-1)
    ap.add_argument("--phase-timeout-sec", type=int, default=0, help="Optional timeout per subprocess phase; 0 disables")
    ap.add_argument("--granular-layers", type=str, default=None, help="Comma-separated activation layers to inspect after coarse mismatch")
    ap.add_argument(
        "--granular-ck-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For granular attribution, stop the generated CK encoder after the inspected layer while llama.cpp still runs the full graph",
    )
    ap.add_argument(
        "--granular-dump-names",
        type=str,
        default="patch_bias,inp_pos_emb,ln1,Qcur,Kcur,Vcur,Qcur_rope,Kcur_rope,kqv_out,attn_out,ffn_inp,ffn_inp_normed,ffn_up_b,ffn_out,layer_out",
    )
    ap.add_argument("--no-granular", action="store_true", help="Only run clean bridge + multitoken parity")
    ap.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True, help="Delete --workdir before running")
    args = ap.parse_args(argv)

    args.decoder_gguf = args.decoder_gguf.resolve()
    args.mmproj_gguf = args.mmproj_gguf.resolve()
    args.image_path = args.image_path.resolve()
    args.workdir = args.workdir.resolve()

    missing = [str(p) for p in (args.decoder_gguf, args.mmproj_gguf, args.image_path) if not p.exists()]
    if missing:
        print("missing required artifact(s): " + ", ".join(missing), file=sys.stderr)
        return 2

    if args.clean and args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(int(args.threads))
    env["OMP_NUM_THREADS"] = str(int(args.threads))

    command_log = args.workdir / "commands.log"
    bridge_dir = args.workdir / "bridge"
    prefix_f32 = args.workdir / "prefix.f32"
    bridge_report = bridge_dir / "bridge_report.json"
    multitoken_report = args.workdir / "multitoken.json"
    encoder_numeric_report = args.workdir / "encoder_numeric.json"
    final_json = args.workdir / "stitched_report.json"
    final_md = args.workdir / "stitched_report.md"

    report: dict[str, Any] = {
        "schema": "ck.v8.stitched_parity.v1",
        "template": args.template,
        "mode": args.mode,
        "status": "running",
        "workdir": str(args.workdir),
        "decoder_gguf": str(args.decoder_gguf),
        "mmproj_gguf": str(args.mmproj_gguf),
        "image_path": str(args.image_path),
        "prompt": args.prompt,
        "ctx_len": int(args.ctx_len),
        "threads": int(args.threads),
        "image_min_tokens": args.image_min_tokens,
        "image_max_tokens": args.image_max_tokens,
        "command_log": str(command_log),
        "bridge_report": str(bridge_report),
        "prefix_f32": str(prefix_f32),
        "encoder_numeric_report": str(encoder_numeric_report),
        "multitoken_report": str(multitoken_report),
        "granular_ck_stop": bool(args.granular_ck_stop),
    }

    bridge_proc = _run_logged(_bridge_command(args, bridge_dir, prefix_f32), log_path=command_log, env=env, timeout_sec=args.phase_timeout_sec)
    if bridge_proc.returncode != 0 or not bridge_report.exists() or not prefix_f32.exists():
        report.update({"status": "setup_fail", "bridge_exit_code": bridge_proc.returncode})
        _write_json(final_json, report)
        _write_markdown(report, final_md)
        print(f"status=setup_fail report={final_json}", file=sys.stderr)
        return 2

    encoder_ok = True
    if not args.skip_encoder_numeric:
        encoder_dir = args.workdir / "encoder_numeric"
        encoder_proc = _run_logged(_encoder_numeric_command(args, encoder_dir, encoder_numeric_report), log_path=command_log, env=env, timeout_sec=args.phase_timeout_sec)
        encoder_numeric = _load_json(encoder_numeric_report)
        encoder_ok = encoder_proc.returncode == 0 and _encoder_numeric_pass(encoder_numeric, args)
        report["encoder_numeric_exit_code"] = int(encoder_proc.returncode)
        report["encoder_numeric_metrics"] = encoder_numeric.get("metrics")
        report["encoder_numeric_pass"] = bool(encoder_ok)
        if not encoder_ok:
            report["status"] = "fail"
            report["failure_stage"] = "encoder_numeric"
            if not args.no_granular:
                granular_reports: list[dict[str, Any]] = []
                for layer in _layers_for_mode(args.mode, args.granular_layers):
                    layer_dir = args.workdir / "granular" / f"layer_{layer}"
                    layer_json = layer_dir / "activation_report.json"
                    proc = _run_logged(_granular_command(args, layer, layer_dir, layer_json), log_path=command_log, env=env, timeout_sec=args.phase_timeout_sec)
                    granular_reports.append(
                        {
                            "layer": int(layer),
                            "exit_code": int(proc.returncode),
                            "report_path": str(layer_json),
                            "report": _load_json(layer_json),
                        }
                    )
                    if isinstance(granular_reports[-1]["report"].get("first_issue"), dict):
                        break
                report["granular_reports"] = [
                    {k: v for k, v in row.items() if k != "report"} for row in granular_reports
                ]
                report["first_granular_issue"] = _first_granular_issue(granular_reports)
            _write_json(final_json, report)
            _write_markdown(report, final_md)
            issue = report.get("first_granular_issue") or {}
            metrics = report.get("encoder_numeric_metrics") or {}
            print(
                "status=fail stage=encoder_numeric "
                f"cosine={metrics.get('cosine')} rmse={metrics.get('rmse')} max_abs={metrics.get('max_abs')} "
                f"granular_layer={issue.get('layer')} granular_op={issue.get('op')} "
                f"report={final_json}"
            )
            return 3

    multi_proc = _run_logged(_multitoken_command(args, bridge_report, prefix_f32, multitoken_report), log_path=command_log, env=env, timeout_sec=args.phase_timeout_sec)
    multitoken = _load_json(multitoken_report)
    report["multitoken_exit_code"] = int(multi_proc.returncode)
    report["multitoken_status"] = multitoken.get("status")
    report["first_divergence"] = multitoken.get("first_divergence")

    if multitoken.get("pass") is True:
        report["status"] = "pass"
        _write_json(final_json, report)
        _write_markdown(report, final_md)
        print(f"status=pass report={final_json}")
        return 0

    report["status"] = "fail"
    if not args.no_granular:
        granular_reports: list[dict[str, Any]] = []
        for layer in _layers_for_mode(args.mode, args.granular_layers):
            layer_dir = args.workdir / "granular" / f"layer_{layer}"
            layer_json = layer_dir / "activation_report.json"
            proc = _run_logged(_granular_command(args, layer, layer_dir, layer_json), log_path=command_log, env=env, timeout_sec=args.phase_timeout_sec)
            granular_reports.append(
                {
                    "layer": int(layer),
                    "exit_code": int(proc.returncode),
                    "report_path": str(layer_json),
                    "report": _load_json(layer_json),
                }
            )
            if isinstance(granular_reports[-1]["report"].get("first_issue"), dict):
                break
        report["granular_reports"] = [
            {k: v for k, v in row.items() if k != "report"} for row in granular_reports
        ]
        report["first_granular_issue"] = _first_granular_issue(granular_reports)

    _write_json(final_json, report)
    _write_markdown(report, final_md)
    first = report.get("first_divergence") or {}
    issue = report.get("first_granular_issue") or {}
    print(
        "status=fail "
        f"step={first.get('step')} "
        f"ck={first.get('ck_next')} llama={first.get('llama_next')} "
        f"granular_layer={issue.get('layer')} granular_op={issue.get('op')} "
        f"report={final_json}"
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
