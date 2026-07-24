#!/usr/bin/env python3
"""Run a private Qwen3-VL image corpus through exact CK/llama.cpp parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BRIDGE = ROOT / "version" / "v8" / "scripts" / "run_multimodal_bridge_v8.py"
PARITY = ROOT / "version" / "v8" / "scripts" / "compare_multimodal_multitoken_logits_v8.py"
PINNED_LLAMA_COMMIT = "f3e182816421c648188b5eab269853bf1531d950"
DEFAULT_PROMPT = "Extract visible form fields as compact JSON."


def _json_write(path: Path, value: Any, *, private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if private:
        temporary.chmod(0o600)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path, *, hash_content: bool) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        result["sha256"] = _sha256_file(resolved)
    return result


def _load_corpus(manifest_path: Path) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("corpus manifest must contain a non-empty samples list")
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} is not an object")
        inputs = sample.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict):
            raise ValueError(f"sample {index} must contain exactly one image input")
        raw_path = inputs[0].get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"sample {index} has no image path")
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"sample {index} image is missing: {image_path}")
        rows.append(
            {
                "index": index,
                "image": image_path,
                "image_sha256": _sha256_file(image_path),
            }
        )
    return rows


def _git_commit(repo: Path) -> str:
    probe = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(probe.stderr.strip() or f"cannot resolve git commit for {repo}")
    return probe.stdout.strip()


def _run_logged(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(command)
    if dry_run:
        print(rendered)
        return 0.0
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        log_path.chmod(0o600)
        stream.write(f"$ {rendered}\n\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}; inspect private log {log_path}"
        )
    return elapsed


def _bridge_command(
    args: argparse.Namespace,
    *,
    image: Path,
    runtime_dir: Path,
    prefix_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(BRIDGE),
        "--decoder-gguf",
        str(args.decoder_gguf),
        "--encoder-gguf",
        str(args.mmproj_gguf),
        "--workdir",
        str(runtime_dir),
        "--prompt",
        args.prompt,
        "--chat-template",
        "qwen3vl",
        "--thinking-mode",
        "suppressed",
        "--image-path",
        str(image),
        "--image-max-tokens",
        str(args.image_max_tokens),
        "--decoder-context-len",
        str(args.context_len),
        "--dump-prefix-f32",
        str(prefix_path),
        "--report-top-k",
        str(args.top_k),
        "--max-tokens",
        "0",
        "--temperature",
        "0",
        "--no-stream-output",
    ]


def _parity_command(
    args: argparse.Namespace,
    *,
    bridge_report: Path,
    prefix_path: Path,
    workdir: Path,
    report_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(PARITY),
        "--bridge-report",
        str(bridge_report),
        "--prefix-f32",
        str(prefix_path),
        "--workdir",
        str(workdir),
        "--reuse-bridge-decoder-runtime-exact",
        "--ctx-len",
        str(args.context_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--top-k",
        str(args.top_k),
        "--threads",
        str(args.threads),
        "--ck-threads",
        str(args.ck_threads),
        "--llama-required-isa",
        args.llama_required_isa,
        "--llama-decode-mode",
        "batched",
        "--append-on-divergence",
        "stop",
        "--json-out",
        str(report_path),
    ]


def _runtime_hash(report: dict[str, Any], key: str) -> str | None:
    runtime = report.get("ck_runtime")
    if not isinstance(runtime, dict):
        return None
    item = runtime.get(key)
    return str(item.get("sha256")) if isinstance(item, dict) and item.get("sha256") else None


def _public_provenance(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "version",
        "cke_commit",
        "manifest_sha256",
        "llama_commit",
        "expected_llama_commit",
        "compiler",
        "prompt_sha256",
        "context_len",
        "image_max_tokens",
        "max_new_tokens",
        "threads",
        "ck_threads",
        "top_k",
        "llama_required_isa",
    )
    return {key: config[key] for key in keys}


def _redacted_row(
    *,
    index: int,
    image_sha256: str,
    prefix_sha256: str,
    report: dict[str, Any],
    elapsed: dict[str, float],
    requested_tokens: int,
) -> dict[str, Any]:
    divergence = report.get("first_divergence")
    first_divergence = None
    if isinstance(divergence, dict):
        first_divergence = {
            "step": divergence.get("step"),
            "ck_next": divergence.get("ck_next"),
            "llama_next": divergence.get("llama_next"),
            "cosine": divergence.get("cosine"),
            "rmse": divergence.get("rmse"),
            "topk_overlap_count": divergence.get("topk_overlap_count"),
        }
    llama = report.get("llama_oracle")
    compiler = report.get("compiler_provenance")
    compiler_summary = None
    if isinstance(compiler, dict):
        compiler_summary = {
            key: compiler.get(key)
            for key in ("status", "decoder_family", "engine_family")
        }
    prefix = report.get("prefix")
    steps = len(report.get("steps") or [])
    prefix_tokens = int(prefix.get("tokens", 0)) if isinstance(prefix, dict) else 0
    prompt_tokens = len(report.get("prompt_tokens_before_image") or []) + len(
        report.get("prompt_tokens_after_image") or []
    )
    prefill_tokens = prefix_tokens + prompt_tokens
    bridge_sec = float(elapsed.get("bridge", 0.0))
    parity_sec = float(elapsed.get("parity", 0.0))
    total_sec = bridge_sec + parity_sec
    return {
        "image_index": int(index),
        "image_sha256": image_sha256,
        "prefix_sha256": prefix_sha256,
        "grid": prefix.get("grid") if isinstance(prefix, dict) else None,
        "status": "pass" if bool(report.get("pass")) else "fail",
        "steps": steps,
        "matched_tokens": steps,
        "requested_tokens": requested_tokens,
        "prefix_tokens": prefix_tokens,
        "prompt_tokens": prompt_tokens,
        "prefill_tokens": prefill_tokens,
        "context_capacity": int(report.get("ctx_len", 0)),
        "context_tokens_after_comparison": prefill_tokens + steps,
        "first_divergence": first_divergence,
        "stop_reason": report.get("stop_reason"),
        "decoder_sha256": _runtime_hash(report, "shared_library"),
        "engine_sha256": _runtime_hash(report, "engine_library"),
        "compiler_provenance": compiler_summary,
        "llama_commit": llama.get("commit") if isinstance(llama, dict) else None,
        "elapsed_sec": {
            **elapsed,
            "total": total_sec,
            "comparison_per_token": parity_sec / steps if steps else None,
        },
    }


def _row_total_sec(row: dict[str, Any]) -> float:
    elapsed = row.get("elapsed_sec")
    return float(elapsed.get("total", 0.0)) if isinstance(elapsed, dict) else 0.0


def _timing_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals = [_row_total_sec(row) for row in rows if _row_total_sec(row) > 0.0]
    if not totals:
        return {
            "total_sec": 0.0,
            "mean_sec_per_image": 0.0,
            "min_sec_per_image": 0.0,
            "max_sec_per_image": 0.0,
        }
    return {
        "total_sec": sum(totals),
        "mean_sec_per_image": sum(totals) / len(totals),
        "min_sec_per_image": min(totals),
        "max_sec_per_image": max(totals),
    }


def _progress_line(
    row: dict[str, Any],
    *,
    completed: int,
    requested: int,
    resumed: bool = False,
) -> str:
    elapsed = row.get("elapsed_sec")
    elapsed = elapsed if isinstance(elapsed, dict) else {}
    matched = int(row.get("matched_tokens", row.get("steps", 0)))
    target = int(row.get("requested_tokens", matched))
    suffix = " resumed" if resumed else ""
    return (
        f"[{completed}/{requested}] image {int(row['image_index']):02d}: "
        f"{str(row.get('status', 'unknown')).upper()} "
        f"matched={matched}/{target} "
        f"prefix={int(row.get('prefix_tokens', 0))} "
        f"prompt={int(row.get('prompt_tokens', 0))} "
        f"prefill={int(row.get('prefill_tokens', 0))} "
        f"context={int(row.get('context_tokens_after_comparison', 0))}/"
        f"{int(row.get('context_capacity', 0))} "
        f"bridge={float(elapsed.get('bridge', 0.0)):.2f}s "
        f"parity={float(elapsed.get('parity', 0.0)):.2f}s "
        f"total={_row_total_sec(row):.2f}s "
        f"compare={float(elapsed.get('comparison_per_token') or 0.0):.3f}s/token-pair"
        f"{suffix}"
    )


def _private_console_enabled(args: argparse.Namespace) -> bool:
    configured = getattr(args, "show_private_details", None)
    if configured is not None:
        return bool(configured)
    return bool(sys.stdout.isatty() and not os.environ.get("CI"))


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _print_private_case_details(
    *,
    sample: dict[str, Any],
    row: dict[str, Any],
    case_dir: Path,
    prompt: str,
) -> None:
    bridge = _load_json_if_present(case_dir / "bridge_report.json")
    parity = _load_json_if_present(case_dir / "parity.json")
    encoder = bridge.get("encoder_report")
    encoder = encoder if isinstance(encoder, dict) else {}
    bridge_timings = bridge.get("timings")
    bridge_timings = bridge_timings if isinstance(bridge_timings, dict) else {}
    source_size = encoder.get("source_image_size")
    source_text = "unknown"
    if isinstance(source_size, list) and len(source_size) == 2:
        source_text = f"{int(source_size[0])}x{int(source_size[1])}"
    image_width = int(encoder.get("image_width", 0) or 0)
    image_height = int(encoder.get("image_height", 0) or 0)
    processed_text = (
        f"{image_width}x{image_height}" if image_width > 0 and image_height > 0 else "unknown"
    )
    grid = row.get("grid")
    grid_text = "unknown"
    if isinstance(grid, list) and len(grid) == 2:
        grid_text = f"{int(grid[0])}x{int(grid[1])}"

    matched = int(row.get("matched_tokens", row.get("steps", 0)))
    requested = int(row.get("requested_tokens", matched))
    exact = str(row.get("status", "")).lower() == "pass"
    label = "EXACT MATCH" if exact else str(row.get("status", "unknown")).upper()
    elapsed = row.get("elapsed_sec")
    elapsed = elapsed if isinstance(elapsed, dict) else {}
    encoder_sec = float(bridge_timings.get("encoder_execute_ms", 0.0) or 0.0) / 1000.0
    prefill_sec = float(bridge_timings.get("decoder_forward_mixed_ms", 0.0) or 0.0) / 1000.0

    print()
    print("=" * 88)
    print(f"QWEN3-VL PRIVATE PARITY | IMAGE {int(sample['index']):02d} | {label}")
    print("-" * 88)
    print(f"Image       : {sample['image']}")
    print(f"Image SHA256: {sample['image_sha256']}")
    print(
        f"Geometry    : source {source_text} -> processed {processed_text} | "
        f"grid {grid_text} | vision tokens {int(row.get('prefix_tokens', 0))}"
    )
    print(f"Prompt      : {prompt}")
    print(
        f"Context     : prefill {int(row.get('prefill_tokens', 0))} + "
        f"compared {matched} = {int(row.get('context_tokens_after_comparison', 0))}/"
        f"{int(row.get('context_capacity', 0))}"
    )
    print(
        f"Comparison  : {matched}/{requested} exact pre-EOS greedy token pairs | "
        f"stop={row.get('stop_reason') or 'token limit'}"
    )
    print(
        f"Timing      : encoder={encoder_sec:.2f}s mixed-prefill={prefill_sec:.2f}s "
        f"bridge={float(elapsed.get('bridge', 0.0)):.2f}s "
        f"parity-pair={float(elapsed.get('parity', 0.0)):.2f}s "
        f"total={_row_total_sec(row):.2f}s"
    )

    shared_text = str(parity.get("generated_shared_text", "") or "")
    print("-" * 88)
    if exact:
        print("Output (CK == llama.cpp for every compared token):")
    else:
        print("Shared output before the first divergence:")
    print(shared_text if shared_text else "<no decoded text>")

    divergence = parity.get("first_divergence")
    if isinstance(divergence, dict):
        print("-" * 88)
        print(
            f"First divergence at step {divergence.get('step')}: "
            f"CK={divergence.get('ck_next')} {divergence.get('ck_next_text')!r} | "
            f"llama.cpp={divergence.get('llama_next')} "
            f"{divergence.get('llama_next_text')!r}"
        )
        print(
            f"cosine={divergence.get('cosine')} rmse={divergence.get('rmse')} "
            f"top-k overlap={divergence.get('topk_overlap_count')}"
        )
    print("=" * 88)


def _summary(
    *,
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_index = {int(row["image_index"]): row for row in rows}
    ordered = [by_index[int(sample["index"])] for sample in selected if int(sample["index"]) in by_index]
    passed = sum(row.get("status") == "pass" for row in ordered)
    failed = sum(row.get("status") != "pass" for row in ordered)
    completed = len(ordered)
    if failed:
        status = "fail"
    elif completed == len(selected):
        status = "pass"
    else:
        status = "incomplete"
    return {
        "status": status,
        "comparison": "exact pre-EOS greedy token parity",
        "requested": len(selected),
        "completed": completed,
        "passed": passed,
        "failed": failed,
        "max_new_tokens": config["max_new_tokens"],
        "config_sha256": _sha256_json(config),
        "provenance": _public_provenance(config),
        "timing": _timing_summary(ordered),
        "rows": ordered,
    }


def _case_config(
    *,
    global_config_sha256: str,
    sample: dict[str, Any],
) -> dict[str, Any]:
    return {
        "global_config_sha256": global_config_sha256,
        "image_index": int(sample["index"]),
        "image_sha256": str(sample["image_sha256"]),
    }


def _resumed_row(case_result: Path, expected_config: dict[str, Any]) -> dict[str, Any] | None:
    if not case_result.is_file():
        return None
    payload = json.loads(case_result.read_text(encoding="utf-8"))
    if payload.get("case_config") != expected_config:
        return None
    row = payload.get("redacted_row")
    if not isinstance(row, dict) or row.get("status") != "pass":
        return None
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decoder-gguf", type=Path, required=True)
    parser.add_argument("--mmproj-gguf", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--expected-llama-commit", default=PINNED_LLAMA_COMMIT)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--image-max-tokens", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--ck-threads", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--llama-required-isa", choices=("auto", "avx2", "avx512"), default="avx2")
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-prefixes", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="ignore matching completed case results and execute the selected cases again",
    )
    private_console = parser.add_mutually_exclusive_group()
    private_console.add_argument(
        "--show-private-details",
        dest="show_private_details",
        action="store_true",
        default=None,
        help="print private image paths, prompt, decoded text, and detailed timings",
    )
    private_console.add_argument(
        "--redacted-console",
        dest="show_private_details",
        action="store_false",
        help="print only redacted progress, even in an interactive local terminal",
    )
    args = parser.parse_args()

    os.umask(0o077)
    args.manifest = args.manifest.expanduser().resolve()
    args.decoder_gguf = args.decoder_gguf.expanduser().resolve()
    args.mmproj_gguf = args.mmproj_gguf.expanduser().resolve()
    args.llama_root = args.llama_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for required in (args.decoder_gguf, args.mmproj_gguf):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not (args.llama_root / "build" / "bin" / "libllama.so").is_file():
        raise FileNotFoundError(f"llama.cpp build is missing libllama.so: {args.llama_root}")
    llama_commit = _git_commit(args.llama_root)
    if args.expected_llama_commit != "any" and llama_commit != args.expected_llama_commit:
        raise RuntimeError(
            "llama.cpp oracle commit mismatch: "
            f"expected={args.expected_llama_commit} actual={llama_commit}"
        )
    corpus = _load_corpus(args.manifest)
    if args.start_index < 1:
        raise ValueError("--start-index must be at least 1")
    selected = [row for row in corpus if int(row["index"]) >= args.start_index]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("the requested corpus range is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)
    runtime_dir = args.output_dir / "runtime"
    config = {
        "version": 1,
        "cke_commit": _git_commit(ROOT),
        "manifest_sha256": _sha256_file(args.manifest),
        "decoder": _file_identity(args.decoder_gguf, hash_content=False),
        "mmproj": _file_identity(args.mmproj_gguf, hash_content=False),
        "llama_root": str(args.llama_root),
        "llama_commit": llama_commit,
        "expected_llama_commit": args.expected_llama_commit,
        "compiler": args.compiler,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "context_len": args.context_len,
        "image_max_tokens": args.image_max_tokens,
        "max_new_tokens": args.max_new_tokens,
        "threads": args.threads,
        "ck_threads": args.ck_threads,
        "top_k": args.top_k,
        "llama_required_isa": args.llama_required_isa,
    }
    config_sha256 = _sha256_json(config)
    _json_write(args.output_dir / "run_config.json", config)
    env = os.environ.copy()
    env.update(
        {
            "CK_LLAMA_CPP_ROOT": str(args.llama_root),
            "CK_V8_COMPILER": args.compiler,
            "CK_V7_COMPILER": args.compiler,
            "CK_NUM_THREADS": str(args.ck_threads),
            "OMP_NUM_THREADS": str(args.ck_threads),
        }
    )

    rows: list[dict[str, Any]] = []
    for sample in selected:
        index = int(sample["index"])
        case_dir = args.output_dir / f"image{index:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        case_dir.chmod(0o700)
        result_path = case_dir / "case_result.json"
        case_config = _case_config(
            global_config_sha256=config_sha256,
            sample=sample,
        )
        resumed = None if args.force_rerun else _resumed_row(result_path, case_config)
        if resumed is not None:
            resumed_report = _load_json_if_present(case_dir / "parity.json")
            if resumed_report:
                resumed = _redacted_row(
                    index=index,
                    image_sha256=sample["image_sha256"],
                    prefix_sha256=str(resumed.get("prefix_sha256", "")),
                    report=resumed_report,
                    elapsed=dict(resumed.get("elapsed_sec") or {}),
                    requested_tokens=args.max_new_tokens,
                )
            rows.append(resumed)
            print(
                _progress_line(
                    resumed,
                    completed=len(rows),
                    requested=len(selected),
                    resumed=True,
                )
            )
            if _private_console_enabled(args):
                _print_private_case_details(
                    sample=sample,
                    row=resumed,
                    case_dir=case_dir,
                    prompt=args.prompt,
                )
            _json_write(args.output_dir / "summary.json", _summary(selected=selected, rows=rows, config=config))
            continue

        prefix_path = case_dir / "prefix.f32"
        bridge_report = case_dir / "bridge_report.json"
        parity_report = case_dir / "parity.json"
        elapsed: dict[str, float] = {}
        try:
            elapsed["bridge"] = _run_logged(
                _bridge_command(
                    args,
                    image=sample["image"],
                    runtime_dir=runtime_dir,
                    prefix_path=prefix_path,
                ),
                env=env,
                log_path=case_dir / "bridge.log",
                dry_run=args.dry_run,
            )
            if args.dry_run:
                _run_logged(
                    _parity_command(
                        args,
                        bridge_report=bridge_report,
                        prefix_path=prefix_path,
                        workdir=case_dir / "parity_work",
                        report_path=parity_report,
                    ),
                    env=env,
                    log_path=case_dir / "parity.log",
                    dry_run=True,
                )
                continue
            source_report = runtime_dir / "bridge_report.json"
            if not source_report.is_file():
                raise FileNotFoundError(f"bridge did not produce {source_report}")
            shutil.copy2(source_report, bridge_report)
            bridge_report.chmod(0o600)
            prefix_sha256 = _sha256_file(prefix_path)
            elapsed["parity"] = _run_logged(
                _parity_command(
                    args,
                    bridge_report=bridge_report,
                    prefix_path=prefix_path,
                    workdir=case_dir / "parity_work",
                    report_path=parity_report,
                ),
                env=env,
                log_path=case_dir / "parity.log",
                dry_run=False,
            )
            report = json.loads(parity_report.read_text(encoding="utf-8"))
            row = _redacted_row(
                index=index,
                image_sha256=sample["image_sha256"],
                prefix_sha256=prefix_sha256,
                report=report,
                elapsed=elapsed,
                requested_tokens=args.max_new_tokens,
            )
            rows.append(row)
            _json_write(
                result_path,
                {
                    "case_config": case_config,
                    "redacted_row": row,
                    "private_artifacts": {
                        "bridge_report": str(bridge_report),
                        "parity_report": str(parity_report),
                        "bridge_log": str(case_dir / "bridge.log"),
                        "parity_log": str(case_dir / "parity.log"),
                    },
                },
            )
            if row["status"] == "pass" and not args.keep_prefixes:
                prefix_path.unlink(missing_ok=True)
            print(_progress_line(row, completed=len(rows), requested=len(selected)))
            if _private_console_enabled(args):
                _print_private_case_details(
                    sample=sample,
                    row=row,
                    case_dir=case_dir,
                    prompt=args.prompt,
                )
            if row["status"] != "pass" and not args.continue_on_failure:
                break
        except Exception as exc:
            row = {
                "image_index": index,
                "image_sha256": sample["image_sha256"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                "elapsed_sec": elapsed,
            }
            rows.append(row)
            _json_write(
                result_path,
                {
                    "case_config": case_config,
                    "redacted_row": row,
                    "private_error": str(exc),
                },
            )
            print(
                f"[{len(rows)}/{len(selected)}] image {index:02d}: "
                f"ERROR {type(exc).__name__}; inspect the local case result",
                file=sys.stderr,
            )
            # --continue-on-failure applies to completed numerical comparisons.
            # Execution/setup errors are global until proven otherwise; fail
            # immediately instead of repeating an expensive broken run.
            break
        finally:
            _json_write(args.output_dir / "summary.json", _summary(selected=selected, rows=rows, config=config))

    if args.dry_run:
        return 0
    summary = _summary(selected=selected, rows=rows, config=config)
    _json_write(args.output_dir / "summary.json", summary)
    print(
        f"status={summary['status']} completed={summary['completed']}/{summary['requested']} "
        f"passed={summary['passed']} failed={summary['failed']} "
        f"total={summary['timing']['total_sec']:.2f}s "
        f"mean={summary['timing']['mean_sec_per_image']:.2f}s/image "
        f"report={args.output_dir / 'summary.json'}"
    )
    return 0 if summary["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
