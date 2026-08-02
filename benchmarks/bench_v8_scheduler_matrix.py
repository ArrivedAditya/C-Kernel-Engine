#!/usr/bin/env python3
"""Resume-safe static/dynamic CKE matrix with final llama.cpp reference lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache" / "ck-engine-v8" / "models"
CK_CLI = ROOT / "build" / "ck-cli-v8"
OCR_BENCH = ROOT / "benchmarks" / "bench_v8_qwen3vl_ocr.py"

MODELS = {
    "qwen2": {
        "name": "Qwen2 0.5B Instruct",
        "cache": CACHE / "qwen2-0_5b-instruct-q4_k_m",
        "gguf": CACHE / "Qwen--Qwen2-0.5B-Instruct-GGUF" / "qwen2-0_5b-instruct-q4_k_m.gguf",
    },
    "qwen3": {
        "name": "Qwen3 0.6B",
        "cache": CACHE / "Qwen3-0.6B-Q8_0",
        "gguf": CACHE / "Qwen--Qwen3-0.6B-GGUF" / "Qwen3-0.6B-Q8_0.gguf",
    },
    "qwen35": {
        "name": "Qwen3.5 0.8B",
        "cache": CACHE / "Qwen3.5-0.8B-Q4_K_M",
        "gguf": CACHE / "unsloth--Qwen3.5-0.8B-GGUF" / "Qwen3.5-0.8B-Q4_K_M.gguf",
    },
    "gemma3": {
        "name": "Gemma 3 270M IT",
        "cache": CACHE / "gemma-3-270m-it-Q5_K_M",
        "gguf": CACHE / "unsloth--gemma-3-270m-it-GGUF" / "gemma-3-270m-it-Q5_K_M.gguf",
    },
}

PROMPTS = {
    "hello": "Hello! Reply with one short sentence.",
    "c_python_sql": "Give me a detailed, correct example using C, Python, and SQL together.",
    "pure_c": "Give me a detailed example of pure C code that reads integers, sorts them, and reports errors.",
    "explain": (
        "Explain in plain English how a persistent heterogeneous CPU thread pool "
        "can balance independent matrix tiles across faster and slower cores."
    ),
    "structured_json": (
        "Return compact JSON describing a benchmark with fields model, prompt_tokens, "
        "decode_tokens, wall_seconds, and status. Do not include prose."
    ),
    "svg": (
        "Generate a small standalone SVG showing four CPU workers taking matrix tiles "
        "from one shared queue. Use only SVG markup, include labels, and do not use JavaScript."
    ),
}

TIMING_RE = re.compile(
    r"prompt eval:\s*(?P<prompt_ms>[0-9.]+) ms /\s*(?P<prompt_tokens>\d+) tokens.*?"
    r"\((?:[^,]+),\s*(?P<prompt_tps>[0-9.]+) tok/s\).*?"
    r"decode:\s*(?P<decode_ms>[0-9.]+) ms /\s*(?P<decode_tokens>\d+) runs.*?"
    r"\((?:[^,]+),\s*(?P<decode_tps>[0-9.]+) tok/s\)",
    re.S,
)
CLI_TIMING_RE = re.compile(
    r"prefill\s+(?P<prompt_tokens>\d+)\s+tok.*?"
    r"(?P<prompt_ms>[0-9.]+)\s+ms\s+(?P<prompt_tps>[0-9.]+)\s+tok/s.*?"
    r"decode\s+(?P<decode_tokens>\d+)\s+tok\s+"
    r"(?P<decode_ms>[0-9.]+)\s+ms\s+(?P<decode_tps>[0-9.]+)\s+tok/s",
    re.S,
)
CLI_PREFILL_RE = re.compile(
    r"prefill\s+(?P<prompt_tokens>\d+)\s+tok.*?"
    r"(?P<prompt_ms>[0-9.]+)\s+ms\s+(?P<prompt_tps>[0-9.]+)\s+tok/s",
    re.S,
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LLAMA_TIMING_RE = re.compile(
    r"prompt eval time\s*=\s*(?P<prompt_ms>[0-9.]+) ms /\s*(?P<prompt_tokens>\d+) tokens.*?"
    r"eval time\s*=\s*(?P<decode_ms>[0-9.]+) ms /\s*(?P<decode_tokens>\d+) runs",
    re.S,
)
LLAMA_CHAT_TIMING_RE = re.compile(
    r"\[\s*Prompt:\s*(?P<prompt_tps>[0-9.]+)\s+t/s\s*\|\s*"
    r"Generation:\s*(?P<decode_tps>[0-9.]+)\s+t/s\s*\]"
)
MTMD_TIMING_RE = re.compile(
    r"prompt eval time\s*=\s*(?P<prompt_ms>[0-9.]+) ms /\s*(?P<prompt_tokens>\d+) tokens.*?"
    r"total time\s*=\s*(?P<total_ms>[0-9.]+) ms",
    re.S,
)


_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def sha256(path: Path) -> str:
    path = path.resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _HASH_CACHE[key] = value
    return value


def run(cmd: list[str], env: dict[str, str], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False,
    )
    return {
        "returncode": proc.returncode,
        "wall_seconds": time.perf_counter() - started,
        "output": proc.stdout,
    }


def load_report(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": "cke.v8.scheduler_matrix", "version": 1, "rows": []}


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(path)


def completed_keys(report: dict[str, Any]) -> set[str]:
    return {str(row["case_key"]) for row in report.get("rows", []) if row.get("status") == "pass"}


def scheduler_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    return env


def parse_ck_timing(output: str) -> dict[str, Any] | None:
    clean = ANSI_RE.sub("", output)
    match = TIMING_RE.search(clean) or CLI_TIMING_RE.search(clean)
    if not match:
        prefill = CLI_PREFILL_RE.search(clean)
        if not prefill:
            return None
        return {
            "prompt_ms": float(prefill.group("prompt_ms")),
            "prompt_tokens": int(prefill.group("prompt_tokens")),
            "prompt_tok_s": float(prefill.group("prompt_tps")),
            "decode_ms": 0.0,
            "decode_tokens": 0,
            "decode_tok_s": 0.0,
        }
    return {
        "prompt_ms": float(match.group("prompt_ms")),
        "prompt_tokens": int(match.group("prompt_tokens")),
        "prompt_tok_s": float(match.group("prompt_tps")),
        "decode_ms": float(match.group("decode_ms")),
        "decode_tokens": int(match.group("decode_tokens")),
        "decode_tok_s": float(match.group("decode_tps")),
    }


def extract_ck_text(output: str) -> str:
    clean = ANSI_RE.sub("", output).replace("\r", "")
    marker = "Type /help for commands, Ctrl+C to stop generation"
    if marker in clean:
        clean = clean.split(marker, 1)[1]
    clean = re.split(r"\nprefill\s+\d+\s+tok", clean, maxsplit=1)[0]
    return clean.strip()


def parse_llama_timing(output: str) -> dict[str, Any] | None:
    match = LLAMA_TIMING_RE.search(output)
    if not match:
        compact = LLAMA_CHAT_TIMING_RE.search(output)
        if not compact:
            return None
        return {
            "prompt_ms": 0.0,
            "prompt_tokens": 0,
            "prompt_tok_s": float(compact.group("prompt_tps")),
            "decode_ms": 0.0,
            "decode_tokens": 0,
            "decode_tok_s": float(compact.group("decode_tps")),
        }
    prompt_tokens = int(match.group("prompt_tokens"))
    decode_tokens = int(match.group("decode_tokens"))
    prompt_ms = float(match.group("prompt_ms"))
    decode_ms = float(match.group("decode_ms"))
    return {
        "prompt_ms": prompt_ms,
        "prompt_tokens": prompt_tokens,
        "prompt_tok_s": 1000.0 * prompt_tokens / prompt_ms if prompt_ms else 0.0,
        "decode_ms": decode_ms,
        "decode_tokens": decode_tokens,
        "decode_tok_s": 1000.0 * decode_tokens / decode_ms if decode_ms else 0.0,
    }


def extract_llama_text(output: str) -> str:
    clean = ANSI_RE.sub("", output).replace("\r", "")
    if "\n> " in clean:
        clean = clean.split("\n> ", 1)[1]
        clean = clean.split("\n", 1)[1] if "\n" in clean else ""
    clean = clean.split("llama_memory_breakdown_print:", 1)[0]
    clean = re.split(r"\n\[\s*Prompt:", clean, maxsplit=1)[0]
    return clean.strip()


def classify_text_result(
    result: dict[str, Any],
    timing: dict[str, Any] | None,
    generated_text: str,
    *,
    require_decode_count: bool,
) -> tuple[str, str]:
    if int(result.get("returncode", -1)) != 0:
        return "fail", "runtime_error"
    if not timing:
        return "fail", "missing_timing"
    if require_decode_count and int(timing.get("decode_tokens", 0) or 0) <= 0:
        return "fail", "no_decoded_tokens"
    if not str(generated_text or "").strip():
        return "fail", "empty_generated_text"
    return "pass", ""


def runtime_provenance(spec: dict[str, Any]) -> dict[str, str]:
    cache = Path(spec["cache"])
    files = {
        "model": cache / "libmodel.so",
        "engine": cache / "libckernel_engine.so",
        "weights": cache / "weights.bump",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing generated runtime artifacts: {missing}")
    current_engine = ROOT / "build" / "libckernel_engine.so"
    hashes = {f"{key}_sha256": sha256(path) for key, path in files.items()}
    hashes["build_engine_sha256"] = sha256(current_engine)
    if hashes["engine_sha256"] != hashes["build_engine_sha256"]:
        raise RuntimeError(
            f"stale engine for {spec['name']}: cache={hashes['engine_sha256']} "
            f"build={hashes['build_engine_sha256']}"
        )
    symbols = subprocess.run(
        ["nm", "-D", str(files["engine"])], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    required_symbols = ("ck_set_gemm_schedule", "ck_threadpool_parallel_for_n")
    missing_symbols = [name for name in required_symbols if name not in symbols.stdout]
    if symbols.returncode != 0 or missing_symbols:
        raise RuntimeError(
            f"runtime engine lacks typed dynamic scheduler ABI {missing_symbols}: "
            f"{files['engine']}"
        )
    hashes["scheduler_symbols"] = list(required_symbols)
    return hashes


def append(report_path: Path, report: dict[str, Any], row: dict[str, Any]) -> None:
    rows = report.setdefault("rows", [])
    rows[:] = [existing for existing in rows if existing.get("case_key") != row.get("case_key")]
    rows.append(row)
    save_report(report_path, report)
    print(
        f"{row['case_key']}: {row['status']} wall={row.get('wall_seconds', 0):.2f}s",
        flush=True,
    )


def run_text_cke(args: argparse.Namespace, report_path: Path, report: dict[str, Any]) -> None:
    done = completed_keys(report)
    case_index = 0
    for model_key in args.models:
        spec = MODELS[model_key]
        provenance = runtime_provenance(spec)
        cache = Path(spec["cache"])
        for prompt_key, prompt in PROMPTS.items():
            order = ("static", "dynamic") if case_index % 2 == 0 else ("dynamic", "static")
            case_index += 1
            for schedule in order:
                for rep in range(args.repetitions):
                    case_key = f"text:cke:{model_key}:{prompt_key}:{schedule}:r{rep}"
                    if case_key in done and not args.force:
                        continue
                    cmd = [
                        str(CK_CLI), "--lib", str(cache / "libmodel.so"),
                        "--weights", str(cache / "weights.bump"),
                        "--manifest", str(cache / "weights_manifest.map"),
                        "--prompt", prompt, "--max-tokens", str(args.text_tokens + 1),
                        "--context", str(args.context), "--temperature", "0",
                        "--no-stream", "--timing", "--gemm-schedule", schedule,
                    ]
                    result = run(cmd, scheduler_env(args.threads), args.timeout)
                    timing = parse_ck_timing(result["output"])
                    generated_text = extract_ck_text(result["output"])
                    status, failure_reason = classify_text_result(
                        result, timing, generated_text, require_decode_count=True
                    )
                    append(report_path, report, {
                        "case_key": case_key, "lane": "text", "backend": "cke",
                        "model_key": model_key, "model": spec["name"],
                        "prompt_key": prompt_key, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "schedule": schedule, "repetition": rep,
                        "status": status, "failure_reason": failure_reason,
                        "wall_seconds": result["wall_seconds"], "timing": timing,
                        "generated_text": generated_text,
                        "output_tail": result["output"][-4000:], "provenance": provenance,
                    })


def run_text_llama(args: argparse.Namespace, report_path: Path, report: dict[str, Any]) -> None:
    done = completed_keys(report)
    llama_cli = args.llama_root / "build" / "bin" / "llama-cli"
    env = scheduler_env(args.threads)
    env["LD_LIBRARY_PATH"] = f"{args.llama_root / 'build' / 'bin'}:{env.get('LD_LIBRARY_PATH', '')}"
    commit = subprocess.check_output(["git", "-C", str(args.llama_root), "rev-parse", "HEAD"], text=True).strip()
    for model_key in args.models:
        spec = MODELS[model_key]
        gguf = Path(spec["gguf"])
        gguf_hash = sha256(gguf)
        for prompt_key, prompt in PROMPTS.items():
            case_key = f"text:llama:{model_key}:{prompt_key}"
            if case_key in done and not args.force:
                continue
            cmd = [
                str(llama_cli), "-m", str(gguf), "-p", prompt,
                "-n", str(args.text_tokens), "-c", str(args.context),
                "-t", str(args.threads), "-tb", str(args.threads), "-ngl", "0",
                "--temp", "0", "--no-display-prompt", "--conversation", "--single-turn",
                "--no-warmup", "--simple-io", "--show-timings",
            ]
            result = run(cmd, env, args.timeout)
            timing = parse_llama_timing(result["output"])
            generated_text = extract_llama_text(result["output"])
            status, failure_reason = classify_text_result(
                result, timing, generated_text, require_decode_count=False
            )
            append(report_path, report, {
                "case_key": case_key, "lane": "text", "backend": "llama.cpp",
                "model_key": model_key, "model": spec["name"],
                "prompt_key": prompt_key, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "schedule": "llama", "status": status, "failure_reason": failure_reason,
                "wall_seconds": result["wall_seconds"], "timing": timing,
                "generated_text": generated_text,
                "output_tail": result["output"][-4000:],
                "provenance": {"commit": commit, "gguf_sha256": gguf_hash},
            })


def corpus_images(manifest: Path) -> list[tuple[int, Path, str]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = []
    for index, sample in enumerate(payload["samples"], 1):
        image = (manifest.parent / sample["inputs"][0]["path"]).resolve()
        rows.append((index, image, sha256(image)))
    return rows


def run_ocr_cke(args: argparse.Namespace, report_path: Path, report: dict[str, Any]) -> None:
    done = completed_keys(report)
    images = corpus_images(args.ocr_manifest)
    engine_hash = sha256(ROOT / "build" / "libckernel_engine.so")
    model_hash = sha256(args.ocr_model)
    mmproj_hash = sha256(args.ocr_mmproj)
    for index, image, image_hash in images:
        order = ("static", "dynamic") if index % 2 else ("dynamic", "static")
        for schedule in order:
            case_key = f"ocr:cke:{index:02d}:{schedule}"
            if case_key in done and not args.force:
                continue
            local_json = args.output.parent / "ocr-cases" / f"{index:02d}-{schedule}.json"
            cmd = [
                sys.executable, str(OCR_BENCH), "--model", str(args.ocr_model),
                "--mmproj", str(args.ocr_mmproj), "--images", str(image),
                "--prompt", args.ocr_prompt, "--threads", str(args.threads),
                "--max-tokens", str(args.ocr_tokens), "--context-len", str(args.ocr_context),
                "--image-max-tokens", str(args.ocr_image_tokens), "--json-out", str(local_json),
                "--gemm-schedule", schedule,
            ]
            result = run(cmd, scheduler_env(args.threads), args.ocr_timeout)
            detail = None
            runtime_hashes: dict[str, str] = {}
            if local_json.exists():
                rows = json.loads(local_json.read_text(encoding="utf-8")).get("results", [])
                detail = rows[0] if rows else None
            if detail and detail.get("report_path"):
                bridge_path = Path(str(detail["report_path"]))
                if bridge_path.is_file():
                    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
                    runtime_hashes["bridge_report_sha256"] = sha256(bridge_path)
                    decoder = bridge.get("decoder_runtime") or {}
                    decoder_so = Path(str(decoder.get("so_path") or ""))
                    if decoder_so.is_file():
                        runtime_hashes["decoder_runtime_sha256"] = sha256(decoder_so)
                    decoder_dir = Path(str(decoder.get("workdir") or ""))
                    decoder_engine = decoder_dir / "libckernel_engine.so"
                    if decoder_engine.is_file():
                        runtime_hashes["decoder_engine_sha256"] = sha256(decoder_engine)
                    encoder = bridge.get("encoder_runtime") or {}
                    encoder_dir = Path(str(encoder.get("workdir") or ""))
                    encoder_so = encoder_dir / "libencoder_v8.so"
                    encoder_engine = encoder_dir / "libckernel_engine.so"
                    if encoder_so.is_file():
                        runtime_hashes["encoder_runtime_sha256"] = sha256(encoder_so)
                    if encoder_engine.is_file():
                        runtime_hashes["encoder_engine_sha256"] = sha256(encoder_engine)
            timing = None if not detail else {
                key: detail.get(key) for key in (
                    "encoder_execute_ms", "decoder_forward_mixed_ms", "decoder_generation_ms",
                    "decoder_generation_tok_s", "steady_state_ms", "prefix_tokens", "total_prefill_tokens",
                )
            }
            append(report_path, report, {
                "case_key": case_key, "lane": "ocr", "backend": "cke",
                "image_index": index, "image_sha256": image_hash, "schedule": schedule,
                "status": "pass" if result["returncode"] == 0 and detail and detail.get("status") == "ok" else "fail",
                "wall_seconds": result["wall_seconds"], "timing": timing,
                "generated_text": "" if not detail else detail.get("generated_text", ""),
                "provenance": {
                    "engine_sha256": engine_hash,
                    "model_sha256": model_hash, "mmproj_sha256": mmproj_hash,
                    **runtime_hashes,
                },
            })


def run_ocr_llama(args: argparse.Namespace, report_path: Path, report: dict[str, Any]) -> None:
    done = completed_keys(report)
    binary = args.llama_root / "build" / "bin" / "llama-mtmd-cli"
    env = scheduler_env(args.threads)
    env["LD_LIBRARY_PATH"] = f"{args.llama_root / 'build' / 'bin'}:{env.get('LD_LIBRARY_PATH', '')}"
    commit = subprocess.check_output(["git", "-C", str(args.llama_root), "rev-parse", "HEAD"], text=True).strip()
    model_hash = sha256(args.ocr_model)
    mmproj_hash = sha256(args.ocr_mmproj)
    for index, image, image_hash in corpus_images(args.ocr_manifest):
        case_key = f"ocr:llama:{index:02d}"
        if case_key in done and not args.force:
            continue
        cmd = [
            str(binary), "-m", str(args.ocr_model), "--mmproj", str(args.ocr_mmproj),
            "--image", str(image), "-p", args.ocr_prompt,
            "-n", str(args.ocr_tokens), "-c", str(args.ocr_context),
            "-t", str(args.threads), "-tb", str(args.threads), "--temp", "0",
            "--no-warmup", "--no-mmproj-offload", "--image-max-tokens",
            str(args.ocr_image_tokens), "--perf",
        ]
        result = run(cmd, env, args.ocr_timeout)
        match = MTMD_TIMING_RE.search(result["output"])
        timing = None if not match else {
            "prompt_ms": float(match.group("prompt_ms")),
            "prompt_tokens": int(match.group("prompt_tokens")),
            "total_ms": float(match.group("total_ms")),
        }
        append(report_path, report, {
            "case_key": case_key, "lane": "ocr", "backend": "llama.cpp",
            "image_index": index, "image_sha256": image_hash, "schedule": "llama",
            "status": "pass" if result["returncode"] == 0 and timing else "fail",
            "wall_seconds": result["wall_seconds"], "timing": timing,
            "output_tail": result["output"][-2000:],
            "provenance": {"commit": commit, "model_sha256": model_hash, "mmproj_sha256": mmproj_hash},
        })


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in report.get("rows", []):
        timing = row.get("timing") or {}
        if row.get("lane") == "text":
            value = timing.get("prompt_ms")
            key = f"text:{row.get('backend')}:{row.get('model_key')}:{row.get('schedule')}"
        elif row.get("backend") == "cke":
            value = timing.get("steady_state_ms")
            key = f"ocr:cke:{row.get('schedule')}"
        else:
            value = timing.get("total_ms")
            key = "ocr:llama"
        if isinstance(value, (int, float)) and value > 0:
            groups.setdefault(key, []).append(float(value))
    return {
        key: {"count": len(values), "mean_ms": statistics.fmean(values), "median_ms": statistics.median(values)}
        for key, values in sorted(groups.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["cke", "llama", "all"], default="all")
    parser.add_argument("--lane", choices=["text", "ocr", "all"], default="all")
    parser.add_argument("--model", dest="models", action="append", choices=sorted(MODELS), default=[])
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--text-tokens", type=int, default=64)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ocr-manifest", type=Path)
    parser.add_argument("--ocr-model", type=Path)
    parser.add_argument("--ocr-mmproj", type=Path)
    parser.add_argument("--ocr-prompt", default="Extract visible form fields as compact JSON.")
    parser.add_argument("--ocr-tokens", type=int, default=1)
    parser.add_argument("--ocr-context", type=int, default=4096)
    parser.add_argument("--ocr-image-tokens", type=int, default=1024)
    parser.add_argument("--ocr-timeout", type=int, default=1800)
    parser.add_argument("--llama-root", type=Path, default=ROOT / "llama.cpp")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "scheduler-matrix" / "matrix.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.models = args.models or list(MODELS)
    args.llama_root = args.llama_root.resolve()
    args.output = args.output.resolve()

    report = load_report(args.output)
    report["config"] = {
        "threads": args.threads, "repetitions": args.repetitions,
        "text_tokens": args.text_tokens, "context": args.context,
        "ocr_tokens": args.ocr_tokens, "ocr_context": args.ocr_context,
        "prompts": PROMPTS,
    }
    save_report(args.output, report)

    if args.lane in {"text", "all"} and args.phase in {"cke", "all"}:
        run_text_cke(args, args.output, report)
    if args.lane in {"ocr", "all"}:
        if not args.ocr_manifest or not args.ocr_model or not args.ocr_mmproj:
            raise SystemExit("OCR lane requires --ocr-manifest, --ocr-model, and --ocr-mmproj")
        args.ocr_manifest = args.ocr_manifest.resolve()
        args.ocr_model = args.ocr_model.resolve()
        args.ocr_mmproj = args.ocr_mmproj.resolve()
        if args.phase in {"cke", "all"}:
            run_ocr_cke(args, args.output, report)
    if args.lane in {"text", "all"} and args.phase in {"llama", "all"}:
        run_text_llama(args, args.output, report)
    if args.lane in {"ocr", "all"} and args.phase in {"llama", "all"}:
        run_ocr_llama(args, args.output, report)

    report["summary"] = summarize(report)
    save_report(args.output, report)
    failed = sum(1 for row in report.get("rows", []) if row.get("status") != "pass")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output}; rows={len(report.get('rows', []))} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
