#!/usr/bin/env python3
"""Standardized CKE v8 vs llama.cpp benchmark runner.

This runner keeps two lanes separate:

* perf: fixed token IDs for CKE and llama-bench pp/tg for llama.cpp. This is
  the kernel/runtime throughput lane.
* prompts: fixed human prompts for output sanity. This is useful for regressions
  in tokenization/chat/output, but it is not a strict performance comparison.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "build" / "reports" / "v8_ck_llama_compare"
CK_MODEL_CACHE = Path.home() / ".cache" / "ck-engine-v8" / "models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    quant: str
    gguf: Path
    ck_model: str
    ck_run_dir: Path | None = None


MODELS: list[ModelSpec] = [
    ModelSpec(
        "gemma3_270m",
        "Gemma3 270M",
        "Q5_K_M",
        Path.home() / ".cache/ck-engine-v8/models/unsloth--gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf",
        "gemma-3-270m-it-Q5_K_M",
    ),
    ModelSpec(
        "qwen2_05b",
        "Qwen2 0.5B",
        "Q4_K_M",
        Path.home() / ".cache/ck-engine-v8/models/Qwen--Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf",
        "qwen2-0_5b-instruct-q4_k_m",
    ),
    ModelSpec(
        "qwen3_06b",
        "Qwen3 0.6B",
        "Q8_0",
        Path.home() / ".cache/ck-engine-v8/models/Qwen--Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
        "Qwen--Qwen3-0.6B-GGUF",
    ),
    ModelSpec(
        "qwen35_08b",
        "Qwen3.5 0.8B",
        "Q4_K_M",
        Path.home() / ".cache/ck-engine-v8/models/unsloth--Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        "unsloth--Qwen3.5-0.8B-GGUF",
    ),
    ModelSpec(
        "nanbeige41_3b",
        "Nanbeige4.1 3B",
        "Q4_K_M",
        Path.home() / ".cache/ck-engine-v8/models/mradermacher--Nanbeige4.1-3B-GGUF/Nanbeige4.1-3B.Q4_K_M.gguf",
        "mradermacher--Nanbeige4.1-3B-GGUF",
    ),
]


PROMPTS: dict[str, str] = {
    "hello": "Hello! Reply with one short sentence.",
    "c_code": "Give me an example of C code that sums an array of integers.",
    "doc_summary": (
        "Summarize this document in three bullet points:\n\n"
        "C-Kernel-Engine is a CPU-only distributed inference and training engine "
        "built from the kernel layer up. It targets sovereign AI and enterprise "
        "deployments where GPU dependency is expensive, scarce, or operationally "
        "risky. The runtime focuses on quantized models, agentic workloads, "
        "long-context memory-heavy serving, and CPU-native training."
    ),
}


def load_prompts(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(PROMPTS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("prompt file must contain a non-empty JSON object")
    prompts: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip():
            raise ValueError("prompt keys and values must be non-empty strings")
        prompts[key] = value
    return prompts


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CK_TIMING_RE = re.compile(
    r"prefill\s+(?P<prompt_tokens>\d+)\s+tok.*?"
    r"(?P<prompt_ms>[0-9.]+)\s+ms\s+(?P<prompt_tok_s>[0-9.]+)\s+tok/s.*?"
    r"decode\s+(?P<decode_tokens>\d+)\s+tok\s+"
    r"(?P<decode_ms>[0-9.]+)\s+ms\s+(?P<decode_tok_s>[0-9.]+)\s+tok/s",
    re.S,
)
LLAMA_TIMING_RE = re.compile(
    r"prompt eval time\s*=\s*[^/]+/\s*(?P<prompt_tokens>\d+)\s+tokens\s*"
    r"\([^,]+,\s*(?P<prompt_tok_s>[0-9.]+)\s+tokens per second\).*?"
    r"eval time\s*=\s*[^/]+/\s*(?P<decode_tokens>\d+)\s+runs\s*"
    r"\([^,]+,\s*(?P<decode_tok_s>[0-9.]+)\s+tokens per second\)",
    re.S,
)


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int | None = None) -> dict[str, Any]:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def clean_text(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


def selected_models(keys: list[str]) -> list[ModelSpec]:
    if not keys or keys == ["all"]:
        return MODELS
    by_key = {model.key: model for model in MODELS}
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise SystemExit(f"Unknown model key(s): {', '.join(missing)}")
    return [by_key[key] for key in keys]


def load_models(path: Path | None) -> list[ModelSpec]:
    if path is None:
        return list(MODELS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("model manifest must contain a non-empty models array")
    result: list[ModelSpec] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each model manifest entry must be an object")
        result.append(
            ModelSpec(
                key=str(row["key"]),
                name=str(row["name"]),
                quant=str(row["quant"]),
                gguf=Path(os.path.expandvars(os.path.expanduser(str(row["gguf"])))),
                ck_model=str(row["ck_model"]),
                ck_run_dir=(
                    Path(os.path.expandvars(os.path.expanduser(str(row["ck_run_dir"]))))
                    if row.get("ck_run_dir")
                    else None
                ),
            )
        )
    if len({row.key for row in result}) != len(result):
        raise ValueError("model manifest keys must be unique")
    return result


def choose_models(catalog: list[ModelSpec], keys: list[str]) -> list[ModelSpec]:
    if not keys or keys == ["all"]:
        return catalog
    by_key = {model.key: model for model in catalog}
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise SystemExit(f"Unknown model key(s): {', '.join(missing)}")
    return [by_key[key] for key in keys]


def ck_runtime_args(model: ModelSpec) -> list[str]:
    run_dir = model.ck_run_dir or (CK_MODEL_CACHE / model.ck_model)
    lib = run_dir / "libmodel.so"
    weights = run_dir / "weights.bump"
    if lib.exists() and weights.exists():
        args = ["--lib", str(lib), "--weights", str(weights)]
        manifest = run_dir / "weights_manifest.map"
        if manifest.exists():
            args.extend(["--manifest", str(manifest)])
        return args
    return ["--model", model.ck_model]


def ck_runtime_context_window(model: ModelSpec) -> int | None:
    run_dir = model.ck_run_dir or (CK_MODEL_CACHE / model.ck_model)
    library = run_dir / "libmodel.so"
    if not library.is_file():
        return None
    runtime = ctypes.CDLL(str(library))
    try:
        getter = runtime.ck_model_get_context_window
    except AttributeError:
        return None
    getter.argtypes = []
    getter.restype = ctypes.c_int
    context = int(getter())
    return context if context > 0 else None


def validate_runtime_capacity(models: list[ModelSpec], required_context: int) -> None:
    for model in models:
        capacity = ck_runtime_context_window(model)
        if capacity is not None and capacity < required_context:
            raise ValueError(
                f"{model.key} runtime capacity is {capacity}, but this run requires "
                f"{required_context}; regenerate the runtime before benchmarking"
            )


def parse_llama_bench(rows: list[dict[str, Any]]) -> dict[str, float]:
    prompt = next((row for row in rows if int(row.get("n_prompt", 0)) > 0), None)
    decode = next((row for row in rows if int(row.get("n_gen", 0)) > 0), None)
    if not prompt or not decode:
        raise ValueError("llama-bench JSON did not contain prompt and decode rows")
    return {
        "prompt_tokens": int(prompt["n_prompt"]),
        "decode_tokens": int(decode["n_gen"]),
        "prompt_tok_s": float(prompt["avg_ts"]),
        "decode_tok_s": float(decode["avg_ts"]),
        "prompt_ms": float(prompt["avg_ns"]) / 1_000_000.0,
        "decode_ms": float(decode["avg_ns"]) / 1_000_000.0,
    }


def token_sequence_sha256(token_id: int, count: int) -> str:
    """Hash the exact little-endian int32 token stream supplied to CKE."""
    digest = hashlib.sha256()
    encoded = int(token_id).to_bytes(4, byteorder="little", signed=True)
    for _ in range(count):
        digest.update(encoded)
    return digest.hexdigest()


def parse_ck_timing(output: str) -> dict[str, float]:
    text = clean_text(output)
    match = CK_TIMING_RE.search(text)
    if not match:
        raise ValueError("CK timing line not found")
    return {
        "prompt_tokens": int(match.group("prompt_tokens")),
        "decode_tokens": int(match.group("decode_tokens")),
        "prompt_ms": float(match.group("prompt_ms")),
        "decode_ms": float(match.group("decode_ms")),
        "prompt_tok_s": float(match.group("prompt_tok_s")),
        "decode_tok_s": float(match.group("decode_tok_s")),
    }


def parse_ck_timing_optional(output: str) -> dict[str, float] | None:
    match = CK_TIMING_RE.search(clean_text(output))
    return parse_ck_timing(match.group(0)) if match else None


def median_metrics(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot aggregate an empty sample set")
    return {
        key: statistics.median(float(sample[key]) for sample in samples)
        for key in samples[0]
    }


def parse_llama_cli_timing(output: str) -> dict[str, float] | None:
    text = clean_text(output)
    match = LLAMA_TIMING_RE.search(text)
    if not match:
        return None
    return {
        "prompt_tokens": int(match.group("prompt_tokens")),
        "decode_tokens": int(match.group("decode_tokens")),
        "prompt_tok_s": float(match.group("prompt_tok_s")),
        "decode_tok_s": float(match.group("decode_tok_s")),
    }


def extract_ck_generated(stdout: str) -> str:
    text = clean_text(stdout)
    marker = "Type /help for commands, Ctrl+C to stop generation"
    if marker in text:
        text = text.split(marker, 1)[1]
    text = re.split(r"\nprefill\s+\d+\s+tok\b", text, maxsplit=1)[0]
    return text.strip()


def extract_llama_generated(output: str) -> str:
    text = clean_text(output)
    # Recent llama-completion writes generated text before its stderr log
    # stream when both are captured together.
    text = text.split("\n\nwarning: no usable GPU", 1)[0]
    if "\n> " in text:
        text = text.split("\n> ", 1)[1]
        text = text.split("\n", 1)[1] if "\n" in text else ""
    text = text.split("llama_memory_breakdown_print:", 1)[0]
    text = re.split(r"\nprompt eval time\s*=", text, maxsplit=1)[0]
    text = re.split(r"\n\[\s*Prompt:", text, maxsplit=1)[0]
    return text.strip()


def run_perf(args: argparse.Namespace, models: list[ModelSpec]) -> list[dict[str, Any]]:
    llama_bench = args.llama_root / "build/bin/llama-bench"
    ck_cli = args.ck_cli
    token_csv = ",".join([str(args.prompt_token_id)] * args.prompt_tokens)
    requested_token_hash = token_sequence_sha256(
        args.prompt_token_id, args.prompt_tokens
    )
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = str(args.omp_threads)

    results: list[dict[str, Any]] = []
    for model in models:
        if not model.gguf.exists():
            print(f"SKIP missing GGUF: {model.gguf}", file=sys.stderr)
            continue

        llama_cmd = [
            str(llama_bench),
            "-m",
            str(model.gguf),
            "-p",
            str(args.prompt_tokens),
            "-n",
            str(args.decode_tokens),
            "-t",
            str(args.threads),
            "-ngl",
            "0",
            "-r",
            "1",
            "-o",
            "json",
        ]
        ck_cmd = [
            str(ck_cli),
            *ck_runtime_args(model),
            "--prompt-tokens",
            token_csv,
            "--max-tokens",
            str(args.decode_tokens + 1),
            "--context",
            str(max(args.context, args.prompt_tokens + args.decode_tokens + 8)),
            "--temperature",
            "0",
            "--ignore-eos",
            "--quiet-output",
            "--no-chat-template",
            "--no-stream",
            "--timing",
        ]
        # Warm CKE once, then alternate backend order to reduce thermal and
        # frequency bias. Each llama-bench sample performs its own warmup.
        run_cmd(ck_cmd, env=env, timeout=args.timeout)
        samples: dict[str, list[dict[str, float]]] = {"cke": [], "llama": []}
        errors: dict[str, str] = {}
        for repetition in range(args.repetitions):
            order = ("cke", "llama") if repetition % 2 == 0 else ("llama", "cke")
            for backend in order:
                command = ck_cmd if backend == "cke" else llama_cmd
                sample = run_cmd(command, env=env, timeout=args.timeout)
                if sample["returncode"] != 0:
                    errors[backend] = sample["stderr"][-2000:]
                    continue
                try:
                    metrics = (
                        parse_ck_timing(sample["stdout"] + sample["stderr"])
                        if backend == "cke"
                        else parse_llama_bench(json.loads(sample["stdout"]))
                    )
                    samples[backend].append(metrics)
                except (ValueError, json.JSONDecodeError) as exc:
                    errors[backend] = str(exc)

        llama_metrics = (
            median_metrics(samples["llama"])
            if len(samples["llama"]) == args.repetitions
            else {"error": errors.get("llama", "missing successful samples")}
        )
        ck_metrics = (
            median_metrics(samples["cke"])
            if len(samples["cke"]) == args.repetitions
            else {"error": errors.get("cke", "missing successful samples")}
        )

        cke_consumed = all(
            int(sample.get("prompt_tokens", -1)) == args.prompt_tokens
            for sample in samples["cke"]
        ) and len(samples["cke"]) == args.repetitions
        llama_consumed = all(
            int(sample.get("prompt_tokens", -1)) == args.prompt_tokens
            for sample in samples["llama"]
        ) and len(samples["llama"]) == args.repetitions
        if not cke_consumed:
            ck_metrics = {
                "error": "CKE did not report consuming the complete requested token stream"
            }
        if not llama_consumed:
            llama_metrics = {
                "error": "llama.cpp did not report consuming the complete requested token count"
            }

        entry = {
            "model_key": model.key,
            "model": model.name,
            "quant": model.quant,
            "llama": llama_metrics,
            "cke": ck_metrics,
            "samples": samples,
            "consumption": {
                "requested_tokens": args.prompt_tokens,
                "requested_token_id": args.prompt_token_id,
                "requested_token_sha256": requested_token_hash,
                "cke_count_verified": cke_consumed,
                "llama_count_verified": llama_consumed,
            },
        }
        if "error" not in llama_metrics and "error" not in ck_metrics:
            entry["ratios"] = {
                "prompt": ck_metrics["prompt_tok_s"] / llama_metrics["prompt_tok_s"],
                "decode": ck_metrics["decode_tok_s"] / llama_metrics["decode_tok_s"],
            }
        results.append(entry)
    return results


def run_prompts(
    args: argparse.Namespace,
    models: list[ModelSpec],
    prompts: dict[str, str],
) -> list[dict[str, Any]]:
    llama_cli = args.llama_root / "build/bin/llama-cli"
    ck_cli = args.ck_cli
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = str(args.omp_threads)

    results: list[dict[str, Any]] = []
    for model in models:
        if not model.gguf.exists():
            print(f"SKIP missing GGUF: {model.gguf}", file=sys.stderr)
            continue
        for prompt_key, prompt in prompts.items():
            ck_cmd = [
                str(ck_cli),
                *ck_runtime_args(model),
                "--prompt",
                prompt,
                "--max-tokens",
                str(args.prompt_max_tokens + 1),
                "--context",
                str(args.context),
                "--temperature",
                "0",
                "--no-stream",
                "--timing",
            ]
            ck_run = run_cmd(ck_cmd, env=env, timeout=args.timeout)
            ck_text = clean_text(ck_run["stdout"] + ck_run["stderr"])

            llama_result: dict[str, Any] | None = None
            if args.prompt_engine in {"llama", "both"}:
                llama_cmd = [
                    str(llama_cli),
                    "-m",
                    str(model.gguf),
                    "-p",
                    prompt,
                    "-n",
                    str(args.prompt_max_tokens),
                    "-c",
                    str(args.context),
                    "-t",
                    str(args.threads),
                    "-tb",
                    str(args.threads),
                    "-ngl",
                    "0",
                    "--temp",
                    "0",
                    "--no-display-prompt",
                    "--conversation",
                    "--single-turn",
                    "--no-warmup",
                    "--simple-io",
                    "--perf",
                ]
                llama_run = run_cmd(llama_cmd, env=env, timeout=args.prompt_timeout)
                llama_text = clean_text(llama_run["stdout"] + llama_run["stderr"])
                llama_result = {
                    "returncode": llama_run["returncode"],
                    "timing": parse_llama_cli_timing(llama_text),
                    "generated": extract_llama_generated(llama_text),
                    "output": llama_text[-6000:],
                }

            results.append(
                {
                    "model_key": model.key,
                    "model": model.name,
                    "quant": model.quant,
                    "prompt_key": prompt_key,
                    "prompt": prompt,
                    "cke": {
                        "returncode": ck_run["returncode"],
                        "timing": parse_ck_timing_optional(ck_text),
                        "generated": extract_ck_generated(ck_run["stdout"]),
                        "output": ck_text[-6000:],
                    },
                    "llama": llama_result,
                }
            )
    return results


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# CKE v8 vs llama.cpp Standard Benchmark",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Configuration",
        "",
        f"- Threads: `{report['config']['threads']}`",
        f"- OMP threads: `{report['config']['omp_threads']}`",
        f"- Perf prompt/decode: `{report['config']['prompt_tokens']}` / `{report['config']['decode_tokens']}`",
                f"- Prompt max tokens: `{report['config']['prompt_max_tokens']}`",
                "- CKE CLI receives `max_tokens + 1` because current v8 timing reports one fewer decode step than the argument.",
                "",
    ]

    perf = report.get("perf") or []
    if perf:
        lines += [
            "## Fixed-Token Perf",
            "",
            "| Model | Quant | llama prompt | CKE prompt | CKE/llama | llama decode | CKE decode | CKE/llama |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in perf:
            llama = row["llama"]
            cke = row["cke"]
            if "error" in llama or "error" in cke:
                lines.append(f"| {row['model']} | {row['quant']} | error | error | - | error | error | - |")
                continue
            ratios = row["ratios"]
            lines.append(
                f"| {row['model']} | {row['quant']} | "
                f"{llama['prompt_tok_s']:.1f} tok/s | {cke['prompt_tok_s']:.1f} tok/s | {ratios['prompt']:.2f}x | "
                f"{llama['decode_tok_s']:.1f} tok/s | {cke['decode_tok_s']:.1f} tok/s | {ratios['decode']:.2f}x |"
            )
        lines.append("")

    prompt_runs = report.get("prompt_runs") or []
    if prompt_runs:
        lines += ["## Standard Prompts", ""]
        for key, prompt in report.get("prompts", {}).items():
            lines += [f"### `{key}`", "", "```text", prompt, "```", ""]
        lines += ["## Prompt Output Smoke", ""]
        for row in prompt_runs:
            lines += [
                f"### {row['model']} / `{row['prompt_key']}`",
                "",
                "**CKE tail output:**",
                "",
                "```text",
                (row["cke"].get("generated") or row["cke"]["output"]).strip()[-2000:],
                "```",
                "",
            ]
            if row.get("llama") is not None:
                lines += [
                    "**llama.cpp tail output:**",
                    "",
                    "```text",
                    row["llama"]["output"].strip()[-2000:],
                    "```",
                    "",
                ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=["perf", "prompts", "both"], default="both")
    parser.add_argument("--model", action="append", default=[], help="Model key to run, or all")
    parser.add_argument("--model-manifest", type=Path, help="JSON model catalog for lab-specific families")
    parser.add_argument("--ck-cli", type=Path, default=PROJECT_ROOT / "build/ck-cli-v8")
    parser.add_argument("--llama-root", type=Path, default=PROJECT_ROOT / "llama.cpp")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prompt-token-id", type=int, default=100)
    parser.add_argument("--prompt-max-tokens", type=int, default=128)
    parser.add_argument("--prompts-file", type=Path, help="JSON object of prompt key to prompt text")
    parser.add_argument("--prompt-engine", choices=["cke", "llama", "both"], default="cke")
    parser.add_argument("--prompt-timeout", type=int, default=90)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="Run Qwen3 only with smaller perf and prompt limits")
    args = parser.parse_args()

    if args.quick:
        args.model = ["qwen3_06b"]
        args.prompt_tokens = min(args.prompt_tokens, 128)
        args.decode_tokens = min(args.decode_tokens, 32)
        args.prompt_max_tokens = min(args.prompt_max_tokens, 24)
        args.repetitions = 1

    models = choose_models(load_models(args.model_manifest), args.model or ["all"])
    prompts = load_prompts(args.prompts_file)
    args.ck_cli = args.ck_cli.expanduser().resolve()
    args.llama_root = args.llama_root.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    required_context = args.context
    if args.lane in {"perf", "both"}:
        required_context = max(
            required_context,
            args.prompt_tokens + args.decode_tokens + 8,
        )
    validate_runtime_capacity(models, required_context)

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "threads": args.threads,
            "omp_threads": args.omp_threads,
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "prompt_token_id": args.prompt_token_id,
            "prompt_max_tokens": args.prompt_max_tokens,
            "prompt_engine": args.prompt_engine,
            "context": args.context,
            "repetitions": args.repetitions,
            "ck_cli": str(args.ck_cli),
            "llama_root": str(args.llama_root),
        },
        "models": [
            model.__dict__
            | {
                "gguf": str(model.gguf),
                "ck_run_dir": str(model.ck_run_dir) if model.ck_run_dir else None,
            }
            for model in models
        ],
        "prompts": prompts,
    }
    if args.lane in {"perf", "both"}:
        report["perf"] = run_perf(args, models)
    if args.lane in {"prompts", "both"}:
        report["prompt_runs"] = run_prompts(args, models, prompts)

    json_path = args.out_dir / "ck_llama_v8_compare.json"
    md_path = args.out_dir / "ck_llama_v8_compare.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    perf_failed = any(
        "error" in row.get("cke", {}) or "error" in row.get("llama", {})
        for row in report.get("perf", [])
    )
    prompt_failed = any(
        int(row.get("cke", {}).get("returncode", -1)) != 0
        or (
            row.get("llama") is not None
            and int(row["llama"].get("returncode", -1)) != 0
        )
        for row in report.get("prompt_runs", [])
    )
    return 1 if perf_failed or prompt_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
