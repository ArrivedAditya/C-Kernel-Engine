#!/usr/bin/env python3
"""Collect perf stat counters for CK v8 and llama.cpp prefill/decode.

This is the safe profiling lane for machines where VTune/Advisor kernel-driver
collections are too risky. It intentionally uses the same fixed-token workload
shape as benchmarks/compare_ck_llama_v8.py so timing and counters can be read
together.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path.home() / ".cache/ck-engine-v8/models/unsloth--Qwen3.5-0.8B-GGUF"
DEFAULT_GGUF = DEFAULT_MODEL_DIR / "Qwen3.5-0.8B-Q4_K_M.gguf"
DEFAULT_OUT = ROOT / "profile_results" / "v8_prefill_perf_stat"

EVENTS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branches",
    "branch-misses",
    "context-switches",
    "cpu-migrations",
    "page-faults",
]

PERF_LINE_RE = re.compile(
    r"^\s*(?P<value>[0-9][0-9,\.]*|<not counted>|<not supported>)\s+"
    r"(?P<event>[A-Za-z0-9_./:-]+)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CK_TIMING_RE = re.compile(
    r"prefill\s+(?P<prompt_tokens>\d+)\s+tok.*?"
    r"(?P<prompt_ms>[0-9.]+)\s+ms\s+(?P<prompt_tok_s>[0-9.]+)\s+tok/s.*?"
    r"decode\s+(?P<decode_tokens>\d+)\s+tok\s+"
    r"(?P<decode_ms>[0-9.]+)\s+ms\s+(?P<decode_tok_s>[0-9.]+)\s+tok/s",
    re.S,
)


def run(cmd: list[str], *, env: dict[str, str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def parse_perf_stat(text: str) -> dict[str, Any]:
    counters: dict[str, Any] = {}
    for line in text.splitlines():
        if "," in line:
            fields = line.split(",")
            if len(fields) < 3:
                continue
            raw_value = fields[0].strip()
            event = fields[2].strip()
        else:
            match = PERF_LINE_RE.match(line)
            if not match:
                continue
            raw_value = match.group("value")
            event = match.group("event")
        if event.startswith("cpu_atom/") and event.endswith("/"):
            event = event[len("cpu_atom/"):-1]
        elif event.startswith("cpu_core/") and event.endswith("/"):
            event = event[len("cpu_core/"):-1]
        if raw_value.startswith("<"):
            counters[event] = raw_value
            continue
        try:
            parsed: Any = int(raw_value.replace(",", ""))
        except ValueError:
            try:
                parsed = float(raw_value.replace(",", ""))
            except ValueError:
                parsed = raw_value
        if isinstance(parsed, (int, float)) and isinstance(counters.get(event), (int, float)):
            counters[event] += parsed
        else:
            counters[event] = parsed
    cycles = counters.get("cycles")
    instructions = counters.get("instructions")
    cache_refs = counters.get("cache-references")
    cache_misses = counters.get("cache-misses")
    if isinstance(cycles, int) and cycles > 0 and isinstance(instructions, int):
        counters["ipc"] = instructions / cycles
    if isinstance(cache_refs, int) and cache_refs > 0 and isinstance(cache_misses, int):
        counters["cache_miss_rate"] = cache_misses / cache_refs
    return counters


def parse_ck_timing(text: str) -> dict[str, Any]:
    clean = ANSI_RE.sub("", text).replace("\r", "")
    match = CK_TIMING_RE.search(clean)
    if not match:
        return {}
    return {
        "prompt_tokens": int(match.group("prompt_tokens")),
        "decode_tokens": int(match.group("decode_tokens")),
        "prompt_ms": float(match.group("prompt_ms")),
        "decode_ms": float(match.group("decode_ms")),
        "prompt_tok_s": float(match.group("prompt_tok_s")),
        "decode_tok_s": float(match.group("decode_tok_s")),
    }


def parse_llama_bench_timing(text: str) -> dict[str, Any]:
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(rows, list):
        return {}
    prompt = next((row for row in rows if int(row.get("n_prompt", 0)) > 0), None)
    decode = next((row for row in rows if int(row.get("n_gen", 0)) > 0), None)
    timing: dict[str, Any] = {}
    if prompt:
        timing.update(
            {
                "prompt_tokens": int(prompt.get("n_prompt", 0)),
                "prompt_ms": float(prompt.get("avg_ns", 0.0)) / 1_000_000.0,
                "prompt_tok_s": float(prompt.get("avg_ts", 0.0)),
            }
        )
    if decode:
        timing.update(
            {
                "decode_tokens": int(decode.get("n_gen", 0)),
                "decode_ms": float(decode.get("avg_ns", 0.0)) / 1_000_000.0,
                "decode_tok_s": float(decode.get("avg_ts", 0.0)),
            }
        )
    return timing


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["ck", "llama", "both"], default="both")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--ck-lib", type=Path, default=DEFAULT_MODEL_DIR / "libmodel.so")
    parser.add_argument("--ck-weights", type=Path, default=DEFAULT_MODEL_DIR / "weights.bump")
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--decode", type=int, default=1)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--token-id", type=int, default=100)
    parser.add_argument("--events", default=",".join(EVENTS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_dir = args.out / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["LD_LIBRARY_PATH"] = f"{ROOT / 'build'}:{ROOT / 'llama.cpp'}:{env.get('LD_LIBRARY_PATH', '')}"

    token_csv = ",".join([str(args.token_id)] * args.prompt)
    context = max(args.prompt + args.decode + 16, 128)
    event_arg = args.events

    commands: dict[str, list[str]] = {
        "ck": [
            "perf", "stat", "-x", ",", "-e", event_arg,
            str(ROOT / "build/ck-cli-v8"),
            str(args.ck_lib),
            str(args.ck_weights),
            "--prompt-tokens", token_csv,
            "--max-tokens", str(args.decode + 1),
            "--context", str(context),
            "--temperature", "0",
            "--ignore-eos",
            "--quiet-output",
            "--no-chat-template",
            "--no-stream",
            "--timing",
        ],
        "llama": [
            "perf", "stat", "-x", ",", "-e", event_arg,
            str(ROOT / "llama.cpp/build/bin/llama-bench"),
            "-m", str(args.gguf),
            "-p", str(args.prompt),
            "-n", str(args.decode),
            "-t", str(args.threads),
            "-ngl", "0",
            "-r", "1",
            "-o", "json",
        ],
    }

    engines = ["ck", "llama"] if args.engine == "both" else [args.engine]
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "prompt": args.prompt,
            "decode": args.decode,
            "threads": args.threads,
            "events": event_arg.split(","),
            "gguf": str(args.gguf),
            "ck_lib": str(args.ck_lib),
            "ck_weights": str(args.ck_weights),
            "ck_env": {
                "CK_ENABLE_Q4K_PACKED_META_2D_PREFILL": env.get("CK_ENABLE_Q4K_PACKED_META_2D_PREFILL", ""),
                "CK_FORCE_Q4K_PACKED_META_2D_PREFILL": env.get("CK_FORCE_Q4K_PACKED_META_2D_PREFILL", ""),
                "CK_Q4K_PACKED_META_TILE_M": env.get("CK_Q4K_PACKED_META_TILE_M", ""),
                "CK_Q4K_PACKED_META_TILE_N": env.get("CK_Q4K_PACKED_META_TILE_N", ""),
                "CK_DISABLE_Q4K_PACKED_META_PREFILL": env.get("CK_DISABLE_Q4K_PACKED_META_PREFILL", ""),
                "CK_FORCE_Q4K_PACKED_META_PREFILL": env.get("CK_FORCE_Q4K_PACKED_META_PREFILL", ""),
                "CK_ENABLE_Q4K_PACKED_META_X8_PREFILL": env.get("CK_ENABLE_Q4K_PACKED_META_X8_PREFILL", ""),
                "CK_FORCE_Q4K_PACKED_META_X8_PREFILL": env.get("CK_FORCE_Q4K_PACKED_META_X8_PREFILL", ""),
                "CK_DISABLE_Q4K_PACKED_META_X8_PREFILL": env.get("CK_DISABLE_Q4K_PACKED_META_X8_PREFILL", ""),
                "CK_ENABLE_Q4K_PACKED_META_X8MT_PREFILL": env.get("CK_ENABLE_Q4K_PACKED_META_X8MT_PREFILL", ""),
                "CK_FORCE_Q4K_PACKED_META_X8MT_PREFILL": env.get("CK_FORCE_Q4K_PACKED_META_X8MT_PREFILL", ""),
                "CK_DISABLE_Q4K_PACKED_META_X8MT_PREFILL": env.get("CK_DISABLE_Q4K_PACKED_META_X8MT_PREFILL", ""),
                "CK_Q4K_PACKED_META_X8MT_TILE_M": env.get("CK_Q4K_PACKED_META_X8MT_TILE_M", ""),
            },
        },
        "runs": {},
    }

    for engine in engines:
        proc = run(commands[engine], env=env, timeout=args.timeout)
        stdout_path = out_dir / f"{engine}.stdout.txt"
        stderr_path = out_dir / f"{engine}.stderr.txt"
        write_text(stdout_path, proc.stdout)
        write_text(stderr_path, proc.stderr)
        report["runs"][engine] = {
            "returncode": proc.returncode,
            "cmd": commands[engine],
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "perf": parse_perf_stat(proc.stderr),
            "timing": parse_ck_timing(proc.stdout)
            if engine == "ck"
            else parse_llama_bench_timing(proc.stdout),
        }

    json_path = out_dir / "perf_stat_v8_prefill_compare.json"
    md_path = out_dir / "perf_stat_v8_prefill_compare.md"
    write_text(json_path, json.dumps(report, indent=2))

    lines = [
        "# v8 Prefill perf stat",
        "",
        f"- Prompt/decode: `{args.prompt}` / `{args.decode}`",
        f"- Threads: `{args.threads}`",
        f"- CK Q4 packed 2D: `{env.get('CK_ENABLE_Q4K_PACKED_META_2D_PREFILL', '')}` "
        f"force=`{env.get('CK_FORCE_Q4K_PACKED_META_2D_PREFILL', '')}` "
        f"tile=`{env.get('CK_Q4K_PACKED_META_TILE_M', '') or 'default'}x"
        f"{env.get('CK_Q4K_PACKED_META_TILE_N', '') or 'default'}`",
        f"- CK Q4 packed-x8: `{env.get('CK_ENABLE_Q4K_PACKED_META_X8_PREFILL', '')}` "
        f"force=`{env.get('CK_FORCE_Q4K_PACKED_META_X8_PREFILL', '')}` "
        f"disabled=`{env.get('CK_DISABLE_Q4K_PACKED_META_X8_PREFILL', '')}`",
        f"- CK Q4 packed-x8mt: `{env.get('CK_ENABLE_Q4K_PACKED_META_X8MT_PREFILL', '')}` "
        f"force=`{env.get('CK_FORCE_Q4K_PACKED_META_X8MT_PREFILL', '')}` "
        f"disabled=`{env.get('CK_DISABLE_Q4K_PACKED_META_X8MT_PREFILL', '')}` "
        f"tile_m=`{env.get('CK_Q4K_PACKED_META_X8MT_TILE_M', '') or 'default'}`",
        "",
        "| Engine | prefill tok/s | prefill ms | decode tok/s | decode ms | cycles | instructions | IPC | cache misses | cache miss rate | ctx switches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for engine in engines:
        perf = report["runs"][engine]["perf"]
        timing = report["runs"][engine].get("timing", {})
        lines.append(
            f"| {engine} | {timing.get('prompt_tok_s', '-')} | {timing.get('prompt_ms', '-')} | "
            f"{timing.get('decode_tok_s', '-')} | {timing.get('decode_ms', '-')} | "
            f"{perf.get('cycles', '-')} | {perf.get('instructions', '-')} | "
            f"{perf.get('ipc', 0.0):.3f} | {perf.get('cache-misses', '-')} | "
            f"{perf.get('cache_miss_rate', 0.0):.3%} | {perf.get('context-switches', '-')} |"
        )
    lines += ["", f"JSON: `{json_path}`", ""]
    write_text(md_path, "\n".join(lines))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
