#!/usr/bin/env python3
"""Compare CKE v8 decoder throughput against llama.cpp for cached GGUFs.

The benchmark intentionally avoids chat templates and tokenizer differences:
llama.cpp uses llama-bench with synthetic prompt/generation sizes, while CKE
uses ck-cli-v8 with repeated prompt token ids.
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
CACHE = Path(os.getenv("CK_V8_MODEL_CACHE", str(Path.home() / ".cache" / "ck-engine-v8" / "models")))
LLAMA_BIN = Path(os.getenv("LLAMA_BENCH", str(ROOT / "llama.cpp" / "build" / "bin" / "llama-bench")))
LLAMA_LIB = Path(os.getenv("LLAMA_LIB_DIR", str(LLAMA_BIN.parent)))
CK_CLI = ROOT / "build" / "ck-cli-v8"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CK_RE = re.compile(
    r"prefill\s+(\d+)\s+tok.*?([0-9.]+)\s+ms\s+([0-9.]+)\s+tok/s"
    r".*?decode\s+(\d+)\s+tok\s+([0-9.]+)\s+ms\s+([0-9.]+)\s+tok/s",
    re.S,
)

MODELS: dict[str, dict[str, str]] = {
    "gemma3-270m-q5_k_m": {
        "label": "Gemma3 270M",
        "quant": "Q5_K_M",
        "run": "unsloth--gemma-3-270m-it-GGUF",
        "gguf": "unsloth--gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf",
    },
    "qwen2-0.5b-q4_k_m": {
        "label": "Qwen2 0.5B",
        "quant": "Q4_K_M",
        "run": "qwen2-0_5b-instruct-q4_k_m",
        "gguf": "Qwen--Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf",
    },
    "qwen3-0.6b-q8_0": {
        "label": "Qwen3 0.6B",
        "quant": "Q8_0",
        "run": "Qwen--Qwen3-0.6B-GGUF",
        "gguf": "Qwen--Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
    },
    "qwen35-0.8b-q4_k_m": {
        "label": "Qwen3.5 0.8B",
        "quant": "Q4_K_M",
        "run": "Qwen3.5-0.8B-Q4_K_M",
        "gguf": "unsloth--Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
    },
    "nanbeige4.1-3b-q4_k_m": {
        "label": "Nanbeige 4.1 3B",
        "quant": "Q4_K_M",
        "run": "mradermacher--Nanbeige4.1-3B-GGUF/.ck_build_v8",
        "gguf": "mradermacher--Nanbeige4.1-3B-GGUF/Nanbeige4.1-3B.Q4_K_M.gguf",
    },
    "gemma4-e4b-q4_k_m": {
        "label": "Gemma4 E4B",
        "quant": "Q4_K_M",
        "run": "unsloth--gemma-4-E4B-it-GGUF",
        "gguf": "unsloth--gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf",
    },
}


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


def _run_llama(gguf: Path, *, prompt: int, decode: int, threads: int, repeats: int, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LLAMA_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [
        str(LLAMA_BIN),
        "-m",
        str(gguf),
        "-p",
        str(prompt),
        "-n",
        str(decode),
        "-t",
        str(threads),
        "-ngl",
        "0",
        "-r",
        str(repeats),
        "-o",
        "json",
    ]
    rc, out = _run(cmd, env=env, timeout=timeout)
    row: dict[str, Any] = {"command": cmd, "returncode": rc, "stdout_tail": "\n".join(out.splitlines()[-40:])}
    if rc != 0:
        row["status"] = "fail"
        return row
    try:
        doc = json.loads(out)
    except json.JSONDecodeError as exc:
        row.update({"status": "fail", "error": f"json parse failed: {exc}"})
        return row
    pp = next((x for x in doc if int(x.get("n_prompt", 0)) > 0), None)
    tg = next((x for x in doc if int(x.get("n_gen", 0)) > 0), None)
    row.update(
        {
            "status": "pass",
            "prompt_tok_s": float(pp["avg_ts"]) if pp else None,
            "decode_tok_s": float(tg["avg_ts"]) if tg else None,
            "raw": doc,
        }
    )
    return row


def _run_cke(run_dir: Path, *, prompt: int, decode: int, threads: int, timeout: int) -> dict[str, Any]:
    lib = run_dir / "libmodel.so"
    weights = run_dir / "weights.bump"
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = f"{ROOT / 'build'}:{run_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    prompt_tokens = ",".join(["100"] * prompt)
    cmd = [
        str(CK_CLI),
        str(lib),
        str(weights),
        "--prompt-tokens",
        prompt_tokens,
        "--max-tokens",
        str(decode),
        "--ignore-eos",
        "--quiet-output",
        "--timing",
    ]
    rc, out = _run(cmd, env=env, timeout=timeout)
    clean = ANSI_RE.sub("", out)
    row: dict[str, Any] = {"command": cmd[:3] + ["--prompt-tokens", f"<{prompt} ids>", *cmd[5:]], "returncode": rc, "stdout_tail": "\n".join(clean.splitlines()[-40:])}
    match = CK_RE.search(clean)
    if rc != 0 or not match:
        row["status"] = "fail"
        return row
    row.update(
        {
            "status": "pass",
            "prompt_tokens": int(match.group(1)),
            "prompt_ms": float(match.group(2)),
            "prompt_tok_s": float(match.group(3)),
            "decode_tokens": int(match.group(4)),
            "decode_ms": float(match.group(5)),
            "decode_tok_s": float(match.group(6)),
        }
    )
    return row


def _fmt(v: float | None) -> str:
    if v is None:
        return "fail"
    return f"{v:.1f}"


def _ratio(a: float | None, b: float | None) -> str:
    if a is None or b in (None, 0):
        return "fail"
    return f"{a / b:.2f}x"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model ids, or 'all'")
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--json-out", type=Path, default=ROOT / "build" / "v8_decoder_matrix.json")
    args = parser.parse_args()

    selected = list(MODELS) if args.models == "all" else [x.strip() for x in args.models.split(",") if x.strip()]
    results: list[dict[str, Any]] = []
    for model_id in selected:
        spec = MODELS[model_id]
        run_dir = CACHE / spec["run"]
        gguf = Path(spec["gguf"])
        if not gguf.is_absolute():
            gguf = CACHE / gguf
        row: dict[str, Any] = {"id": model_id, "label": spec["label"], "quant": spec["quant"], "run_dir": str(run_dir), "gguf": str(gguf)}
        missing = [str(p) for p in [CK_CLI, LLAMA_BIN, run_dir / "libmodel.so", run_dir / "weights.bump", gguf] if not p.exists()]
        if missing:
            row.update({"status": "skip", "missing": missing})
            results.append(row)
            continue
        print(f"== {spec['label']} {spec['quant']} ==", flush=True)
        row["llama"] = _run_llama(gguf, prompt=args.prompt, decode=args.decode, threads=args.threads, repeats=args.repeats, timeout=args.timeout)
        row["cke"] = _run_cke(run_dir, prompt=args.prompt, decode=args.decode, threads=args.threads, timeout=args.timeout)
        results.append(row)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    arg_doc = dict(vars(args))
    arg_doc["json_out"] = str(args.json_out)
    args.json_out.write_text(json.dumps({"args": arg_doc, "results": results}, indent=2), encoding="utf-8")

    print("\n| Model | Quant | llama prompt | CKE prompt | CKE/llama | llama decode | CKE decode | CKE/llama |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        llama = row.get("llama") or {}
        cke = row.get("cke") or {}
        lp = llama.get("prompt_tok_s")
        cp = cke.get("prompt_tok_s")
        ld = llama.get("decode_tok_s")
        cd = cke.get("decode_tok_s")
        print(
            f"| {row['label']} | {row['quant']} | {_fmt(lp)} tok/s | {_fmt(cp)} tok/s | {_ratio(cp, lp)} | "
            f"{_fmt(ld)} tok/s | {_fmt(cd)} tok/s | {_ratio(cd, ld)} |"
        )
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
