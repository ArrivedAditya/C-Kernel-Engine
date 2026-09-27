#!/usr/bin/env python3
from __future__ import annotations

"""
ARM/x86 v8 baseline certification probe (NEW CERTIFICATION lane).

Drives a compiled v8 runtime through the exact ck_chat.py inference path and
records a deterministic, cross-machine-comparable report:

- formatted prompt text and prompt token IDs (C tokenizer path)
- first-prefill logits: sha256, argmax, top-k ids/values (optionally the raw
  fp32 vector via --logits-out for offline max-abs/cosine diffs)
- greedy decode trajectory token IDs (temperature 0, pure argmax, no sampler
  filters so the trajectory depends only on the numerics)
- prefill/decode timings and peak RSS of the probe process

Usage:
  version/v8/scripts/arm_baseline_probe_v8.py \
    --model-dir ~/.cache/ck-engine-v8/models/<model-dir> \
    --prompt "hello" --chat-template auto --max-tokens 32 \
    --json-out probe.json --logits-out first_logits.npy
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"


def _load_ck_chat():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "ck_chat_probe", SCRIPTS_DIR / "ck_chat.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPTS_DIR / 'ck_chat.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _logits_sha256(logits: np.ndarray) -> str:
    canonical = np.ascontiguousarray(logits, dtype=np.dtype("<f4"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _topk(logits: np.ndarray, k: int) -> list[dict]:
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    return [{"id": int(i), "value": float(logits[i])} for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="v8 baseline certification probe (ARM/x86)")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--prompt", default="hello")
    ap.add_argument("--chat-template", default="auto")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--logits-out", type=Path, default=None,
                    help="optional .npy dump of the raw first-prefill logits (fp32)")
    args = ap.parse_args()

    ck_chat = _load_ck_chat()
    model = ck_chat.CKModel(str(args.model_dir))
    if not model.load(chat_template=str(args.chat_template)):
        raise RuntimeError(f"model load failed: {args.model_dir}")

    formatted = model.format_chat_conversation([("user", str(args.prompt))])
    prompt_ids = [int(t) for t in model.encode(formatted)]

    if not (model.has_kv_decode and model.kv_cache_enable()):
        raise RuntimeError("probe requires KV decode support")
    model.kv_cache_reset()

    t0 = time.perf_counter()
    logits = np.asarray(model.prefill(prompt_ids), dtype=np.float32)
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    first_logits = logits.copy()

    trajectory: list[int] = []
    eos_index: int | None = None
    t0 = time.perf_counter()
    for step in range(int(args.max_tokens)):
        next_token = int(np.argmax(logits))
        trajectory.append(next_token)
        if eos_index is None and model.is_eos_token(next_token):
            eos_index = step
        logits = np.asarray(model.decode_step(next_token), dtype=np.float32)
    decode_ms = (time.perf_counter() - t0) * 1000.0
    model.free()

    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports ru_maxrss in KiB.
    peak_rss_bytes = peak_rss * 1024 if sys.platform.startswith("linux") else peak_rss

    report = {
        "probe": "arm_baseline_probe_v8",
        "label": "NEW CERTIFICATION (DGX Spark AArch64 baseline; not recovered TDA4VM parity)",
        "model_dir": str(args.model_dir),
        "prompt": str(args.prompt),
        "chat_template": str(args.chat_template),
        "formatted_prompt": formatted,
        "prompt_token_ids": prompt_ids,
        "first_token_logits": {
            "vocab": int(first_logits.size),
            "sha256": _logits_sha256(first_logits),
            "argmax": int(np.argmax(first_logits)),
            "min": float(first_logits.min()),
            "max": float(first_logits.max()),
            "topk": _topk(first_logits, int(args.top_k)),
        },
        "decode": {
            "max_tokens": int(args.max_tokens),
            "token_ids": trajectory,
            "first_eos_index": eos_index,
        },
        "timing": {
            "prefill_ms": prefill_ms,
            "prefill_tokens": len(prompt_ids),
            "decode_ms": decode_ms,
            "decode_tokens": len(trajectory),
            "decode_ms_per_token": decode_ms / max(1, len(trajectory)),
        },
        "peak_rss_bytes": peak_rss_bytes,
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
    }

    if args.logits_out:
        args.logits_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.logits_out, first_logits)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
