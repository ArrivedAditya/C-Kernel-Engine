#!/usr/bin/env python3
from __future__ import annotations

"""Nemotron-H safetensors PyTorch-vs-CK first-token guardrail.

This is the correctness workflow entry point after compiler/lowering guardrails:

1. Tokenize a prompt, or accept explicit token ids.
2. Run the safetensors model in PyTorch with the local CPU Nemotron shims.
3. Run the converted CK/BUMP runtime on the same token ids.
4. Compare final-token logits and top-k ranking.
5. Optionally dump PyTorch hidden states for the next layer-by-layer pass.

The script intentionally exits nonzero when requested semantic thresholds fail.
That makes it useful for nightly/local guardrails while still producing a JSON
artifact with enough evidence for the next debugging step.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from version.v8.scripts.compare_first_token_logits_v8 import compare_logits, discover_ck_model_dir, load_ck_logits
from version.v8.scripts.export_ck_hidden_v8 import export_ck_hidden
from version.v8.scripts.export_nemotron_torch_hidden_v8 import (
    _install_nemotron_cpu_shims,
    _parse_hidden_indices,
    _parse_tokens,
    _tensor_stats,
)


def _finite_stats(x: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(x)
    return {
        "size": int(x.size),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(x).sum()),
        "inf": int(np.isinf(x).sum()),
        "min": float(np.min(x[finite])) if finite.any() else None,
        "max": float(np.max(x[finite])) if finite.any() else None,
    }


def _top_texts(tokenizer: Any, ids: list[int]) -> list[str]:
    out: list[str] = []
    for tok in ids:
        try:
            out.append(str(tokenizer.decode([int(tok)])))
        except Exception:
            out.append("")
    return out


def _top_values(logits: np.ndarray, ids: list[int]) -> list[float]:
    return [float(logits[int(i)]) for i in ids]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safetensors", required=True, type=Path, help="Nemotron-H safetensors directory")
    ap.add_argument("--ck-model-dir", required=True, type=Path, help="CK run dir containing libmodel.so/weights.bump")
    ap.add_argument("--prompt", default="Hello", help="Prompt text; ignored when --tokens is set")
    ap.add_argument("--tokens", help="Comma-separated token ids; overrides --prompt")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--ck-prefill-mode", default="auto", choices=("auto", "batched", "sequential", "hybrid"))
    ap.add_argument("--require-top1-match", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-topk-overlap", type=float, default=0.50)
    ap.add_argument("--max-abs-threshold", type=float, default=1.0e9)
    ap.add_argument("--hidden-indices", default="0,1,2,5,10,20,30,40,-1")
    ap.add_argument("--dump-dir", type=Path, help="Optional directory for PyTorch .npy dumps and CK hidden snapshots")
    ap.add_argument("--ck-hidden-mode", choices=("decode", "prefill"), default="decode")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    _install_nemotron_cpu_shims()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    safetensors_dir = args.safetensors.expanduser().resolve()
    ck_model_dir = discover_ck_model_dir(args.ck_model_dir)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    load_t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(safetensors_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        safetensors_dir,
        trust_remote_code=True,
        dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.time() - load_t0

    token_ids = _parse_tokens(args.tokens)
    if token_ids is None:
        encoded = tokenizer(args.prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        prompt_text = args.prompt
        token_ids = [int(x) for x in input_ids[0].tolist()]
    else:
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        prompt_text = tokenizer.decode(token_ids)

    fwd_t0 = time.time()
    with torch.inference_mode():
        torch_out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
    torch_seconds = time.time() - fwd_t0

    torch_logits = torch_out.logits[0, -1].float().contiguous().cpu().numpy().astype(np.float32, copy=False)
    ck = load_ck_logits(ck_model_dir, token_ids, ck_prefill_mode=args.ck_prefill_mode)
    ck_logits = ck["logits"].astype(np.float32, copy=False)
    cmp = compare_logits(ck_logits, torch_logits, int(args.top_k))

    ck_ids = cmp["ck_topk_ids"]
    torch_ids = cmp["llama_topk_ids"]
    cmp["ck_topk_texts"] = _top_texts(tokenizer, ck_ids)
    cmp["torch_topk_texts"] = _top_texts(tokenizer, torch_ids)
    cmp["ck_topk_values"] = _top_values(ck_logits, ck_ids)
    cmp["torch_topk_values"] = _top_values(torch_logits, torch_ids)

    hidden_count = len(torch_out.hidden_states)
    hidden_indices = _parse_hidden_indices(args.hidden_indices, hidden_count)
    hidden_report: dict[str, Any] = {}

    ck_hidden_dir = None
    ck_hidden_files: list[str] = []
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_dir / "input_ids.npy", input_ids.cpu().numpy().astype(np.int64, copy=False))
        np.save(args.dump_dir / "torch_logits_last.npy", torch_logits)
        np.save(args.dump_dir / "ck_logits_last.npy", ck_logits)
        ck_hidden_dir = args.dump_dir / "ck_hidden"
        export_ck_hidden(
            model_dir=ck_model_dir,
            tokens=token_ids,
            out_dir=ck_hidden_dir,
            mode=args.ck_hidden_mode,
        )
        ck_hidden_files = [p.name for p in sorted(ck_hidden_dir.glob("*.f32"))]

    for idx in hidden_indices:
        h_last = torch_out.hidden_states[idx][0, -1].float().contiguous()
        hidden_report[str(idx)] = _tensor_stats(h_last)
        if args.dump_dir:
            np.save(args.dump_dir / f"torch_hidden_{idx:03d}_last.npy", h_last.cpu().numpy().astype(np.float32, copy=False))

    overlap_ok = cmp["topk_overlap_ratio"] >= float(args.min_topk_overlap)
    top1_threshold_ok = (not bool(args.require_top1_match)) or bool(cmp["top1_match"])
    max_abs_ok = cmp["max_abs_diff"] <= float(args.max_abs_threshold)
    ck_finite = _finite_stats(ck_logits)
    torch_finite = _finite_stats(torch_logits)
    finite_ok = ck_finite["finite"] == int(ck_logits.size) and torch_finite["finite"] == int(torch_logits.size)
    semantic_top1_match = bool(cmp["top1_match"])
    semantic_topk_overlap_ok = cmp["topk_overlap_ratio"] >= 0.50
    passed = bool(top1_threshold_ok and overlap_ok and max_abs_ok and finite_ok)
    next_step = (
        "top logits match; proceed to longer prompts"
        if semantic_top1_match and semantic_topk_overlap_ok
        else "run layer/op hidden parity around the first mismatching boundary"
    )

    report: dict[str, Any] = {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "safetensors": str(safetensors_dir),
        "ck_model_dir": str(ck_model_dir),
        "dtype": args.dtype,
        "prompt": prompt_text,
        "input_ids": [int(x) for x in token_ids],
        "timing": {
            "torch_load_seconds": load_seconds,
            "torch_forward_seconds": torch_seconds,
        },
        "thresholds": {
            "require_top1_match": bool(args.require_top1_match),
            "min_topk_overlap": float(args.min_topk_overlap),
            "max_abs_threshold": float(args.max_abs_threshold),
        },
        "ck": {
            "vocab": int(ck["vocab"]),
            "active_tokens": int(ck["active_tokens"]),
            "logits_stride": int(ck["stride"]),
            "prefill_policy": str(ck.get("prefill_policy", "")),
            "contract_prefill_policy": str(ck.get("contract_prefill_policy", "")),
            "finite": ck_finite,
        },
        "torch": {
            "logits_shape": [int(x) for x in torch_out.logits.shape],
            "hidden_count": int(hidden_count),
            "finite": torch_finite,
        },
        "compare": cmp,
        "semantic": {
            "top1_match": semantic_top1_match,
            "topk_overlap_ok": semantic_topk_overlap_ok,
            "topk_overlap_ratio": float(cmp["topk_overlap_ratio"]),
        },
        "threshold_result": {
            "top1_threshold_ok": bool(top1_threshold_ok),
            "overlap_threshold_ok": bool(overlap_ok),
            "max_abs_threshold_ok": bool(max_abs_ok),
            "finite_ok": bool(finite_ok),
        },
        "torch_hidden": hidden_report,
        "dump_artifacts": {
            "dump_dir": str(args.dump_dir) if args.dump_dir else None,
            "ck_hidden_dir": str(ck_hidden_dir) if ck_hidden_dir else None,
            "ck_hidden_file_count": int(len(ck_hidden_files)),
            "ck_hidden_files_sample": ck_hidden_files[:20],
        },
        "next_step": next_step,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
