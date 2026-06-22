#!/usr/bin/env python3
from __future__ import annotations

"""GLM4 safetensors PyTorch-vs-CK logit parity guardrail.

This script compares next-token logits for the same teacher-forced prefixes:

1. Load the official GLM4 safetensors model with Transformers/PyTorch.
2. Tokenize either --prompt or explicit --tokens.
3. Run one causal PyTorch forward over the full prefix.
4. Replay selected prefixes through an already-built CK v8 runtime.
5. Compare final-token logits, top-k overlap, and tie-aware top-1 ranking.

Use this for CK safetensors-to-BUMP parity first. It can also be pointed at a
GGUF-converted CK runtime, but that compares quantized CK weights against
unquantized PyTorch weights and should be interpreted as a smoke/diagnostic,
not exact numerical parity.
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

from version.v8.scripts.compare_first_token_logits_v8 import (  # noqa: E402
    compare_logits,
    discover_ck_model_dir,
    load_ck_logits,
    parse_tokens_csv,
)


def _torch_dtype(name: str) -> torch.dtype:
    n = str(name).strip().lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp32", "float32"}:
        return torch.float32
    if n in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


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


def _prefixes(spec: str, token_count: int) -> list[int]:
    if token_count <= 0:
        return []
    text = str(spec or "auto").strip().lower()
    if text in {"all", "every"}:
        return list(range(1, token_count + 1))
    if text in {"auto", ""}:
        vals = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        vals = [v for v in vals if v <= token_count]
        if token_count not in vals:
            vals.append(token_count)
        return vals
    vals: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v < 0:
            v = token_count + v + 1
        if 1 <= v <= token_count:
            vals.append(v)
    return sorted(set(vals))


def _parse_tokens_or_prompt(args: argparse.Namespace, tokenizer: Any) -> tuple[list[int], str]:
    if args.tokens:
        ids = parse_tokens_csv(args.tokens)
        return ids, tokenizer.decode(ids)
    enc = tokenizer(
        args.prompt,
        add_special_tokens=not bool(args.no_special_tokens),
        return_tensors="pt",
    )
    ids = [int(x) for x in enc["input_ids"][0].tolist()]
    if args.max_input_tokens and len(ids) > int(args.max_input_tokens):
        ids = ids[: int(args.max_input_tokens)]
    return ids, args.prompt


def _row_report(
    *,
    tokenizer: Any,
    prefix_len: int,
    ck_logits: np.ndarray,
    torch_logits: np.ndarray,
    top_k: int,
    require_top1: bool,
    min_topk_overlap: float,
    max_abs_threshold: float,
    tie_margin: float,
) -> dict[str, Any]:
    cmp = compare_logits(ck_logits.astype(np.float32, copy=False), torch_logits.astype(np.float32, copy=False), top_k)
    ck_ids = [int(x) for x in cmp["ck_topk_ids"]]
    torch_ids = [int(x) for x in cmp["llama_topk_ids"]]
    cmp["ck_topk_texts"] = _top_texts(tokenizer, ck_ids)
    cmp["torch_topk_texts"] = _top_texts(tokenizer, torch_ids)
    cmp["ck_topk_values"] = _top_values(ck_logits, ck_ids)
    cmp["torch_topk_values"] = _top_values(torch_logits, torch_ids)

    torch_top1_margin = float(cmp.get("llama_top1_margin", 0.0))
    top1_match = bool(cmp["top1_match"])
    tie_aware_top1 = top1_match or (torch_top1_margin <= float(tie_margin) and int(cmp["top1_ck"]) in torch_ids)
    finite = _finite_stats(ck_logits)
    torch_finite = _finite_stats(torch_logits)
    finite_ok = finite["finite"] == int(ck_logits.size) and torch_finite["finite"] == int(torch_logits.size)
    top1_ok = (not bool(require_top1)) or tie_aware_top1
    overlap_ok = float(cmp["topk_overlap_ratio"]) >= float(min_topk_overlap)
    max_abs_ok = float(cmp["max_abs_diff"]) <= float(max_abs_threshold)
    passed = bool(finite_ok and top1_ok and overlap_ok and max_abs_ok)
    return {
        "prefix_len": int(prefix_len),
        "status": "pass" if passed else "fail",
        "pass": passed,
        "finite_ok": bool(finite_ok),
        "top1_match": top1_match,
        "tie_aware_top1_ok": bool(tie_aware_top1),
        "overlap_ok": bool(overlap_ok),
        "max_abs_ok": bool(max_abs_ok),
        "ck_finite": finite,
        "torch_finite": torch_finite,
        "compare": cmp,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safetensors", required=True, type=Path, help="Official GLM4 HF safetensors directory")
    ap.add_argument("--ck-model-dir", required=True, type=Path, help="CK v8 run dir containing libmodel.so/weights.bump")
    ap.add_argument("--prompt", default="Give me a detailed example of C code.")
    ap.add_argument("--tokens", help="Comma-separated token ids; overrides --prompt")
    ap.add_argument("--prefixes", default="auto", help="auto, all, or comma-separated prefix lengths")
    ap.add_argument("--max-input-tokens", type=int, default=0)
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "fp16"), default="bf16")
    ap.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--ck-prefill-mode", default="auto", choices=("auto", "batched", "sequential", "hybrid"))
    ap.add_argument("--require-top1-match", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tie-margin", type=float, default=0.05)
    ap.add_argument("--min-topk-overlap", type=float, default=0.50)
    ap.add_argument("--max-abs-threshold", type=float, default=1.0e9)
    ap.add_argument("--dump-dir", type=Path, help="Optional directory for logits .npy dumps")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    safetensors_dir = args.safetensors.expanduser().resolve()
    ck_model_dir = discover_ck_model_dir(args.ck_model_dir)
    dtype = _torch_dtype(args.dtype)

    load_t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(safetensors_dir, trust_remote_code=bool(args.trust_remote_code))
    model = AutoModelForCausalLM.from_pretrained(
        safetensors_dir,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=bool(args.low_cpu_mem_usage),
    )
    model.eval()
    load_seconds = time.time() - load_t0

    token_ids, prompt_text = _parse_tokens_or_prompt(args, tokenizer)
    if not token_ids:
        raise SystemExit("empty token sequence")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    wanted_prefixes = _prefixes(args.prefixes, len(token_ids))
    if not wanted_prefixes:
        raise SystemExit("no valid prefixes selected")

    fwd_t0 = time.time()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False, return_dict=True)
    torch_seconds = time.time() - fwd_t0
    torch_logits_all = out.logits[0].float().contiguous().cpu().numpy().astype(np.float32, copy=False)

    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_dir / "input_ids.npy", input_ids.cpu().numpy().astype(np.int64, copy=False))

    rows: list[dict[str, Any]] = []
    ck_seconds_total = 0.0
    for prefix_len in wanted_prefixes:
        torch_logits = torch_logits_all[int(prefix_len) - 1]
        ck_t0 = time.time()
        ck = load_ck_logits(ck_model_dir, token_ids[: int(prefix_len)], ck_prefill_mode=args.ck_prefill_mode)
        ck_seconds = time.time() - ck_t0
        ck_seconds_total += ck_seconds
        row = _row_report(
            tokenizer=tokenizer,
            prefix_len=int(prefix_len),
            ck_logits=ck["logits"],
            torch_logits=torch_logits,
            top_k=int(args.top_k),
            require_top1=bool(args.require_top1_match),
            min_topk_overlap=float(args.min_topk_overlap),
            max_abs_threshold=float(args.max_abs_threshold),
            tie_margin=float(args.tie_margin),
        )
        row["ck_seconds"] = ck_seconds
        row["ck_active_tokens"] = int(ck.get("active_tokens", 0))
        row["ck_prefill_policy"] = str(ck.get("prefill_policy", ""))
        rows.append(row)
        if args.dump_dir:
            np.save(args.dump_dir / f"torch_logits_prefix_{int(prefix_len):04d}.npy", torch_logits)
            np.save(args.dump_dir / f"ck_logits_prefix_{int(prefix_len):04d}.npy", ck["logits"].astype(np.float32, copy=False))

    passed = all(bool(r["pass"]) for r in rows)
    first_fail = next((r for r in rows if not bool(r["pass"])), None)
    report: dict[str, Any] = {
        "status": "pass" if passed else "fail",
        "pass": bool(passed),
        "safetensors": str(safetensors_dir),
        "ck_model_dir": str(ck_model_dir),
        "dtype": args.dtype,
        "prompt": prompt_text,
        "input_ids": [int(x) for x in token_ids],
        "prefixes": [int(x) for x in wanted_prefixes],
        "thresholds": {
            "require_top1_match": bool(args.require_top1_match),
            "tie_margin": float(args.tie_margin),
            "min_topk_overlap": float(args.min_topk_overlap),
            "max_abs_threshold": float(args.max_abs_threshold),
        },
        "timing": {
            "torch_load_seconds": float(load_seconds),
            "torch_forward_seconds": float(torch_seconds),
            "ck_total_seconds": float(ck_seconds_total),
        },
        "rows": rows,
        "first_failure": first_fail,
        "next_step": "GLM4 CK-vs-PyTorch logits pass for selected prefixes" if passed else "dump layer boundaries around the first failing prefix",
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
