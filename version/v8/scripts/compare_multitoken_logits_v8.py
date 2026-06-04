#!/usr/bin/env python3
from __future__ import annotations

"""
Tokenizer-free multi-token greedy parity probe.

This script repeatedly compares CK and llama.cpp logits for the same explicit
token prefix, appends the shared greedy top-1 token, and stops at the first
top-1 divergence. It is deliberately deterministic and sampler-free so that
generation collapse can be separated from sampling/template issues.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from compare_first_token_logits_v8 import (  # type: ignore
    compare_logits,
    discover_ck_model_dir,
    discover_gguf,
    load_ck_logits,
    load_ck_logits_segmented,
    load_runtime_contract,
    parse_tokens_csv,
    run_llama_logits,
    run_llama_logits_segmented,
)


def run_multitoken_parity(
    *,
    model_dir: Path,
    gguf_path: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    ctx_len: int,
    top_k: int,
    threads: int,
    append_on_divergence: str,
    ck_prefill_mode: str,
    llama_decode_mode: str,
    llama_no_repack: bool,
) -> dict[str, Any]:
    tokens = [int(t) for t in prompt_tokens]
    steps: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None

    for step in range(max(1, int(max_new_tokens))):
        if llama_decode_mode == "hybrid":
            ll = run_llama_logits_segmented(
                gguf_path,
                [int(t) for t in prompt_tokens],
                [int(t) for t in tokens[len(prompt_tokens) :]],
                int(ctx_len),
                int(top_k),
                int(threads),
                prefix_decode_mode="batched",
                decode_mode="sequential",
                no_repack=llama_no_repack,
            )
        else:
            ll = run_llama_logits(
                gguf_path,
                tokens,
                int(ctx_len),
                int(top_k),
                int(threads),
                decode_mode=llama_decode_mode,
                no_repack=llama_no_repack,
            )
        generated_tokens = [int(t) for t in tokens[len(prompt_tokens) :]]
        if ck_prefill_mode == "hybrid":
            ck = load_ck_logits_segmented(
                model_dir=model_dir,
                prompt_tokens=[int(t) for t in prompt_tokens],
                decode_tokens=generated_tokens,
                ck_prefill_mode="hybrid",
            )
        else:
            ck = load_ck_logits(model_dir, tokens, ck_prefill_mode=ck_prefill_mode)
        cmp = compare_logits(ck["logits"], ll["logits"], int(top_k))
        ck_next = int(cmp["top1_ck"])
        llama_next = int(cmp["top1_llama"])
        top1_match = bool(ck_next == llama_next)

        row = {
            "step": int(step),
            "prefix_len": int(len(tokens)),
            "ck_next": ck_next,
            "llama_next": llama_next,
            "top1_match": top1_match,
            "cosine": float(cmp["cosine"]),
            "rmse": float(cmp["rmse"]),
            "mean_abs_diff": float(cmp["mean_abs_diff"]),
            "max_abs_diff": float(cmp["max_abs_diff"]),
            "ck_top1_margin": float(cmp.get("ck_top1_margin", 0.0)),
            "llama_top1_margin": float(cmp.get("llama_top1_margin", 0.0)),
            "ck_llama_winner_delta_in_ck": float(cmp.get("ck_llama_winner_delta_in_ck", 0.0)),
            "llama_winner_delta_in_llama": float(cmp.get("llama_winner_delta_in_llama", 0.0)),
            "topk_overlap_count": int(cmp["topk_overlap_count"]),
            "topk_overlap_ratio": float(cmp["topk_overlap_ratio"]),
            "ck_topk_ids": list(cmp["ck_topk_ids"]),
            "llama_topk_ids": list(cmp["llama_topk_ids"]),
            "topk_logits": list(cmp.get("topk_logits", [])),
        }
        steps.append(row)

        if not top1_match and first_divergence is None:
            first_divergence = row
            if append_on_divergence == "stop":
                break

        if top1_match or append_on_divergence == "llama":
            tokens.append(llama_next)
        elif append_on_divergence == "ck":
            tokens.append(ck_next)
        else:
            break

    return {
        "status": "pass" if first_divergence is None else "fail",
        "pass": first_divergence is None,
        "model_dir": str(model_dir),
        "gguf_path": str(gguf_path),
        "initial_tokens": [int(t) for t in prompt_tokens],
        "final_prefix": tokens,
        "max_new_tokens": int(max_new_tokens),
        "ctx_len": int(ctx_len),
        "top_k": int(top_k),
        "threads": int(threads),
        "append_on_divergence": str(append_on_divergence),
        "ck_prefill_mode": str(ck_prefill_mode),
        "llama_decode_mode": str(llama_decode_mode),
        "llama_no_repack": bool(llama_no_repack),
        "first_divergence": first_divergence,
        "steps": steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tokenizer-free multi-token greedy parity (CK vs llama.cpp)")
    ap.add_argument("--model-dir", required=True, type=Path, help="run dir or .ck_build dir containing libmodel.so")
    ap.add_argument("--gguf", default=None, type=Path, help="GGUF path for llama.cpp runtime")
    ap.add_argument("--tokens", required=True, help="comma-separated prompt token IDs")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--ctx-len", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument(
        "--llama-decode-mode",
        choices=["auto", "batched", "sequential", "hybrid"],
        default="auto",
        help="llama.cpp replay mode; hybrid batches the initial prompt then decodes generated tokens sequentially.",
    )
    ap.add_argument(
        "--ck-prefill-mode",
        choices=["auto", "sequential", "batched", "hybrid"],
        default="auto",
        help=(
            "CK replay mode. auto follows runtime_contract; sequential feeds every token through decode; "
            "batched runs the whole prefix through ck_model_forward; hybrid batches the initial prompt "
            "then decodes generated tokens one by one."
        ),
    )
    ap.add_argument(
        "--llama-no-repack",
        action="store_true",
        help="Disable llama.cpp CPU tensor repacking in the replay helper for accumulation-order attribution.",
    )
    ap.add_argument(
        "--append-on-divergence",
        choices=["stop", "llama", "ck"],
        default="stop",
        help="What to append after first top-1 mismatch.",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--summary", action="store_true", help="Print a compact one-line result instead of full JSON.")
    args = ap.parse_args()

    model_dir = discover_ck_model_dir(args.model_dir)
    gguf_path = discover_gguf(args.gguf, model_dir)
    prompt_tokens = parse_tokens_csv(args.tokens)
    runtime_contract = load_runtime_contract(model_dir)
    llama_decode_mode = str(args.llama_decode_mode)
    if llama_decode_mode == "auto":
        prefill_policy = str(runtime_contract.get("prefill_policy") or "batched").strip().lower()
        llama_decode_mode = "hybrid" if prefill_policy == "sequential_decode" else "batched"
    report = run_multitoken_parity(
        model_dir=model_dir,
        gguf_path=gguf_path,
        prompt_tokens=prompt_tokens,
        max_new_tokens=int(args.max_new_tokens),
        ctx_len=int(args.ctx_len),
        top_k=int(args.top_k),
        threads=int(args.threads),
        append_on_divergence=str(args.append_on_divergence),
        ck_prefill_mode=str(args.ck_prefill_mode),
        llama_decode_mode=llama_decode_mode,
        llama_no_repack=bool(args.llama_no_repack),
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.summary:
        first = report.get("first_divergence")
        if first:
            print(
                "status=fail "
                f"step={first['step']} prefix_len={first['prefix_len']} "
                f"ck_next={first['ck_next']} llama_next={first['llama_next']} "
                f"llama_mode={llama_decode_mode} "
                f"ck_mode={args.ck_prefill_mode} "
                f"llama_no_repack={bool(args.llama_no_repack)} "
                f"cosine={first['cosine']:.6f} rmse={first['rmse']:.6f} "
                f"ck_margin={first['ck_top1_margin']:.6f} llama_margin={first['llama_top1_margin']:.6f} "
                f"topk_overlap={first['topk_overlap_count']}/{args.top_k}"
            )
        else:
            print(
                "status=pass "
                f"llama_mode={llama_decode_mode} "
                f"ck_mode={args.ck_prefill_mode} "
                f"llama_no_repack={bool(args.llama_no_repack)} "
                f"steps={len(report.get('steps', []))} "
                f"final_prefix_len={len(report.get('final_prefix', []))}"
            )
    else:
        print(json.dumps(report))
    return 0 if report.get("pass") else 3


if __name__ == "__main__":
    raise SystemExit(main())
