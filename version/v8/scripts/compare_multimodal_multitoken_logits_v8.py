#!/usr/bin/env python3
from __future__ import annotations

"""
Tokenizer-free multimodal multi-token greedy parity probe.

This is the multimodal counterpart to compare_multitoken_logits_v8.py.  It
starts from a bridge_report.json + prefix.f32 produced by run_multimodal_bridge_v8,
does one CK mixed visual/text prefill, then advances CK through real
ck_model_decode calls while llama.cpp replays the same generated suffix.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import decoder_first_token_parity_v8 as first_token  # type: ignore  # noqa: E402
from gguf_tokenizer import GGUFTokenizer  # type: ignore  # noqa: E402


def _ck_logits_from_buffer(buf: Any, vocab_size: int) -> np.ndarray:
    return np.ctypeslib.as_array(buf, shape=(int(vocab_size),)).astype(np.float32, copy=True)


def _decode_topk(logits: np.ndarray, tokenizer: GGUFTokenizer, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(logits.size)
    k = max(1, min(int(top_k), n))
    top = np.argpartition(-logits, k - 1)[:k]
    top = top[np.argsort(-logits[top])]
    for idx in top.tolist():
        rows.append(
            {
                "token_id": int(idx),
                "logit": float(logits[int(idx)]),
                "token_text": tokenizer.decode([int(idx)], skip_special=False),
            }
        )
    return rows



def _run_llama_greedy_sequence(inputs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    helper = first_token.compare_first_token_logits_v7.ensure_llama_helper()
    with tempfile.TemporaryDirectory(prefix="llama_token_replay_v8_seq_") as td:
        tmp = Path(td)
        logits_out = tmp / "llama_final.f32"
        seq_out = tmp / "llama_seq.f32"
        cmd = [
            str(helper),
            "--model",
            str(inputs["gguf_path"]),
            "--ctx",
            str(int(inputs["ctx_len"])),
            "--top-k",
            str(int(args.top_k)),
            "--decode-mode",
            str(args.llama_decode_mode),
            "--logits-out",
            str(logits_out),
            "--logits-seq-out",
            str(seq_out),
            "--greedy-steps",
            str(int(args.max_new_tokens)),
        ]
        if inputs["tokens_before"]:
            cmd.extend(["--tokens-before", ",".join(str(t) for t in inputs["tokens_before"])])
            if inputs["tokens_after"]:
                cmd.extend(["--tokens-after", ",".join(str(t) for t in inputs["tokens_after"])])
        elif inputs["tokens_after"]:
            cmd.extend(["--tokens", ",".join(str(t) for t in inputs["tokens_after"])])
        if inputs["llama_prefix_path"] is not None:
            cmd.extend(["--prefix-f32", str(inputs["llama_prefix_path"])])
        if inputs["prefix_grid"] is not None:
            gx, gy = inputs["prefix_grid"]
            cmd.extend(["--prefix-grid-x", str(int(gx)), "--prefix-grid-y", str(int(gy))])
        if int(inputs["prefix_row_dim"]) > 0:
            cmd.extend(["--prefix-row-dim", str(int(inputs["prefix_row_dim"]))])
        cmd.extend(["--prefix-text-pos", str(int(inputs["prefix_text_pos"]))])
        if int(args.threads) > 0:
            cmd.extend(["--threads", str(int(args.threads))])
        if bool(args.llama_no_repack):
            cmd.append("--no-repack")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "llama_token_replay greedy sequence failed\n"
                f"cmd: {' '.join(cmd)}\nrc: {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"
            )
        payload = proc.stdout.strip().splitlines()[-1]
        meta = json.loads(payload)
        if not isinstance(meta, dict) or not meta.get("ok"):
            raise RuntimeError(f"llama_token_replay returned invalid payload: {payload}")
        n_vocab = int(meta.get("n_vocab", 0))
        steps = int(meta.get("greedy_steps", args.max_new_tokens) or 0)
        logits = np.fromfile(seq_out, dtype=np.float32)
        expected = int(steps) * int(n_vocab)
        if logits.size != expected:
            raise RuntimeError(f"llama sequence logits size mismatch: got={logits.size} expected={expected}")
        return {
            "meta": meta,
            "logits": logits.reshape((steps, n_vocab)),
        }

def _prepare_inputs(args: argparse.Namespace, *, parity_dump: bool = False) -> dict[str, Any]:
    bridge_report = first_token._load_bridge_report(args.bridge_report.resolve())
    gguf_value = str(((bridge_report.get("decoder_runtime") or {}).get("gguf") or "")).strip()
    if not gguf_value:
        raise ValueError("bridge report does not include decoder_runtime.gguf; pass a fresh report")
    gguf_path = Path(gguf_value).resolve()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    decoder_dir = workdir / "decoder"

    requested_ctx_len = int(args.ctx_len or bridge_report.get("decoder_context_len") or 256)
    runtime = first_token.bridge_runner_v8._prepare_decoder_runtime(
        gguf_path,
        decoder_dir,
        parity_dump=bool(parity_dump),
        context_override=max(1, requested_ctx_len),
    )
    tokenizer = GGUFTokenizer.from_gguf(str(gguf_path))
    _, tokens_before, tokens_after, prompt_meta = first_token._resolve_prompt_token_segments(
        tokenizer,
        prompt=None,
        tokens_csv=None,
        tokens_before_csv=None,
        tokens_after_csv=None,
        bridge_report=bridge_report,
    )
    prefix_path = args.prefix_f32.resolve() if args.prefix_f32 is not None else None
    if prefix_path is None:
        report_prefix = str(bridge_report.get("prefix_dump_path") or "").strip()
        if report_prefix:
            prefix_path = Path(report_prefix).resolve()
    prefix_embeddings, prefix_tokens, prefix_row_dim, prefix_source = first_token._load_prefix_embeddings(
        prefix_path,
        0,
        int(runtime["embed_dim"]),
        int(runtime.get("input_embed_dim", 0) or 0),
        int(args.prefix_row_dim) if args.prefix_row_dim is not None else None,
    )
    total_prompt_tokens = len(tokens_before) + len(tokens_after)
    resolved_ctx_len = max(int(requested_ctx_len), int(prefix_tokens) + total_prompt_tokens, 1)
    if resolved_ctx_len != requested_ctx_len:
        runtime = first_token.bridge_runner_v8._prepare_decoder_runtime(
            gguf_path,
            decoder_dir,
            parity_dump=bool(parity_dump),
            context_override=resolved_ctx_len,
        )

    bridge_grid_x = None if bridge_report.get("prefix_grid_x") is None else int(bridge_report.get("prefix_grid_x"))
    bridge_grid_y = None if bridge_report.get("prefix_grid_y") is None else int(bridge_report.get("prefix_grid_y"))
    prefix_grid = first_token._resolve_prefix_grid(
        int(prefix_tokens),
        int(args.prefix_grid_x) if args.prefix_grid_x is not None else bridge_grid_x,
        int(args.prefix_grid_y) if args.prefix_grid_y is not None else bridge_grid_y,
    )
    prefix_text_pos = (
        int(args.prefix_text_pos)
        if args.prefix_text_pos is not None
        else (
            int(bridge_report.get("prefix_text_pos"))
            if bridge_report.get("prefix_text_pos") is not None
            else (
                len(tokens_before) + max(int(prefix_grid[0]), int(prefix_grid[1]))
                if prefix_grid is not None
                else len(tokens_before) + int(prefix_tokens)
            )
        )
    )
    llama_prefix_path = first_token._materialize_llama_prefix(
        prefix_embeddings,
        int(prefix_tokens),
        workdir,
        prefix_path=prefix_path,
    )
    return {
        "bridge_report": bridge_report,
        "gguf_path": gguf_path,
        "workdir": workdir,
        "runtime": runtime,
        "tokenizer": tokenizer,
        "tokens_before": [int(t) for t in tokens_before],
        "tokens_after": [int(t) for t in tokens_after],
        "prompt_meta": prompt_meta,
        "prefix_embeddings": prefix_embeddings,
        "prefix_tokens": int(prefix_tokens),
        "prefix_row_dim": int(prefix_row_dim),
        "prefix_source": prefix_source,
        "prefix_path": prefix_path,
        "llama_prefix_path": llama_prefix_path,
        "prefix_grid": prefix_grid,
        "prefix_text_pos": int(prefix_text_pos),
        "ctx_len": int(resolved_ctx_len),
        "requested_ctx_len": int(requested_ctx_len),
    }


def _capture_step_dump(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    step_index = int(args.dump_step)
    steps = list(report.get("steps") or [])
    if step_index < 0 or step_index >= len(steps):
        raise ValueError(f"--dump-step {step_index} is outside captured steps [0, {max(0, len(steps) - 1)}]")
    dump_dir = args.dump_dir
    if dump_dir is None:
        dump_dir = args.workdir / f"dump_step_{step_index:04d}"

    inputs = _prepare_inputs(args, parity_dump=True)
    row = dict(steps[step_index])
    generated_prefix = [int(t) for t in list(row.get("generated_prefix") or [])]
    replay_tokens_after = [int(t) for t in inputs["tokens_after"]] + generated_prefix
    ck, dump_report = first_token._capture_dump_compare(
        inputs["gguf_path"],
        inputs["runtime"],
        inputs["prefix_embeddings"],
        int(inputs["prefix_tokens"]),
        replay_tokens_after,
        tokens_before=inputs["tokens_before"],
        prefix_row_dim=int(inputs["prefix_row_dim"]),
        ctx_len=int(inputs["ctx_len"]),
        top_k=int(args.top_k),
        threads=int(args.threads),
        dump_root=dump_dir.resolve(),
        dump_names=str(args.dump_names),
        dump_pass=str(args.dump_pass),
        dump_atol=float(args.dump_atol),
        dump_rtol=float(args.dump_rtol),
        prefix_grid=inputs["prefix_grid"],
        prefix_text_pos=int(inputs["prefix_text_pos"]),
        ck_strict_parity=bool(args.ck_strict_parity),
    )
    return {
        "step": int(step_index),
        "step_row": row,
        "dump_dir": str(dump_dir.resolve()),
        "replay_tokens_after_count": int(len(replay_tokens_after)),
        "generated_prefix": generated_prefix,
        "ck_top1": int((ck.get("comparison") or {}).get("top1_ck", ck.get("ck_top1", -1))),
        "dump": dump_report,
    }


def _init_ck_state(inputs: dict[str, Any], strict_parity: bool) -> tuple[Any, Any, int]:
    runtime = inputs["runtime"]
    model_so = Path(runtime["so_path"])
    lib = first_token.bridge_runner_v8._load_decoder_lib(model_so)
    rc = lib.ck_model_init_with_manifest(
        str(runtime["weights_bump"]).encode(),
        str(runtime["manifest_map"]).encode(),
    )
    if rc != 0:
        raise RuntimeError(f"decoder init failed with rc={rc}")
    if hasattr(lib, "ck_set_strict_parity"):
        lib.ck_set_strict_parity(1 if strict_parity else 0)

    vocab_size = int(lib.ck_model_get_vocab_size())
    if vocab_size <= 0:
        vocab_size = int(runtime["vocab_size"])
    logits = (ctypes.c_float * vocab_size)()

    tokens_before = inputs["tokens_before"]
    tokens_after = inputs["tokens_after"]
    prefix_tokens = int(inputs["prefix_tokens"])
    prefix_row_dim = int(inputs["prefix_row_dim"])
    prefix_embeddings: array = inputs["prefix_embeddings"]

    prefix_ptr = None
    if prefix_tokens > 0:
        expected = prefix_tokens * prefix_row_dim
        if len(prefix_embeddings) != expected:
            raise RuntimeError(f"prefix float count mismatch: got={len(prefix_embeddings)} expected={expected}")
        prefix_ptr = (ctypes.c_float * len(prefix_embeddings))(*prefix_embeddings)
    before_arr = (ctypes.c_int32 * len(tokens_before))(*tokens_before) if tokens_before else None
    after_arr = (ctypes.c_int32 * len(tokens_after))(*tokens_after) if tokens_after else None
    grid = inputs["prefix_grid"]
    if tokens_before and hasattr(lib, "ck_model_forward_segments_grid_ex"):
        grid_x, grid_y = grid if grid is not None else (0, 0)
        rc = lib.ck_model_forward_segments_grid_ex(
            before_arr,
            len(tokens_before),
            prefix_ptr,
            prefix_tokens,
            prefix_row_dim,
            int(grid_x),
            int(grid_y),
            int(inputs["prefix_text_pos"]),
            after_arr,
            len(tokens_after),
            logits,
        )
    elif hasattr(lib, "ck_model_forward_mixed_grid_ex") and grid is not None:
        grid_x, grid_y = grid
        rc = lib.ck_model_forward_mixed_grid_ex(
            prefix_ptr,
            prefix_tokens,
            prefix_row_dim,
            int(grid_x),
            int(grid_y),
            int(inputs["prefix_text_pos"]),
            after_arr,
            len(tokens_after),
            logits,
        )
    elif hasattr(lib, "ck_model_forward_mixed_ex"):
        rc = lib.ck_model_forward_mixed_ex(prefix_ptr, prefix_tokens, prefix_row_dim, after_arr, len(tokens_after), logits)
    else:
        rc = lib.ck_model_forward_mixed(prefix_ptr, prefix_tokens, after_arr, len(tokens_after), logits)
    if rc != 0:
        raise RuntimeError(f"decoder forward_mixed failed rc={rc}")
    return lib, logits, vocab_size


def run_multimodal_multitoken_parity(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _prepare_inputs(args)
    tokenizer: GGUFTokenizer = inputs["tokenizer"]
    llama_seq = _run_llama_greedy_sequence(inputs, args)
    lib, logits_buf, vocab_size = _init_ck_state(inputs, bool(args.ck_strict_parity))
    generated: list[int] = []
    llama_generated = [int(t) for t in list((llama_seq.get("meta") or {}).get("greedy_generated", []))]
    steps: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    try:
        for step in range(max(1, int(args.max_new_tokens))):
            ck_logits = _ck_logits_from_buffer(logits_buf, vocab_size)
            if step >= int(llama_seq["logits"].shape[0]):
                raise RuntimeError(f"llama sequence ended before step={step}")
            llama_logits = llama_seq["logits"][step]
            cmp = first_token.compare_first_token_logits_v7.compare_logits(ck_logits, llama_logits, int(args.top_k))
            ck_next = int(cmp["top1_ck"])
            llama_next = int(llama_generated[step]) if step < len(llama_generated) else int(cmp["top1_llama"])
            top1_match = ck_next == llama_next
            row = {
                "step": int(step),
                "generated_prefix": [int(t) for t in generated],
                "prefix_len_after_image": int(len(inputs["tokens_after"]) + len(generated)),
                "ck_next": ck_next,
                "llama_next": llama_next,
                "ck_next_text": tokenizer.decode([ck_next], skip_special=False),
                "llama_next_text": tokenizer.decode([llama_next], skip_special=False),
                "top1_match": bool(top1_match),
                "llama_no_repack": bool(args.llama_no_repack),
                "cosine": float(cmp["cosine"]),
                "rmse": float(cmp["rmse"]),
                "mean_abs_diff": float(cmp["mean_abs_diff"]),
                "max_abs_diff": float(cmp["max_abs_diff"]),
                "ck_top1_margin": float(cmp.get("ck_top1_margin", 0.0)),
                "llama_top1_margin": float(cmp.get("llama_top1_margin", 0.0)),
                "topk_overlap_count": int(cmp["topk_overlap_count"]),
                "topk_overlap_ratio": float(cmp["topk_overlap_ratio"]),
                "ck_topk_ids": list(cmp["ck_topk_ids"]),
                "llama_topk_ids": list(cmp["llama_topk_ids"]),
                "ck_topk": _decode_topk(ck_logits, tokenizer, int(args.top_k)),
                "llama_topk": _decode_topk(llama_logits, tokenizer, int(args.top_k)),
                "topk_logits": list(cmp.get("topk_logits", [])),
            }
            steps.append(row)
            if not top1_match and first_divergence is None:
                first_divergence = row
                if args.append_on_divergence == "stop":
                    break
            if top1_match or args.append_on_divergence == "llama":
                next_token = llama_next
            elif args.append_on_divergence == "ck":
                next_token = ck_next
            else:
                break
            generated.append(int(next_token))
            if step + 1 >= int(args.max_new_tokens):
                break
            rc = lib.ck_model_decode(ctypes.c_int32(int(next_token)), logits_buf)
            if rc != 0:
                raise RuntimeError(f"ck_model_decode failed rc={rc} at step={step}")
    finally:
        if hasattr(lib, "ck_model_free"):
            try:
                lib.ck_model_free()
            except Exception:
                pass

    return {
        "status": "pass" if first_divergence is None else "fail",
        "pass": first_divergence is None,
        "bridge_report_path": str(args.bridge_report.resolve()),
        "gguf_path": str(inputs["gguf_path"]),
        "workdir": str(inputs["workdir"]),
        "ctx_len": int(inputs["ctx_len"]),
        "requested_ctx_len": int(inputs["requested_ctx_len"]),
        "threads": int(args.threads),
        "top_k": int(args.top_k),
        "llama_decode_mode": str(args.llama_decode_mode),
        "llama_greedy_generated": llama_generated,
        "append_on_divergence": str(args.append_on_divergence),
        "prompt_tokens_before_image": inputs["tokens_before"],
        "prompt_tokens_after_image": inputs["tokens_after"],
        "prefix": {
            "source": str(inputs["prefix_source"]),
            "tokens": int(inputs["prefix_tokens"]),
            "row_dim": int(inputs["prefix_row_dim"]),
            "path": None if inputs["prefix_path"] is None else str(inputs["prefix_path"]),
            "grid": None if inputs["prefix_grid"] is None else [int(inputs["prefix_grid"][0]), int(inputs["prefix_grid"][1])],
            "text_pos": int(inputs["prefix_text_pos"]),
        },
        "generated_shared_tokens": [int(t) for t in generated],
        "generated_shared_text": tokenizer.decode(generated, skip_special=False) if generated else "",
        "first_divergence": first_divergence,
        "steps": steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Multimodal multi-token greedy parity (CK vs llama.cpp)")
    ap.add_argument("--bridge-report", required=True, type=Path)
    ap.add_argument("--prefix-f32", type=Path, default=None)
    ap.add_argument("--prefix-row-dim", type=int, default=None)
    ap.add_argument("--prefix-grid-x", type=int, default=None)
    ap.add_argument("--prefix-grid-y", type=int, default=None)
    ap.add_argument("--prefix-text-pos", type=int, default=None)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--ctx-len", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--llama-decode-mode", choices=["batched", "sequential"], default="sequential")
    ap.add_argument("--llama-no-repack", action="store_true", help="Disable llama.cpp CPU tensor repacking in the replay helper")
    ap.add_argument("--llama-repack-fallback", action=argparse.BooleanOptionalAction, default=True, help="Retry a failed llama replay step with --no-repack")
    ap.add_argument("--append-on-divergence", choices=["stop", "llama", "ck"], default="stop")
    ap.add_argument("--ck-strict-parity", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--dump-step", type=int, default=None, help="Capture CK-vs-llama tensor dumps at this generated step")
    ap.add_argument("--dump-dir", type=Path, default=None, help="Directory for --dump-step tensor dumps")
    ap.add_argument(
        "--dump-names",
        type=str,
        default="Qcur-0,Kcur-0,Vcur-0,Qcur_normed-0,Kcur_normed-0,kqv_out-0",
        help="Comma-separated llama dump names for --dump-step",
    )
    ap.add_argument("--dump-pass", choices=("all", "prefill", "decode"), default="decode")
    ap.add_argument("--dump-atol", type=float, default=1.0e-4)
    ap.add_argument("--dump-rtol", type=float, default=1.0e-3)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.threads > 0:
        os.environ["CK_NUM_THREADS"] = str(int(args.threads))
        os.environ["OMP_NUM_THREADS"] = str(int(args.threads))
    report = run_multimodal_multitoken_parity(args)
    if args.dump_step is not None:
        report["step_dump"] = _capture_step_dump(report, args)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.summary:
        first = report.get("first_divergence")
        if first:
            print(
                "status=fail "
                f"step={first['step']} ck_next={first['ck_next']}({first['ck_next_text']!r}) "
                f"llama_next={first['llama_next']}({first['llama_next_text']!r}) "
                f"cosine={first['cosine']:.6f} rmse={first['rmse']:.6f} "
                f"topk_overlap={first['topk_overlap_count']}/{args.top_k}"
            )
        else:
            print(
                "status=pass "
                f"steps={len(report.get('steps', []))} "
                f"generated={report.get('generated_shared_text', '')!r}"
            )
        if report.get("step_dump"):
            dump = (report["step_dump"] or {}).get("dump") or {}
            first_issue = dump.get("first_issue")
            summary = dump.get("summary") or {}
            if first_issue:
                print(
                    "dump=fail "
                    f"step={report['step_dump']['step']} "
                    f"layer={first_issue.get('layer')} op={first_issue.get('op')} "
                    f"status={first_issue.get('status')} "
                    f"max_abs_diff={float(first_issue.get('max_abs_diff', 0.0)):.6g} "
                    f"summary={summary}"
                )
            else:
                print(f"dump={dump.get('status', 'ok')} step={report['step_dump']['step']} summary={summary}")
    else:
        print(json.dumps(report, indent=2))
    return 0 if report.get("pass") else 3


if __name__ == "__main__":
    raise SystemExit(main())
