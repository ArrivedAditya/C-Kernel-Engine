#!/usr/bin/env python3
from __future__ import annotations

"""Gemma4 answer-level PyTorch-vs-CK parity/debug harness.

This is deliberately deterministic: it uses the HF tokenizer once, feeds the
same token IDs to PyTorch and CK, uses greedy argmax generation, and records the
first generated-token divergence.  When requested, it also exports CK hidden
snapshots and runs the Gemma4 PyTorch-vs-CK hidden comparator for the exact
prefix at the divergence point.

Important: current CK Gemma4 artifacts may be Q4 GGUF-derived BUMP, while the
PyTorch path is BF16 safetensors.  Exact full-answer equality is expected only
once CK runs the same safetensors/BF16 BUMP weights.  Until then this script is
used to separate prompt/template/sampler bugs, CK prefill/decode consistency,
and structural layer drift from ordinary quantization differences.
"""

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_ck_prefill_decode_logits_v8 import _extract_logits, _init_model  # type: ignore
from compare_first_token_logits_v8 import discover_ck_model_dir  # type: ignore

DEFAULT_PROMPT = "Give me a detailed example of a C, Python, and SQL program."


def _torch_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(name)


def _topk(logits: np.ndarray, k: int = 10) -> list[dict[str, float | int]]:
    k = max(1, min(int(k), int(logits.size)))
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return [{"token": int(i), "logit": float(logits[int(i)])} for i in idx]


def _compare_logits(a: np.ndarray, b: np.ndarray, top_k: int = 20) -> dict[str, Any]:
    n = min(int(a.size), int(b.size))
    af = a[:n].astype(np.float64)
    bf = b[:n].astype(np.float64)
    diff = af - bf
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    a_top = [x["token"] for x in _topk(a, top_k)]
    b_top = [x["token"] for x in _topk(b, top_k)]
    return {
        "top1_a": int(np.argmax(a)),
        "top1_b": int(np.argmax(b)),
        "top1_match": int(np.argmax(a)) == int(np.argmax(b)),
        "cosine": float(np.dot(af, bf) / denom) if denom else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "topk_overlap": int(len(set(a_top).intersection(b_top))),
        "topk": int(top_k),
        "a_top": _topk(a, top_k),
        "b_top": _topk(b, top_k),
    }


def _load_tokenizer(hf_model: Path, trust_remote_code: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_model, trust_remote_code=trust_remote_code)


def _prompt_tokens(tokenizer: Any, prompt: str, chat_template: bool, add_special_tokens: bool) -> list[int]:
    if chat_template:
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "keys") and "input_ids" in ids.keys():
            ids = ids["input_ids"]
        elif isinstance(ids, dict):
            ids = ids.get("input_ids")
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]
    return [int(x) for x in tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]]


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _run_torch_greedy(
    hf_model: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    dtype: str,
    trust_remote_code: bool,
    eos_tokens: set[int],
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=_torch_dtype(dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated: list[int] = []
    step_logits: list[np.ndarray] = []
    past = None
    cur = input_ids
    with torch.no_grad():
        for step in range(max_new_tokens):
            out = model(
                input_ids=cur,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = out.logits[0, -1, :].detach().float().cpu().numpy().astype(np.float32, copy=True)
            step_logits.append(logits)
            tok = int(np.argmax(logits))
            generated.append(tok)
            if tok in eos_tokens:
                break
            past = out.past_key_values
            cur = torch.tensor([[tok]], dtype=torch.long, device=device)
    return {"tokens": generated, "logits": step_logits}


def _load_ck(model_dir: Path) -> tuple[ctypes.CDLL, int]:
    model_dir = discover_ck_model_dir(model_dir)
    lib = ctypes.CDLL(str(model_dir / "libmodel.so"), mode=ctypes.RTLD_GLOBAL)
    _init_model(lib, model_dir)
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    vocab = int(lib.ck_model_get_vocab_size())
    if vocab <= 0:
        raise RuntimeError(f"invalid CK vocab size: {vocab}")
    if hasattr(lib, "ck_model_kv_cache_reset"):
        lib.ck_model_kv_cache_reset.argtypes = []
        lib.ck_model_kv_cache_reset.restype = None
    if hasattr(lib, "ck_model_free"):
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    return lib, vocab


def _run_ck_greedy(
    ck_model_dir: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    eos_tokens: set[int],
) -> dict[str, Any]:
    lib, vocab = _load_ck(ck_model_dir)
    generated: list[int] = []
    step_logits: list[np.ndarray] = []
    try:
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        arr = (ctypes.c_int32 * len(prompt_tokens))(*[int(t) for t in prompt_tokens])
        rc = lib.ck_model_embed_tokens(arr, len(prompt_tokens))
        if rc != 0:
            raise RuntimeError(f"ck_model_embed_tokens failed rc={rc}")
        rc = lib.ck_model_forward(None)
        if rc != 0:
            raise RuntimeError(f"ck_model_forward failed rc={rc}")
        for step in range(max_new_tokens):
            logits, _stride, _active = _extract_logits(lib, vocab, len(prompt_tokens) + step)
            step_logits.append(logits)
            tok = int(np.argmax(logits))
            generated.append(tok)
            if tok in eos_tokens:
                break
            rc = lib.ck_model_decode(ctypes.c_int32(tok), None)
            if rc != 0:
                raise RuntimeError(f"ck_model_decode failed rc={rc} token={tok}")
    finally:
        if hasattr(lib, "ck_model_free"):
            try:
                lib.ck_model_free()
            except Exception:
                pass
    return {"tokens": generated, "logits": step_logits}


def _run_subprocess(cmd: list[str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def _hidden_debug(
    hf_model: Path,
    ck_model_dir: Path,
    prefix_tokens: list[int],
    prompt_token_count: int,
    out_dir: Path,
    dtype: str,
    trust_remote_code: bool,
    cosine_threshold: float,
    max_abs_threshold: float,
) -> dict[str, Any]:
    ck_hidden = out_dir / "ck_hidden"
    torch_hidden = out_dir / "torch_hidden"
    compare_json = out_dir / "hidden_compare.json"
    token_csv = ",".join(str(t) for t in prefix_tokens)
    export_mode = "prefill-decode" if len(prefix_tokens) > int(prompt_token_count) else "prefill"
    export_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "export_ck_hidden_v8.py"),
        "--model-dir",
        str(ck_model_dir),
        "--tokens",
        token_csv,
        "--out-dir",
        str(ck_hidden),
        "--mode",
        export_mode,
    ]
    if export_mode == "prefill-decode":
        export_cmd.extend(["--prompt-length", str(prompt_token_count)])
    export_result = _run_subprocess(export_cmd)

    ck_token_index = 0
    token_re = re.compile(r"tok_(\d+)_")
    token_indices: list[int] = []
    for path in ck_hidden.glob("tok_*.f32"):
        m = token_re.search(path.name)
        if m:
            token_indices.append(int(m.group(1)))
    if token_indices:
        ck_token_index = max(token_indices)

    compare_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "compare_gemma4_torch_ck_hidden_v8.py"),
        "--model",
        str(hf_model),
        "--ck-hidden-dir",
        str(ck_hidden),
        "--tokens",
        token_csv,
        "--token-index",
        str(len(prefix_tokens) - 1),
        "--ck-token-index",
        str(ck_token_index),
        "--dtype",
        dtype,
        "--torch-out",
        str(torch_hidden),
        "--json-out",
        str(compare_json),
        "--cosine-threshold",
        str(cosine_threshold),
        "--max-abs-threshold",
        str(max_abs_threshold),
        "--layer-source",
        "hooks",
        "--include-op-hooks",
    ]
    if trust_remote_code:
        compare_cmd.append("--trust-remote-code")
    compare_result = _run_subprocess(compare_cmd)
    parsed = None
    if compare_json.exists():
        parsed = json.loads(compare_json.read_text(encoding="utf-8"))
    return {"export_ck_hidden": export_result, "compare_hidden": compare_result, "parsed": parsed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-model", required=True, type=Path, help="local HF Gemma4 safetensors directory")
    ap.add_argument("--ck-model-dir", required=True, type=Path, help="CK model runtime directory")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--chat-template", action="store_true", help="use tokenizer.apply_chat_template for the user prompt")
    ap.add_argument("--add-special-tokens", action="store_true", help="only applies when --chat-template is not used")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--debug-hidden", action="store_true")
    ap.add_argument("--debug-prefix", choices=("prompt", "first-divergence"), default="first-divergence")
    ap.add_argument("--cosine-threshold", type=float, default=0.90)
    ap.add_argument("--max-abs-threshold", type=float, default=10.0)
    ap.add_argument("--out-dir", type=Path, default=Path("build/gemma4_answer_parity_debug"))
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok = _load_tokenizer(args.hf_model, args.trust_remote_code)
    prompt_tokens = _prompt_tokens(tok, args.prompt, args.chat_template, args.add_special_tokens)
    eos_tokens = set(int(x) for x in tok.all_special_ids if x is not None)

    torch_run = _run_torch_greedy(
        args.hf_model,
        prompt_tokens,
        args.max_new_tokens,
        args.dtype,
        args.trust_remote_code,
        eos_tokens,
    )
    ck_run = _run_ck_greedy(args.ck_model_dir, prompt_tokens, args.max_new_tokens, eos_tokens)

    steps = min(len(torch_run["tokens"]), len(ck_run["tokens"]))
    first_divergence = None
    step_reports: list[dict[str, Any]] = []
    for i in range(steps):
        t_tok = int(torch_run["tokens"][i])
        c_tok = int(ck_run["tokens"][i])
        cmp = _compare_logits(torch_run["logits"][i], ck_run["logits"][i], args.top_k)
        row = {
            "step": i,
            "torch_token": t_tok,
            "ck_token": c_tok,
            "match": t_tok == c_tok,
            "torch_piece": _decode(tok, [t_tok]),
            "ck_piece": _decode(tok, [c_tok]),
            "logits": cmp,
        }
        step_reports.append(row)
        if first_divergence is None and t_tok != c_tok:
            first_divergence = row
    if first_divergence is None and len(torch_run["tokens"]) != len(ck_run["tokens"]):
        first_divergence = {"step": steps, "reason": "different generation lengths"}

    torch_answer = _decode(tok, torch_run["tokens"])
    ck_answer = _decode(tok, ck_run["tokens"])
    report: dict[str, Any] = {
        "hf_model": str(args.hf_model),
        "ck_model_dir": str(args.ck_model_dir),
        "prompt": args.prompt,
        "chat_template": bool(args.chat_template),
        "prompt_tokens": prompt_tokens,
        "prompt_token_count": len(prompt_tokens),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "torch": {"tokens": torch_run["tokens"], "answer": torch_answer},
        "ck": {"tokens": ck_run["tokens"], "answer": ck_answer},
        "first_divergence": first_divergence,
        "steps": step_reports,
    }

    if args.debug_hidden:
        if args.debug_prefix == "prompt" or first_divergence is None or not isinstance(first_divergence.get("step"), int):
            prefix = list(prompt_tokens)
        else:
            div_step = int(first_divergence["step"])
            # Compare the state used to predict the divergent token.  Up to that
            # point both systems should have consumed the same generated tokens.
            prefix = list(prompt_tokens) + [int(t) for t in torch_run["tokens"][:div_step]]
        report["hidden_debug_prefix_tokens"] = prefix
        report["hidden_debug"] = _hidden_debug(
            args.hf_model,
            args.ck_model_dir,
            prefix,
            len(prompt_tokens),
            args.out_dir / "hidden_debug",
            args.dtype,
            args.trust_remote_code,
            args.cosine_threshold,
            args.max_abs_threshold,
        )

    json_out = args.json_out or (args.out_dir / "answer_parity_report.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"prompt_tokens={len(prompt_tokens)} torch_new={len(torch_run['tokens'])} ck_new={len(ck_run['tokens'])}")
    print(f"first_divergence={first_divergence}")
    print("\n[PyTorch answer]\n" + torch_answer)
    print("\n[CK answer]\n" + ck_answer)
    print(f"\nreport={json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
