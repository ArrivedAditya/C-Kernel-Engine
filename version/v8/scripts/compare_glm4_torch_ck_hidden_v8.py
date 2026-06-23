#!/usr/bin/env python3
from __future__ import annotations

"""Compare GLM4 PyTorch layer/op tensors against CK hidden dumps.

This is a diagnostic companion to compare_glm4_torch_ck_v8.py.  CK exports
last-token prefill snapshots through CK_DEBUG_EXPORT_HIDDEN; this script hooks
the matching Hugging Face GLM4 modules and reports where the hidden stream first
opens up.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_tokens_csv(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise SystemExit("--tokens did not contain any token IDs")
    return out


def _torch_dtype(name: str):
    import torch

    n = str(name).lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp32", "float32"}:
        return torch.float32
    if n in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(name)


def _to_np(x: Any, token_index: int) -> np.ndarray:
    import torch

    if isinstance(x, tuple):
        x = x[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(type(x).__name__)
    t = x.detach().float().cpu()
    if t.ndim == 3:
        t = t[0, token_index, :]
    elif t.ndim == 2:
        t = t[token_index, :]
    elif t.ndim == 4:
        if t.shape[1] > token_index:
            t = t[0, token_index, :, :]
        elif t.shape[2] > token_index:
            t = t[0, :, token_index, :]
        else:
            t = t.reshape(-1)
    else:
        t = t.reshape(-1)
    return t.numpy().astype(np.float32, copy=False).reshape(-1)


def _cmp(torch_vec: np.ndarray, ck_vec: np.ndarray) -> dict[str, Any]:
    a = torch_vec.reshape(-1).astype(np.float64)
    b = ck_vec.reshape(-1).astype(np.float64)
    n = min(int(a.size), int(b.size))
    if n <= 0:
        return {"compared": 0, "error": "empty"}
    a = a[:n]
    b = b[:n]
    d = a - b
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    max_idx = int(np.argmax(np.abs(d)))
    return {
        "torch_size": int(torch_vec.size),
        "ck_size": int(ck_vec.size),
        "compared": int(n),
        "cosine": float(np.dot(a, b) / denom) if denom else float("nan"),
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "max_idx": max_idx,
        "torch_at_max": float(a[max_idx]),
        "ck_at_max": float(b[max_idx]),
    }


def _ck_file(ck_dir: Path, ck_token_index: int, layer: int, label: str) -> Path:
    return ck_dir / f"tok_{ck_token_index:04d}_layer_{layer:03d}_{label}.f32"


def _load_ck(ck_dir: Path, ck_token_index: int, layer: int, label: str) -> np.ndarray | None:
    labels = [label]
    if not label.endswith("_last"):
        labels.append(f"{label}_last")
    for lab in labels:
        path = _ck_file(ck_dir, ck_token_index, layer, lab)
        if path.exists():
            return np.fromfile(path, dtype=np.float32)
    return None


def _get_attr(obj: Any, dotted: str) -> Any | None:
    cur = obj
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _load_input(args: argparse.Namespace, tokenizer: Any) -> tuple[list[int], str]:
    if args.tokens:
        ids = _parse_tokens_csv(args.tokens)
        return ids, tokenizer.decode(ids)
    enc = tokenizer(args.prompt, add_special_tokens=not args.no_special_tokens, return_tensors="pt")
    ids = [int(x) for x in enc["input_ids"][0].tolist()]
    if args.max_input_tokens and len(ids) > int(args.max_input_tokens):
        ids = ids[: int(args.max_input_tokens)]
    return ids, args.prompt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safetensors", required=True, type=Path)
    ap.add_argument("--ck-hidden-dir", required=True, type=Path)
    ap.add_argument("--prompt", default="Give me a detailed example of C, Python, and SQL code.")
    ap.add_argument("--tokens")
    ap.add_argument("--max-input-tokens", type=int, default=0)
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "fp16"), default="bf16")
    ap.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--token-index", type=int, default=-1, help="PyTorch token row to compare")
    ap.add_argument("--ck-token-index", type=int, default=0, help="CK hidden dump token prefix, usually 0 for prefill last-row exports")
    ap.add_argument("--layers", default="all", help="all or comma-separated layer ids")
    ap.add_argument("--fail-max", type=float, default=2.0)
    ap.add_argument("--fail-cos", type=float, default=0.995)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = args.safetensors.expanduser().resolve()
    ck_dir = args.ck_hidden_dir.expanduser().resolve()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=bool(args.trust_remote_code))
    token_ids, prompt_text = _load_input(args, tok)
    token_index = int(args.token_index)
    if token_index < 0:
        token_index = len(token_ids) + token_index
    if token_index < 0 or token_index >= len(token_ids):
        raise SystemExit(f"token index out of range: {args.token_index} for {len(token_ids)} tokens")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=_torch_dtype(args.dtype),
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    layers = _get_attr(model, "model.layers")
    if layers is None:
        raise SystemExit("unable to find model.layers")

    selected_layers: list[int]
    if str(args.layers).strip().lower() == "all":
        selected_layers = list(range(len(layers)))
    else:
        selected_layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]

    captures: dict[tuple[int, str], np.ndarray] = {}
    layer_inputs: dict[int, np.ndarray] = {}
    post_self_norm: dict[int, np.ndarray] = {}
    after_attn: dict[int, np.ndarray] = {}
    handles = []

    def save(layer_idx: int, label: str, value: Any) -> None:
        captures[(layer_idx, label)] = _to_np(value, token_index)

    for layer_idx in selected_layers:
        layer = layers[layer_idx]

        def make_pre(idx: int):
            def pre(_module: Any, inputs: tuple[Any, ...]) -> None:
                if inputs:
                    layer_inputs[idx] = _to_np(inputs[0], token_index)
            return pre

        handles.append(layer.register_forward_pre_hook(make_pre(layer_idx)))
        for attr, label in (
            ("self_attn.q_proj", "q_proj"),
            ("self_attn.k_proj", "k_proj"),
            ("self_attn.v_proj", "v_proj"),
            ("self_attn.o_proj", "out_proj"),
            ("post_self_attn_layernorm", "post_attn_norm"),
            ("mlp.gate_up_proj", "mlp_gate_up"),
            ("mlp.activation_fn", "mlp_activation"),
            ("mlp.down_proj", "mlp_down"),
            ("post_mlp_layernorm", "post_ffn_norm"),
        ):
            module = _get_attr(layer, attr)
            if module is None:
                continue

            def make_hook(idx: int, lab: str):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                    vec = _to_np(output, token_index)
                    if lab == "mlp_gate_up":
                        half = vec.size // 2
                        captures[(idx, "mlp_gate")] = vec[:half].copy()
                        captures[(idx, "mlp_up")] = vec[half:].copy()
                    else:
                        captures[(idx, lab)] = vec
                        if lab == "post_attn_norm":
                            post_self_norm[idx] = vec
                return hook

            handles.append(module.register_forward_hook(make_hook(layer_idx, label)))

        def make_layer_hook(idx: int):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                captures[(idx, "layer_out")] = _to_np(output, token_index)
                if idx in layer_inputs and idx in post_self_norm:
                    aa = layer_inputs[idx] + post_self_norm[idx]
                    after_attn[idx] = aa
                    captures[(idx, "after_attn")] = aa
                    captures[(idx, "ffn_residual")] = aa
            return hook

        handles.append(layer.register_forward_hook(make_layer_hook(layer_idx)))

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        _ = model(input_ids=input_ids, use_cache=False, return_dict=True)
    for h in handles:
        h.remove()

    rows: list[dict[str, Any]] = []
    first_bad: dict[str, Any] | None = None
    labels = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "post_attn_norm",
        "after_attn",
        "ffn_residual",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
        "post_ffn_norm",
        "layer_out",
    ]
    for layer_idx in selected_layers:
        for label in labels:
            tv = captures.get((layer_idx, label))
            if tv is None:
                continue
            cv = _load_ck(ck_dir, int(args.ck_token_index), layer_idx, label)
            if cv is None:
                continue
            c = _cmp(tv, cv)
            row = {"layer": layer_idx, "label": label, **c}
            bad = bool(float(c["max_abs"]) > float(args.fail_max) or float(c["cosine"]) < float(args.fail_cos))
            row["status"] = "fail" if bad else "pass"
            rows.append(row)
            if bad and first_bad is None:
                first_bad = row

    report = {
        "status": "pass" if first_bad is None else "fail",
        "prompt": prompt_text,
        "input_ids": token_ids,
        "token_index": int(token_index),
        "ck_token_index": int(args.ck_token_index),
        "ck_hidden_dir": str(ck_dir),
        "thresholds": {"fail_max": float(args.fail_max), "fail_cos": float(args.fail_cos)},
        "rows": rows,
        "first_bad": first_bad,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if first_bad is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
