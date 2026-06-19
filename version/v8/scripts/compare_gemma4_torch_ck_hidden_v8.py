#!/usr/bin/env python3
from __future__ import annotations

"""Compare Gemma4 PyTorch layer-boundary tensors against CK hidden dumps.

This script is intentionally model-file based: pass a local HF/Transformers
Gemma4 safetensors directory plus a CK_DEBUG_EXPORT_HIDDEN output directory.
It dumps the PyTorch last-token layer stream and compares labels that CK already
exports.  The first pass focuses on stable graph boundaries, because those are
the right places to distinguish stitching bugs from lower-level kernel drift.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_tokens_csv(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise SystemExit("--tokens did not contain any token IDs")
    return out


def _to_numpy_f32(x: Any) -> np.ndarray:
    return x.detach().float().cpu().numpy().astype(np.float32, copy=False)


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    n = min(int(af.size), int(bf.size))
    if n <= 0:
        return {"compared": 0, "error": "empty input"}
    af = af[:n]
    bf = bf[:n]
    diff = af - bf
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    max_idx = int(np.argmax(np.abs(diff)))
    return {
        "a_size": int(a.size),
        "b_size": int(b.size),
        "compared": n,
        "cosine": float(np.dot(af, bf) / denom) if denom else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "max_idx": max_idx,
        "torch_at_max": float(af[max_idx]),
        "ck_at_max": float(bf[max_idx]),
    }


def _ck_name(token_index: int, layer: int, label: str) -> str:
    if layer < 0:
        return f"tok_{token_index:04d}_layer_-01_{label}.f32"
    return f"tok_{token_index:04d}_layer_{layer:03d}_{label}.f32"


def _load_ck_vector(ck_dir: Path, ck_token_index: int, layer: int, label: str) -> tuple[np.ndarray, Path] | tuple[None, None]:
    candidates = [ck_dir / _ck_name(ck_token_index, layer, label)]
    if not label.endswith("_last"):
        candidates.append(ck_dir / _ck_name(ck_token_index, layer, f"{label}_last"))
    for path in candidates:
        if path.exists():
            return np.fromfile(path, dtype=np.float32), path
    return None, None


def _torch_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(name)


def _load_input_ids(args: argparse.Namespace):
    import torch

    if args.tokens:
        ids = _parse_tokens_csv(args.tokens)
    else:
        if not args.prompt:
            raise SystemExit("provide either --tokens or --prompt")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SystemExit("transformers is required for --prompt tokenization") from exc
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        ids = tok(args.prompt, add_special_tokens=not args.no_special_tokens)["input_ids"]
    if args.max_tokens and len(ids) > args.max_tokens:
        ids = ids[: int(args.max_tokens)]
    return torch.tensor([ids], dtype=torch.long), ids


def _get_attr_path(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _find_language_layers(model: Any) -> Any | None:
    for path in ("model.language_model.layers", "language_model.layers", "model.layers"):
        layers = _get_attr_path(model, path)
        if layers is not None:
            return layers
    return None


def _last_token_vec_from_output(x: Any, token_index: int) -> np.ndarray:
    if isinstance(x, tuple):
        x = x[0]
    t = x.detach().float().cpu()
    if t.ndim == 3:
        t = t[0, token_index, :]
    elif t.ndim == 4:
        # Gemma4 attention internals are usually [B, T, H, D] before
        # transpose and [B, H, T, D] after transpose. Hooked q/k/v norm
        # modules use the former; this fallback keeps later attention hooks
        # usable if Transformers changes one of those local layouts.
        if t.shape[1] > token_index:
            t = t[0, token_index, :, :]
        elif t.shape[2] > token_index:
            t = t[0, :, token_index, :]
        else:
            t = t.reshape(-1)
    elif t.ndim == 2:
        t = t[token_index, :]
    else:
        t = t.reshape(-1)
    return t.numpy().astype(np.float32, copy=False)


def _maybe_get_logits(outputs: Any) -> np.ndarray | None:
    logits = getattr(outputs, "logits", None)
    if logits is None:
        return None
    return _to_numpy_f32(logits[0, -1, :])


def export_and_compare(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    model_dir = Path(args.model)
    ck_dir = Path(args.ck_hidden_dir)
    torch_out = Path(args.torch_out) if args.torch_out else None
    if torch_out:
        torch_out.mkdir(parents=True, exist_ok=True)

    dtype = _torch_dtype(args.dtype)
    input_ids, ids = _load_input_ids(args)
    token_index = int(args.token_index)
    if token_index < 0:
        token_index = len(ids) + token_index
    if token_index < 0 or token_index >= len(ids):
        raise SystemExit(f"--token-index out of range for {len(ids)} tokens: {args.token_index}")
    ck_token_index = token_index if args.ck_token_index is None else int(args.ck_token_index)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    layer_hook_outputs: dict[int, np.ndarray] = {}
    op_hook_outputs: dict[tuple[int, str], np.ndarray] = {}
    input_hook_outputs: dict[tuple[int, str], np.ndarray] = {}
    hook_handles = []
    if args.layer_source == "hooks":
        layers = _find_language_layers(model)
        if layers is None:
            raise SystemExit("unable to find language model layers for --layer-source hooks")
        for idx, layer_module in enumerate(layers):
            def make_hook(layer_idx: int):
                def hook(_module: Any, _inp: Any, out: Any) -> None:
                    layer_hook_outputs[layer_idx] = _last_token_vec_from_output(out, token_index)
                return hook
            hook_handles.append(layer_module.register_forward_hook(make_hook(idx)))
            if args.include_op_hooks:
                op_map = (
                    ("self_attn.q_proj", "q_proj"),
                    ("self_attn.k_proj", "k_proj"),
                    ("self_attn.v_proj", "v_proj"),
                    ("self_attn.q_norm", "qk_norm_q"),
                    ("self_attn.k_norm", "qk_norm_k"),
                    ("self_attn.v_norm", "v_norm"),
                    ("self_attn.o_proj", "out_proj"),
                    ("self_attn", "attn_module"),
                    ("post_attention_layernorm", "post_attn_norm"),
                    ("pre_feedforward_layernorm", "ffn_norm"),
                    ("mlp", "mlp_down"),
                    ("post_feedforward_layernorm", "post_ffn_norm"),
                    ("post_per_layer_input_norm", "post_per_layer_input_norm"),
                )
                for attr, label in op_map:
                    module = _get_attr_path(layer_module, attr)
                    if module is None:
                        continue
                    def make_op_hook(layer_idx: int, hook_label: str):
                        def hook(_module: Any, _inp: Any, out: Any) -> None:
                            op_hook_outputs[(layer_idx, hook_label)] = _last_token_vec_from_output(out, token_index)
                        return hook
                    hook_handles.append(module.register_forward_hook(make_op_hook(idx, label)))
                    if attr == "self_attn.o_proj":
                        def make_input_hook(layer_idx: int):
                            def hook(_module: Any, inp: Any) -> None:
                                if inp:
                                    input_hook_outputs[(layer_idx, "attn_out")] = _last_token_vec_from_output(inp[0], token_index)
                            return hook
                        hook_handles.append(module.register_forward_pre_hook(make_input_hook(idx)))

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    for handle in hook_handles:
        handle.remove()

    hidden_states = list(getattr(outputs, "hidden_states", ()) or ())
    rows: list[dict[str, Any]] = []

    def add_row(label: str, layer: int, torch_vec: np.ndarray, ck_label: str) -> None:
        if torch_out:
            torch_vec.astype(np.float32, copy=False).tofile(torch_out / _ck_name(token_index, layer, label))
        ck_vec, ck_path = _load_ck_vector(ck_dir, ck_token_index, layer, ck_label)
        row: dict[str, Any] = {
            "layer": layer,
            "torch": label,
            "ck": ck_label,
            "ck_path": str(ck_path) if ck_path else "",
            "token_index": token_index,
            "token_id": int(ids[token_index]),
            "status": "missing_ck" if ck_vec is None else "ok",
        }
        if ck_vec is not None:
            row.update(_compare(torch_vec, ck_vec))
        rows.append(row)

    if hidden_states:
        # hidden_states[0] is post-token-embedding; hidden_states[i+1] is layer i output
        # in the standard HF decoder API.
        emb = _to_numpy_f32(hidden_states[0][0, token_index, :])
        add_row("embedding", -2, emb, args.ck_embedding_label)

    if args.layer_source == "hooks":
        for layer, label in sorted(input_hook_outputs):
            add_row(label, layer, input_hook_outputs[(layer, label)], label)
        for layer, label in sorted(op_hook_outputs):
            add_row(label, layer, op_hook_outputs[(layer, label)], label)
        for layer in sorted(layer_hook_outputs):
            add_row("layer_out", layer, layer_hook_outputs[layer], args.ck_layer_out_label)
    elif hidden_states:
        for layer, hs in enumerate(hidden_states[1:]):
            add_row("layer_out", layer, _to_numpy_f32(hs[0, token_index, :]), args.ck_layer_out_label)

    logits = _maybe_get_logits(outputs)
    if logits is not None:
        if torch_out:
            logits.astype(np.float32, copy=False).tofile(torch_out / _ck_name(token_index, -1, "logits"))
        ck_logits, ck_logits_path = _load_ck_vector(ck_dir, ck_token_index, -1, "logits")
        row = {
            "layer": -1,
            "torch": "logits",
            "ck": "logits",
            "ck_path": str(ck_logits_path) if ck_logits_path else "",
            "token_index": token_index,
            "token_id": int(ids[token_index]),
            "status": "missing_ck" if ck_logits is None else "ok",
        }
        if ck_logits is not None:
            row.update(_compare(logits, ck_logits))
            row["torch_top1"] = int(np.argmax(logits))
            row["ck_top1"] = int(np.argmax(ck_logits))
        rows.append(row)

    first_bad: dict[str, Any] | None = None
    for row in rows:
        if row.get("status") != "ok":
            continue
        if float(row.get("cosine", 1.0)) < float(args.cosine_threshold) or float(row.get("max_abs", 0.0)) > float(args.max_abs_threshold):
            first_bad = row
            break

    return {
        "model": str(model_dir),
        "ck_hidden_dir": str(ck_dir),
        "tokens": ids,
        "token_index": token_index,
        "ck_token_index": ck_token_index,
        "num_rows": len(rows),
        "first_bad": first_bad,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path, help="local HF/Transformers safetensors model directory")
    ap.add_argument("--ck-hidden-dir", required=True, type=Path, help="directory produced by CK_DEBUG_EXPORT_HIDDEN")
    ap.add_argument("--tokens", help="comma-separated token IDs; avoids tokenizer/template ambiguity")
    ap.add_argument("--prompt", help="prompt text to tokenize with the HF tokenizer")
    ap.add_argument("--no-special-tokens", action="store_true", help="do not add tokenizer special tokens for --prompt")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--token-index", type=int, default=-1, help="PyTorch token row to compare; default is last token")
    ap.add_argument("--ck-token-index", type=int, help="CK dump token index to read; useful because batched prefill last-row dumps use tok_0000")
    ap.add_argument("--ck-layer-out-label", default="gemma4_per_layer_embed", help="CK label corresponding to the PyTorch decoder layer output")
    ap.add_argument("--ck-embedding-label", default="embedding_scaled", help="CK label corresponding to PyTorch embedding hidden state")
    ap.add_argument("--layer-source", choices=("hooks", "hidden_states"), default="hooks", help="Use decoder-layer module hooks or HF output_hidden_states for layer outputs")
    ap.add_argument("--include-op-hooks", action="store_true", help="with hook-based layer output, also compare stable Gemma4 submodule boundaries")
    ap.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--torch-out", type=Path, help="optional directory for PyTorch .f32 dumps")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--cosine-threshold", type=float, default=0.999)
    ap.add_argument("--max-abs-threshold", type=float, default=1e-2)
    args = ap.parse_args()

    result = export_and_compare(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
