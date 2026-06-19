#!/usr/bin/env python3
from __future__ import annotations

"""Convert HF safetensors language weights into v8 BUMPWGT5 artifacts.

This is a BUMP-first importer.  GGUF and safetensors are source formats; CK
runtime consumes weights.bump + sidecars.  The first supported target is the
Gemma4 text/language model path, using CK-internal manifest names so existing
v8 IR/codegen can consume the result.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_gguf_to_bump_v8 import (  # type: ignore
    BUMP_META_FOOTER_MAGIC,
    BUMP_VERSION_V5,
    CACHE_ALIGN,
    CK_DT_BF16,
    CK_DT_FP16,
    CK_DT_FP32,
    DATA_START,
    EXT_METADATA_SIZE,
    HEADER_SIZE,
    _canonical_json_bytes,
    _inject_runtime_config_defaults,
    apply_model_contract_overrides,
    build_bumpv5_metadata,
    calculate_manifest_hash,
    calculate_metadata_hash,
    calculate_template_hash,
    load_template_for_arch,
    write_bumpv5_footer,
)

DTYPE_TO_CK = {
    "F32": ("fp32", CK_DT_FP32, 4),
    "BF16": ("bf16", CK_DT_BF16, 2),
    "F16": ("fp16", CK_DT_FP16, 2),
    "float32": ("fp32", CK_DT_FP32, 4),
    "bfloat16": ("bf16", CK_DT_BF16, 2),
    "float16": ("fp16", CK_DT_FP16, 2),
}


def align_up(n: int, a: int = CACHE_ALIGN) -> int:
    return ((int(n) + int(a) - 1) // int(a)) * int(a)


@dataclass(frozen=True)
class TensorRef:
    ck_name: str
    source_names: tuple[str, ...]
    dtype: str | None = None
    synth: str | None = None
    shape: tuple[int, ...] | None = None


@dataclass
class HeaderTensor:
    name: str
    dtype: str
    shape: list[int]
    shard: Path


class HashingWriter:
    def __init__(self, f):
        self.f = f
        self.h = hashlib.sha256()
        self.bytes_written = 0

    def write(self, data: bytes) -> None:
        self.f.write(data)
        self.h.update(data)
        self.bytes_written += len(data)

    def digest(self) -> bytes:
        return self.h.digest()


def _load_safetensors_headers(model_dir: Path) -> dict[str, HeaderTensor]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors package is required") from exc

    index = model_dir / "model.safetensors.index.json"
    out: dict[str, HeaderTensor] = {}
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        shards = sorted({str(v) for v in weight_map.values()})
        for shard_name in shards:
            shard = model_dir / shard_name
            with safe_open(shard, framework="pt") as sf:
                for key in sf.keys():
                    sl = sf.get_slice(key)
                    out[key] = HeaderTensor(key, str(sl.get_dtype()), [int(x) for x in sl.get_shape()], shard)
        return out

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"No safetensors files found in {model_dir}")
    for shard in files:
        with safe_open(shard, framework="pt") as sf:
            for key in sf.keys():
                sl = sf.get_slice(key)
                out[key] = HeaderTensor(key, str(sl.get_dtype()), [int(x) for x in sl.get_shape()], shard)
    return out


def _load_tensor(model_dir: Path, headers: dict[str, HeaderTensor], name: str):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors package is required") from exc
    h = headers[name]
    with safe_open(h.shard, framework="pt") as sf:
        return sf.get_tensor(name).detach().cpu().contiguous()


def _torch_to_bytes(t, dtype_policy: str) -> tuple[bytes, str, list[int]]:
    import torch

    shape = [int(x) for x in t.shape]
    if dtype_policy == "fp32":
        arr = t.to(dtype=torch.float32).numpy().astype(np.float32, copy=False)
        return arr.tobytes(order="C"), "fp32", shape
    if dtype_policy == "bf16":
        u16 = t.to(dtype=torch.bfloat16).view(torch.uint16).numpy()
        return u16.tobytes(order="C"), "bf16", shape
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes(order="C"), "bf16", shape
    if t.dtype == torch.float16:
        return t.numpy().view(np.uint16).tobytes(order="C"), "fp16", shape
    arr = t.to(dtype=torch.float32).numpy().astype(np.float32, copy=False)
    return arr.tobytes(order="C"), "fp32", shape


def _dtype_code(dtype_name: str) -> int:
    if dtype_name == "fp32":
        return CK_DT_FP32
    if dtype_name == "bf16":
        return CK_DT_BF16
    if dtype_name == "fp16":
        return CK_DT_FP16
    raise ValueError(f"unsupported dtype {dtype_name}")


def _header_dtype_to_ck(dtype: str) -> tuple[str, int]:
    row = DTYPE_TO_CK.get(str(dtype))
    if row:
        return row[0], row[2]
    return "fp32", 4


def _synth_bytes(kind: str, shape: tuple[int, ...], dtype_policy: str) -> tuple[bytes, str, list[int]]:
    n = int(np.prod(shape, dtype=np.int64))
    if kind == "zeros_fp32":
        return np.zeros(n, dtype=np.float32).tobytes(), "fp32", list(shape)
    if kind == "ones_fp32":
        return np.ones(n, dtype=np.float32).tobytes(), "fp32", list(shape)
    raise ValueError(f"unknown synth tensor kind {kind}")


def _hf_config(model_dir: Path) -> dict[str, Any]:
    p = model_dir / "config.json"
    if not p.exists():
        raise SystemExit(f"Missing config.json in {model_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_runtime_config_template(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "config" in data and isinstance(data["config"], dict):
        return dict(data["config"])
    return dict(data)


def _gemma4_text_refs(config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    num_layers = int(config.get("num_layers") or config.get("num_hidden_layers") or 0)
    embed_dim = int(config.get("embed_dim") or config.get("hidden_size") or 0)
    intermediate = int(config.get("intermediate_size") or 0)
    per_layer_dim = int(config.get("per_layer_dim") or 0)
    if num_layers <= 0 or embed_dim <= 0 or intermediate <= 0:
        raise SystemExit("Gemma4 config missing num_layers/embed_dim/intermediate_size")
    if per_layer_dim <= 0:
        # HF tensor shape is [vocab, layers, per_layer_dim] or equivalent in current checkpoints.
        h = headers.get("model.language_model.per_layer_projection_norm.weight")
        if h and h.shape:
            per_layer_dim = int(h.shape[-1])
        else:
            raise SystemExit("Gemma4 config missing per_layer_dim and cannot infer it")
        config["per_layer_dim"] = per_layer_dim

    refs: list[TensorRef] = [
        TensorRef("token_emb", ("model.language_model.embed_tokens.weight",)),
        TensorRef("per_layer_token_emb", ("model.language_model.embed_tokens_per_layer.weight",)),
        TensorRef("per_layer_model_proj", ("model.language_model.per_layer_model_projection.weight",)),
        TensorRef("per_layer_proj_norm", ("model.language_model.per_layer_projection_norm.weight",), dtype="fp32"),
        TensorRef("rope_freqs", (), dtype="fp32", synth="ones_fp32", shape=(int(config.get("max_rotary_dim") or config.get("rotary_dim") or 256),)),
    ]
    for layer in range(num_layers):
        p = f"model.language_model.layers.{layer}"
        refs.extend([
            TensorRef(f"layer.{layer}.ln1_gamma", (f"{p}.input_layernorm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.ln2_gamma", (f"{p}.pre_feedforward_layernorm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.post_attention_norm", (f"{p}.post_attention_layernorm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.post_ffn_norm", (f"{p}.post_feedforward_layernorm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.wq", (f"{p}.self_attn.q_proj.weight",)),
            TensorRef(f"layer.{layer}.bq", (), dtype="fp32", synth="zeros_fp32", shape=(int(config.get("layer_q_dim", [embed_dim])[layer]) if isinstance(config.get("layer_q_dim"), list) else embed_dim,)),
            TensorRef(f"layer.{layer}.q_norm", (f"{p}.self_attn.q_norm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.wk", (f"{p}.self_attn.k_proj.weight",)),
            TensorRef(f"layer.{layer}.bk", (), dtype="fp32", synth="zeros_fp32", shape=(int(config.get("layer_kv_dim", [embed_dim])[layer]) if isinstance(config.get("layer_kv_dim"), list) else int(config.get("num_key_value_heads", config.get("num_kv_heads", 2))) * int(config.get("head_dim", 256)),)),
            TensorRef(f"layer.{layer}.k_norm", (f"{p}.self_attn.k_norm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.wv", (f"{p}.self_attn.v_proj.weight",)),
            TensorRef(f"layer.{layer}.bv", (), dtype="fp32", synth="zeros_fp32", shape=(int(config.get("layer_kv_dim", [embed_dim])[layer]) if isinstance(config.get("layer_kv_dim"), list) else int(config.get("num_key_value_heads", config.get("num_kv_heads", 2))) * int(config.get("head_dim", 256)),)),
            TensorRef(f"layer.{layer}.wo", (f"{p}.self_attn.o_proj.weight",)),
            TensorRef(f"layer.{layer}.bo", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
            TensorRef(f"layer.{layer}.w1", (f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight")),
            TensorRef(f"layer.{layer}.b1", (), dtype="fp32", synth="zeros_fp32", shape=(2 * intermediate,)),
            TensorRef(f"layer.{layer}.w2", (f"{p}.mlp.down_proj.weight",)),
            TensorRef(f"layer.{layer}.b2", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
            TensorRef(f"layer.{layer}.per_layer_inp_gate", (f"{p}.per_layer_input_gate.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.per_layer_proj", (f"{p}.per_layer_projection.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.per_layer_post_norm", (f"{p}.post_per_layer_input_norm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.layer_output_scale", (f"{p}.layer_scalar",), dtype="fp32"),
        ])
    refs.extend([
        TensorRef("final_ln_weight", ("model.language_model.norm.weight",), dtype="fp32"),
        TensorRef("final_ln_bias", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
    ])
    if "model.language_model.lm_head.weight" in headers:
        refs.append(TensorRef("output.weight", ("model.language_model.lm_head.weight",)))
    elif "lm_head.weight" in headers:
        refs.append(TensorRef("output.weight", ("lm_head.weight",)))
    return refs


def _first_existing(headers: dict[str, HeaderTensor], names: Iterable[str]) -> str | None:
    for name in names:
        if name in headers:
            return name
    return None


def _require_existing(headers: dict[str, HeaderTensor], names: Iterable[str], desc: str) -> str:
    found = _first_existing(headers, names)
    if found is None:
        raise SystemExit(f"Missing required safetensors tensor for {desc}: tried {list(names)}")
    return found


def _language_prefix(headers: dict[str, HeaderTensor]) -> str:
    if "model.embed_tokens.weight" in headers:
        return "model."
    if "model.language_model.embed_tokens.weight" in headers:
        return "model.language_model."
    return "model."


def _maybe_tensor_ref(
    headers: dict[str, HeaderTensor],
    ck_name: str,
    source_names: tuple[str, ...],
    *,
    dtype: str | None = None,
) -> TensorRef | None:
    found = _first_existing(headers, source_names)
    if found is None:
        return None
    return TensorRef(ck_name, (found,), dtype=dtype)


def _llama_family_text_refs(config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    num_layers = int(config.get("num_layers") or config.get("num_hidden_layers") or 0)
    embed_dim = int(config.get("embed_dim") or config.get("hidden_size") or 0)
    intermediate = int(config.get("intermediate_size") or 0)
    num_heads = int(config.get("num_heads") or config.get("num_attention_heads") or 0)
    num_kv_heads = int(config.get("num_kv_heads") or config.get("num_key_value_heads") or num_heads or 0)
    head_dim = int(config.get("head_dim") or (embed_dim // max(1, num_heads)))
    if num_layers <= 0 or embed_dim <= 0 or intermediate <= 0 or num_heads <= 0:
        raise SystemExit("Llama-family config missing num_layers/embed_dim/intermediate_size/num_heads")

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    prefix = _language_prefix(headers)
    refs: list[TensorRef] = [
        TensorRef("token_emb", (_require_existing(headers, (f"{prefix}embed_tokens.weight", "model.embed_tokens.weight"), "token_emb"),)),
    ]

    for layer in range(num_layers):
        layer_prefix = f"{prefix}layers.{layer}"
        refs.extend([
            TensorRef(f"layer.{layer}.ln1_gamma", (_require_existing(headers, (f"{layer_prefix}.input_layernorm.weight",), f"layer {layer} input norm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.ln2_gamma", (_require_existing(headers, (f"{layer_prefix}.post_attention_layernorm.weight", f"{layer_prefix}.pre_feedforward_layernorm.weight"), f"layer {layer} ffn norm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.wq", (_require_existing(headers, (f"{layer_prefix}.self_attn.q_proj.weight",), f"layer {layer} q_proj"),)),
            _maybe_tensor_ref(headers, f"layer.{layer}.bq", (f"{layer_prefix}.self_attn.q_proj.bias",), dtype="fp32")
            or TensorRef(f"layer.{layer}.bq", (), dtype="fp32", synth="zeros_fp32", shape=(q_dim,)),
        ])
        q_norm = _maybe_tensor_ref(headers, f"layer.{layer}.q_norm", (f"{layer_prefix}.self_attn.q_norm.weight",), dtype="fp32")
        if q_norm is not None:
            refs.append(q_norm)
        refs.extend([
            TensorRef(f"layer.{layer}.wk", (_require_existing(headers, (f"{layer_prefix}.self_attn.k_proj.weight",), f"layer {layer} k_proj"),)),
            _maybe_tensor_ref(headers, f"layer.{layer}.bk", (f"{layer_prefix}.self_attn.k_proj.bias",), dtype="fp32")
            or TensorRef(f"layer.{layer}.bk", (), dtype="fp32", synth="zeros_fp32", shape=(kv_dim,)),
        ])
        k_norm = _maybe_tensor_ref(headers, f"layer.{layer}.k_norm", (f"{layer_prefix}.self_attn.k_norm.weight",), dtype="fp32")
        if k_norm is not None:
            refs.append(k_norm)
        refs.extend([
            TensorRef(f"layer.{layer}.wv", (_require_existing(headers, (f"{layer_prefix}.self_attn.v_proj.weight",), f"layer {layer} v_proj"),)),
            _maybe_tensor_ref(headers, f"layer.{layer}.bv", (f"{layer_prefix}.self_attn.v_proj.bias",), dtype="fp32")
            or TensorRef(f"layer.{layer}.bv", (), dtype="fp32", synth="zeros_fp32", shape=(kv_dim,)),
            TensorRef(f"layer.{layer}.wo", (_require_existing(headers, (f"{layer_prefix}.self_attn.o_proj.weight",), f"layer {layer} o_proj"),)),
            _maybe_tensor_ref(headers, f"layer.{layer}.bo", (f"{layer_prefix}.self_attn.o_proj.bias",), dtype="fp32")
            or TensorRef(f"layer.{layer}.bo", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
            TensorRef(
                f"layer.{layer}.w1",
                (
                    _require_existing(headers, (f"{layer_prefix}.mlp.gate_proj.weight",), f"layer {layer} gate_proj"),
                    _require_existing(headers, (f"{layer_prefix}.mlp.up_proj.weight",), f"layer {layer} up_proj"),
                ),
            ),
            TensorRef(f"layer.{layer}.b1", (), dtype="fp32", synth="zeros_fp32", shape=(2 * intermediate,)),
            TensorRef(f"layer.{layer}.w2", (_require_existing(headers, (f"{layer_prefix}.mlp.down_proj.weight",), f"layer {layer} down_proj"),)),
            TensorRef(f"layer.{layer}.b2", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
        ])

    refs.extend([
        TensorRef("final_ln_weight", (_require_existing(headers, (f"{prefix}norm.weight", "model.norm.weight"), "final norm"),), dtype="fp32"),
        TensorRef("final_ln_bias", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
    ])
    lm_head = _first_existing(headers, ("lm_head.weight", f"{prefix}lm_head.weight", "model.lm_head.weight"))
    if lm_head is not None:
        refs.append(TensorRef("output.weight", (lm_head,)))
    return refs


def _infer_arch(hf: dict[str, Any]) -> str:
    text = hf.get("text_config") if isinstance(hf.get("text_config"), dict) else hf
    mt = str(text.get("model_type") or hf.get("model_type") or "").lower()
    if "gemma" in mt and "4" in mt:
        return "gemma4"
    if "gemma" in mt and "3" in mt:
        return "gemma3"
    if "qwen3" in mt:
        return "qwen3"
    if "qwen2" in mt or mt == "qwen":
        return "qwen2"
    if "llama" in mt:
        return "llama"
    return mt


def _refs_for_arch(arch: str, config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    if arch == "gemma4":
        return _gemma4_text_refs(config, headers)
    if arch in {"llama", "qwen2", "qwen3", "gemma3"}:
        return _llama_family_text_refs(config, headers)
    raise SystemExit(f"Unsupported safetensors arch for v8 importer: {arch}")


def _build_config(model_dir: Path, arch: str, config_template: Path | None) -> dict[str, Any]:
    hf = _hf_config(model_dir)
    base = _load_runtime_config_template(config_template)
    text = hf.get("text_config") if isinstance(hf.get("text_config"), dict) else hf
    cfg = dict(base)
    cfg.setdefault("model", arch)
    cfg.setdefault("model_type", arch)
    cfg.setdefault("num_layers", text.get("num_hidden_layers"))
    cfg.setdefault("num_hidden_layers", text.get("num_hidden_layers"))
    cfg.setdefault("embed_dim", text.get("hidden_size"))
    cfg.setdefault("hidden_size", text.get("hidden_size"))
    cfg.setdefault("intermediate_size", text.get("intermediate_size"))
    cfg.setdefault("num_heads", text.get("num_attention_heads"))
    cfg.setdefault("num_attention_heads", text.get("num_attention_heads"))
    cfg.setdefault("num_kv_heads", text.get("num_key_value_heads"))
    cfg.setdefault("num_key_value_heads", text.get("num_key_value_heads"))
    cfg.setdefault("head_dim", text.get("head_dim") or (int(text.get("hidden_size", 0)) // max(1, int(text.get("num_attention_heads", 1)))))
    cfg.setdefault("vocab_size", text.get("vocab_size"))
    cfg.setdefault("context_length", text.get("max_position_embeddings") or text.get("sliding_window"))
    rope_params = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), dict) else {}
    full_rope = rope_params.get("full_attention") if isinstance(rope_params.get("full_attention"), dict) else {}
    sliding_rope = rope_params.get("sliding_attention") if isinstance(rope_params.get("sliding_attention"), dict) else {}
    cfg.setdefault("rope_theta", full_rope.get("rope_theta", text.get("rope_theta", 1000000.0)))
    cfg.setdefault("rope_theta_swa", sliding_rope.get("rope_theta", 10000.0))
    cfg.setdefault("rope_layout", "split")
    cfg.setdefault("rope_param_mode", "per_layer_direct")
    cfg.setdefault("rms_eps", text.get("rms_norm_eps", text.get("rms_norm_epsilon", 1e-6)))
    cfg.setdefault("rms_norm_eps", cfg.get("rms_eps"))
    cfg.setdefault("tie_word_embeddings", bool(text.get("tie_word_embeddings", True)))
    cfg = _inject_runtime_config_defaults(cfg, arch)
    if arch == "gemma4" and "layer_kinds" not in cfg:
        raise SystemExit("Gemma4 safetensors conversion currently requires --config-template with explicit layer_kinds/shared KV policy")
    return cfg


def _entry_size_from_header(ref: TensorRef, headers: dict[str, HeaderTensor], dtype_policy: str) -> tuple[str, int, list[int]]:
    if ref.synth:
        data, dt, shape = _synth_bytes(ref.synth, ref.shape or (), dtype_policy)
        return dt, len(data), shape
    total = 0
    out_dtype: str | None = ref.dtype
    out_shape: list[int] = []
    for src in ref.source_names:
        if src not in headers:
            raise KeyError(src)
        h = headers[src]
        shape = h.shape
        if ref.dtype:
            dtype = ref.dtype
            elem = {"fp32": 4, "bf16": 2, "fp16": 2}[dtype]
        else:
            dtype, elem = _header_dtype_to_ck(h.dtype)
        total += int(np.prod(shape, dtype=np.int64)) * elem
        out_dtype = out_dtype or dtype
        out_shape = shape if not out_shape else out_shape
    return out_dtype or "fp32", total, out_shape


def _write_ref(w: HashingWriter, model_dir: Path, headers: dict[str, HeaderTensor], ref: TensorRef, dtype_policy: str) -> tuple[str, int, list[int]]:
    if ref.synth:
        data, dt, shape = _synth_bytes(ref.synth, ref.shape or (), dtype_policy)
        w.write(data)
        return dt, len(data), shape
    total = 0
    out_dtype: str | None = ref.dtype
    out_shape: list[int] = []
    for src in ref.source_names:
        t = _load_tensor(model_dir, headers, src)
        policy = ref.dtype or dtype_policy
        data, dt, shape = _torch_to_bytes(t, policy)
        w.write(data)
        total += len(data)
        out_dtype = out_dtype or dt
        out_shape = shape if not out_shape else out_shape
    return out_dtype or "fp32", total, out_shape


def _copy_tokenizer_sidecars(model_dir: Path, out_dir: Path) -> None:
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        src = model_dir / name
        if src.exists():
            dst = out_dir / name
            if src.resolve() != dst.resolve():
                dst.write_bytes(src.read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert HF safetensors language weights to v8 BUMPWGT5")
    ap.add_argument("--checkpoint", required=True, type=Path, help="HF safetensors model directory")
    ap.add_argument("--output", required=True, type=Path, help="output weights.bump")
    ap.add_argument("--config-out", required=True, type=Path)
    ap.add_argument("--manifest-out", required=True, type=Path)
    ap.add_argument("--arch", default="auto", choices=["auto", "gemma4", "gemma3", "llama", "qwen2", "qwen3"])
    ap.add_argument("--config-template", type=Path, help="existing v8 config/manifest to reuse explicit runtime policy")
    ap.add_argument("--dtype", default="preserve", choices=["preserve", "bf16", "fp32"])
    ap.add_argument("--dry-run", action="store_true", help="validate mapping and write JSON reports only; do not write BUMP")
    args = ap.parse_args()

    model_dir = args.checkpoint.resolve()
    headers = _load_safetensors_headers(model_dir)
    hf = _hf_config(model_dir)
    arch = args.arch
    if arch == "auto":
        arch = _infer_arch(hf)

    config = _build_config(model_dir, arch, args.config_template)
    refs = _refs_for_arch(arch, config, headers)
    missing: list[str] = []
    entries_preview: list[dict[str, Any]] = []
    dtype_table: list[int] = []
    offset = DATA_START + 4 + len(refs)
    for ref in refs:
        for src in ref.source_names:
            if src not in headers:
                missing.append(src)
        if missing:
            continue
        dt, size, shape = _entry_size_from_header(ref, headers, args.dtype)
        dtype_table.append(_dtype_code(dt))
        entries_preview.append({
            "name": ref.ck_name,
            "dtype": dt,
            "file_offset": offset,
            "size": size,
            "source_name": "+".join(ref.source_names) if ref.source_names else f"synthetic:{ref.synth}",
            "shape": shape,
        })
        offset += size
    if missing:
        uniq = sorted(set(missing))
        raise SystemExit("Missing required safetensors tensors:\n  " + "\n  ".join(uniq[:80]))

    template = apply_model_contract_overrides(
        load_template_for_arch(arch),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", True)),
        has_untied_output_weight=any(e["name"] == "output.weight" for e in entries_preview),
    )
    quant_summary: dict[str, Any] = {"source": "safetensors", "dtype_policy": args.dtype}
    for e in entries_preview:
        if e["name"].startswith("layer."):
            layer_key = ".".join(e["name"].split(".")[:2])
            quant_summary.setdefault(layer_key, {})[e["name"].split(".")[-1]] = e["dtype"]
        else:
            quant_summary[e["name"]] = e["dtype"]

    manifest = {
        "version": 5,
        "model": arch,
        "source_format": "safetensors",
        "bump_layout": {
            "header_size": HEADER_SIZE,
            "ext_metadata_size": EXT_METADATA_SIZE,
            "data_start": DATA_START,
        },
        "config": config,
        "template": template,
        "quant_summary": quant_summary,
        "num_layers": int(config.get("num_layers", 0)),
        "embed_dim": int(config.get("embed_dim", 0)),
        "num_heads": int(config.get("num_heads", 0)),
        "num_kv_heads": int(config.get("num_kv_heads", 0)),
        "head_dim": int(config.get("head_dim", 0)),
        "intermediate_size": int(config.get("intermediate_size", 0)),
        "vocab_size": int(config.get("vocab_size", 0)),
        "context_length": int(config.get("context_length", 0) or 0),
        "has_attention_biases": False,
        "has_qk_norm": True,
        "entries": entries_preview,
    }

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _copy_tokenizer_sidecars(model_dir, args.output.parent)

    print(f"[safetensors->bump] arch={arch} tensors={len(refs)} entries={len(entries_preview)} dry_run={args.dry_run}")
    print(f"[safetensors->bump] estimated weights payload={(offset - (DATA_START + 4 + len(refs))) / 1024 / 1024:.1f} MiB")
    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w+b") as f:
        f.write(b"\x00" * HEADER_SIZE)
        f.write(b"\x00" * EXT_METADATA_SIZE)
        w = HashingWriter(f)
        w.write(struct.pack("<I", len(dtype_table)))
        w.write(bytes(dtype_table))
        entries: list[dict[str, Any]] = []
        current_offset = DATA_START + 4 + len(dtype_table)
        for ref in refs:
            start = current_offset
            dt, size, shape = _write_ref(w, model_dir, headers, ref, args.dtype)
            current_offset += size
            entries.append({
                "name": ref.ck_name,
                "dtype": dt,
                "file_offset": start,
                "size": size,
                "source_name": "+".join(ref.source_names) if ref.source_names else f"synthetic:{ref.synth}",
                "shape": shape,
            })
        checksum = w.digest()
        manifest["entries"] = entries
        args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_hash = calculate_manifest_hash(manifest)
        metadata = build_bumpv5_metadata(template, config, quant_summary, manifest_hash, "convert_safetensors_to_bump_v8.py")
        metadata["template_hash"] = calculate_template_hash(template)
        metadata_bytes = _canonical_json_bytes(metadata)
        meta_hash = calculate_metadata_hash(metadata)

        f.flush()
        f.seek(0)
        f.write(b"BUMPWGT5")
        f.write(struct.pack("<I", BUMP_VERSION_V5))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", int(config.get("num_layers", 0))))
        f.write(struct.pack("<I", int(config.get("vocab_size", 0))))
        f.write(struct.pack("<I", int(config.get("embed_dim", 0))))
        f.write(struct.pack("<I", int(config.get("intermediate_size", 0))))
        f.write(struct.pack("<I", int(config.get("context_length", 0) or 0)))
        f.write(struct.pack("<I", int(config.get("num_heads", 0))))
        f.write(struct.pack("<I", int(config.get("num_kv_heads", 0))))
        f.write(struct.pack("<I", int(config.get("head_dim", 0))))
        for value in (
            int(config.get("embed_dim", 0)),
            int(config.get("head_dim", 0)),
            int(config.get("intermediate_size", 0)),
            int(config.get("context_length", 0) or 0),
        ):
            f.write(struct.pack("<Q", value))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(checksum)
        f.seek(0, os.SEEK_END)
        f.write(metadata_bytes)
        write_bumpv5_footer(f, len(metadata_bytes), meta_hash)
    print(f"[safetensors->bump] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
