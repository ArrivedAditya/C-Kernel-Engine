#!/usr/bin/env python3
from __future__ import annotations

"""Convert HF safetensors language weights into v8 BUMPWGT5 artifacts.

This is a BUMP-first importer.  GGUF and safetensors are source formats; CK
runtime consumes weights.bump + sidecars.  Supported targets map source tensor
names into CK-internal manifest names so existing v8 IR/codegen can consume
safetensors and GGUF through the same BUMP runtime contract.
"""

import argparse
import hashlib
import json
import os
import shutil
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
    build_qwen35_execution_plan,
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


def _qwen35_text_refs(config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    num_layers = int(config.get("num_layers") or config.get("num_hidden_layers") or text.get("num_hidden_layers") or 0)
    embed_dim = int(config.get("embed_dim") or config.get("hidden_size") or text.get("hidden_size") or 0)
    intermediate = int(config.get("intermediate_size") or text.get("intermediate_size") or 0)
    num_heads = int(config.get("num_heads") or text.get("num_attention_heads") or 0)
    num_kv_heads = int(config.get("num_kv_heads") or text.get("num_key_value_heads") or 0)
    head_dim = int(config.get("head_dim") or text.get("head_dim") or (embed_dim // max(1, num_heads)))
    if num_layers <= 0 or embed_dim <= 0 or intermediate <= 0 or num_heads <= 0:
        raise SystemExit("Qwen3.5 config missing num_layers/embed_dim/intermediate_size/num_heads")

    prefix = "model.language_model."
    layer_types = list(config.get("layer_types") or text.get("layer_types") or [])
    if layer_types and len(layer_types) != num_layers:
        raise SystemExit(f"Qwen3.5 layer_types length {len(layer_types)} != num_layers {num_layers}")

    refs: list[TensorRef] = [
        TensorRef("token_emb", (_require_existing(headers, (f"{prefix}embed_tokens.weight",), "token_emb"),)),
    ]

    for layer in range(num_layers):
        layer_prefix = f"{prefix}layers.{layer}"
        layer_type = str(layer_types[layer] if layer_types else ("full_attention" if (layer + 1) % 4 == 0 else "linear_attention"))
        refs.extend([
            TensorRef(f"layer.{layer}.attn_norm", (_require_existing(headers, (f"{layer_prefix}.input_layernorm.weight",), f"layer {layer} input norm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.post_attention_norm", (_require_existing(headers, (f"{layer_prefix}.post_attention_layernorm.weight",), f"layer {layer} post attention norm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.ffn_gate", (_require_existing(headers, (f"{layer_prefix}.mlp.gate_proj.weight",), f"layer {layer} gate_proj"),)),
            TensorRef(f"layer.{layer}.ffn_up", (_require_existing(headers, (f"{layer_prefix}.mlp.up_proj.weight",), f"layer {layer} up_proj"),)),
            TensorRef(f"layer.{layer}.ffn_down", (_require_existing(headers, (f"{layer_prefix}.mlp.down_proj.weight",), f"layer {layer} down_proj"),)),
        ])
        if layer_type in {"linear_attention", "recurrent"}:
            refs.extend([
                TensorRef(f"layer.{layer}.attn_qkv", (_require_existing(headers, (f"{layer_prefix}.linear_attn.in_proj_qkv.weight",), f"layer {layer} recurrent qkv"),)),
                TensorRef(f"layer.{layer}.attn_gate", (_require_existing(headers, (f"{layer_prefix}.linear_attn.in_proj_z.weight",), f"layer {layer} recurrent gate"),)),
                TensorRef(f"layer.{layer}.ssm_alpha", (_require_existing(headers, (f"{layer_prefix}.linear_attn.in_proj_a.weight",), f"layer {layer} recurrent alpha"),)),
                TensorRef(f"layer.{layer}.ssm_beta", (_require_existing(headers, (f"{layer_prefix}.linear_attn.in_proj_b.weight",), f"layer {layer} recurrent beta"),)),
                TensorRef(f"layer.{layer}.ssm_conv1d", (_require_existing(headers, (f"{layer_prefix}.linear_attn.conv1d.weight",), f"layer {layer} recurrent conv1d"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.ssm_dt_bias", (_require_existing(headers, (f"{layer_prefix}.linear_attn.dt_bias",), f"layer {layer} recurrent dt bias"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.ssm_a", (_require_existing(headers, (f"{layer_prefix}.linear_attn.A_log",), f"layer {layer} recurrent A_log"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.ssm_norm", (_require_existing(headers, (f"{layer_prefix}.linear_attn.norm.weight",), f"layer {layer} recurrent norm"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.ssm_out", (_require_existing(headers, (f"{layer_prefix}.linear_attn.out_proj.weight",), f"layer {layer} recurrent out"),)),
            ])
        elif layer_type == "full_attention":
            refs.extend([
                TensorRef(f"layer.{layer}.attn_q_gate", (_require_existing(headers, (f"{layer_prefix}.self_attn.q_proj.weight",), f"layer {layer} q_proj"),)),
                TensorRef(f"layer.{layer}.attn_k", (_require_existing(headers, (f"{layer_prefix}.self_attn.k_proj.weight",), f"layer {layer} k_proj"),)),
                TensorRef(f"layer.{layer}.attn_v", (_require_existing(headers, (f"{layer_prefix}.self_attn.v_proj.weight",), f"layer {layer} v_proj"),)),
                TensorRef(f"layer.{layer}.attn_output", (_require_existing(headers, (f"{layer_prefix}.self_attn.o_proj.weight",), f"layer {layer} o_proj"),)),
                TensorRef(f"layer.{layer}.attn_q_norm", (_require_existing(headers, (f"{layer_prefix}.self_attn.q_norm.weight",), f"layer {layer} q_norm"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.attn_k_norm", (_require_existing(headers, (f"{layer_prefix}.self_attn.k_norm.weight",), f"layer {layer} k_norm"),), dtype="fp32"),
            ])
        else:
            raise SystemExit(f"Unsupported Qwen3.5 layer_type at layer {layer}: {layer_type}")

    refs.append(TensorRef("final_ln_weight", (_require_existing(headers, (f"{prefix}norm.weight",), "final norm"),), dtype="fp32"))
    lm_head = _first_existing(headers, ("lm_head.weight", f"{prefix}lm_head.weight"))
    if lm_head is not None:
        refs.append(TensorRef("output.weight", (lm_head,)))
    return refs


def _parse_nemotron_h_pattern(pattern: str, num_layers: int) -> list[str]:
    mapping = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}
    chars = [ch for ch in str(pattern or "").strip() if not ch.isspace()]
    if not chars:
        return ["mamba"] * num_layers
    if len(chars) != num_layers:
        raise SystemExit(f"Nemotron-H hybrid_override_pattern length {len(chars)} != num_layers {num_layers}")
    try:
        return [mapping[ch] for ch in chars]
    except KeyError as exc:
        raise SystemExit(f"Unsupported Nemotron-H layer pattern character: {exc.args[0]!r}") from exc




def _nemotron_dt_limit(config: dict[str, Any]) -> tuple[float, float]:
    """Return CK runtime dt clamp bounds for Nemotron-H.

    Nemotron-H uses time_step_min/max for initialization only. Runtime forward
    clamps softplus(dt + bias) with time_step_limit; the common (0, inf) limit
    is represented as (0, 0) so the CK kernel disables clamping.
    """
    limit = config.get("time_step_limit")
    if isinstance(limit, (list, tuple)) and len(limit) >= 2:
        lo = float(limit[0] or 0.0)
        try:
            hi = float(limit[1])
        except Exception:
            hi = float("inf")
        if hi == float("inf"):
            return 0.0, 0.0
        return lo, hi
    return 0.0, 0.0

def _nemotron_h_text_refs(config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    num_layers = int(text.get("num_hidden_layers") or config.get("num_layers") or 0)
    hidden = int(text.get("hidden_size") or config.get("hidden_size") or 0)
    intermediate = int(text.get("intermediate_size") or config.get("intermediate_size") or 0)
    moe_intermediate = int(text.get("moe_intermediate_size") or intermediate)
    shared_intermediate = int(text.get("moe_shared_expert_intermediate_size") or moe_intermediate)
    n_experts = int(text.get("n_routed_experts") or 0)
    layer_kinds = list(config.get("layer_kinds") or _parse_nemotron_h_pattern(str(text.get("hybrid_override_pattern") or ""), num_layers))
    if num_layers <= 0 or hidden <= 0 or intermediate <= 0:
        raise SystemExit("Nemotron-H config missing num_hidden_layers/hidden_size/intermediate_size")
    if len(layer_kinds) != num_layers:
        raise SystemExit(f"Nemotron-H layer_kinds length {len(layer_kinds)} != num_layers {num_layers}")

    refs: list[TensorRef] = [
        TensorRef("token_emb", (_require_existing(headers, ("backbone.embeddings.weight", "model.embed_tokens.weight"), "token_emb"),)),
    ]

    for layer, kind in enumerate(layer_kinds):
        pfx = f"backbone.layers.{layer}"
        refs.append(TensorRef(f"layer.{layer}.block_norm", (_require_existing(headers, (f"{pfx}.norm.weight",), f"layer {layer} block norm"),), dtype="fp32"))
        mp = f"{pfx}.mixer"
        if kind == "mamba":
            refs.extend([
                TensorRef(f"layer.{layer}.mamba_in_proj", (_require_existing(headers, (f"{mp}.in_proj.weight",), f"layer {layer} mamba in_proj"),)),
                TensorRef(f"layer.{layer}.mamba_conv1d", (_require_existing(headers, (f"{mp}.conv1d.weight",), f"layer {layer} mamba conv1d"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_conv1d_bias", (_require_existing(headers, (f"{mp}.conv1d.bias",), f"layer {layer} mamba conv1d bias"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_dt_bias", (_require_existing(headers, (f"{mp}.dt_bias",), f"layer {layer} mamba dt_bias"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_a", (_require_existing(headers, (f"{mp}.A_log",), f"layer {layer} mamba A_log"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_d", (_require_existing(headers, (f"{mp}.D",), f"layer {layer} mamba D"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_norm", (_require_existing(headers, (f"{mp}.norm.weight",), f"layer {layer} mamba norm"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_out_proj", (_require_existing(headers, (f"{mp}.out_proj.weight",), f"layer {layer} mamba out_proj"),)),
            ])
        elif kind == "attention":
            refs.extend([
                TensorRef(f"layer.{layer}.attn_q", (_require_existing(headers, (f"{mp}.q_proj.weight",), f"layer {layer} q_proj"),)),
                TensorRef(f"layer.{layer}.attn_k", (_require_existing(headers, (f"{mp}.k_proj.weight",), f"layer {layer} k_proj"),)),
                TensorRef(f"layer.{layer}.attn_v", (_require_existing(headers, (f"{mp}.v_proj.weight",), f"layer {layer} v_proj"),)),
                TensorRef(f"layer.{layer}.attn_o", (_require_existing(headers, (f"{mp}.o_proj.weight",), f"layer {layer} o_proj"),)),
            ])
        elif kind == "mlp":
            refs.extend([
                TensorRef(f"layer.{layer}.mlp_up", (_require_existing(headers, (f"{mp}.up_proj.weight",), f"layer {layer} mlp up_proj"),)),
                TensorRef(f"layer.{layer}.mlp_down", (_require_existing(headers, (f"{mp}.down_proj.weight",), f"layer {layer} mlp down_proj"),)),
            ])
        elif kind == "moe":
            if n_experts <= 0:
                raise SystemExit("Nemotron-H MoE layer requires n_routed_experts")
            refs.extend([
                TensorRef(f"layer.{layer}.moe_router", (_require_existing(headers, (f"{mp}.gate.weight",), f"layer {layer} moe router"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.moe_router_bias", (_require_existing(headers, (f"{mp}.gate.e_score_correction_bias",), f"layer {layer} moe router correction bias"),), dtype="fp32"),
            ])
            for expert in range(n_experts):
                refs.extend([
                    TensorRef(f"layer.{layer}.moe_expert.{expert}.up", (_require_existing(headers, (f"{mp}.experts.{expert}.up_proj.weight",), f"layer {layer} expert {expert} up_proj"),)),
                    TensorRef(f"layer.{layer}.moe_expert.{expert}.down", (_require_existing(headers, (f"{mp}.experts.{expert}.down_proj.weight",), f"layer {layer} expert {expert} down_proj"),)),
                ])
            refs.extend([
                TensorRef(f"layer.{layer}.moe_shared_up", (_require_existing(headers, (f"{mp}.shared_experts.up_proj.weight",), f"layer {layer} shared expert up_proj"),)),
                TensorRef(f"layer.{layer}.moe_shared_down", (_require_existing(headers, (f"{mp}.shared_experts.down_proj.weight",), f"layer {layer} shared expert down_proj"),)),
            ])
        else:
            raise SystemExit(f"Unsupported Nemotron-H layer kind at {layer}: {kind}")

    refs.append(TensorRef("final_ln_weight", (_require_existing(headers, ("backbone.norm_f.weight", "model.norm.weight"), "final norm"),), dtype="fp32"))
    refs.append(TensorRef("final_ln_bias", (), dtype="fp32", synth="zeros_fp32", shape=(hidden,)))
    lm_head = _first_existing(headers, ("lm_head.weight", "backbone.lm_head.weight"))
    if lm_head is not None:
        refs.append(TensorRef("output.weight", (lm_head,)))
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
    if "qwen3_5" in mt or "qwen3.5" in mt or "qwen35" in mt:
        return "qwen35"
    if "qwen3" in mt:
        return "qwen3"
    if "qwen2" in mt or mt == "qwen":
        return "qwen2"
    if "llama" in mt:
        return "llama"
    if "nemotron_h" in mt or "nemotron-h" in mt:
        return "nemotron_h"
    return mt


def _refs_for_arch(arch: str, config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    if arch == "gemma4":
        return _gemma4_text_refs(config, headers)
    if arch == "qwen35":
        return _qwen35_text_refs(config, headers)
    if arch == "nemotron_h":
        return _nemotron_h_text_refs(config, headers)
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
    rope_parameters = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), dict) else {}
    cfg.setdefault("rope_theta", full_rope.get("rope_theta", rope_parameters.get("rope_theta", text.get("rope_theta", 1000000.0))))
    cfg.setdefault("rope_theta_swa", sliding_rope.get("rope_theta", 10000.0))
    mrope_sections_raw = rope_parameters.get("mrope_section") or []
    rope_interleaved = bool(rope_parameters.get("mrope_interleaved", False))
    if mrope_sections_raw:
        cfg.setdefault("rope_layout", "multi_section_1d")
    else:
        cfg.setdefault("rope_layout", "pairwise" if rope_interleaved else "split")
    cfg.setdefault("rope_param_mode", "per_layer_direct")
    cfg.setdefault("rms_eps", text.get("rms_norm_eps", text.get("rms_norm_epsilon", 1e-6)))
    cfg.setdefault("rms_norm_eps", cfg.get("rms_eps"))
    cfg.setdefault("tie_word_embeddings", bool(text.get("tie_word_embeddings", True)))
    if arch == "nemotron_h":
        layer_kinds = _parse_nemotron_h_pattern(str(text.get("hybrid_override_pattern") or ""), int(cfg.get("num_layers") or 0))
        mamba_num_heads = int(text.get("mamba_num_heads") or 0)
        mamba_head_dim = int(text.get("mamba_head_dim") or 0)
        ssm_state_size = int(text.get("ssm_state_size") or 0)
        ssm_group_count = int(text.get("n_groups") or 0)
        ssm_conv_kernel = int(text.get("conv_kernel") or 0)
        ssm_inner_size = mamba_num_heads * mamba_head_dim
        mamba_conv_dim = ssm_inner_size + 2 * ssm_group_count * ssm_state_size
        mamba_projection_size = ssm_inner_size + mamba_conv_dim + mamba_num_heads
        cfg.update({
            "model": "nemotron_h",
            "arch": "nemotron_h",
            "model_type": "nemotron_h",
            "layer_kinds": layer_kinds,
            "hybrid_block_pattern": layer_kinds[:],
            "layer_state_policy": ["mamba2" if k == "mamba" else "none" for k in layer_kinds],
            "layer_attention_policy": ["full_attention" if k == "attention" else "none" for k in layer_kinds],
            "layer_recurrent_policy": ["mamba2" if k == "mamba" else "none" for k in layer_kinds],
            "layer_moe_policy": ["routed_relu2" if k == "moe" else "none" for k in layer_kinds],
            "layer_mlp_policy": ["relu2" if k == "mlp" else "none" for k in layer_kinds],
            "layer_kv_policy": ["attention_kv_cache" if k == "attention" else "none" for k in layer_kinds],
            "intermediate_dim": int(text.get("intermediate_size") or 0),
            "mamba_num_heads": mamba_num_heads,
            "mamba_head_dim": mamba_head_dim,
            "ssm_state_size": ssm_state_size,
            "ssm_conv_kernel": ssm_conv_kernel,
            "ssm_group_count": ssm_group_count,
            "mamba_norm_group_size": int(ssm_inner_size // ssm_group_count) if ssm_group_count else int(mamba_head_dim),
            "chunk_size": int(text.get("chunk_size") or 0),
            "mamba_dt_min": _nemotron_dt_limit(text)[0],
            "mamba_dt_max": _nemotron_dt_limit(text)[1],
            "ssm_inner_size": ssm_inner_size,
            "ssm_conv_channels": mamba_conv_dim,
            "ssm_conv_history": max(ssm_conv_kernel, 0),
            "mamba_conv_dim": mamba_conv_dim,
            "mamba_projection_size": mamba_projection_size,
            "moe_intermediate_size": int(text.get("moe_intermediate_size") or 0),
            "moe_shared_expert_intermediate_size": int(text.get("moe_shared_expert_intermediate_size") or 0),
            "n_routed_experts": int(text.get("n_routed_experts") or 0),
            "num_experts_per_tok": int(text.get("num_experts_per_tok") or 0),
            "n_group": int(text.get("n_group") or text.get("n_groups") or 0),
            "topk_group": int(text.get("topk_group") or 0),
            "norm_topk_prob": bool(text.get("norm_topk_prob", True)),
            "routed_scaling_factor": float(text.get("routed_scaling_factor") or 1.0),
            "rope_layout": "split",
            "rope_theta": float(text.get("rope_theta") or 10000.0),
            "rope_freq_base": float(text.get("rope_theta") or 10000.0),
            "use_rope_freq_factors": 0,
            "has_attention_biases": bool(text.get("attention_bias", False)),
            "has_qk_norm": False,
            "prefill_policy": "batched",
        })

    if arch == "qwen35":
        headers = _load_safetensors_headers(model_dir)
        q_gate_proj_dim = 0
        attn_out_dim = 0
        recurrent_q_dim = int(text.get("linear_num_key_heads", 0) or 0) * int(text.get("linear_key_head_dim", 0) or 0)
        recurrent_k_dim = recurrent_q_dim
        recurrent_v_dim = int(text.get("linear_num_value_heads", 0) or 0) * int(text.get("linear_value_head_dim", 0) or 0)
        ssm_time_step_rank = 0
        for name, h in headers.items():
            if name.endswith(".self_attn.q_proj.weight") and len(h.shape) >= 2:
                q_gate_proj_dim = int(h.shape[0])
                if q_gate_proj_dim % 2 == 0:
                    attn_out_dim = q_gate_proj_dim // 2
            elif name.endswith(".linear_attn.in_proj_qkv.weight") and len(h.shape) >= 2:
                total = int(h.shape[0])
                if recurrent_q_dim <= 0 or recurrent_k_dim <= 0 or recurrent_v_dim <= 0:
                    recurrent_q_dim = total // 3
                    recurrent_k_dim = total // 3
                    recurrent_v_dim = total - recurrent_q_dim - recurrent_k_dim
            elif name.endswith(".linear_attn.in_proj_a.weight") and len(h.shape) >= 2:
                ssm_time_step_rank = max(ssm_time_step_rank, int(h.shape[0]))
        if attn_out_dim <= 0:
            attn_out_dim = int(text.get("num_attention_heads", cfg.get("num_heads", 0))) * int(text.get("head_dim", cfg.get("head_dim", 0)))
        if q_gate_proj_dim <= 0:
            q_gate_proj_dim = attn_out_dim * 2
        if ssm_time_step_rank <= 0:
            ssm_time_step_rank = int(text.get("linear_num_key_heads") or text.get("linear_num_value_heads") or 0)

        layer_types = [str(x) for x in (text.get("layer_types") or [])]
        layer_kinds = ["full_attention" if x == "full_attention" else "recurrent" for x in layer_types]
        if not layer_kinds:
            n_layers = int(cfg.get("num_layers") or 0)
            interval = int(text.get("full_attention_interval") or 4)
            layer_kinds = ["full_attention" if (i + 1) % interval == 0 else "recurrent" for i in range(n_layers)]
        execution_plan = build_qwen35_execution_plan(layer_kinds)
        cfg.update({
            "model": "qwen35",
            "arch": "qwen35",
            "model_type": "qwen35",
            "attn_out_dim": int(attn_out_dim),
            "attn_q_gate_proj_dim": int(q_gate_proj_dim),
            "intermediate_dim": int(cfg.get("intermediate_size") or 0),
            "max_seq_len": int(cfg.get("context_length") or 0),
            "rotary_dim": int(int(text.get("head_dim", cfg.get("head_dim", 0))) * float(rope_parameters.get("partial_rotary_factor", 1.0))),
            "has_qk_norm": any(kind == "full_attention" for kind in layer_kinds),
            "has_attention_biases": False,
            "layer_types": layer_types,
            "layer_kinds": layer_kinds,
            "hybrid_block_pattern": layer_kinds[:],
            "layer_execution_plan": execution_plan["layer_execution_plan"],
            "layer_state_policy": execution_plan["layer_state_policy"],
            "layer_attention_policy": execution_plan["layer_attention_policy"],
            "layer_recurrent_policy": execution_plan["layer_recurrent_policy"],
            "layer_kv_policy": execution_plan["layer_kv_policy"],
            "prefill_policy": "batched",
            "full_attention_interval": int(text.get("full_attention_interval") or 4),
            "ssm_conv_kernel": int(text.get("linear_conv_kernel_dim") or 4),
            "ssm_state_size": int(text.get("linear_key_head_dim") or max(1, recurrent_q_dim)),
            "ssm_group_count": int(text.get("linear_num_key_heads") or max(1, recurrent_q_dim // max(1, int(text.get("linear_key_head_dim") or recurrent_q_dim)))),
            "ssm_time_step_rank": int(ssm_time_step_rank),
            "ssm_inner_size": int(recurrent_v_dim),
            "q_dim": int(recurrent_q_dim),
            "k_dim": int(recurrent_k_dim),
            "v_dim": int(recurrent_v_dim),
            "gate_dim": int(ssm_time_step_rank),
            "ssm_conv_channels": int(recurrent_q_dim + recurrent_k_dim + recurrent_v_dim),
            "recurrent_num_heads": int(ssm_time_step_rank),
            "recurrent_head_dim": int(recurrent_v_dim // max(1, ssm_time_step_rank)) if ssm_time_step_rank else int(text.get("linear_key_head_dim") or 0),
            "sampler_defaults": {"repeat_penalty": 1.12, "repeat_last_n": 96, "no_repeat_ngram_size": 4},
        })
        mrope = list(rope_parameters.get("mrope_section") or [])
        if mrope:
            cfg["mrope_sections"] = [int(v) for v in mrope] + ([0] if len(mrope) == 3 else [])
            cfg["mrope_n_dims"] = int(sum(int(v) for v in mrope))
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


def _is_qwen35_shifted_norm_ref(ref: TensorRef) -> bool:
    if not ref.source_names:
        return False
    src = ref.source_names[0]
    if not src.endswith("norm.weight"):
        return False
    # llama.cpp shifts Qwen3.5 norm weights by +1, except the recurrent
    # linear_attn.norm.weight used inside the DeltaNet norm-gate.
    if src.endswith("linear_attn.norm.weight"):
        return False
    return ref.ck_name == "final_ln_weight" or ref.ck_name.endswith((
        ".attn_norm",
        ".post_attention_norm",
        ".attn_q_norm",
        ".attn_k_norm",
    ))


def _ref_transform(ref: TensorRef) -> str | None:
    if ref.ck_name.endswith(".ssm_a") or ref.ck_name.endswith(".mamba_a"):
        return "neg_exp_a_log"
    if _is_qwen35_shifted_norm_ref(ref):
        return "qwen35_norm_plus_one"
    return None


def _ignored_source_tensor(arch: str, name: str) -> str | None:
    if name.startswith("mtp.") or name.startswith("model.mtp."):
        return "mtp_decoder_block_not_in_main_pass"
    if arch == "qwen35" and (name.startswith("model.visual.") or name.startswith("visual.")):
        return "vision_tower_not_in_decoder_pass"
    if arch == "qwen35" and name.startswith("model.vision_model."):
        return "vision_tower_not_in_decoder_pass"
    return None


def _build_source_audit(arch: str, headers: dict[str, HeaderTensor], refs: list[TensorRef]) -> dict[str, Any]:
    consumed: dict[str, list[str]] = {}
    synthetic: list[str] = []
    transforms: list[dict[str, str]] = []
    for ref in refs:
        transform = _ref_transform(ref)
        if ref.source_names:
            for src in ref.source_names:
                consumed.setdefault(src, []).append(ref.ck_name)
                if transform:
                    transforms.append({"source": src, "target": ref.ck_name, "transform": transform})
        else:
            synthetic.append(ref.ck_name)
    ignored: list[dict[str, str]] = []
    unmapped: list[str] = []
    for name in sorted(headers):
        if name in consumed:
            continue
        reason = _ignored_source_tensor(arch, name)
        if reason:
            ignored.append({"source": name, "reason": reason})
        else:
            unmapped.append(name)
    return {
        "arch": arch,
        "source_tensor_count": len(headers),
        "consumed_source_count": len(consumed),
        "consumed_source_tensors": sorted(consumed),
        "source_to_targets": {name: sorted(targets) for name, targets in sorted(consumed.items())},
        "entry_count": len(refs),
        "synthetic_entries": synthetic,
        "ignored_source_tensors": ignored,
        "unmapped_source_tensors": unmapped,
        "transforms": transforms,
        "verdict": "fail" if unmapped else "pass",
    }


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
        transform = _ref_transform(ref)
        if transform == "neg_exp_a_log":
            import torch
            t = -torch.exp(t.to(dtype=torch.float32))
        elif transform == "qwen35_norm_plus_one":
            import torch
            t = t.to(dtype=torch.float32) + 1.0
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


def _resolve_output_path(output: Path | None, *, ram_output: bool, ram_dir: Path, checkpoint: Path) -> Path:
    if not ram_output:
        if output is None:
            raise SystemExit("--output is required unless --ram-output is used")
        return output

    ram_root = ram_dir.resolve()
    if output is not None and output.is_absolute() and str(output.resolve()).startswith(str(ram_root)):
        return output

    name = checkpoint.resolve().name or "model"
    base = ram_root / "ck-engine-v8" / name
    if output is None:
        return base / "weights.bump"
    return base / output.name




def _is_proc_fd_path(path: Path) -> bool:
    text = str(path)
    return text.startswith("/proc/self/fd/") or (text.startswith("/proc/") and "/fd/" in text)

def _check_output_capacity(path: Path, estimated_bytes: int, *, ram_output: bool) -> None:
    if _is_proc_fd_path(path):
        return
    probe = path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    reserve = max(256 * 1024 * 1024, int(estimated_bytes * 0.02))
    required = int(estimated_bytes) + reserve
    if usage.free < required:
        kind = "RAM/tmpfs" if ram_output else "filesystem"
        raise SystemExit(
            f"Not enough free space in {kind} for BUMP output: need about "
            f"{required / 1024 / 1024 / 1024:.2f} GiB including reserve, "
            f"free {usage.free / 1024 / 1024 / 1024:.2f} GiB at {path.parent}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert HF safetensors language weights to v8 BUMPWGT5")
    ap.add_argument("--checkpoint", required=True, type=Path, help="HF safetensors model directory")
    ap.add_argument("--output", type=Path, help="output weights.bump")
    ap.add_argument("--ram-output", action="store_true", help="write weights.bump under a RAM-backed tmpfs path instead of the checkpoint/output directory")
    ap.add_argument("--ram-dir", type=Path, default=Path("/dev/shm"), help="tmpfs directory for --ram-output; default: /dev/shm")
    ap.add_argument("--config-out", required=True, type=Path)
    ap.add_argument("--manifest-out", required=True, type=Path)
    ap.add_argument("--arch", default="auto", choices=["auto", "gemma4", "gemma3", "llama", "qwen2", "qwen3", "qwen35", "nemotron_h"])
    ap.add_argument("--config-template", type=Path, help="existing v8 config/manifest to reuse explicit runtime policy")
    ap.add_argument("--dtype", default="preserve", choices=["preserve", "bf16", "fp32"])
    ap.add_argument("--dry-run", action="store_true", help="validate mapping and write JSON reports only; do not write BUMP")
    ap.add_argument("--audit-out", type=Path, help="write safetensors source coverage audit JSON; default is manifest directory/conversion_audit.json")
    ap.add_argument("--allow-unmapped", action="store_true", help="allow non-ignored source tensors to remain unmapped")
    args = ap.parse_args()

    model_dir = args.checkpoint.resolve()
    args.output = _resolve_output_path(args.output, ram_output=bool(args.ram_output), ram_dir=args.ram_dir, checkpoint=model_dir)
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
        preview_entry = {
            "name": ref.ck_name,
            "dtype": dt,
            "file_offset": offset,
            "size": size,
            "source_name": "+".join(ref.source_names) if ref.source_names else f"synthetic:{ref.synth}",
            "shape": shape,
        }
        transform = _ref_transform(ref)
        if transform:
            preview_entry["transform"] = transform
        entries_preview.append(preview_entry)
        offset += size
    if missing:
        uniq = sorted(set(missing))
        raise SystemExit("Missing required safetensors tensors:\n  " + "\n  ".join(uniq[:80]))

    audit = _build_source_audit(arch, headers, refs)
    audit_out = args.audit_out or (args.manifest_out.parent / "conversion_audit.json")
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["unmapped_source_tensors"] and not args.allow_unmapped:
        sample = "\n  ".join(audit["unmapped_source_tensors"][:80])
        raise SystemExit(f"Unmapped safetensors source tensors; see {audit_out}:\n  {sample}")

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

    def _int_or_zero(value: Any) -> int:
        return int(value or 0)

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
        "num_layers": _int_or_zero(config.get("num_layers")),
        "embed_dim": _int_or_zero(config.get("embed_dim")),
        "num_heads": _int_or_zero(config.get("num_heads")),
        "num_kv_heads": _int_or_zero(config.get("num_kv_heads")),
        "head_dim": _int_or_zero(config.get("head_dim")),
        "intermediate_size": _int_or_zero(config.get("intermediate_size")),
        "vocab_size": _int_or_zero(config.get("vocab_size")),
        "context_length": _int_or_zero(config.get("context_length")),
        "has_attention_biases": False,
        "has_qk_norm": True,
        "source_audit": audit,
        "entries": entries_preview,
    }

    print(f"[safetensors->bump] arch={arch} tensors={len(refs)} entries={len(entries_preview)} dry_run={args.dry_run}")
    estimated_bytes = int(offset - DATA_START)
    print(f"[safetensors->bump] output={args.output}")
    if args.ram_output:
        print(f"[safetensors->bump] ram_output=True ram_dir={args.ram_dir}")
    print(f"[safetensors->bump] estimated weights payload={(offset - (DATA_START + 4 + len(refs))) / 1024 / 1024:.1f} MiB")

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        return 0

    _check_output_capacity(args.output, estimated_bytes, ram_output=bool(args.ram_output))
    if not _is_proc_fd_path(args.output):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _copy_tokenizer_sidecars(model_dir, args.output.parent)
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
            entry = {
                "name": ref.ck_name,
                "dtype": dt,
                "file_offset": start,
                "size": size,
                "source_name": "+".join(ref.source_names) if ref.source_names else f"synthetic:{ref.synth}",
                "shape": shape,
            }
            transform = _ref_transform(ref)
            if transform:
                entry["transform"] = transform
            entries.append(entry)
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
        f.write(struct.pack("<I", _int_or_zero(config.get("num_layers"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("vocab_size"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("embed_dim"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("intermediate_size"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("context_length"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("num_heads"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("num_kv_heads"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("head_dim"))))
        for value in (
            _int_or_zero(config.get("embed_dim")),
            _int_or_zero(config.get("head_dim")),
            _int_or_zero(config.get("intermediate_size")),
            _int_or_zero(config.get("context_length")),
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
