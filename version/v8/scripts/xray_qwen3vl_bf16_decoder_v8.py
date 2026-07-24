#!/usr/bin/env python3
"""Bounded PyTorch-vs-CK BF16 decoder X-ray at one teacher-forced position."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import struct
import sys
from array import array
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
try:
    import xray_numerical_parity_v8 as xray
    from normalize_xray_ranking_report_v8 import normalize as normalize_ranking
finally:
    sys.path.pop(0)

BOUNDARIES = (
    "block_rmsnorm",
    "qk_norm_q",
    "qk_norm_k",
    "rope_q",
    "rope_k",
    "attn_pregate",
    "out_proj",
    "after_attn",
    "ffn_norm",
    "mlp_gate",
    "mlp_up",
    "mlp_swiglu",
    "mlp_down",
    "layer_out",
)

CK_EXPORT_NAMES = (
    "block_rmsnorm",
    "qk_norm_q",
    "qk_norm_k",
    "rope_q",
    "rope_k",
    "attn_pregate",
    "out_proj",
    "after_attn",
    "ffn_norm",
    "mlp_gate_up",
    "mlp_swiglu",
    "mlp_down",
    "layer_out",
)

PREFILL_PROBE_NAMES = (
    "layer_input", "block_rmsnorm", "q_proj", "k_proj", "v_proj",
    "qk_norm_q", "qk_norm_k", "rope_q", "rope_k", "attn_pregate",
    "out_proj", "after_attn", "ffn_norm", "mlp_gate", "mlp_up",
    "mlp_swiglu", "mlp_down", "layer_out",
)
PREFILL_HEAD_MAJOR_NAMES = {
    "qk_norm_q", "qk_norm_k", "rope_q", "rope_k", "attn_pregate",
}

BOUNDARY_OP_OCCURRENCE = {
    "block_rmsnorm": ("rmsnorm", 0),
    "qk_norm_q": ("qk_norm", 0),
    "qk_norm_k": ("qk_norm", 0),
    "rope_q": ("rope_qk", 0),
    "rope_k": ("rope_qk", 0),
    "attn_pregate": ("attn", 0),
    "out_proj": ("out_proj", 0),
    "after_attn": ("residual_add", 0),
    "ffn_norm": ("rmsnorm", 1),
    "mlp_gate": ("mlp_gate_up", 0),
    "mlp_up": ("mlp_gate_up", 0),
    "mlp_swiglu": ("silu_mul", 0),
    "mlp_down": ("mlp_down", 0),
    "layer_out": ("residual_add", 1),
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_f32(path: Path, tensor: Any) -> dict[str, Any]:
    if hasattr(tensor, "detach"):
        values = tensor.detach().float().cpu().contiguous().numpy()
    else:
        values = np.ascontiguousarray(tensor, dtype=np.float32)
    values = np.ascontiguousarray(values.reshape(1, -1), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    values.tofile(path)
    return {"path": str(path.resolve()), "shape": list(values.shape), "sha256": _sha256(path)}


def _save_bf16_bits(path: Path, tensor: Any) -> dict[str, Any]:
    import torch

    values = tensor.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    values.tofile(path)
    return {"path": str(path.resolve()), "shape": list(values.shape), "sha256": _sha256(path)}


def _first_failed_position(result: dict[str, Any]) -> int:
    position = result.get("first_teacher_forced_top1_divergence", result.get("first_divergence"))
    if position is None:
        raise ValueError("corpus result has no teacher-forced divergence")
    position = int(position)
    if position <= 0:
        raise ValueError("decoder checkpoint adapter currently requires a failure after position zero")
    return position


def _force_processor(tokens: list[int]):
    from transformers import LogitsProcessor

    class ForceReferenceTokens(LogitsProcessor):
        def __init__(self, values: list[int]):
            self.values = values
            self.index = 0

        def __call__(self, input_ids, scores):
            if self.index < len(self.values):
                token = int(self.values[self.index])
                scores.fill_(float("-inf"))
                scores[:, token] = 0.0
            self.index += 1
            return scores

    return ForceReferenceTokens(tokens)


def _capture_pytorch(
    checkpoint: Path,
    image_path: Path,
    prompt: str,
    forced: list[int],
    target_position: int,
    threads: int,
    output_dir: Path,
    *,
    capture_prefix: bool = False,
    cache_layers: set[int] | None = None,
    prefill_probe: tuple[int, int] | None = None,
) -> tuple[dict[int, dict[str, dict[str, Any]]], array | None, dict[str, Any]]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, LogitsProcessorList, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3vl

    torch.set_num_threads(threads)
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(checkpoint), torch_dtype=torch.bfloat16, local_files_only=True
    )
    model.eval()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt},
    ]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_text], images=[Image.open(image_path).convert("RGB")],
        return_tensors="pt", min_pixels=1, max_pixels=1048576,
    )
    captures: dict[int, dict[str, Any]] = {index: {} for index in range(len(model.model.language_model.layers))}
    decode_calls = [0] * len(captures)
    active_layer = [-1]
    active_cache: list[Any | None] = [None] * len(captures)
    cache_captures: dict[int, dict[str, Any]] = {}
    prefill_captures: dict[str, dict[str, Any]] = {}
    handles = []

    def current_row(value: Any):
        if isinstance(value, tuple):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            return None
        if value.ndim < 2:
            return None
        return value.detach().float().cpu().contiguous().reshape(-1)

    def layer_pre(layer: int) -> Callable:
        def hook(_module, args, _kwargs):
            value = next((item for item in args if isinstance(item, torch.Tensor)), None)
            if (
                decode_calls[layer] == 0
                and prefill_probe is not None
                and layer == prefill_probe[0]
                and value is not None
                and value.ndim >= 3
            ):
                token = int(prefill_probe[1])
                if token < value.shape[1]:
                    row = value[:, token, ...].detach().float().cpu().contiguous().reshape(-1)
                    prefill_captures["layer_input"] = _save_f32(
                        output_dir / f"layer_{layer:03d}_prefill_token_{token:04d}_layer_input.f32",
                        row,
                    )
            if value is not None and value.ndim >= 2 and value.shape[-2] == 1:
                decode_calls[layer] += 1
                if decode_calls[layer] == target_position:
                    captures[layer]["layer_input"] = current_row(value)
        return hook

    def module_output(layer: int, name: str) -> Callable:
        def hook(_module, _args, output):
            if decode_calls[layer] == 0 and prefill_probe is not None and layer == prefill_probe[0]:
                value = output[0] if isinstance(output, tuple) else output
                if isinstance(value, torch.Tensor) and value.ndim >= 3:
                    token = int(prefill_probe[1])
                    if token < value.shape[1]:
                        row = value[:, token, ...].detach().float().cpu().contiguous().reshape(-1)
                        prefill_captures[name] = _save_f32(
                            output_dir / f"layer_{layer:03d}_prefill_token_{token:04d}_{name}.f32", row
                        )
            if decode_calls[layer] != target_position:
                return
            row = current_row(output)
            if row is not None:
                captures[layer][name] = row
        return hook

    def module_input_output(layer: int, input_name: str, output_name: str) -> Callable:
        def hook(_module, args, output):
            if decode_calls[layer] == 0 and prefill_probe is not None and layer == prefill_probe[0]:
                token = int(prefill_probe[1])
                input_value = next((item for item in args if isinstance(item, torch.Tensor)), None)
                output_value = output[0] if isinstance(output, tuple) else output
                for name, value in ((input_name, input_value), (output_name, output_value)):
                    if isinstance(value, torch.Tensor) and value.ndim >= 3 and token < value.shape[1]:
                        row = value[:, token, ...].detach().float().cpu().contiguous().reshape(-1)
                        prefill_captures[name] = _save_f32(
                            output_dir / f"layer_{layer:03d}_prefill_token_{token:04d}_{name}.f32",
                            row,
                        )
            if decode_calls[layer] != target_position:
                return
            input_row = current_row(next((item for item in args if isinstance(item, torch.Tensor)), None))
            output_row = current_row(output)
            if input_row is not None:
                captures[layer][input_name] = input_row
            if output_row is not None:
                captures[layer][output_name] = output_row
        return hook

    def attention_pre(layer: int) -> Callable:
        def hook(_module, _args, kwargs):
            active_layer[0] = layer
            if decode_calls[layer] in (0, target_position):
                cache = kwargs.get("past_key_values")
                if cache is None:
                    cache = kwargs.get("past_key_value")
                active_cache[layer] = cache
        return hook

    def attention_cache_output(layer: int) -> Callable:
        def hook(_module, _args, _output):
            if active_cache[layer] is None or (cache_layers is not None and layer not in cache_layers):
                return
            cache_layer = active_cache[layer].layers[layer]
            if decode_calls[layer] == 0:
                stage = "prefill"
            elif decode_calls[layer] == target_position:
                stage = "decode"
            else:
                return
            cache_captures.setdefault(layer, {})[stage] = {
                "key": _save_bf16_bits(
                    output_dir / f"layer_{layer:03d}_{stage}_kv_key.bf16", cache_layer.keys
                ),
                "value": _save_bf16_bits(
                    output_dir / f"layer_{layer:03d}_{stage}_kv_value.bf16", cache_layer.values
                ),
            }
        return hook

    for layer_index, layer in enumerate(model.model.language_model.layers):
        handles.append(layer.register_forward_pre_hook(layer_pre(layer_index), with_kwargs=True))
        handles.append(layer.register_forward_hook(module_output(layer_index, "layer_out")))
        handles.append(layer.input_layernorm.register_forward_hook(module_output(layer_index, "block_rmsnorm")))
        handles.append(layer.self_attn.q_proj.register_forward_hook(module_output(layer_index, "q_proj")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(module_output(layer_index, "k_proj")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(module_output(layer_index, "v_proj")))
        handles.append(layer.self_attn.register_forward_pre_hook(attention_pre(layer_index), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(attention_cache_output(layer_index)))
        handles.append(layer.self_attn.q_norm.register_forward_hook(module_output(layer_index, "qk_norm_q")))
        handles.append(layer.self_attn.k_norm.register_forward_hook(module_output(layer_index, "qk_norm_k")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(
            module_input_output(layer_index, "attn_pregate", "out_proj")
        ))
        handles.append(layer.post_attention_layernorm.register_forward_hook(
            module_input_output(layer_index, "after_attn", "ffn_norm")
        ))
        handles.append(layer.mlp.gate_proj.register_forward_hook(module_output(layer_index, "mlp_gate")))
        handles.append(layer.mlp.up_proj.register_forward_hook(module_output(layer_index, "mlp_up")))
        handles.append(layer.mlp.down_proj.register_forward_hook(
            module_input_output(layer_index, "mlp_swiglu", "mlp_down")
        ))

    original_rope = qwen3vl.apply_rotary_pos_emb

    def capture_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_out, k_out = original_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
        layer = active_layer[0]
        if (
            layer >= 0
            and decode_calls[layer] == 0
            and prefill_probe is not None
            and layer == prefill_probe[0]
            and q_out.ndim >= 4
        ):
            token = int(prefill_probe[1])
            if token < q_out.shape[-2] and token < k_out.shape[-2]:
                prefill_captures["rope_q"] = _save_f32(
                    output_dir / f"layer_{layer:03d}_prefill_token_{token:04d}_rope_q.f32",
                    q_out[..., token, :].detach().float().cpu().contiguous().reshape(-1),
                )
                prefill_captures["rope_k"] = _save_f32(
                    output_dir / f"layer_{layer:03d}_prefill_token_{token:04d}_rope_k.f32",
                    k_out[..., token, :].detach().float().cpu().contiguous().reshape(-1),
                )
        if layer >= 0 and decode_calls[layer] == target_position and q_out.shape[-2] == 1:
            captures[layer]["rope_q"] = q_out.detach().float().cpu().contiguous().reshape(-1)
            captures[layer]["rope_k"] = k_out.detach().float().cpu().contiguous().reshape(-1)
        return q_out, k_out

    qwen3vl.apply_rotary_pos_emb = capture_rope
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=target_position + 1,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                logits_processor=LogitsProcessorList([_force_processor(forced[:target_position])]),
            )
            if capture_prefix:
                image_embeds, deepstack = model.get_image_features(
                    inputs["pixel_values"], inputs["image_grid_thw"]
                )
    finally:
        qwen3vl.apply_rotary_pos_emb = original_rope
        for handle in handles:
            handle.remove()

    stored: dict[int, dict[str, dict[str, Any]]] = {}
    for layer, tensors in captures.items():
        stored[layer] = {}
        for name, value in tensors.items():
            stored[layer][name] = _save_f32(output_dir / f"layer_{layer:03d}_{name}.f32", value)
    prefix_values = None
    prefix_shape = None
    if capture_prefix:
        prefix = torch.cat([image_embeds[0], *deepstack], dim=-1).float().cpu().contiguous()
        prefix_values = array("f", prefix.numpy().reshape(-1))
        prefix_shape = list(prefix.shape)
    metadata = {
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "grid_thw": [int(value) for value in inputs["image_grid_thw"][0].tolist()],
        "prefix_shape": prefix_shape,
        "forced_prefix_matches": generated[0, -target_position - 1:-1].tolist() == forced[:target_position],
        "decode_calls": decode_calls,
        "kv_cache": cache_captures,
        "prefill_probe": {
            "layer": prefill_probe[0], "token": prefill_probe[1], "tensors": prefill_captures
        } if prefill_probe is not None else None,
    }
    return stored, prefix_values, metadata


def _load_certified_prefix(request: dict[str, Any]) -> array:
    path = Path(str(request["prefix_f32"]))
    expected = int(request["prefix_tokens"]) * int(request["prefix_embed_dim"])
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, expected)
        if handle.read(1):
            raise ValueError(f"certified prefix contains more than {expected} FP32 values: {path}")
    if len(values) != expected:
        raise ValueError(f"certified prefix has {len(values)} FP32 values, expected {expected}: {path}")
    return values


def _compare_prefix_arrays(actual: array, reference: array) -> dict[str, Any]:
    if len(actual) != len(reference):
        return {
            "byte_exact": False,
            "actual_elements": len(actual),
            "reference_elements": len(reference),
            "shape_match": False,
        }
    actual_values = np.frombuffer(actual.tobytes(), dtype=np.float32)
    reference_values = np.frombuffer(reference.tobytes(), dtype=np.float32)
    delta = actual_values.astype(np.float64) - reference_values.astype(np.float64)
    absolute = np.abs(delta)
    rmse = float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0
    reference_norm = float(np.sqrt(np.mean(np.square(reference_values, dtype=np.float64))))
    denominator = float(
        np.linalg.norm(actual_values.astype(np.float64))
        * np.linalg.norm(reference_values.astype(np.float64))
    )
    return {
        "byte_exact": actual.tobytes() == reference.tobytes(),
        "shape_match": True,
        "actual_elements": len(actual),
        "reference_elements": len(reference),
        "exact_elements": int(np.count_nonzero(actual_values == reference_values)),
        "max_abs": float(np.max(absolute)) if absolute.size else 0.0,
        "rmse": rmse,
        "relative_rmse": rmse / reference_norm if reference_norm else (0.0 if rmse == 0.0 else None),
        "cosine": float(
            np.dot(actual_values.astype(np.float64), reference_values.astype(np.float64)) / denominator
        ) if denominator else 1.0,
    }


def _capture_ck(
    runtime: Path,
    request: dict[str, Any],
    prefix: array,
    forced: list[int],
    target_position: int,
    threads: int,
    output_dir: Path,
    *,
    export_layer: int | None = None,
    import_layer_input: tuple[int, Path] | None = None,
    export_kv_layer: int | None = None,
    prefill_probe_layer: int | None = None,
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    bridge = _load_module("cke_xray_bf16_decoder_bridge", SCRIPT_DIR / "run_multimodal_bridge_v8.py")
    os.environ["CK_NUM_THREADS"] = str(threads)
    lib = bridge._load_decoder_lib(runtime / "libmodel.so", engine_so=runtime / "libckernel_engine.so")
    rc = lib.ck_model_init_with_manifest(
        str(runtime / "weights.bump").encode(), str(runtime / "weights_manifest.map").encode()
    )
    if rc != 0:
        raise RuntimeError(f"CK model init failed: rc={rc}")
    try:
        prefix_buf = (ctypes.c_float * len(prefix))(*prefix)
        before = [int(value) for value in request["tokens_before"]]
        after = [int(value) for value in request["tokens_after"]]
        before_buf = (ctypes.c_int32 * len(before))(*before)
        after_buf = (ctypes.c_int32 * len(after))(*after)
        vocab = int(lib.ck_model_get_vocab_size())
        logits = (ctypes.c_float * vocab)()
        prefill_probe_dir = output_dir / "prefill_probe"
        if prefill_probe_layer is not None:
            prefill_probe_dir.mkdir(parents=True, exist_ok=True)
            os.environ["CK_DEBUG_EXPORT_HIDDEN"] = str(prefill_probe_dir)
            os.environ["CK_DEBUG_EXPORT_HIDDEN_NAMES"] = ",".join(PREFILL_PROBE_NAMES)
            os.environ["CK_DEBUG_EXPORT_HIDDEN_LAYER"] = str(prefill_probe_layer)
        rc = lib.ck_model_forward_segments_grid_ex(
            before_buf, len(before), prefix_buf, int(request["prefix_tokens"]),
            int(request["prefix_embed_dim"]), int(request["prefix_grid"][0]),
            int(request["prefix_grid"][1]), int(request["prefix_text_pos"]),
            after_buf, len(after), logits,
        )
        if prefill_probe_layer is not None:
            os.environ.pop("CK_DEBUG_EXPORT_HIDDEN", None)
            os.environ.pop("CK_DEBUG_EXPORT_HIDDEN_NAMES", None)
            os.environ.pop("CK_DEBUG_EXPORT_HIDDEN_LAYER", None)
        if rc != 0:
            raise RuntimeError(f"CK mixed prefill failed: rc={rc}")
        output_dir.mkdir(parents=True, exist_ok=True)
        prefill_kv_export = None
        if export_kv_layer is not None:
            if not hasattr(lib, "ck_model_debug_export_kv_f16"):
                raise RuntimeError("runtime lacks bounded ck_model_debug_export_kv_f16 X-ray ABI")
            lib.ck_model_debug_export_kv_f16.argtypes = [ctypes.c_char_p, ctypes.c_int]
            lib.ck_model_debug_export_kv_f16.restype = ctypes.c_int
            prefill_kv_path = output_dir / f"layer_{export_kv_layer:03d}_prefill_kv_cache.bin"
            prefill_kv_rc = int(
                lib.ck_model_debug_export_kv_f16(str(prefill_kv_path).encode(), int(export_kv_layer))
            )
            if prefill_kv_rc != 0:
                raise RuntimeError(
                    f"CK bounded prefill KV export failed: layer={export_kv_layer} rc={prefill_kv_rc}"
                )
            prefill_kv_export = {
                "path": str(prefill_kv_path.resolve()), "sha256": _sha256(prefill_kv_path)
            }
        for step, token in enumerate(forced[:target_position]):
            capture = step + 1 == target_position
            if capture:
                os.environ["CK_DEBUG_EXPORT_HIDDEN"] = str(output_dir)
                os.environ["CK_DEBUG_EXPORT_HIDDEN_NAMES"] = ",".join(CK_EXPORT_NAMES)
                if export_layer is None:
                    os.environ.pop("CK_DEBUG_EXPORT_HIDDEN_LAYER", None)
                else:
                    os.environ["CK_DEBUG_EXPORT_HIDDEN_LAYER"] = str(export_layer)
                if import_layer_input is not None:
                    import_layer, import_path = import_layer_input
                    os.environ["CK_DEBUG_IMPORT_HIDDEN"] = str(import_path)
                    os.environ["CK_DEBUG_IMPORT_LAYER"] = str(import_layer)
                    os.environ["CK_DEBUG_IMPORT_CHECKPOINT"] = "layer_input"
            rc = lib.ck_model_decode(ctypes.c_int32(int(token)), logits)
            if capture:
                os.environ.pop("CK_DEBUG_EXPORT_HIDDEN", None)
                os.environ.pop("CK_DEBUG_EXPORT_HIDDEN_NAMES", None)
                os.environ.pop("CK_DEBUG_EXPORT_HIDDEN_LAYER", None)
                os.environ.pop("CK_DEBUG_IMPORT_HIDDEN", None)
                os.environ.pop("CK_DEBUG_IMPORT_LAYER", None)
                os.environ.pop("CK_DEBUG_IMPORT_CHECKPOINT", None)
            if rc != 0:
                raise RuntimeError(f"CK decode failed at input step {step}: rc={rc}")
        top1 = int(np.argmax(np.ctypeslib.as_array(logits)))
        decode_kv_export = None
        if export_kv_layer is not None:
            if not hasattr(lib, "ck_model_debug_export_kv_f16"):
                raise RuntimeError("runtime lacks bounded ck_model_debug_export_kv_f16 X-ray ABI")
            lib.ck_model_debug_export_kv_f16.argtypes = [ctypes.c_char_p, ctypes.c_int]
            lib.ck_model_debug_export_kv_f16.restype = ctypes.c_int
            kv_path = output_dir / f"layer_{export_kv_layer:03d}_kv_cache.bin"
            kv_rc = int(lib.ck_model_debug_export_kv_f16(str(kv_path).encode(), int(export_kv_layer)))
            if kv_rc != 0:
                raise RuntimeError(f"CK bounded KV export failed: layer={export_kv_layer} rc={kv_rc}")
            decode_kv_export = {"path": str(kv_path.resolve()), "sha256": _sha256(kv_path)}
    finally:
        os.environ.pop("CK_DEBUG_EXPORT_HIDDEN", None)
        os.environ.pop("CK_DEBUG_EXPORT_HIDDEN_NAMES", None)
        os.environ.pop("CK_DEBUG_EXPORT_HIDDEN_LAYER", None)
        os.environ.pop("CK_DEBUG_IMPORT_HIDDEN", None)
        os.environ.pop("CK_DEBUG_IMPORT_LAYER", None)
        os.environ.pop("CK_DEBUG_IMPORT_CHECKPOINT", None)
        lib.ck_model_free()

    stored, intermediate_dim = _load_ck_capture_dir(output_dir)
    return stored, {
        "top1": top1,
        "captured_layers": sorted(stored),
        "intermediate_dim": intermediate_dim,
        "imported_layer_input": import_layer_input is not None,
        "imported_layer": import_layer_input[0] if import_layer_input is not None else None,
        "kv_cache": {"prefill": prefill_kv_export, "decode": decode_kv_export}
        if export_kv_layer is not None else None,
        "prefill_probe_dir": str(prefill_probe_dir.resolve()) if prefill_probe_layer is not None else None,
    }


def _load_ck_capture_dir(output_dir: Path) -> tuple[dict[int, dict[str, dict[str, Any]]], int | None]:
    files: dict[int, dict[str, Path]] = {}
    for path in output_dir.glob("tok_*_layer_*_*.f32"):
        parts = path.stem.split("_")
        layer = int(parts[3])
        name = "_".join(parts[4:])
        files.setdefault(layer, {})[name] = path
    stored: dict[int, dict[str, dict[str, Any]]] = {}
    intermediate_dim = None
    for layer, tensors in files.items():
        stored[layer] = {}
        for name, path in tensors.items():
            values = np.fromfile(path, dtype=np.float32)
            if name == "mlp_gate_up":
                if values.size % 2:
                    raise RuntimeError(f"odd fused gate/up extent: {path}")
                intermediate_dim = values.size // 2
                stored[layer]["mlp_gate"] = _save_f32(
                    output_dir / f"canonical_layer_{layer:03d}_mlp_gate.f32", values[:intermediate_dim]
                )
                stored[layer]["mlp_up"] = _save_f32(
                    output_dir / f"canonical_layer_{layer:03d}_mlp_up.f32", values[intermediate_dim:]
                )
                continue
            stored[layer][name] = {
                "path": str(path.resolve()), "shape": [1, int(values.size)], "sha256": _sha256(path)
            }
    return stored, intermediate_dim


def _load_pytorch_capture_dir(output_dir: Path) -> dict[int, dict[str, dict[str, Any]]]:
    stored: dict[int, dict[str, dict[str, Any]]] = {}
    for path in output_dir.glob("layer_*_*.f32"):
        parts = path.stem.split("_")
        layer = int(parts[1])
        name = "_".join(parts[2:])
        values = np.fromfile(path, dtype=np.float32)
        stored.setdefault(layer, {})[name] = {
            "path": str(path.resolve()),
            "shape": [1, int(values.size)],
            "sha256": _sha256(path),
        }
    return stored


def _compare_kv_cache(ck_path: Path, torch_cache: dict[str, Any]) -> dict[str, Any]:
    with ck_path.open("rb") as handle:
        header = struct.unpack("<8I", handle.read(32))
        raw = np.fromfile(handle, dtype=np.uint16)
    magic, version, layer, token_count, heads, _capacity, head_dim, _reserved = header
    if magic != 0x564B5843 or version != 1:
        raise ValueError(f"invalid CK KV X-ray header: {ck_path}")
    per_tensor = int(heads) * int(token_count) * int(head_dim)
    if raw.size != 2 * per_tensor:
        raise ValueError(f"CK KV payload has {raw.size} uint16 values, expected {2 * per_tensor}")
    ck_values = {
        "key": raw[:per_tensor].reshape(heads, token_count, head_dim),
        "value": raw[per_tensor:].reshape(heads, token_count, head_dim),
    }
    report: dict[str, Any] = {"layer": int(layer), "token_count": int(token_count), "heads": int(heads), "head_dim": int(head_dim)}
    total_exact = 0
    total = 0
    first_difference = None
    for kind in ("key", "value"):
        reference_meta = torch_cache[kind]
        reference = np.fromfile(reference_meta["path"], dtype=np.uint16)
        expected_shape = tuple(int(value) for value in reference_meta["shape"])
        reference = reference.reshape(expected_shape)
        if reference.ndim == 4 and reference.shape[0] == 1:
            reference = reference[0]
        actual = ck_values[kind]
        if actual.shape != reference.shape:
            raise ValueError(f"{kind} KV shape mismatch: CK {actual.shape}, PyTorch {reference.shape}")
        same = actual == reference
        exact = int(np.count_nonzero(same))
        total_exact += exact
        total += int(actual.size)
        diff_indices = np.argwhere(~same)
        first = None
        if diff_indices.size:
            h, token, channel = (int(value) for value in diff_indices[0])
            first = {
                "head": h, "token": token, "channel": channel,
                "ck_bits": int(actual[h, token, channel]),
                "pytorch_bits": int(reference[h, token, channel]),
            }
            if first_difference is None:
                first_difference = {"kind": kind, **first}
        actual_f32 = (actual.astype(np.uint32) << 16).view(np.float32)
        reference_f32 = (reference.astype(np.uint32) << 16).view(np.float32)
        absolute = np.abs(actual_f32 - reference_f32)
        report[kind] = {
            "exact_elements": exact,
            "total_elements": int(actual.size),
            "exact_ratio": exact / int(actual.size),
            "max_abs": float(np.max(absolute)) if absolute.size else 0.0,
            "rmse": float(np.sqrt(np.mean(np.square(absolute, dtype=np.float64)))) if absolute.size else 0.0,
            "first_difference": first,
        }
    report["byte_exact"] = total_exact == total
    report["exact_elements"] = total_exact
    report["total_elements"] = total
    report["first_difference"] = first_difference
    return report


def _compare_prefill_probe(
    ck_dir: Path,
    torch_probe: dict[str, Any],
    token_count: int,
    head_dim: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name in PREFILL_PROBE_NAMES:
        reference_meta = (torch_probe.get("tensors") or {}).get(name)
        matches = sorted(ck_dir.glob(f"tok_*_layer_*_{name}.f32"))
        if reference_meta is None or len(matches) != 1:
            rows.append({"name": name, "status": "missing", "ck_matches": len(matches)})
            continue
        reference = np.fromfile(reference_meta["path"], dtype=np.float32)
        values = np.fromfile(matches[0], dtype=np.float32)
        if values.size != token_count * reference.size:
            raise ValueError(
                f"CK prefill {name} has {values.size} values, expected {token_count * reference.size}"
            )
        token = int(torch_probe["token"])
        if name in PREFILL_HEAD_MAJOR_NAMES:
            if head_dim <= 0 or reference.size % head_dim:
                raise ValueError(f"invalid head-major prefill shape for {name}: {reference.size}")
            heads = reference.size // head_dim
            actual = values.reshape(heads, token_count, head_dim)[:, token, :].reshape(-1).copy()
        else:
            actual = values.reshape(token_count, reference.size)[token].copy()
        actual_path = output_dir / f"token_{token:04d}_{name}.f32"
        actual.tofile(actual_path)
        delta = actual.astype(np.float64) - reference.astype(np.float64)
        absolute = np.abs(delta)
        reference_norm = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        denominator = float(np.linalg.norm(actual.astype(np.float64)) * np.linalg.norm(reference.astype(np.float64)))
        rows.append({
            "name": name,
            "status": "exact" if np.array_equal(actual, reference) else "different",
            "exact_elements": int(np.count_nonzero(actual == reference)),
            "total_elements": int(actual.size),
            "max_abs": float(np.max(absolute)) if absolute.size else 0.0,
            "rmse": rmse,
            "relative_rmse": rmse / reference_norm if reference_norm else (0.0 if rmse == 0.0 else None),
            "cosine": float(np.dot(actual.astype(np.float64), reference.astype(np.float64)) / denominator)
            if denominator else 1.0,
            "ck_path": str(actual_path.resolve()),
            "pytorch_path": reference_meta["path"],
        })
    first = next((row for row in rows if row["status"] == "different"), None)
    return {
        "layer": int(torch_probe["layer"]),
        "token": int(torch_probe["token"]),
        "token_count": int(token_count),
        "first_non_exact": first,
        "comparisons": rows,
    }


def _operation_metadata(call_ir: dict[str, Any], layer: int, boundary: str) -> dict[str, Any]:
    op_name, occurrence = BOUNDARY_OP_OCCURRENCE[boundary]
    candidates = [
        operation for operation in call_ir.get("operations", [])
        if int(operation.get("layer", -1)) == layer and str(operation.get("op")) == op_name
    ]
    if occurrence >= len(candidates):
        raise ValueError(f"call IR has no {op_name} occurrence {occurrence} at layer {layer}")
    operation = candidates[occurrence]
    resolved = operation.get("resolved_contract") or {}
    return {
        "producer": op_name,
        "resolved_contract_id": str(
            resolved.get("resolved_contract_id") or resolved.get("contract_id") or "unresolved"
        ),
        "kernel_id": str((operation.get("call_abi") or {}).get("kernel_id") or resolved.get("kernel_id") or "unresolved"),
        "function": str(operation.get("function") or resolved.get("function") or "unresolved"),
    }


def _manifest(
    backend: str,
    tensors: dict[int, dict[str, dict[str, Any]]],
    call_ir: dict[str, Any],
    selected: list[tuple[int, str]],
    phase: str,
    source: str,
) -> dict[str, Any]:
    checkpoints = []
    for layer, boundary in selected:
        tensor = tensors.get(layer, {}).get(boundary)
        if tensor is None:
            continue
        metadata = _operation_metadata(call_ir, layer, boundary)
        checkpoint_id = f"decoder.layer.{layer}.{boundary}"
        checkpoints.append({
            "checkpoint_id": checkpoint_id,
            "producer": metadata["producer"],
            "phase": phase,
            "layer": layer,
            "tensor_path": tensor["path"],
            "storage_dtype": "bf16",
            "exported_dtype": "fp32",
            "logical_shape": tensor["shape"],
            "physical_shape": tensor["shape"],
            "logical_layout": "token_major_flattened",
            "axis_names": ["token", "channel"],
            "physical_axis_names": ["token", "channel"],
            "resolved_contract_id": metadata["resolved_contract_id"],
            "kernel_id": metadata["kernel_id"],
            "function": metadata["function"],
            "sha256": tensor["sha256"],
        })
    return {
        "schema": "cke.checkpoint_manifest",
        "schema_version": 1,
        "backend": backend,
        "run": {"model": "qwen3vl", "phase": phase, "source": source},
        "checkpoints": checkpoints,
    }


def _profile(order: list[str], interval_expansions: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "schema": "cke.parity_profile",
        "schema_version": 1,
        "name": "qwen3vl_pytorch_bf16_decoder_runtime",
        "backend": "pytorch",
        "contract_schema_version": 1,
        "required_match_fields": [
            "checkpoint_id", "producer", "logical_layout", "axis_names",
            "resolved_contract_id", "kernel_id", "function",
        ],
        "observed_storage": {"default": "bf16", "checkpoints": {}},
        "dtype_thresholds": {
            "bf16": {
                "cosine_min": 0.999,
                "rmse_max": 0.02,
                "relative_rmse_max": 0.02,
                "max_abs_max": 0.25,
                "finite_required": True,
            },
        },
        "checkpoint_order": order,
        "interval_expansions": interval_expansions,
        "backend_mappings": {},
    }


def _largest_layer_delta(rows: list[dict[str, Any]]) -> int:
    numeric = [row for row in rows if isinstance(row.get("metrics"), dict)]
    if not numeric:
        raise ValueError("no comparable layer outputs")
    previous = None
    best = None
    for row in numeric:
        current = float(row["metrics"]["relative_rmse"])
        if previous is not None:
            candidate = (current - previous[1], int(row["checkpoint_id"].split(".")[2]))
            if best is None or candidate[0] > best[0]:
                best = candidate
        previous = (row, current)
    return int(best[1] if best is not None else numeric[0]["checkpoint_id"].split(".")[2])


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = _read_json(args.corpus_result)
    request = _read_json(args.decoder_request)
    call_ir = _read_json(args.call_ir)
    target = args.position if args.position is not None else _first_failed_position(result)
    forced = [int(value) for value in request.get("forced_generation_token_ids") or []]
    if len(forced) < target:
        raise ValueError(f"request has {len(forced)} forced tokens but position {target} requires {target}")
    if result.get("prefix_metrics", {}).get("exact_elements") != result.get("prefix_metrics", {}).get("elements"):
        raise ValueError("decoder X-ray requires a byte-exact certified visual prefix")

    reuse = bool(getattr(args, "reuse_existing_captures", False))
    prefix: array | None = None
    if reuse:
        torch_tensors = _load_pytorch_capture_dir(args.output_dir / "pytorch")
        ck_tensors, ck_intermediate_dim = _load_ck_capture_dir(args.output_dir / "ck")
        if not torch_tensors or not ck_tensors:
            raise ValueError("--reuse-existing-captures requires populated pytorch/ and ck/ directories")
        torch_meta = {"reused_existing_captures": True}
        ck_meta = {
            "reused_existing_captures": True,
            "captured_layers": sorted(ck_tensors),
            "intermediate_dim": ck_intermediate_dim,
        }
    else:
        prefix_path = Path(str(request["prefix_f32"]))
        torch_tensors, regenerated_prefix, torch_meta = _capture_pytorch(
            args.checkpoint,
            args.image,
            args.prompt,
            forced,
            target,
            args.threads,
            args.output_dir / "pytorch",
            capture_prefix=True,
            cache_layers={int(args.target_layer)} if args.target_layer is not None else None,
            prefill_probe=(int(args.target_layer), int(args.prefill_probe_token))
            if args.target_layer is not None and args.prefill_probe_token is not None else None,
        )
        if regenerated_prefix is None:
            raise RuntimeError("PyTorch did not regenerate the visual prefix")
        expected = int(request["prefix_tokens"]) * int(request["prefix_embed_dim"])
        if len(regenerated_prefix) != expected:
            raise ValueError(
                f"regenerated prefix has {len(regenerated_prefix)} FP32 values, expected {expected}"
            )
        certified_prefix = _load_certified_prefix(request) if prefix_path.is_file() else None
        prefix_comparison = (
            _compare_prefix_arrays(certified_prefix, regenerated_prefix)
            if certified_prefix is not None else None
        )
        if args.ck_prefix_source == "pytorch":
            prefix = regenerated_prefix
            prefix_source = "regenerated_pytorch"
        elif certified_prefix is not None:
            prefix = certified_prefix
            prefix_source = "certified_corpus_artifact"
        else:
            prefix = regenerated_prefix
            prefix_source = "regenerated_pytorch"
        torch_meta["prefix_source"] = prefix_source
        torch_meta["certified_vs_regenerated_prefix"] = prefix_comparison
        prefix_capture_path = args.output_dir / "certified_prefix.f32"
        prefix_capture_path.parent.mkdir(parents=True, exist_ok=True)
        with prefix_capture_path.open("wb") as handle:
            prefix.tofile(handle)
        torch_meta["prefix_capture"] = {
            "path": str(prefix_capture_path.resolve()),
            "sha256": _sha256(prefix_capture_path),
            "elements": len(prefix),
        }
        ck_tensors, ck_meta = _capture_ck(
            args.runtime, request, prefix, forced, target, args.threads, args.output_dir / "ck"
        )
    layer_count = min(len(torch_tensors), len(ck_tensors))
    all_layers = [(layer, "layer_out") for layer in range(layer_count)]
    sparse_layers = sorted(set([0, 8, 16, 24, 32, layer_count - 1]))
    sparse = [(layer, "layer_out") for layer in sparse_layers]
    all_order = [f"decoder.layer.{layer}.layer_out" for layer in range(layer_count)]
    sparse_order = [f"decoder.layer.{layer}.layer_out" for layer in sparse_layers]
    intervals = {}
    for lower, upper in zip(sparse_layers, sparse_layers[1:]):
        intervals[f"decoder.layer.{lower}.layer_out->decoder.layer.{upper}.layer_out"] = [
            f"decoder.layer.{layer}.layer_out" for layer in range(lower + 1, upper)
        ] or [f"decoder.layer.{upper}.layer_out"]

    ck_all_manifest = _manifest("ck", ck_tensors, call_ir, all_layers, "teacher_forced", str(args.runtime))
    torch_all_manifest = _manifest("pytorch", torch_tensors, call_ir, all_layers, "teacher_forced", str(args.checkpoint))
    ck_sparse_manifest = _manifest("ck", ck_tensors, call_ir, sparse, "teacher_forced", str(args.runtime))
    torch_sparse_manifest = _manifest("pytorch", torch_tensors, call_ir, sparse, "teacher_forced", str(args.checkpoint))
    ranking = normalize_ranking(result, "teacher_forced")
    sparse_report = xray.compare_manifests(
        ck_sparse_manifest, torch_sparse_manifest, _profile(sparse_order, intervals),
        ranking_report=ranking, checkpoint_order=sparse_order,
    )
    all_report = xray.compare_manifests(
        ck_all_manifest, torch_all_manifest, _profile(all_order, {}), checkpoint_order=all_order,
    )
    target_layer = (
        int(args.target_layer)
        if getattr(args, "target_layer", None) is not None
        else _largest_layer_delta(all_report["comparisons"])
    )
    if target_layer < 0 or target_layer >= layer_count:
        raise ValueError(f"target layer {target_layer} is outside decoder layer range 0..{layer_count - 1}")
    granular = [(target_layer, boundary) for boundary in BOUNDARIES]
    granular_order = [f"decoder.layer.{target_layer}.{boundary}" for boundary in BOUNDARIES]
    ck_granular = _manifest("ck", ck_tensors, call_ir, granular, "teacher_forced", str(args.runtime))
    torch_granular = _manifest("pytorch", torch_tensors, call_ir, granular, "teacher_forced", str(args.checkpoint))
    granular_report = xray.compare_manifests(
        ck_granular, torch_granular, _profile(granular_order, {}), checkpoint_order=granular_order,
    )

    torch_layer_input = torch_tensors.get(target_layer, {}).get("layer_input")
    if torch_layer_input is None:
        raise RuntimeError(f"PyTorch did not capture decoder layer {target_layer} input")
    if reuse:
        ck_exact_tensors, ck_exact_intermediate_dim = _load_ck_capture_dir(args.output_dir / "ck_exact_input")
        if target_layer not in ck_exact_tensors:
            raise ValueError(
                f"--reuse-existing-captures has no exact-input capture for decoder layer {target_layer}"
            )
        ck_exact_meta = {
            "reused_existing_captures": True,
            "captured_layers": sorted(ck_exact_tensors),
            "intermediate_dim": ck_exact_intermediate_dim,
            "imported_layer_input": True,
            "imported_layer": target_layer,
        }
    else:
        assert prefix is not None
        ck_exact_tensors, ck_exact_meta = _capture_ck(
            args.runtime,
            request,
            prefix,
            forced,
            target,
            args.threads,
            args.output_dir / "ck_exact_input",
            export_layer=target_layer,
            import_layer_input=(target_layer, Path(torch_layer_input["path"])),
            export_kv_layer=target_layer,
            prefill_probe_layer=target_layer if args.prefill_probe_token is not None else None,
        )
    ck_exact_granular = _manifest(
        "ck", ck_exact_tensors, call_ir, granular, "teacher_forced", str(args.runtime)
    )
    torch_exact_granular = _manifest(
        "pytorch", torch_tensors, call_ir, granular, "teacher_forced", str(args.checkpoint)
    )
    exact_input_report = xray.compare_manifests(
        ck_exact_granular,
        torch_exact_granular,
        _profile(granular_order, {}),
        checkpoint_order=granular_order,
    )
    kv_cache_reports: dict[str, Any] = {}
    torch_kv = (torch_meta.get("kv_cache") or {}).get(target_layer) or {}
    ck_kv = ck_exact_meta.get("kv_cache")
    if ck_kv is not None:
        for stage in ("prefill", "decode"):
            if torch_kv.get(stage) is not None and ck_kv.get(stage) is not None:
                kv_cache_reports[stage] = _compare_kv_cache(
                    Path(ck_kv[stage]["path"]), torch_kv[stage]
                )
    prefill_probe_report = None
    torch_probe = torch_meta.get("prefill_probe")
    ck_probe_dir = ck_exact_meta.get("prefill_probe_dir")
    if torch_probe is not None and ck_probe_dir is not None:
        token_count = len(request["tokens_before"]) + int(request["prefix_tokens"]) + len(request["tokens_after"])
        prefill_probe_report = _compare_prefill_probe(
            Path(ck_probe_dir),
            torch_probe,
            token_count,
            int((call_ir.get("config") or {}).get("head_dim", 0) or 0),
            args.output_dir / "prefill_probe_rows",
        )

    manifests = {
        "ck_all": ck_all_manifest,
        "pytorch_all": torch_all_manifest,
        "ck_granular": ck_granular,
        "pytorch_granular": torch_granular,
        "ck_exact_input_granular": ck_exact_granular,
        "pytorch_exact_input_granular": torch_exact_granular,
    }
    for name, manifest in manifests.items():
        _write_json(args.output_dir / f"{name}.checkpoints.json", manifest)
    _write_json(args.output_dir / "ranking.json", ranking)
    _write_json(args.output_dir / "sparse_report.json", sparse_report)
    _write_json(args.output_dir / "all_layers_report.json", all_report)
    _write_json(args.output_dir / "granular_report.json", granular_report)
    _write_json(args.output_dir / "exact_input_granular_report.json", exact_input_report)
    if kv_cache_reports:
        _write_json(args.output_dir / "kv_cache_report.json", kv_cache_reports)
    if prefill_probe_report is not None:
        _write_json(args.output_dir / "prefill_probe_report.json", prefill_probe_report)
    report = {
        "schema": "cke.xray.qwen3vl_bf16_decoder",
        "schema_version": 1,
        "status": "attributed",
        "teacher_forced_position": target,
        "image_sha256": _sha256(args.image),
        "prefix_certified_byte_exact": (
            (torch_meta.get("certified_vs_regenerated_prefix") or {}).get("byte_exact")
            if not reuse else None
        ),
        "torch": torch_meta,
        "ck": ck_meta,
        "sparse_report": str(args.output_dir / "sparse_report.json"),
        "all_layers_report": str(args.output_dir / "all_layers_report.json"),
        "strongest_amplification_layer": target_layer,
        "granular_report": str(args.output_dir / "granular_report.json"),
        "granular_first_non_exact": granular_report.get("first_non_exact_checkpoint"),
        "granular_first_material": granular_report.get("first_divergence"),
        "exact_input": ck_exact_meta,
        "exact_input_granular_report": str(args.output_dir / "exact_input_granular_report.json"),
        "exact_input_first_non_exact": exact_input_report.get("first_non_exact_checkpoint"),
        "exact_input_first_material": exact_input_report.get("first_divergence"),
        "kv_cache_report": str(args.output_dir / "kv_cache_report.json") if kv_cache_reports else None,
        "prefill_kv_cache_byte_exact": (kv_cache_reports.get("prefill") or {}).get("byte_exact"),
        "prefill_kv_cache_first_difference": (kv_cache_reports.get("prefill") or {}).get("first_difference"),
        "decode_kv_cache_byte_exact": (kv_cache_reports.get("decode") or {}).get("byte_exact"),
        "decode_kv_cache_first_difference": (kv_cache_reports.get("decode") or {}).get("first_difference"),
        "prefill_probe_report": str(args.output_dir / "prefill_probe_report.json")
        if prefill_probe_report is not None else None,
        "prefill_probe_first_non_exact": prefill_probe_report.get("first_non_exact")
        if prefill_probe_report is not None else None,
    }
    _write_json(args.output_dir / "xray_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--call-ir", type=Path)
    parser.add_argument("--corpus-result", type=Path, required=True)
    parser.add_argument("--decoder-request", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="Extract visible form fields as compact JSON.")
    parser.add_argument("--position", type=int)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--prefill-probe-token", type=int)
    parser.add_argument(
        "--ck-prefix-source",
        choices=("certified", "pytorch"),
        default="certified",
        help="Visual prefix supplied to CK; PyTorch mode guarantees same-input decoder attribution.",
    )
    parser.add_argument("--reuse-existing-captures", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.runtime = args.runtime.resolve()
    args.call_ir = (args.call_ir or (args.runtime / "lowered_decode_call.json")).resolve()
    args.corpus_result = args.corpus_result.resolve()
    args.decoder_request = args.decoder_request.resolve()
    args.image = args.image.resolve()
    args.output_dir = args.output_dir.resolve()
    report = run(args)
    print(f"position={report['teacher_forced_position']}")
    print(f"strongest_layer={report['strongest_amplification_layer']}")
    print(f"first_non_exact={(report.get('granular_first_non_exact') or {}).get('checkpoint_id')}")
    print(f"first_material={(report.get('granular_first_material') or {}).get('checkpoint_id')}")
    print(f"report={args.output_dir / 'xray_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
