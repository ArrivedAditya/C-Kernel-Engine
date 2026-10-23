#!/usr/bin/env python3
"""Author the connected codec pre-transformer ingress through first-layer Q/K/V RoPE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_preconv_circuit import build_circuit as preconv_circuit
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_qkv_bounded.json"
PREFIX = "codec.decoder.pre_transformer"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64) -> dict:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (hidden_width, heads, head_dim)):
        raise ValueError("hidden width, heads and head dimension must be positive integers")
    if head_dim % 2:
        raise ValueError("Q/K/V geometry requires an even head dimension")
    projected_width = heads * head_dim
    graph = preconv_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width)
    graph["name"] = "qwen3_tts_codec_qkv_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_qkv_bounded"
    graph["contract"]["runtime_invariants"]["pretransformer_qkv_rope"] = True
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
    exports = graph["semantic_checkpoints"]["exports"]
    for name, shape in (
        ("preconv_token", [frames, latent_width]),
        ("transformer_input", [frames, hidden_width]),
        ("first_input_norm", [frames, hidden_width]),
        ("first_q", [frames, projected_width]),
        ("first_k", [frames, projected_width]),
        ("first_v", [frames, projected_width]),
        ("first_q_head", [heads, frames, head_dim]),
        ("first_k_head", [heads, frames, head_dim]),
        ("first_v_head", [heads, frames, head_dim]),
    ):
        shapes[name] = {"shape": shape}
        bindings[name] = name

    def append(op: dict, layout: str = "token_major") -> None:
        ops.append(op)
        contract = checkpoint(op, layout)
        if layout == "head_major":
            for item in contract["checkpoints"]:
                item["axis_names"] = ["head", "frame", "channel"]
        if op["id"] == "first_rope_qk":
            contract["checkpoints"].append({
                **contract["checkpoints"][0],
                "id": "qwen3_tts.codec.first_rope_k",
                "tensor": "first_k_head",
            })
        exports[op["id"]] = contract

    def linear(name: str, source: str, input_width: int, output_width: int,
               weight: str, *, bias: bool = False) -> None:
        refs = {"weight": f"{PREFIX}.{weight}.weight"}
        constants = {
            "linear_input_elements": frames * input_width,
            "linear_input_stride": input_width,
            "linear_weight_elements": output_width * input_width,
            "linear_weight_stride": input_width,
            "linear_output_elements": frames * output_width,
            "linear_output_stride": output_width,
            "linear_rows": frames,
            "linear_input_channels": input_width,
            "linear_output_channels": output_width,
        }
        if bias:
            refs["bias"] = f"{PREFIX}.{weight}.bias"
            constants["linear_bias_elements"] = output_width
        append({
            "id": name,
            "op": "linear_rows_checked" if bias else "linear_rows_checked_no_bias",
            "kernel": "linear_rows_checked_f32" if bias else
                      "linear_rows_checked_no_bias_f32",
            "returns_status": True,
            "weight_refs": refs,
            "graph_slots": {"inputs": {"input": source},
                            "outputs": {"output": name}},
            "params": {"M": frames, "K": input_width, "N": output_width,
                       "call_constants": constants},
        })

    append({
        "id": "preconv_to_tokens", "op": "transpose_strided_checked",
        "kernel": "transpose_strided_f32_checked", "returns_status": True,
        "graph_slots": {"inputs": {"input": "preconv_output"},
                        "outputs": {"output": "preconv_token"}},
        "params": {"R": latent_width, "C": frames, "call_constants": {
            "transpose_input_elements": frames * latent_width,
            "transpose_input_stride": frames,
            "transpose_output_elements": frames * latent_width,
            "transpose_output_stride": latent_width,
            "transpose_rows": latent_width,
            "transpose_columns": frames}},
    })
    linear("transformer_input", "preconv_token", latent_width, hidden_width,
           "input_proj", bias=True)
    append({
        "id": "first_input_norm", "op": "rmsnorm_rows_checked",
        "kernel": "rmsnorm_rows_checked_f32", "returns_status": True,
        "weight_refs": {
            "gamma": f"{PREFIX}.layers.0.input_layernorm.weight"},
        "graph_slots": {"inputs": {"input": "transformer_input"},
                        "outputs": {"output": "first_input_norm"}},
        "params": {"R": frames, "C": hidden_width, "eps": 1e-5,
                   "call_constants": {
            "input_elements": frames * hidden_width,
            "input_stride": hidden_width,
            "gamma_elements": hidden_width,
            "output_elements": frames * hidden_width,
            "output_stride": hidden_width,
            "rows": frames, "width": hidden_width}},
    })
    for name in ("q", "k", "v"):
        linear(f"first_{name}", "first_input_norm", hidden_width,
               projected_width, f"layers.0.self_attn.{name}_proj")
        append({
            "id": f"first_{name}_head", "op": "token_head_layout_checked",
            "kernel": "layout_token_to_head_checked_f32",
            "returns_status": True,
            "graph_slots": {"inputs": {"input": f"first_{name}"},
                            "outputs": {"output": f"first_{name}_head"}},
            "params": {"T": frames, "H": heads, "D": head_dim,
                       "call_constants": {
                "input_elements": frames * projected_width,
                "output_elements": frames * projected_width,
                "tokens": frames, "heads": heads, "head_dim": head_dim}},
        }, "head_major")
    append({
        "id": "first_rope_qk", "op": "rope_qk_bounded",
        "kernel": "rope_qk_split_bounded_f32", "returns_status": True,
        "graph_slots": {"inputs": {"q": "first_q_head", "k": "first_k_head"},
                        "outputs": {"q": "first_q_head", "k": "first_k_head"}},
        "params": {"H": heads, "KV": heads, "T": frames,
                   "D": head_dim, "freq_base": 10000.0,
                   "call_constants": {
            "q_elements": frames * projected_width,
            "k_elements": frames * projected_width,
            "heads": heads, "kv_heads": heads,
            "tokens": frames, "head_dim": head_dim, "pos_offset": 0}},
    }, "head_major")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec Q/K/V circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
