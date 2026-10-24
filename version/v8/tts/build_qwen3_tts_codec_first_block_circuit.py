#!/usr/bin/env python3
"""Extend the connected codec prefix through its first complete transformer block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_attention_circuit import build_circuit as attention_circuit
from build_qwen3_tts_codec_qkv_circuit import PREFIX
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_first_block_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024) -> dict:
    if (isinstance(intermediate_width, bool) or
            not isinstance(intermediate_width, int) or intermediate_width < 1):
        raise ValueError("intermediate width must be a positive integer")
    graph = attention_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window)
    graph["name"] = "qwen3_tts_codec_first_block_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_first_block_bounded"
    graph["contract"]["runtime_invariants"]["first_codec_block_complete"] = True
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
    exports = graph["semantic_checkpoints"]["exports"]
    for name, width in (
        ("first_attention_token", heads * head_dim),
        ("first_attention_proj", hidden_width),
        ("first_attention_scaled", hidden_width),
        ("first_post_attention", hidden_width),
        ("first_post_attention_norm", hidden_width),
        ("first_gate", intermediate_width),
        ("first_up", intermediate_width),
        ("first_gated_mlp", intermediate_width),
        ("first_down", hidden_width),
        ("first_mlp_scaled", hidden_width),
        ("first_block_output", hidden_width),
    ):
        shapes[name] = {"shape": [frames, width]}
        bindings[name] = name

    def append(name, kind, kernel, inputs, outputs, *, weights=None,
               params=None, layout="token_major"):
        op = {"id": name, "op": kind, "kernel": kernel,
              "returns_status": True,
              "graph_slots": {"inputs": inputs, "outputs": outputs}}
        if weights:
            op["weight_refs"] = weights
        if params:
            op["params"] = params
        ops.append(op)
        exports[name] = checkpoint(op, layout)

    def linear(name, source, input_width, result_width, weight):
        append(name, "linear_rows_checked_no_bias",
               "linear_rows_checked_no_bias_f32", {"input": source},
               {"output": name},
               weights={"weight": f"{PREFIX}.layers.0.{weight}.weight"},
               params={"M": frames, "K": input_width, "N": result_width,
                       "call_constants": {
                   "linear_input_elements": frames * input_width,
                   "linear_input_stride": input_width,
                   "linear_weight_elements": result_width * input_width,
                   "linear_weight_stride": input_width,
                   "linear_output_elements": frames * result_width,
                   "linear_output_stride": result_width,
                   "linear_rows": frames,
                   "linear_input_channels": input_width,
                   "linear_output_channels": result_width}})

    def scale(name, source, weight):
        append(name, "channel_scale_rows_checked",
               "channel_scale_rows_checked_f32", {"input": source},
               {"output": name},
               weights={"scale": f"{PREFIX}.layers.0.{weight}.scale"},
               params={"R": frames, "C": hidden_width,
                       "call_constants": {
                   "input_elements": frames * hidden_width,
                   "input_stride": hidden_width,
                   "scale_elements": hidden_width,
                   "output_elements": frames * hidden_width,
                   "output_stride": hidden_width,
                   "rows": frames, "channels": hidden_width}})

    def add(name, left, right):
        count = frames * hidden_width
        append(name, "add_rows_checked", "add_rows_checked_f32",
               {"left": left, "right": right}, {"output": name},
               params={"N": count, "call_constants": {
                   "left_elements": count, "right_elements": count,
                   "output_elements": count, "elements": count}})

    append("first_attention_token", "token_head_layout_checked",
           "layout_head_to_token_checked_f32",
           {"input": "first_attention_head"},
           {"output": "first_attention_token"},
           params={"T": frames, "H": heads, "D": head_dim,
                   "call_constants": {
               "input_elements": frames * heads * head_dim,
               "output_elements": frames * heads * head_dim,
               "tokens": frames, "heads": heads, "head_dim": head_dim}})
    linear("first_attention_proj", "first_attention_token", heads * head_dim,
           hidden_width, "self_attn.o_proj")
    scale("first_attention_scaled", "first_attention_proj",
          "self_attn_layer_scale")
    add("first_post_attention", "transformer_input",
        "first_attention_scaled")
    append("first_post_attention_norm", "rmsnorm_rows_checked",
           "rmsnorm_rows_checked_f32", {"input": "first_post_attention"},
           {"output": "first_post_attention_norm"},
           weights={"gamma": f"{PREFIX}.layers.0.post_attention_layernorm.weight"},
           params={"R": frames, "C": hidden_width, "eps": 1e-5,
                   "call_constants": {
               "input_elements": frames * hidden_width,
               "input_stride": hidden_width,
               "gamma_elements": hidden_width,
               "output_elements": frames * hidden_width,
               "output_stride": hidden_width,
               "rows": frames, "width": hidden_width}})
    linear("first_gate", "first_post_attention_norm", hidden_width,
           intermediate_width, "mlp.gate_proj")
    linear("first_up", "first_post_attention_norm", hidden_width,
           intermediate_width, "mlp.up_proj")
    count = frames * intermediate_width
    append("first_gated_mlp", "silu_mul_rows_checked",
           "silu_mul_rows_checked_f32", {"gate": "first_gate", "up": "first_up"},
           {"output": "first_gated_mlp"},
           params={"N": count, "call_constants": {
               "gate_elements": count, "up_elements": count,
               "output_elements": count, "elements": count}})
    linear("first_down", "first_gated_mlp", intermediate_width,
           hidden_width, "mlp.down_proj")
    scale("first_mlp_scaled", "first_down", "mlp_layer_scale")
    add("first_block_output", "first_post_attention", "first_mlp_scaled")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec first-block circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
