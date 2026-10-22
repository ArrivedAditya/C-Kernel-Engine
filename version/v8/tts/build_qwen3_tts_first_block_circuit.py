#!/usr/bin/env python3
"""Author one FP32 talker block with explicit, checked operation edges.

The initial block input is a pinned oracle tensor; the text-ID projection is
separately generated. This circuit does not yet construct role/codec embeddings
or claim complete synthesis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_text_projection_circuit import ROOT

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_first_block_bounded.json"
HIDDEN = 1024
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 3072
PREFIX = "talker.model.layers.0"


def operation(name, kind, kernel, inputs, outputs, *, weights=None,
              params=None, returns_status=True):
    value = {"id": name, "op": kind, "kernel": kernel,
             "returns_status": returns_status,
             "graph_slots": {"inputs": inputs, "outputs": outputs}}
    if weights:
        value["weight_refs"] = weights
    if params:
        value["params"] = params
    return value


def norm(name, source, output, width, rows, weight):
    return operation(name, "rmsnorm_rows_checked", "rmsnorm_rows_checked_f32",
                     {"input": source}, {"output": output},
                     weights={"gamma": f"{PREFIX}.{weight}"},
                     params={"R": rows, "C": width, "eps": 1e-6,
                             "call_constants": {
                         "input_elements": rows * width,
                         "input_stride": width, "gamma_elements": width,
                         "output_elements": rows * width,
                         "output_stride": width, "rows": rows,
                         "width": width}})


def linear(name, source, output, rows, input_width, output_width, weight):
    return operation(name, "linear_rows_checked",
                     "linear_rows_checked_no_bias_f32",
                     {"input": source}, {"output": output},
                     weights={"weight": f"{PREFIX}.{weight}"},
                     params={"M": rows, "K": input_width, "N": output_width,
                             "call_constants": {
                         "linear_input_elements": rows * input_width,
                         "linear_input_stride": input_width,
                         "linear_weight_elements": input_width * output_width,
                         "linear_weight_stride": input_width,
                         "linear_output_elements": rows * output_width,
                         "linear_output_stride": output_width,
                         "linear_rows": rows,
                         "linear_input_channels": input_width,
                         "linear_output_channels": output_width}})


def layout(name, source, output, tokens, heads, *, reverse=False):
    return operation(name, "token_head_layout_checked",
                     ("layout_head_to_token_checked_f32" if reverse else
                      "layout_token_to_head_checked_f32"),
                     {"input": source}, {"output": output},
                     params={"T": tokens, "H": heads, "D": HEAD_DIM,
                             "call_constants": {
                         "input_elements": tokens * heads * HEAD_DIM,
                         "output_elements": tokens * heads * HEAD_DIM,
                         "tokens": tokens, "heads": heads,
                         "head_dim": HEAD_DIM}})


def add(name, left, right, output, tokens):
    elements = tokens * HIDDEN
    return operation(name, "add_rows_checked", "add_rows_checked_f32",
                     {"left": ("external:block_input" if left == "block_input"
                               else left), "right": right}, {"output": output},
                     params={"N": elements, "call_constants": {
                         "left_elements": elements,
                         "right_elements": elements,
                         "output_elements": elements,
                         "elements": elements}})


def build_circuit(tokens=18):
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1 <= tokens <= 128:
        raise ValueError("first-block tokens must be an integer from 1 to 128")
    shapes = {
        "block_input": [tokens, HIDDEN],
        "input_norm": [tokens, HIDDEN],
        "q_proj": [tokens, HEADS * HEAD_DIM],
        "k_proj": [tokens, KV_HEADS * HEAD_DIM],
        "v_proj": [tokens, KV_HEADS * HEAD_DIM],
        "q_norm": [tokens, HEADS * HEAD_DIM],
        "k_norm": [tokens, KV_HEADS * HEAD_DIM],
        "q_head": [HEADS, tokens, HEAD_DIM],
        "k_head": [KV_HEADS, tokens, HEAD_DIM],
        "v_head": [KV_HEADS, tokens, HEAD_DIM],
        "attention_work": [HEADS, tokens, HEAD_DIM],
        "attention_head": [HEADS, tokens, HEAD_DIM],
        "attention_token": [tokens, HEADS * HEAD_DIM],
        "attention_proj": [tokens, HIDDEN],
        "post_attention": [tokens, HIDDEN],
        "post_attention_norm": [tokens, HIDDEN],
        "gate_proj": [tokens, INTERMEDIATE],
        "up_proj": [tokens, INTERMEDIATE],
        "gated_mlp": [tokens, INTERMEDIATE],
        "down_proj": [tokens, HIDDEN],
        "block_output": [tokens, HIDDEN],
    }
    ops = [
        norm("input_norm", "external:block_input", "input_norm", HIDDEN,
             tokens, "input_layernorm.weight"),
        linear("q_proj", "input_norm", "q_proj", tokens, HIDDEN,
               HEADS * HEAD_DIM, "self_attn.q_proj.weight"),
        linear("k_proj", "input_norm", "k_proj", tokens, HIDDEN,
               KV_HEADS * HEAD_DIM, "self_attn.k_proj.weight"),
        linear("v_proj", "input_norm", "v_proj", tokens, HIDDEN,
               KV_HEADS * HEAD_DIM, "self_attn.v_proj.weight"),
        norm("q_norm", "q_proj", "q_norm", HEAD_DIM,
             tokens * HEADS, "self_attn.q_norm.weight"),
        norm("k_norm", "k_proj", "k_norm", HEAD_DIM,
             tokens * KV_HEADS, "self_attn.k_norm.weight"),
        layout("q_head", "q_norm", "q_head", tokens, HEADS),
        layout("k_head", "k_norm", "k_head", tokens, KV_HEADS),
        layout("v_head", "v_proj", "v_head", tokens, KV_HEADS),
        operation("rope_qk", "rope_qk_bounded", "rope_qk_split_bounded_f32",
                  {"q": "q_head", "k": "k_head"},
                  {"q": "q_head", "k": "k_head"},
                  params={"H": HEADS, "KV": KV_HEADS, "T": tokens,
                          "D": HEAD_DIM, "freq_base": 1e6,
                          "call_constants": {
                              "q_elements": HEADS * tokens * HEAD_DIM,
                              "k_elements": KV_HEADS * tokens * HEAD_DIM,
                              "heads": HEADS, "kv_heads": KV_HEADS,
                              "tokens": tokens, "head_dim": HEAD_DIM,
                              "pos_offset": 0}}),
        operation("attention_head", "attention_causal_bounded",
                  "attention_causal_head_major_bounded_f32",
                  {"q": "q_head", "k": "k_head", "v": "v_head"},
                  {"work": "attention_work", "out": "attention_head"},
                  params={"H": HEADS, "KV": KV_HEADS, "T": tokens,
                          "D": HEAD_DIM, "call_constants": {
                              "q_elements": HEADS * tokens * HEAD_DIM,
                              "k_elements": KV_HEADS * tokens * HEAD_DIM,
                              "v_elements": KV_HEADS * tokens * HEAD_DIM,
                              "work_elements": HEADS * tokens * HEAD_DIM,
                              "output_elements": HEADS * tokens * HEAD_DIM,
                              "heads": HEADS, "kv_heads": KV_HEADS,
                              "tokens": tokens, "head_dim": HEAD_DIM}}),
        layout("attention_token", "attention_head", "attention_token",
               tokens, HEADS, reverse=True),
        linear("attention_proj", "attention_token", "attention_proj",
               tokens, HEADS * HEAD_DIM, HIDDEN,
               "self_attn.o_proj.weight"),
        add("post_attention", "block_input", "attention_proj",
            "post_attention", tokens),
        norm("post_attention_norm", "post_attention",
             "post_attention_norm", HIDDEN, tokens,
             "post_attention_layernorm.weight"),
        linear("gate_proj", "post_attention_norm", "gate_proj", tokens,
               HIDDEN, INTERMEDIATE, "mlp.gate_proj.weight"),
        linear("up_proj", "post_attention_norm", "up_proj", tokens,
               HIDDEN, INTERMEDIATE, "mlp.up_proj.weight"),
        operation("gated_mlp", "silu_mul_rows_checked",
                  "silu_mul_rows_checked_f32",
                  {"gate": "gate_proj", "up": "up_proj"},
                  {"output": "gated_mlp"},
                  params={"N": tokens * INTERMEDIATE, "call_constants": {
                      "gate_elements": tokens * INTERMEDIATE,
                      "up_elements": tokens * INTERMEDIATE,
                      "output_elements": tokens * INTERMEDIATE,
                      "elements": tokens * INTERMEDIATE}}),
        linear("down_proj", "gated_mlp", "down_proj", tokens,
               INTERMEDIATE, HIDDEN, "mlp.down_proj.weight"),
        add("block_output", "post_attention", "down_proj",
            "block_output", tokens),
    ]
    return {
        "version": 3, "name": "qwen3_tts_first_block_bounded",
        "family": "bounded_graph", "checked_native_entry": True,
        "contract": {"runtime_invariants": {
            "inference_only": True, "production_kernel_heap_allocation": False,
            "tokens": tokens, "oracle_fed_block_input": True,
            "complete_talker": False, "complete_synthesis": False}},
        "activation_buffers": {name: {"shape": shape}
                               for name, shape in shapes.items()},
        "activation_bindings": {name: name for name in shapes},
        "native_entry": {"function": "ck_qwen3_tts_first_block_bounded",
                         "params": [{"c_type": "uint8_t *", "name": "arena"},
                                    {"c_type": "size_t", "name": "arena_bytes"}],
                         "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
        "sequence": ["first_block"],
        "block_types": {"first_block": {
            "sequence": ["header", "body", "footer"], "header": [],
            "body": {"type": "dense", "ops": ops}, "footer": []}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("first talker block circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
