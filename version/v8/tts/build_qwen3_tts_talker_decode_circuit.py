#!/usr/bin/env python3
"""Declare one checked Qwen3-TTS talker continuation step.

The caller supplies the next input embedding and persistent per-layer KV state.
This graph owns all 28 neural layers and the output head; autonomous token and
codebook generation are separate milestones.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_first_block_circuit import build_circuit as first_block
    from .build_qwen3_tts_talker_prefill_circuit import ROOT, LAYERS, HIDDEN, CODEC_VOCAB
else:
    from build_qwen3_tts_first_block_circuit import build_circuit as first_block
    from build_qwen3_tts_talker_prefill_circuit import ROOT, LAYERS, HIDDEN, CODEC_VOCAB

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_talker_decode_bounded.json"
PROMPT_ROWS = 18
CACHE_CAPACITY = 21
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128


def build_circuit(*, layers: int = LAYERS) -> dict:
    if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= LAYERS:
        raise ValueError("layers must be in 1..28")
    graph = first_block(1)
    graph["name"] = "qwen3_tts_talker_decode_bounded" if layers == LAYERS else f"qwen3_tts_talker_decode_{layers}_layer_fixture"
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update({
        "oracle_fed_block_input": True,
        "valid_decode_positions": [PROMPT_ROWS, CACHE_CAPACITY - 1],
        "cache_capacity": CACHE_CAPACITY,
        "complete_talker_decode": layers == LAYERS,
        "autonomous_code_generation": False,
    })
    graph["runtime_constants"] = {
        "value_elements": 1, "phoneme_count": 1,
        "max_duration": CACHE_CAPACITY - PROMPT_ROWS - 1,
        "expanded_capacity": CACHE_CAPACITY - PROMPT_ROWS - 1,
    }
    graph["runtime_lengths"] = {
        "step_index": {"producer": "check_step_index", "result": "valid_extent",
                       "capacity": CACHE_CAPACITY - PROMPT_ROWS - 1, "allow_zero": True},
        "decode_position": {"producer": "check_position", "result": "valid_extent",
                     "capacity": CACHE_CAPACITY - 1, "allow_zero": False},
    }
    graph["activation_buffers"]["step_index_input"] = {"shape": [1]}
    graph["activation_buffers"]["step_index_checked"] = {"shape": [1]}
    graph["activation_buffers"]["position_checked"] = {"shape": [1]}
    for name in ("step_index_input", "step_index_checked", "position_checked"):
        graph["activation_bindings"][name] = name
    block = graph["block_types"]["first_block"]
    block["header"] = [
        {"id": "check_step_index", "op": "runtime_extent_sum",
         "kernel": "runtime_extent_sum_i32", "returns_status": True,
         "produces_runtime_lengths": {"step_index": "valid_extent"},
         "graph_slots": {"inputs": {"values": "external:step_index_input"},
                         "outputs": {"valid_extent": "step_index_checked"}}},
        {"id": "check_position", "op": "runtime_extent_affine",
         "kernel": "runtime_extent_affine_i32", "returns_status": True,
         "consumes_runtime_lengths": ["step_index"],
         "produces_runtime_lengths": {"decode_position": "valid_extent"},
         "runtime_scalar_bindings": {"extent_affine_source": "step_index"},
         "params": {"call_constants": {
             "extent_affine_factor": 1, "extent_affine_offset": PROMPT_ROWS,
             "extent_affine_capacity": CACHE_CAPACITY - 1}},
         "graph_slots": {"inputs": {}, "outputs": {"valid_extent": "position_checked"}}},
    ]
    original = copy.deepcopy(block["body"]["ops"])
    original_outputs = sorted({value for op in original for value in op["graph_slots"]["outputs"].values()})
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    previous = "block_output"
    for layer in range(layers):
        prefix = "" if layer == 0 else f"layer{layer}_"
        ops = copy.deepcopy(original)
        if layer == 0:
            block["body"]["ops"] = ops
        if layer:
            for name in original_outputs:
                shapes[prefix + name] = copy.deepcopy(shapes[name])
                bindings[prefix + name] = prefix + name
            for op in ops:
                op["id"] = prefix + op["id"]
                op["graph_slots"]["inputs"] = {
                    port: (previous if value in ("block_input", "external:block_input") else
                           prefix + value if value in original_outputs else value)
                    for port, value in op["graph_slots"]["inputs"].items()}
                op["graph_slots"]["outputs"] = {
                    port: (prefix + value if value in original_outputs else value)
                    for port, value in op["graph_slots"]["outputs"].items()}
                if op.get("weight_refs"):
                    op["weight_refs"] = {port: name.replace(
                        "talker.model.layers.0.", f"talker.model.layers.{layer}.")
                        for port, name in op["weight_refs"].items()}
        cache_k = f"talker_layer{layer}_cache_k"
        cache_v = f"talker_layer{layer}_cache_v"
        for name in (cache_k, cache_v):
            shapes[name] = {"shape": [KV_HEADS, CACHE_CAPACITY, HEAD_DIM]}
            bindings[name] = name
        shapes[prefix + "attention_decode_scores"] = {"shape": [HEADS, CACHE_CAPACITY]}
        shapes[prefix + "attention_decode_staged"] = {"shape": [HEADS, HEAD_DIM]}
        bindings[prefix + "attention_decode_scores"] = prefix + "attention_decode_scores"
        bindings[prefix + "attention_decode_staged"] = prefix + "attention_decode_staged"
        rope = next(op for op in ops if op["op"] == "rope_qk_bounded")
        rope["params"]["call_constants"].pop("pos_offset")
        rope["consumes_runtime_lengths"] = ["decode_position"]
        rope["runtime_scalar_bindings"] = {"pos_offset": "decode_position"}
        attention = next(op for op in ops if op["op"] == "attention_causal_bounded")
        attention.update({
            "op": "attention_decode_append_checked",
            "kernel": "attention_decode_append_checked_f32",
            "consumes_runtime_lengths": ["decode_position"],
            "runtime_scalar_bindings": {"position": "decode_position"},
            "graph_slots": {"inputs": {
                "q": prefix + "q_head", "k_new": prefix + "k_head",
                "v_new": prefix + "v_head", "cache_k": "external:" + cache_k,
                "cache_v": "external:" + cache_v},
                "outputs": {"work_scores": prefix + "attention_decode_scores",
                            "staged": prefix + "attention_decode_staged",
                            "out": prefix + "attention_head",
                            "cache_k": "external:" + cache_k,
                            "cache_v": "external:" + cache_v}},
            "params": {"H": HEADS, "KV": KV_HEADS, "S": CACHE_CAPACITY, "D": HEAD_DIM,
                       "call_constants": {
                           "q_elements": HEADS * HEAD_DIM,
                           "k_new_elements": KV_HEADS * HEAD_DIM,
                           "v_new_elements": KV_HEADS * HEAD_DIM,
                           "k_cache_elements": KV_HEADS * CACHE_CAPACITY * HEAD_DIM,
                           "v_cache_elements": KV_HEADS * CACHE_CAPACITY * HEAD_DIM,
                           "score_elements": HEADS * CACHE_CAPACITY,
                           "staged_output_elements": HEADS * HEAD_DIM,
                           "output_elements": HEADS * HEAD_DIM,
                           "heads": HEADS, "kv_heads": KV_HEADS,
                           "cache_capacity": CACHE_CAPACITY, "head_dim": HEAD_DIM}}
        })
        if layer:
            block_name = f"talker_layer_{layer}"
            graph["sequence"].append(block_name)
            graph["block_types"][block_name] = {
                "sequence": ["header", "body", "footer"], "header": [],
                "body": {"type": "dense", "ops": ops}, "footer": []}
        previous = prefix + "block_output"
    shapes["talker_final_norm"] = {"shape": [1, HIDDEN]}
    shapes["talker_logits"] = {"shape": [1, CODEC_VOCAB]}
    bindings["talker_final_norm"] = "talker_final_norm"
    bindings["talker_logits"] = "talker_logits"
    graph["sequence"].append("talker_head")
    graph["block_types"]["talker_head"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": [
            {"id": "talker_final_norm", "op": "rmsnorm_rows_checked",
             "kernel": "rmsnorm_rows_checked_f32", "returns_status": True,
             "weight_refs": {"gamma": "talker.model.norm.weight"},
             "graph_slots": {"inputs": {"input": previous},
                             "outputs": {"output": "talker_final_norm"}},
             "params": {"R": 1, "C": HIDDEN, "eps": 1e-6,
                        "call_constants": {"input_elements": HIDDEN,
                            "input_stride": HIDDEN, "gamma_elements": HIDDEN,
                            "output_elements": HIDDEN, "output_stride": HIDDEN,
                            "rows": 1, "width": HIDDEN}}},
            {"id": "talker_logits", "op": "linear_rows_checked_no_bias",
             "kernel": "linear_rows_checked_no_bias_f32", "returns_status": True,
             "weight_refs": {"weight": "talker.codec_head.weight"},
             "graph_slots": {"inputs": {"input": "talker_final_norm"},
                             "outputs": {"output": "talker_logits"}},
             "params": {"M": 1, "K": HIDDEN, "N": CODEC_VOCAB,
                        "call_constants": {"linear_input_elements": HIDDEN,
                            "linear_input_stride": HIDDEN,
                            "linear_weight_elements": HIDDEN * CODEC_VOCAB,
                            "linear_weight_stride": HIDDEN,
                            "linear_output_elements": CODEC_VOCAB,
                            "linear_output_stride": CODEC_VOCAB,
                            "linear_rows": 1, "linear_input_channels": HIDDEN,
                            "linear_output_channels": CODEC_VOCAB}}},
        ]}, "footer": []}
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("talker decode circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
