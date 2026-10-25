#!/usr/bin/env python3
"""Declare the complete 28-layer Qwen3-TTS talker prompt prefill.

This repeats the already certified first-block operation graph with distinct
weight bindings and checkpoint IDs. It does not implement continuation cache,
code prediction, EOS or autonomous speech.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_prompt_first_block_circuit import (
        HIDDEN, ROOT, build_circuit as prompt_first_block,
    )
else:
    from build_qwen3_tts_prompt_first_block_circuit import (
        HIDDEN, ROOT, build_circuit as prompt_first_block,
    )

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_talker_prefill_bounded.json"
LAYERS = 28
CODEC_VOCAB = 3072


def build_circuit(*, context_tokens: int = 7, layers: int = LAYERS) -> dict:
    if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= LAYERS:
        raise ValueError("talker layers must be an integer from 1 to 28")
    graph = prompt_first_block(context_tokens)
    rows = context_tokens + 11
    graph["name"] = ("qwen3_tts_talker_prefill_bounded" if layers == LAYERS else
                     f"qwen3_tts_talker_prefill_{layers}_layer_fixture")
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update({
        "talker_layers": layers,
        "complete_talker_prefill": layers == LAYERS,
        "continuation_cache": False,
        "autonomous_code_generation": False,
    })
    original = graph["block_types"]["first_block"]["body"]["ops"]
    original_outputs = sorted({
        value for op in original
        for value in op["graph_slots"]["outputs"].values()
    })
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    exports = graph["semantic_checkpoints"]["exports"]
    previous = "block_output"
    for layer in range(1, layers):
        prefix = f"layer{layer}_"
        for name in original_outputs:
            target = prefix + name
            shapes[target] = copy.deepcopy(shapes[name])
            bindings[target] = target
        cloned = copy.deepcopy(original)
        for op in cloned:
            old_id = op["id"]
            op["id"] = prefix + old_id
            op["graph_slots"]["inputs"] = {
                port: (previous if value == "block_input" else
                       prefix + value if value in original_outputs else value)
                for port, value in op["graph_slots"]["inputs"].items()
            }
            op["graph_slots"]["outputs"] = {
                port: prefix + value
                for port, value in op["graph_slots"]["outputs"].items()
            }
            op["weight_refs"] = {
                port: name.replace("talker.model.layers.0.",
                                   f"talker.model.layers.{layer}.")
                for port, name in op.get("weight_refs", {}).items()
            } if op.get("weight_refs") else op.get("weight_refs", {})
            if not op["weight_refs"]:
                op.pop("weight_refs", None)
        block_name = f"talker_layer_{layer}"
        graph["sequence"].append(block_name)
        graph["block_types"][block_name] = {
            "sequence": ["header", "body", "footer"],
            "header": [], "body": {"type": "dense", "ops": cloned},
            "footer": [],
        }
        previous = prefix + "block_output"
        exports[previous] = {
            "section": "body", "template_op_id": previous,
            "op": "add_rows_checked",
            "checkpoints": [{"id": f"qwen3_tts.talker.{previous}",
                             "producer": previous, "tensor": previous,
                             "logical_layout": "token_major",
                             "axis_names": ["token", "channel"],
                             "storage_dtype": "fp32"}],
        }

    shapes["talker_final_norm"] = {"shape": [rows, HIDDEN]}
    shapes["talker_logits"] = {"shape": [rows, CODEC_VOCAB]}
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
             "params": {"R": rows, "C": HIDDEN, "eps": 1e-6,
                        "call_constants": {
                            "input_elements": rows * HIDDEN,
                            "input_stride": HIDDEN,
                            "gamma_elements": HIDDEN,
                            "output_elements": rows * HIDDEN,
                            "output_stride": HIDDEN,
                            "rows": rows, "width": HIDDEN}}},
            {"id": "talker_logits", "op": "linear_rows_checked_no_bias",
             "kernel": "linear_rows_checked_no_bias_f32",
             "returns_status": True,
             "weight_refs": {"weight": "talker.codec_head.weight"},
             "graph_slots": {"inputs": {"input": "talker_final_norm"},
                             "outputs": {"output": "talker_logits"}},
             "params": {"M": rows, "K": HIDDEN, "N": CODEC_VOCAB,
                        "call_constants": {
                            "linear_input_elements": rows * HIDDEN,
                            "linear_input_stride": HIDDEN,
                            "linear_weight_elements": HIDDEN * CODEC_VOCAB,
                            "linear_weight_stride": HIDDEN,
                            "linear_output_elements": rows * CODEC_VOCAB,
                            "linear_output_stride": CODEC_VOCAB,
                            "linear_rows": rows,
                            "linear_input_channels": HIDDEN,
                            "linear_output_channels": CODEC_VOCAB}}},
        ]}, "footer": [],
    }
    for name, kind in (
        ("talker_final_norm", "rmsnorm_rows_checked"),
        ("talker_logits", "linear_rows_checked_no_bias"),
    ):
        exports[name] = {
            "section": "body", "template_op_id": name, "op": kind,
            "checkpoints": [{"id": f"qwen3_tts.talker.{name}",
                             "producer": name, "tensor": name,
                             "logical_layout": "token_major",
                             "axis_names": ["token", "channel"],
                             "storage_dtype": "fp32"}],
        }
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("talker prefill circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
