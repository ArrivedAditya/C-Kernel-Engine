#!/usr/bin/env python3
"""Five-layer Qwen3-TTS code-predictor prefill with checked KV seeding."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_first_block_circuit import (
        build_circuit as first_block, norm, linear,
    )
    from .build_qwen3_tts_talker_prefill_circuit import ROOT, HIDDEN
else:
    from build_qwen3_tts_first_block_circuit import (
        build_circuit as first_block, norm, linear,
    )
    from build_qwen3_tts_talker_prefill_circuit import ROOT, HIDDEN

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_code_predictor_prefill_bounded.json"
LAYERS, ROWS, CACHE_CAPACITY = 5, 2, 16
KV_HEADS, HEAD_DIM, CODE_VOCAB = 8, 128, 2048


def build_circuit(*, layers: int = LAYERS) -> dict:
    if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= LAYERS:
        raise ValueError("code-predictor layers must be in 1..5")
    graph = first_block(ROWS)
    graph["name"] = ("qwen3_tts_code_predictor_prefill_bounded" if layers == LAYERS
                     else f"qwen3_tts_code_predictor_prefill_{layers}_layer_fixture")
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update({
        "predictor_rows": ROWS, "predictor_layers": layers,
        "cache_capacity": CACHE_CAPACITY, "oracle_fed_initial_embeddings": True,
        "autonomous_codebook_generation": False,
    })
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]
    original = copy.deepcopy(graph["block_types"]["first_block"]["body"]["ops"])
    outputs = sorted({value for op in original
                      for value in op["graph_slots"]["outputs"].values()})
    previous = "block_output"
    for layer in range(layers):
        prefix = "" if layer == 0 else f"layer{layer}_"
        ops = copy.deepcopy(original)
        if layer:
            for name in outputs:
                shapes[prefix + name] = copy.deepcopy(shapes[name])
                bindings[prefix + name] = prefix + name
        for op in ops:
            if layer:
                op["id"] = prefix + op["id"]
                op["graph_slots"]["inputs"] = {
                    port: (previous if value in ("block_input", "external:block_input")
                           else prefix + value if value in outputs else value)
                    for port, value in op["graph_slots"]["inputs"].items()}
                op["graph_slots"]["outputs"] = {
                    port: prefix + value if value in outputs else value
                    for port, value in op["graph_slots"]["outputs"].items()}
            if op.get("weight_refs"):
                op["weight_refs"] = {
                    port: name.replace("talker.model.layers.0.",
                                       f"talker.code_predictor.model.layers.{layer}.")
                    for port, name in op["weight_refs"].items()}
        if layer:
            name = f"predictor_layer_{layer}"
            graph["sequence"].append(name)
            graph["block_types"][name] = {
                "sequence": ["header", "body", "footer"], "header": [],
                "body": {"type": "dense", "ops": ops}, "footer": []}
        else:
            graph["block_types"]["first_block"]["body"]["ops"] = ops
        previous = prefix + "block_output"

    shapes["predictor_final_norm"] = {"shape": [ROWS, HIDDEN]}
    shapes["predictor_codebook0_logits"] = {"shape": [ROWS, CODE_VOCAB]}
    bindings["predictor_final_norm"] = "predictor_final_norm"
    bindings["predictor_codebook0_logits"] = "predictor_codebook0_logits"
    final_norm = norm("predictor_final_norm", previous,
                      "predictor_final_norm", HIDDEN, ROWS, "unused")
    final_norm["weight_refs"]["gamma"] = "talker.code_predictor.model.norm.weight"
    head = linear("predictor_codebook0_logits", "predictor_final_norm",
                  "predictor_codebook0_logits", ROWS, HIDDEN, CODE_VOCAB,
                  "unused")
    head["weight_refs"]["weight"] = "talker.code_predictor.lm_head.0.weight"
    graph["sequence"].append("predictor_head")
    graph["block_types"]["predictor_head"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": [final_norm, head]}, "footer": []}

    seed_ops = []
    for layer in range(layers):
        prefix = "" if layer == 0 else f"layer{layer}_"
        cache_k = f"predictor_layer{layer}_cache_k"
        cache_v = f"predictor_layer{layer}_cache_v"
        for name in (cache_k, cache_v):
            shapes[name] = {"shape": [KV_HEADS, CACHE_CAPACITY, HEAD_DIM]}
            bindings[name] = name
        seed_ops.append({
            "id": f"predictor_layer{layer}_cache_seed",
            "op": "kv_cache_seed_checked", "kernel": "kv_cache_seed_checked_f32",
            "returns_status": True,
            "graph_slots": {"inputs": {"k": prefix + "k_head", "v": prefix + "v_head",
                                      "cache_k": "external:" + cache_k,
                                      "cache_v": "external:" + cache_v},
                            "outputs": {"cache_k": "external:" + cache_k,
                                        "cache_v": "external:" + cache_v}},
            "params": {"KV": KV_HEADS, "T": ROWS, "S": CACHE_CAPACITY,
                       "D": HEAD_DIM, "call_constants": {
                           "k_elements": KV_HEADS * ROWS * HEAD_DIM,
                           "v_elements": KV_HEADS * ROWS * HEAD_DIM,
                           "k_cache_elements": KV_HEADS * CACHE_CAPACITY * HEAD_DIM,
                           "v_cache_elements": KV_HEADS * CACHE_CAPACITY * HEAD_DIM,
                           "kv_heads": KV_HEADS, "tokens": ROWS,
                           "cache_capacity": CACHE_CAPACITY, "head_dim": HEAD_DIM}},
        })
    graph["sequence"].append("predictor_cache_seed")
    graph["block_types"]["predictor_cache_seed"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": seed_ops}, "footer": []}
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("predictor prefill circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
