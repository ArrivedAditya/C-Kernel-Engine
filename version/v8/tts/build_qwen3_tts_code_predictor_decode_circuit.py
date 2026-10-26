#!/usr/bin/env python3
"""One checked code-predictor continuation call through normal generated C.

The caller supplies a projected embedding and the per-frame KV state. The
codebook index selects a declared head weight; this graph does not implement
sampling or autonomously feed its selected code into the next call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_talker_decode_circuit import build_circuit as talker_decode
    from .build_qwen3_tts_talker_prefill_circuit import ROOT
else:
    from build_qwen3_tts_talker_decode_circuit import build_circuit as talker_decode
    from build_qwen3_tts_talker_prefill_circuit import ROOT

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_code_predictor_decode_bounded.json"
LAYERS, CACHE_CAPACITY = 5, 16
KV_HEADS, HEAD_DIM, CODE_VOCAB = 8, 128, 2048


def build_circuit(*, codebook_index: int = 1, layers: int = LAYERS) -> dict:
    if isinstance(codebook_index, bool) or not isinstance(codebook_index, int) or not 1 <= codebook_index <= 14:
        raise ValueError("continuation codebook index must be in 1..14")
    graph = talker_decode(layers=layers)
    graph["name"] = ("qwen3_tts_code_predictor_decode_bounded" if
                     codebook_index == 1 and layers == LAYERS else
                     f"qwen3_tts_code_predictor_decode_head{codebook_index}_{layers}_layers")
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update({
        "predictor_layers": layers, "codebook_index": codebook_index,
        "cache_capacity": CACHE_CAPACITY, "initial_cache_rows": 2,
        "oracle_fed_continuation_embedding": True,
        "autonomous_codebook_generation": False,
    })
    graph["contract"]["runtime_invariants"].pop("complete_talker_decode", None)
    graph["contract"]["runtime_invariants"].pop("autonomous_code_generation", None)
    graph["contract"]["runtime_invariants"].pop("valid_decode_positions", None)
    graph["contract"]["runtime_invariants"]["valid_decode_positions"] = [2, 15]
    graph["runtime_constants"].update({
        "max_duration": 14, "expanded_capacity": 14,
    })
    graph["runtime_lengths"]["step_index"]["capacity"] = 14
    graph["runtime_lengths"]["decode_position"]["capacity"] = 15
    header = graph["block_types"]["first_block"]["header"]
    header[1]["params"]["call_constants"].update({
        "extent_affine_offset": 2, "extent_affine_capacity": 15,
    })

    def rename(value):
        if isinstance(value, str):
            return (value.replace("talker.model.layers.", "talker.code_predictor.model.layers.")
                    .replace("talker.model.norm.weight", "talker.code_predictor.model.norm.weight")
                    .replace("talker.codec_head.weight", f"talker.code_predictor.lm_head.{codebook_index}.weight")
                    .replace("talker_layer", "predictor_layer")
                    .replace("talker_final_norm", "predictor_final_norm")
                    .replace("talker_logits", "predictor_logits"))
        if isinstance(value, dict):
            return {rename(key): rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        return value

    graph = rename(graph)
    graph["activation_buffers"]["predictor_logits"] = {"shape": [1, CODE_VOCAB]}
    for layer in range(layers):
        for kind in ("k", "v"):
            graph["activation_buffers"][f"predictor_layer{layer}_cache_{kind}"] = {
                "shape": [KV_HEADS, CACHE_CAPACITY, HEAD_DIM]}
        ops = graph["block_types"]["first_block" if layer == 0 else
                                   f"predictor_layer_{layer}"]["body"]["ops"]
        attention = next(op for op in ops if op["op"] == "attention_decode_append_checked")
        attention["params"]["S"] = CACHE_CAPACITY
        constants = attention["params"]["call_constants"]
        constants.update({
            "k_cache_elements": KV_HEADS * CACHE_CAPACITY * HEAD_DIM,
            "v_cache_elements": KV_HEADS * CACHE_CAPACITY * HEAD_DIM,
            "score_elements": 16 * CACHE_CAPACITY,
            "cache_capacity": CACHE_CAPACITY,
        })
        prefix = "" if layer == 0 else f"layer{layer}_"
        graph["activation_buffers"][prefix + "attention_decode_scores"] = {
            "shape": [16, CACHE_CAPACITY]}
    head = graph["block_types"]["talker_head"]["body"]["ops"][-1]
    head["params"]["N"] = CODE_VOCAB
    head["params"]["call_constants"].update({
        "linear_weight_elements": 1024 * CODE_VOCAB,
        "linear_output_elements": CODE_VOCAB,
        "linear_output_stride": CODE_VOCAB,
        "linear_output_channels": CODE_VOCAB,
    })
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("predictor decode circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
