#!/usr/bin/env python3
"""Connect generated predictor prefill ID to its first continuation embedding.

The pinned two-token prefill input remains caller supplied. All subsequent
selection, lookup, five-layer continuation and KV append execute in one
generated entry. This graph proves one feedback edge, not a full speech frame.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_code_predictor_prefill_circuit import build_circuit as prefill
    from .build_qwen3_tts_code_predictor_decode_circuit import build_circuit as decode
    from .build_qwen3_tts_talker_prefill_circuit import ROOT
else:
    from build_qwen3_tts_code_predictor_prefill_circuit import build_circuit as prefill
    from build_qwen3_tts_code_predictor_decode_circuit import build_circuit as decode
    from build_qwen3_tts_talker_prefill_circuit import ROOT

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_predictor_first_feedback_bounded.json"
VOCAB, HIDDEN = 2048, 1024


def build_circuit() -> dict:
    graph, continuation = prefill(), decode(codebook_index=1)
    graph["name"] = "qwen3_tts_predictor_first_feedback_bounded"
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update({
        "generated_predictor_feedback_edges": 1,
        "oracle_fed_initial_embeddings": True,
        "autonomous_complete_speech_frame": False,
    })
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]
    shapes["predictor_codebook0_selected_id"] = {"shape": [1]}
    shapes["feedback_embedding"] = {"shape": [1, HIDDEN]}
    bindings["predictor_codebook0_selected_id"] = "predictor_codebook0_selected_id"
    bindings["feedback_embedding"] = "feedback_embedding"
    graph["sequence"].append("predictor_feedback_select")
    graph["block_types"]["predictor_feedback_select"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": [
            {"id": "predictor_feedback_argmax", "op": "argmax_last_row_checked",
             "kernel": "audio_argmax_last_row_checked_f32", "returns_status": True,
             "graph_slots": {"inputs": {"values": "predictor_codebook0_logits"},
                             "outputs": {"index": "predictor_codebook0_selected_id"}},
             "params": {"R": 2, "C": VOCAB, "call_constants": {
                 "values_elements": 2 * VOCAB, "rows": 2, "channels": VOCAB,
                 "row_stride": VOCAB, "selected_elements": 1}}},
            {"id": "predictor_feedback_embedding",
             "op": "embedding_lookup_token_major_checked",
             "kernel": "audio_text_embedding_token_major_f32", "returns_status": True,
             "weight_refs": {"table": "talker.code_predictor.model.codec_embedding.0.weight"},
             "graph_slots": {"inputs": {"ids": "predictor_codebook0_selected_id"},
                             "outputs": {"output": "feedback_embedding"}},
             "params": {"T": 1, "V": VOCAB, "C": HIDDEN, "call_constants": {
                 "id_elements": 1, "table_elements": VOCAB * HIDDEN,
                 "vocabulary": VOCAB, "channels": HIDDEN,
                 "table_stride": HIDDEN, "output_elements": HIDDEN,
                 "tokens": 1, "output_stride": HIDDEN}}},
        ]}, "footer": []}

    caches = {f"predictor_layer{layer}_cache_{kind}"
              for layer in range(5) for kind in ("k", "v")}

    def slot(value: str) -> str:
        external = value.startswith("external:")
        name = value.removeprefix("external:")
        if name == "block_input":
            return "feedback_embedding"
        if name in caches:
            return value
        return ("external:" if external else "") + "feedback_" + name

    for name, shape in continuation["activation_buffers"].items():
        if name == "block_input" or name in caches:
            continue
        mapped = slot(name)
        shapes[mapped] = copy.deepcopy(shape)
        bindings[mapped] = mapped
    graph["runtime_constants"] = copy.deepcopy(continuation["runtime_constants"])
    graph["runtime_lengths"] = copy.deepcopy(continuation["runtime_lengths"])
    for item in graph["runtime_lengths"].values():
        item["producer"] = "feedback_" + item["producer"]
    for block_name in continuation["sequence"]:
        block = copy.deepcopy(continuation["block_types"][block_name])
        for op in block.get("header", []) + block.get("body", {}).get("ops", []) + block.get("footer", []):
            op["id"] = "feedback_" + op["id"]
            for direction in ("inputs", "outputs"):
                slots = op.get("graph_slots", {}).get(direction, {})
                op["graph_slots"][direction] = {
                    port: slot(name) for port, name in slots.items()}
        name = "feedback_" + block_name
        graph["sequence"].append(name)
        graph["block_types"][name] = block
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("predictor feedback circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
