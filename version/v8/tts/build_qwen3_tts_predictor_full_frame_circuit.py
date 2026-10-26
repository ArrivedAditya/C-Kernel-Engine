#!/usr/bin/env python3
"""Unroll one Qwen3-TTS predictor frame in a circuit-owned generated entry.

The first two-token predictor input is supplied by the caller. All fifteen
greedy predictor codebook IDs, lookup feedback, five shared layers per step,
and KV appends are scheduled by this circuit and executed by generated C.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_predictor_feedback_circuit import build_circuit as first_feedback
    from .build_qwen3_tts_code_predictor_decode_circuit import build_circuit as decode
    from .build_qwen3_tts_talker_prefill_circuit import ROOT
else:
    from build_qwen3_tts_predictor_feedback_circuit import build_circuit as first_feedback
    from build_qwen3_tts_code_predictor_decode_circuit import build_circuit as decode
    from build_qwen3_tts_talker_prefill_circuit import ROOT

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_predictor_full_frame_bounded.json"
VOCAB, HIDDEN = 2048, 1024


def _argmax(logits: str, output: str, op_id: str) -> dict:
    return {"id": op_id, "op": "argmax_last_row_checked",
            "kernel": "audio_argmax_last_row_checked_f32", "returns_status": True,
            "graph_slots": {"inputs": {"values": logits},
                            "outputs": {"index": output}},
            "params": {"R": 1, "C": VOCAB, "call_constants": {
                "values_elements": VOCAB, "rows": 1, "channels": VOCAB,
                "row_stride": VOCAB, "selected_elements": 1}}}


def build_circuit(*, continuation_steps: int = 14) -> dict:
    if isinstance(continuation_steps, bool) or not isinstance(continuation_steps, int) \
            or not 2 <= continuation_steps <= 14:
        raise ValueError("continuation_steps must be in 2..14")
    graph = first_feedback()
    graph["name"] = ("qwen3_tts_predictor_full_frame_bounded" if
                     continuation_steps == 14 else
                     f"qwen3_tts_predictor_{continuation_steps + 1}_codes_fixture")
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update({
        "generated_predictor_feedback_edges": continuation_steps,
        "generated_predictor_codebook_ids": continuation_steps + 1,
        "autonomous_complete_speech_frame": continuation_steps == 14,
        "oracle_fed_initial_embeddings": True,
    })
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]
    caches = {f"predictor_layer{layer}_cache_{kind}"
              for layer in range(5) for kind in ("k", "v")}
    previous_logits = "feedback_predictor_logits"
    for step in range(2, continuation_steps + 1):
        selected = f"step{step - 1}_selected_id"
        embedding = f"step{step}_embedding"
        shapes[selected] = {"shape": [1]}
        shapes[embedding] = {"shape": [1, HIDDEN]}
        bindings[selected] = selected
        bindings[embedding] = embedding
        graph["sequence"].append(f"step{step}_feedback_select")
        graph["block_types"][f"step{step}_feedback_select"] = {
            "sequence": ["header", "body", "footer"], "header": [],
            "body": {"type": "dense", "ops": [
                _argmax(previous_logits, selected, f"step{step}_argmax"),
                {"id": f"step{step}_embedding_lookup",
                 "op": "embedding_lookup_token_major_checked",
                 "kernel": "audio_text_embedding_token_major_f32",
                 "returns_status": True,
                 "weight_refs": {"table":
                     f"talker.code_predictor.model.codec_embedding.{step - 1}.weight"},
                 "graph_slots": {"inputs": {"ids": selected},
                                 "outputs": {"output": embedding}},
                 "params": {"T": 1, "V": VOCAB, "C": HIDDEN,
                            "call_constants": {"id_elements": 1,
                                "table_elements": VOCAB * HIDDEN,
                                "vocabulary": VOCAB, "channels": HIDDEN,
                                "table_stride": HIDDEN, "output_elements": HIDDEN,
                                "tokens": 1, "output_stride": HIDDEN}}},
            ]}, "footer": []}
        continuation = decode(codebook_index=step)
        prefix = f"step{step}_"

        def slot(value: str) -> str:
            external = value.startswith("external:")
            name = value.removeprefix("external:")
            if name == "block_input":
                return embedding
            if name in caches:
                return value
            return ("external:" if external else "") + prefix + name

        for name, shape in continuation["activation_buffers"].items():
            if name in caches or name in ("block_input", "step_index_input",
                                         "step_index_checked", "position_checked"):
                continue
            mapped = slot(name)
            shapes[mapped] = copy.deepcopy(shape)
            bindings[mapped] = mapped
        for block_name in continuation["sequence"]:
            block = copy.deepcopy(continuation["block_types"][block_name])
            block["header"] = []  # compile-time position is checked by provider
            for op in block.get("body", {}).get("ops", []):
                op["id"] = prefix + op["id"]
                for direction in ("inputs", "outputs"):
                    op["graph_slots"][direction] = {
                        port: slot(name)
                        for port, name in op["graph_slots"][direction].items()}
                if op["op"] == "rope_qk_bounded":
                    op.pop("consumes_runtime_lengths", None)
                    op.pop("runtime_scalar_bindings", None)
                    op["params"]["call_constants"]["pos_offset"] = step + 1
                if op["op"] == "attention_decode_append_checked":
                    op.pop("consumes_runtime_lengths", None)
                    op.pop("runtime_scalar_bindings", None)
                    op["params"]["call_constants"]["position"] = step + 1
            name = prefix + block_name
            graph["sequence"].append(name)
            graph["block_types"][name] = block
        previous_logits = prefix + "predictor_logits"
    final_id = f"step{continuation_steps}_selected_id"
    shapes[final_id] = {"shape": [1]}
    bindings[final_id] = final_id
    graph["sequence"].append("predictor_final_select")
    graph["block_types"]["predictor_final_select"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": [
            _argmax(previous_logits, final_id, "predictor_final_argmax")
        ]}, "footer": []}
    shapes["predictor_valid_code_count"] = {"shape": [1]}
    bindings["predictor_valid_code_count"] = "predictor_valid_code_count"
    graph["block_types"]["first_block"]["header"].insert(0, {
        "id": "predictor_clear_code_count",
        "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
        "returns_status": True,
        "params": {"call_constants": {
            "extent_affine_source": 0, "extent_affine_factor": 1,
            "extent_affine_offset": 0,
            "extent_affine_capacity": continuation_steps + 1}},
        "graph_slots": {"inputs": {},
                        "outputs": {"valid_extent": "predictor_valid_code_count"}},
    })
    graph["sequence"].append("predictor_publish_count")
    graph["block_types"]["predictor_publish_count"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": [{
            "id": "predictor_publish_code_count",
            "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
            "returns_status": True, "consumes_runtime_lengths": ["step_index"],
            "runtime_scalar_bindings": {"extent_affine_source": "step_index"},
            "params": {"call_constants": {
                "extent_affine_factor": 1,
                "extent_affine_offset": continuation_steps + 1,
                "extent_affine_capacity": continuation_steps + 1}},
            "graph_slots": {"inputs": {},
                            "outputs": {"valid_extent": "predictor_valid_code_count"}}},
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
            raise SystemExit("predictor full-frame circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
