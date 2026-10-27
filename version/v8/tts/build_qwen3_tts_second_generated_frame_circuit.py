#!/usr/bin/env python3
"""Connect the first generated talker continuation to a second codebook frame.

The first frame, its feedback, the next talker ID and the second predictor
frame are circuit edges. EOS loop control and waveform decoding remain separate.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_generated_talker_continuation_circuit import (
        ROOT, HIDDEN, build_circuit as first_continuation,
    )
    from .build_qwen3_tts_predictor_full_frame_circuit import (
        build_circuit as predictor_frame,
    )
else:
    from build_qwen3_tts_generated_talker_continuation_circuit import (
        ROOT, HIDDEN, build_circuit as first_continuation,
    )
    from build_qwen3_tts_predictor_full_frame_circuit import (
        build_circuit as predictor_frame,
    )

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_second_generated_frame_bounded.json"
TALKER_VOCAB = 3072
TALKER_EOS_ID = 2150
PREFIX = "second_"


def build_circuit(*, talker_layers: int = 28,
                  continuation_steps: int = 14) -> dict:
    graph = first_continuation(talker_layers=talker_layers,
                               continuation_steps=continuation_steps)
    predictor = predictor_frame(continuation_steps=continuation_steps)
    full = talker_layers == 28 and continuation_steps == 14
    graph["name"] = ("qwen3_tts_second_generated_frame_bounded" if full else
                     f"qwen3_tts_second_frame_{talker_layers}_layer_"
                     f"{continuation_steps + 1}_code_fixture")
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update({
        "generated_speech_code_frames": 2,
        "second_frame_requires_non_eos_talker_id": True,
        "second_frame_pinned_talker_position": 18,
        "later_continuation_positions_safety_only": [19, 20],
        "generation_policy": "greedy_fixed_non_eos_prefix_with_repetition_penalty",
        "repetition_penalty": "105/100 sign-dependent on generated-code history",
        "minimum_new_frames": 2,
        "maximum_new_frames": 64,
        "codec_eos_token_id": TALKER_EOS_ID,
        "oracle_fed_model_intermediates": False,
        "complete_text_to_wav": False,
    })
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]
    for name, shape in (
        ("second_talker_adjusted_logits", [TALKER_VOCAB]),
        ("second_talker_first_code_id", [1]),
        ("second_talker_first_code_embedding", [1, HIDDEN]),
        ("second_bridge_left_indices", [2]),
        ("second_bridge_right_indices", [2]),
        ("second_pred_block_input", [2, HIDDEN]),
    ):
        shapes[name] = {"shape": shape}
        bindings[name] = name
    graph["sequence"].append("second_talker_predictor_bridge")
    graph["block_types"]["second_talker_predictor_bridge"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": [
            {"id": "second_talker_repetition_penalty",
             "op": "repetition_penalty_last_row_checked",
             "kernel": "repetition_penalty_last_row_checked_f32",
             "returns_status": True,
             "graph_slots": {"inputs": {"values": "cont_talker_logits",
                                        "history": "talker_first_code_id"},
                             "outputs": {"output": "second_talker_adjusted_logits"}},
             "params": {"R": 1, "C": TALKER_VOCAB, "H": 1,
                        "call_constants": {
                            "values_elements": TALKER_VOCAB, "rows": 1,
                            "channels": TALKER_VOCAB,
                            "row_stride": TALKER_VOCAB,
                            "history_elements": 1, "history_count": 1,
                            "penalty_numerator": 105,
                            "penalty_denominator": 100,
                            "output_elements": TALKER_VOCAB}}},
            {"id": "second_talker_select_first_code",
             "op": "select_next_id_checked",
             "kernel": "select_next_id_checked_f32",
             "returns_status": True,
             "graph_slots": {"inputs": {"values": "second_talker_adjusted_logits"},
                             "outputs": {"index": "second_talker_first_code_id"}},
             "params": {"R": 1, "C": TALKER_VOCAB, "call_constants": {
                 "values_elements": TALKER_VOCAB, "rows": 1,
                 "channels": TALKER_VOCAB, "row_stride": TALKER_VOCAB,
                 "emitted_frames": 1, "min_frames": 2,
                 "max_frames": 64, "stop_id": TALKER_EOS_ID,
                 "suppressed_begin": TALKER_VOCAB - 1024,
                 "suppressed_end": TALKER_VOCAB,
                 "selected_elements": 1}}},
            {"id": "second_talker_first_code_embedding_lookup",
             "op": "embedding_lookup_token_major_checked",
             "kernel": "audio_text_embedding_token_major_f32",
             "returns_status": True,
             "weight_refs": {"table": "talker.model.codec_embedding.weight"},
             "graph_slots": {"inputs": {"ids": "second_talker_first_code_id"},
                             "outputs": {"output": "second_talker_first_code_embedding"}},
             "params": {"T": 1, "V": TALKER_VOCAB, "C": HIDDEN,
                        "call_constants": {
                            "id_elements": 1, "table_elements": TALKER_VOCAB * HIDDEN,
                            "vocabulary": TALKER_VOCAB, "channels": HIDDEN,
                            "table_stride": HIDDEN, "output_elements": HIDDEN,
                            "tokens": 1, "output_stride": HIDDEN}}},
            {"id": "second_predictor_input_rows",
             "op": "indexed_sum_rows_checked",
             "kernel": "indexed_sum_rows_checked_f32", "returns_status": True,
             "graph_slots": {"inputs": {
                 "left": "cont_talker_final_norm",
                 "right": "second_talker_first_code_embedding",
                 "left_indices": "external:second_bridge_left_indices",
                 "right_indices": "external:second_bridge_right_indices"},
                 "outputs": {"output": "second_pred_block_input"}},
             "params": {"L": 1, "K": 1, "R": 2, "C": HIDDEN,
                        "call_constants": {
                            "left_elements": HIDDEN, "left_stride": HIDDEN,
                            "left_rows": 1, "right_elements": HIDDEN,
                            "right_stride": HIDDEN, "right_rows": 1,
                            "left_index_elements": 2, "right_index_elements": 2,
                            "output_elements": 2 * HIDDEN,
                            "output_stride": HIDDEN, "rows": 2,
                            "channels": HIDDEN}}},
        ]}, "footer": [],
    }

    def slot(value: str) -> str:
        external = value.startswith("external:")
        name = value.removeprefix("external:")
        if name == "block_input":
            return "second_pred_block_input"
        return ("external:" if external else "") + PREFIX + name

    for name, shape in predictor["activation_buffers"].items():
        if name == "block_input":
            continue
        mapped = PREFIX + name
        if mapped in shapes:
            raise ValueError(f"second predictor slot collides: {mapped}")
        shapes[mapped] = copy.deepcopy(shape)
        bindings[mapped] = mapped
    if predictor["runtime_constants"] != graph["runtime_constants"]:
        raise ValueError("second predictor runtime constants differ")
    for name, item in predictor["runtime_lengths"].items():
        mapped = copy.deepcopy(item)
        mapped["producer"] = PREFIX + item["producer"]
        graph["runtime_lengths"][PREFIX + name] = mapped
    for block_name in predictor["sequence"]:
        block = copy.deepcopy(predictor["block_types"][block_name])
        for op in (block.get("header", []) + block.get("body", {}).get("ops", []) +
                   block.get("footer", [])):
            op["id"] = PREFIX + op["id"]
            for direction in ("inputs", "outputs"):
                op["graph_slots"][direction] = {
                    port: slot(value)
                    for port, value in op["graph_slots"][direction].items()}
            if op.get("consumes_runtime_lengths"):
                op["consumes_runtime_lengths"] = [
                    PREFIX + name for name in op["consumes_runtime_lengths"]]
            if op.get("produces_runtime_lengths"):
                op["produces_runtime_lengths"] = {
                    PREFIX + name: result for name, result in
                    op["produces_runtime_lengths"].items()}
            if op.get("runtime_scalar_bindings"):
                op["runtime_scalar_bindings"] = {
                    argument: PREFIX + name for argument, name in
                    op["runtime_scalar_bindings"].items()}
        mapped_name = PREFIX + block_name
        graph["sequence"].append(mapped_name)
        graph["block_types"][mapped_name] = block
    # A failed first frame must not leave a prior second-frame extent valid.
    graph["block_types"][graph["sequence"][0]]["header"].insert(0, {
        "id": "clear_second_predictor_code_count",
        "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
        "returns_status": True,
        "params": {"call_constants": {
            "extent_affine_source": 0, "extent_affine_factor": 1,
            "extent_affine_offset": 0,
            "extent_affine_capacity": continuation_steps + 1}},
        "graph_slots": {"inputs": {},
                        "outputs": {"valid_extent":
                                    "second_predictor_valid_code_count"}},
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
            raise SystemExit("second-frame circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
