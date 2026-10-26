#!/usr/bin/env python3
"""Connect generated Qwen talker prefill to one complete predictor frame.

The circuit selects the first codec ID from talker logits, looks up that ID,
and assembles the predictor's two input rows from the last talker hidden row
and the selected embedding. The caller supplies only the declared prompt IDs,
prompt row plan, and fixed bridge row plan; no model intermediate is supplied.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_talker_prefill_circuit import (
        CODEC_VOCAB, HIDDEN, ROOT, build_circuit as talker_prefill,
    )
    from .build_qwen3_tts_predictor_full_frame_circuit import (
        build_circuit as predictor_frame,
    )
else:
    from build_qwen3_tts_talker_prefill_circuit import (
        CODEC_VOCAB, HIDDEN, ROOT, build_circuit as talker_prefill,
    )
    from build_qwen3_tts_predictor_full_frame_circuit import (
        build_circuit as predictor_frame,
    )

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_talker_predictor_frame_bounded.json"
PREDICTOR_PREFIX = "pred_"


def bridge_plan(prompt_rows: int) -> dict[str, list[int]]:
    if isinstance(prompt_rows, bool) or not isinstance(prompt_rows, int) or prompt_rows < 2:
        raise ValueError("prompt_rows must be an integer of at least two")
    return {"left_indices": [prompt_rows - 1, -1],
            "right_indices": [-1, 0]}


def build_circuit(*, context_tokens: int = 7, talker_layers: int = 28,
                  continuation_steps: int = 14) -> dict:
    graph = talker_prefill(context_tokens=context_tokens, layers=talker_layers)
    predictor = predictor_frame(continuation_steps=continuation_steps)
    rows = context_tokens + 11
    full = talker_layers == 28 and continuation_steps == 14 and context_tokens == 7
    graph["name"] = ("qwen3_tts_talker_predictor_frame_bounded" if full else
                     f"qwen3_tts_talker_{talker_layers}_predictor_{continuation_steps + 1}_fixture")
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update({
        "generated_talker_layers": talker_layers,
        "generated_predictor_codebook_ids": continuation_steps + 1,
        "predictor_bridge_plan": bridge_plan(rows),
        "oracle_fed_model_intermediates": False,
        "complete_text_to_wav": False,
    })
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]
    for name, shape in (
        ("talker_first_code_id", [1]),
        ("talker_first_code_embedding", [1, HIDDEN]),
        ("bridge_left_indices", [2]),
        ("bridge_right_indices", [2]),
        ("pred_block_input", [2, HIDDEN]),
    ):
        shapes[name] = {"shape": shape}
        bindings[name] = name
    graph["sequence"].append("talker_predictor_bridge")
    graph["block_types"]["talker_predictor_bridge"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": [
            {"id": "talker_select_first_code", "op": "argmax_last_row_checked",
             "kernel": "audio_argmax_last_row_checked_f32", "returns_status": True,
             "graph_slots": {"inputs": {"values": "talker_logits"},
                             "outputs": {"index": "talker_first_code_id"}},
             "params": {"R": rows, "C": CODEC_VOCAB, "call_constants": {
                 "values_elements": rows * CODEC_VOCAB, "rows": rows,
                 "channels": CODEC_VOCAB, "row_stride": CODEC_VOCAB,
                 "selected_elements": 1}}},
            {"id": "talker_first_code_embedding_lookup",
             "op": "embedding_lookup_token_major_checked",
             "kernel": "audio_text_embedding_token_major_f32", "returns_status": True,
             "weight_refs": {"table": "talker.model.codec_embedding.weight"},
             "graph_slots": {"inputs": {"ids": "talker_first_code_id"},
                             "outputs": {"output": "talker_first_code_embedding"}},
             "params": {"T": 1, "V": CODEC_VOCAB, "C": HIDDEN,
                        "call_constants": {"id_elements": 1,
                            "table_elements": CODEC_VOCAB * HIDDEN,
                            "vocabulary": CODEC_VOCAB, "channels": HIDDEN,
                            "table_stride": HIDDEN, "output_elements": HIDDEN,
                            "tokens": 1, "output_stride": HIDDEN}}},
            {"id": "talker_predictor_input_rows", "op": "indexed_sum_rows_checked",
             "kernel": "indexed_sum_rows_checked_f32", "returns_status": True,
             "graph_slots": {"inputs": {
                 "left": "talker_final_norm",
                 "right": "talker_first_code_embedding",
                 "left_indices": "external:bridge_left_indices",
                 "right_indices": "external:bridge_right_indices"},
                 "outputs": {"output": "pred_block_input"}},
             "params": {"L": rows, "K": 1, "R": 2, "C": HIDDEN,
                        "call_constants": {
                            "left_elements": rows * HIDDEN,
                            "left_stride": HIDDEN, "left_rows": rows,
                            "right_elements": HIDDEN, "right_stride": HIDDEN,
                            "right_rows": 1, "left_index_elements": 2,
                            "right_index_elements": 2,
                            "output_elements": 2 * HIDDEN,
                            "output_stride": HIDDEN, "rows": 2,
                            "channels": HIDDEN}}},
        ]}, "footer": [],
    }

    def slot(value: str) -> str:
        external = value.startswith("external:")
        name = value.removeprefix("external:")
        if name == "block_input":
            return "pred_block_input"
        return ("external:" if external else "") + PREDICTOR_PREFIX + name

    for name, shape in predictor["activation_buffers"].items():
        if name == "block_input":
            continue
        mapped = PREDICTOR_PREFIX + name
        if mapped in shapes:
            raise ValueError(f"predictor slot collides with talker: {mapped}")
        shapes[mapped] = copy.deepcopy(shape)
        bindings[mapped] = mapped
    graph["runtime_constants"] = copy.deepcopy(predictor["runtime_constants"])
    graph["runtime_lengths"] = copy.deepcopy(predictor["runtime_lengths"])
    for item in graph["runtime_lengths"].values():
        item["producer"] = PREDICTOR_PREFIX + item["producer"]
    for block_name in predictor["sequence"]:
        block = copy.deepcopy(predictor["block_types"][block_name])
        for op in block.get("header", []) + block.get("body", {}).get("ops", []) + block.get("footer", []):
            op["id"] = PREDICTOR_PREFIX + op["id"]
            for direction in ("inputs", "outputs"):
                slots = op.get("graph_slots", {}).get(direction, {})
                op["graph_slots"][direction] = {
                    port: slot(name) for port, name in slots.items()}
        mapped = PREDICTOR_PREFIX + block_name
        graph["sequence"].append(mapped)
        graph["block_types"][mapped] = block
    # The talker can reject a request before the predictor's own header runs.
    # Invalidate the complete-frame result before any request operation.
    graph["block_types"][graph["sequence"][0]]["header"].insert(0, {
        "id": "clear_talker_predictor_code_count",
        "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
        "returns_status": True,
        "params": {"call_constants": {
            "extent_affine_source": 0, "extent_affine_factor": 1,
            "extent_affine_offset": 0,
            "extent_affine_capacity": continuation_steps + 1}},
        "graph_slots": {"inputs": {},
                        "outputs": {"valid_extent": "pred_predictor_valid_code_count"}},
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
            raise SystemExit("talker-predictor circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
