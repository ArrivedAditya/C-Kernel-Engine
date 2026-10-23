#!/usr/bin/env python3
"""Author checked split-RVQ codebook reconstruction for Qwen3-TTS.

The semantic codebook and acoustic codebooks are explicit circuit inputs.
Their lookup and ordered sums precede separate output projections and a
final join. No model arithmetic is scheduled by the native host.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_semantic_circuit import build_circuit as semantic_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_rvq_bounded.json"


def checkpoint(op: dict, layout: str) -> dict:
    return {
        "section": "body", "template_op_id": op["id"], "op": op["op"],
        "checkpoints": [{
            "id": f"qwen3_tts.codec.{op['id']}", "producer": op["id"],
            "tensor": next(iter(op["graph_slots"]["outputs"].values())),
            "logical_layout": layout,
            "axis_names": (["channel", "frame"] if layout == "channel_major"
                           else ["frame", "channel"]),
            "storage_dtype": "fp32"}],
    }


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15) -> dict:
    if isinstance(acoustic_codebooks, bool) or not isinstance(acoustic_codebooks, int) or not 1 <= acoustic_codebooks <= 63:
        raise ValueError("acoustic_codebooks must be an integer from 1 to 63")
    graph = semantic_circuit(frames=frames, vocabulary=vocabulary,
                             codebook_width=codebook_width,
                             output_width=output_width)
    graph["name"] = "qwen3_tts_codec_rvq_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_rvq_bounded"
    graph["contract"]["runtime_invariants"]["acoustic_codebooks"] = acoustic_codebooks
    graph["contract"]["runtime_invariants"]["complete_rvq"] = True
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
    exports = graph["semantic_checkpoints"]["exports"]

    def append(op: dict, layout: str) -> None:
        ops.append(op)
        exports[op["id"]] = checkpoint(op, layout)

    previous = None
    elements = frames * codebook_width
    for index in range(acoustic_codebooks):
        ids = f"acoustic_ids_{index}"
        channel = f"acoustic_channel_{index}"
        shapes[ids] = {"shape": [frames]}
        shapes[channel] = {"shape": [codebook_width, frames]}
        bindings[ids] = ids
        bindings[channel] = channel
        append({"id": f"acoustic_lookup_{index}",
                "op": "embedding_lookup_checked",
                "kernel": "audio_text_embedding_channel_major_f32",
                "returns_status": True,
                "weight_refs": {"table": f"codec.acoustic.{index}.effective_embedding"},
                "graph_slots": {"inputs": {"ids": f"external:{ids}"},
                                "outputs": {"output": channel}},
                "params": {"T": frames, "V": vocabulary, "C": codebook_width,
                           "T_capacity": frames, "call_constants": {
                               "id_elements": frames,
                               "table_elements": vocabulary * codebook_width,
                               "vocabulary": vocabulary,
                               "channels": codebook_width,
                               "table_stride": codebook_width,
                               "output_elements": elements,
                               "tokens": frames, "output_stride": frames}}},
               "channel_major")
        if previous is None:
            previous = channel
            continue
        summed = f"acoustic_sum_{index}"
        shapes[summed] = {"shape": [codebook_width, frames]}
        bindings[summed] = summed
        append({"id": summed, "op": "add_rows_checked",
                "kernel": "add_rows_checked_f32", "returns_status": True,
                "graph_slots": {"inputs": {"left": previous, "right": channel},
                                "outputs": {"output": summed}},
                "params": {"N": elements, "call_constants": {
                    "left_elements": elements, "right_elements": elements,
                    "output_elements": elements, "elements": elements}}},
               "channel_major")
        previous = summed

    shapes["acoustic_token"] = {"shape": [frames, codebook_width]}
    shapes["acoustic_projected"] = {"shape": [frames, output_width]}
    shapes["rvq_output"] = {"shape": [frames, output_width]}
    for name in ("acoustic_token", "acoustic_projected", "rvq_output"):
        bindings[name] = name
    append({"id": "acoustic_transpose", "op": "transpose_strided_checked",
            "kernel": "transpose_strided_f32_checked", "returns_status": True,
            "graph_slots": {"inputs": {"input": previous},
                            "outputs": {"output": "acoustic_token"}},
            "params": {"R": codebook_width, "C": frames, "call_constants": {
                "transpose_input_elements": elements,
                "transpose_input_stride": frames,
                "transpose_output_elements": elements,
                "transpose_output_stride": codebook_width,
                "transpose_rows": codebook_width,
                "transpose_columns": frames}}}, "token_major")
    append({"id": "acoustic_projection", "op": "linear_rows_checked_no_bias",
            "kernel": "linear_rows_checked_no_bias_f32", "returns_status": True,
            "weight_refs": {"weight": "codec.acoustic.output_projection.weight"},
            "graph_slots": {"inputs": {"input": "acoustic_token"},
                            "outputs": {"output": "acoustic_projected"}},
            "params": {"M": frames, "K": codebook_width, "N": output_width,
                       "call_constants": {
                           "linear_input_elements": elements,
                           "linear_input_stride": codebook_width,
                           "linear_weight_elements": output_width * codebook_width,
                           "linear_weight_stride": codebook_width,
                           "linear_output_elements": frames * output_width,
                           "linear_output_stride": output_width,
                           "linear_rows": frames,
                           "linear_input_channels": codebook_width,
                           "linear_output_channels": output_width}}},
           "token_major")
    projected = frames * output_width
    append({"id": "rvq_join", "op": "add_rows_checked",
            "kernel": "add_rows_checked_f32", "returns_status": True,
            "graph_slots": {"inputs": {
                "left": "semantic_projected", "right": "acoustic_projected"},
                "outputs": {"output": "rvq_output"}},
            "params": {"N": projected, "call_constants": {
                "left_elements": projected, "right_elements": projected,
                "output_elements": projected, "elements": projected}}},
           "token_major")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec RVQ circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
