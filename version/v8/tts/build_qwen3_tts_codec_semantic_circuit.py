#!/usr/bin/env python3
"""Author the checked first-codebook → semantic projection codec prefix.

The first RVQ codebook is independent of the remaining acoustic codebooks.
Its effective table is exported from embedding_sum / max(cluster_usage, epsilon);
the ordinary circuit performs lookup, layout conversion, and projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_semantic_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512) -> dict:
    for name, value in (("frames", frames), ("vocabulary", vocabulary),
                        ("codebook_width", codebook_width),
                        ("output_width", output_width)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if frames > 4096:
        raise ValueError("bounded codec prefix supports at most 4096 frames")
    width = codebook_width
    ops = [
        {"id": "semantic_lookup", "op": "embedding_lookup_checked",
         "kernel": "audio_text_embedding_channel_major_f32",
         "returns_status": True,
         "weight_refs": {"table": "codec.semantic.effective_embedding"},
         "graph_slots": {"inputs": {"ids": "external:semantic_ids"},
                         "outputs": {"output": "semantic_channel"}},
         "params": {"T": frames, "V": vocabulary, "C": width,
                    "T_capacity": frames, "call_constants": {
                        "id_elements": frames, "table_elements": vocabulary * width,
                        "vocabulary": vocabulary, "channels": width,
                        "table_stride": width,
                        "output_elements": width * frames,
                        "tokens": frames, "output_stride": frames}}},
        {"id": "semantic_transpose", "op": "transpose_strided_checked",
         "kernel": "transpose_strided_f32_checked", "returns_status": True,
         "graph_slots": {"inputs": {"input": "semantic_channel"},
                         "outputs": {"output": "semantic_token"}},
         "params": {"R": width, "C": frames, "call_constants": {
             "transpose_input_elements": width * frames,
             "transpose_input_stride": frames,
             "transpose_output_elements": width * frames,
             "transpose_output_stride": width,
             "transpose_rows": width, "transpose_columns": frames}}},
        {"id": "semantic_projection", "op": "linear_rows_checked_no_bias",
         "kernel": "linear_rows_checked_no_bias_f32", "returns_status": True,
         "weight_refs": {"weight": "codec.semantic.output_projection.weight"},
         "graph_slots": {"inputs": {"input": "semantic_token"},
                         "outputs": {"output": "semantic_projected"}},
         "params": {"M": frames, "K": width, "N": output_width,
                    "call_constants": {
                        "linear_input_elements": frames * width,
                        "linear_input_stride": width,
                        "linear_weight_elements": output_width * width,
                        "linear_weight_stride": width,
                        "linear_output_elements": frames * output_width,
                        "linear_output_stride": output_width,
                        "linear_rows": frames,
                        "linear_input_channels": width,
                        "linear_output_channels": output_width}}},
    ]
    shapes = {"semantic_ids": [frames],
              "semantic_channel": [width, frames],
              "semantic_token": [frames, width],
              "semantic_projected": [frames, output_width]}
    return {
        "version": 3, "name": "qwen3_tts_codec_semantic_bounded",
        "family": "bounded_graph", "checked_native_entry": True,
        "contract": {"runtime_invariants": {
            "inference_only": True, "production_kernel_heap_allocation": False,
            "frames": frames, "complete_codec": False,
            "complete_synthesis": False}},
        "activation_buffers": {name: {"shape": shape}
                               for name, shape in shapes.items()},
        "activation_bindings": {name: name for name in shapes},
        "native_entry": {"function": "ck_qwen3_tts_codec_semantic_bounded",
                         "params": [{"c_type": "uint8_t *", "name": "arena"},
                                    {"c_type": "size_t", "name": "arena_bytes"}],
                         "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
        "sequence": ["codec_semantic"],
        "block_types": {"codec_semantic": {
            "sequence": ["header", "body", "footer"],
            "header": [], "body": {"type": "dense", "ops": ops},
            "footer": []}},
        "semantic_checkpoints": {
            "schema": "cke.semantic_checkpoint_contract", "schema_version": 1,
            "exports": {op["id"]: {
                "section": "body", "template_op_id": op["id"],
                "op": op["op"], "checkpoints": [{
                    "id": f"qwen3_tts.codec.{op['id']}",
                    "producer": op["id"],
                    "tensor": next(iter(op["graph_slots"]["outputs"].values())),
                    "logical_layout": ("channel_major" if op["id"] == "semantic_lookup"
                                       else "token_major"),
                    "axis_names": (["channel", "frame"] if op["id"] == "semantic_lookup"
                                   else ["frame", "channel"]),
                    "storage_dtype": "fp32"}]}
                for op in ops}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec semantic circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
