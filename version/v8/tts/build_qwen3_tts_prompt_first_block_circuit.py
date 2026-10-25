#!/usr/bin/env python3
"""Connect pinned CustomVoice prompt assembly to the first generated talker block.

The row plan is a declared model input. It selects projected text rows and
codec rows; the indexed-sum provider performs the native arithmetic. The host
loads these declared indices and the request's token IDs but does not execute
neural layers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_first_block_circuit import build_circuit as first_block
    from .build_qwen3_tts_text_projection_circuit import (
        ROOT, VOCAB, TEXT, HIDDEN, build_circuit as text_projection,
    )
else:
    from build_qwen3_tts_first_block_circuit import build_circuit as first_block
    from build_qwen3_tts_text_projection_circuit import (
        ROOT, VOCAB, TEXT, HIDDEN, build_circuit as text_projection,
    )

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_prompt_first_block_bounded.json"
CODEC_VOCAB = 3072


def prompt_plan(context_tokens: int) -> dict[str, list[int]]:
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or not 1 <= context_tokens <= 117:
        raise ValueError("context tokens must be an integer from 1 to 117")
    # Text rows: role(3), pad, BOS, EOS, content(context_tokens).
    # Codec rows: think, think_BOS, language, think_EOS, speaker, pad, BOS.
    left = [0, 1, 2] + [3] * 5 + [4] + list(range(6, 6 + context_tokens)) + [5, 3]
    right = [-1] * 3 + list(range(6)) + [5] * (context_tokens + 1) + [6]
    assert len(left) == len(right) == context_tokens + 11
    return {"left_indices": left, "right_indices": right}


def build_circuit(context_tokens: int = 7) -> dict:
    plan = prompt_plan(context_tokens)
    text_rows = context_tokens + 6
    prompt_rows = context_tokens + 11
    graph = text_projection(tokens=text_rows, vocab=VOCAB,
                            text_width=TEXT, hidden_width=HIDDEN)
    block = first_block(prompt_rows)
    graph["name"] = "qwen3_tts_prompt_first_block_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_prompt_first_block_bounded"
    graph["contract"]["runtime_invariants"].update({
        "context_tokens": context_tokens,
        "prompt_rows": prompt_rows,
        "complete_talker_block": True,
        "complete_talker": False,
        "complete_synthesis": False,
        "prompt_index_plan": plan,
        "reference_scope": "CustomVoice English Ryan non-streaming prefill",
    })
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    shapes.update({
        "codec_ids": {"shape": [7]},
        "codec_embedding_channel": {"shape": [HIDDEN, 7]},
        "codec_embedding_token": {"shape": [7, HIDDEN]},
        "left_indices": {"shape": [prompt_rows]},
        "right_indices": {"shape": [prompt_rows]},
        "block_input": {"shape": [prompt_rows, HIDDEN]},
    })
    bindings.update({name: name for name in (
        "codec_ids", "codec_embedding_channel", "codec_embedding_token",
        "left_indices", "right_indices", "block_input")})
    shapes.update(block["activation_buffers"])
    bindings.update(block["activation_bindings"])

    ops = graph["block_types"]["text_projection"]["body"]["ops"]
    ops.extend([
        {"id": "codec_embedding", "op": "embedding_lookup_checked",
         "kernel": "audio_text_embedding_channel_major_f32",
         "returns_status": True,
         "weight_refs": {"table": "talker.model.codec_embedding.weight"},
         "graph_slots": {"inputs": {"ids": "external:codec_ids"},
                         "outputs": {"output": "codec_embedding_channel"}},
         "params": {"T": 7, "V": CODEC_VOCAB, "C": HIDDEN,
                    "T_capacity": 7, "call_constants": {
                        "id_elements": 7, "table_elements": CODEC_VOCAB * HIDDEN,
                        "vocabulary": CODEC_VOCAB, "channels": HIDDEN,
                        "table_stride": HIDDEN,
                        "output_elements": 7 * HIDDEN,
                        "tokens": 7, "output_stride": 7}}},
        {"id": "codec_embedding_transpose", "op": "transpose_strided_checked",
         "kernel": "transpose_strided_f32_checked", "returns_status": True,
         "graph_slots": {"inputs": {"input": "codec_embedding_channel"},
                         "outputs": {"output": "codec_embedding_token"}},
         "params": {"R": HIDDEN, "C": 7, "call_constants": {
             "transpose_input_elements": HIDDEN * 7,
             "transpose_input_stride": 7,
             "transpose_output_elements": HIDDEN * 7,
             "transpose_output_stride": HIDDEN,
             "transpose_rows": HIDDEN, "transpose_columns": 7}}},
        {"id": "prompt_assembly", "op": "indexed_sum_rows_checked",
         "kernel": "indexed_sum_rows_checked_f32", "returns_status": True,
         "graph_slots": {"inputs": {
             "left": "projection_output", "right": "codec_embedding_token",
             "left_indices": "external:left_indices",
             "right_indices": "external:right_indices"},
             "outputs": {"output": "block_input"}},
         "params": {"L": text_rows, "K": 7, "R": prompt_rows, "C": HIDDEN,
                    "call_constants": {
                        "left_elements": text_rows * HIDDEN,
                        "left_stride": HIDDEN, "left_rows": text_rows,
                        "right_elements": 7 * HIDDEN,
                        "right_stride": HIDDEN, "right_rows": 7,
                        "left_index_elements": prompt_rows,
                        "right_index_elements": prompt_rows,
                        "output_elements": prompt_rows * HIDDEN,
                        "output_stride": HIDDEN,
                        "rows": prompt_rows, "channels": HIDDEN}}},
    ])
    block_ops = block["block_types"]["first_block"]["body"]["ops"]
    for op in block_ops:
        op["graph_slots"]["inputs"] = {
            port: ("block_input" if value == "external:block_input" else value)
            for port, value in op["graph_slots"]["inputs"].items()}
    graph["sequence"].append("first_block")
    graph["block_types"]["first_block"] = block["block_types"]["first_block"]
    exports = graph["semantic_checkpoints"]["exports"]
    for op_id, kind, tensor, layout, axes in (
        ("codec_embedding", "embedding_lookup_checked", "codec_embedding_channel",
         "channel_major", ["channel", "token"]),
        ("prompt_assembly", "indexed_sum_rows_checked", "block_input",
         "token_major", ["token", "channel"]),
        ("block_output", "add_rows_checked", "block_output",
         "token_major", ["token", "channel"]),
    ):
        exports[op_id] = {
            "section": "body", "template_op_id": op_id, "op": kind,
            "checkpoints": [{"id": f"qwen3_tts.talker.{op_id}",
                             "producer": op_id, "tensor": tensor,
                             "logical_layout": layout, "axis_names": axes,
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
            raise SystemExit("prompt/first-block circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
