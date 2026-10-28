#!/usr/bin/env python3
"""Join two CKE-generated Qwen speech-code frames to one codec invocation.

This is a fixed, non-EOS prefix for diagnosing connected waveform arithmetic.
It does not implement autonomous stopping or useful utterance synthesis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_second_generated_frame_circuit import (
        ROOT, build_circuit as generated_frames,
    )
    from .build_qwen3_tts_first_frame_waveform_circuit import attach_codec
else:
    from build_qwen3_tts_second_generated_frame_circuit import (
        ROOT, build_circuit as generated_frames,
    )
    from build_qwen3_tts_first_frame_waveform_circuit import attach_codec

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_two_frame_waveform_bounded.json"
FRAMES = 2
SAMPLES = 3840


def build_circuit() -> dict:
    graph = generated_frames()
    graph["name"] = "qwen3_tts_two_frame_waveform_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_two_frame_waveform_bounded"
    # This graph appends codec calls after the second frame. Its full entry owns
    # that additional schedule; the frame-only segment boundaries do not.
    graph["native_entry"].pop("segments", None)
    graph["contract"]["runtime_invariants"].update({
        "generated_codec_frames": FRAMES,
        "generated_sample_count": SAMPLES,
        "two_frame_codec_connected": True,
        "generation_policy": "greedy_fixed_non_eos_prefix_with_repetition_penalty",
        "complete_text_to_wav": False,
        "audible_speech": False,
    })
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]
    ids = {}
    pairs = [("semantic_ids", "talker_first_code_id",
              "second_talker_first_code_id")]
    pairs += [(f"acoustic_ids_{i}",
               "pred_predictor_codebook0_selected_id" if i == 0 else
               f"pred_step{i}_selected_id",
               "second_predictor_codebook0_selected_id" if i == 0 else
               f"second_step{i}_selected_id") for i in range(15)]
    ops = []
    for name, first, second in pairs:
        packed = "packed_" + name
        if shapes[first]["shape"] != [1] or shapes[second]["shape"] != [1]:
            raise ValueError(f"codebook scalar geometry differs: {name}")
        shapes[packed] = {"shape": [FRAMES]}
        bindings[packed] = packed
        ids[name] = packed
        ops.append({
            "id": "pack_" + name,
            "op": "concat_i32_checked", "kernel": "concat_i32_checked",
            "returns_status": True,
            "graph_slots": {"inputs": {"left": first, "right": second},
                            "outputs": {"output": packed}},
            "params": {"L": 1, "R": 1, "O": FRAMES,
                       "call_constants": {
                           "left_elements": 1, "left_count": 1,
                           "right_elements": 1, "right_count": 1,
                           "output_elements": FRAMES}},
        })
    graph["sequence"].append("pack_generated_codec_frames")
    graph["block_types"]["pack_generated_codec_frames"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": ops}, "footer": [],
    }
    return attach_codec(graph, frames=FRAMES, ids=ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("two-frame waveform circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
