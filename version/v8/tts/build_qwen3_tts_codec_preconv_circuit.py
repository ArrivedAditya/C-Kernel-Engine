#!/usr/bin/env python3
"""Extend checked split-RVQ reconstruction through the causal pre-convolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_rvq_circuit import build_circuit as rvq_circuit, checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_preconv_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15,
                  latent_width: int = 1024) -> dict:
    if isinstance(latent_width, bool) or not isinstance(latent_width, int) or latent_width < 1:
        raise ValueError("latent_width must be a positive integer")
    graph = rvq_circuit(frames=frames, vocabulary=vocabulary,
                        codebook_width=codebook_width,
                        output_width=output_width,
                        acoustic_codebooks=acoustic_codebooks)
    graph["name"] = "qwen3_tts_codec_preconv_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_preconv_bounded"
    graph["contract"]["runtime_invariants"]["causal_preconv"] = True
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
    exports = graph["semantic_checkpoints"]["exports"]
    for name, shape in (
        ("rvq_channel", [output_width, frames]),
        ("preconv_padded", [output_width, frames + 2]),
        ("preconv_output", [latent_width, frames]),
    ):
        shapes[name] = {"shape": shape}
        bindings[name] = name

    def append(op: dict, layout: str) -> None:
        ops.append(op)
        exports[op["id"]] = checkpoint(op, layout)

    append({
        "id": "rvq_to_channels", "op": "transpose_strided_checked",
        "kernel": "transpose_strided_f32_checked", "returns_status": True,
        "graph_slots": {"inputs": {"input": "rvq_output"},
                        "outputs": {"output": "rvq_channel"}},
        "params": {"R": frames, "C": output_width, "call_constants": {
            "transpose_input_elements": frames * output_width,
            "transpose_input_stride": output_width,
            "transpose_output_elements": frames * output_width,
            "transpose_output_stride": frames,
            "transpose_rows": frames, "transpose_columns": output_width}},
    }, "channel_major")
    append({
        "id": "causal_left_zero_pad", "op": "audio_zero_pad1d_left_checked",
        "kernel": "audio_zero_pad1d_left_channel_major_f32_checked",
        "returns_status": True,
        "graph_slots": {"inputs": {"input": "rvq_channel"},
                        "outputs": {"output": "preconv_padded"}},
        "params": {"C": output_width, "T": frames, "U": frames + 2,
                   "call_constants": {
            "zero_pad_input_elements": output_width * frames,
            "zero_pad_input_stride": frames,
            "zero_pad_output_elements": output_width * (frames + 2),
            "zero_pad_output_stride": frames + 2,
            "zero_pad_channels": output_width,
            "zero_pad_input_frames": frames,
            "zero_pad_left_padding": 2,
            "zero_pad_output_frames": frames + 2}},
    }, "channel_major")
    append({
        "id": "preconv", "op": "audio_conv1d_checked",
        "kernel": "audio_conv1d_checked_channel_major_f32",
        "returns_status": True,
        "weight_refs": {"weight": "codec.decoder.pre_conv.conv.weight",
                        "bias": "codec.decoder.pre_conv.conv.bias"},
        "graph_slots": {"inputs": {"input": "preconv_padded"},
                        "outputs": {"output": "preconv_output"}},
        "params": {"I": output_width, "O": latent_width,
                   "T": frames + 2, "K": 3, "U": frames,
                   "Q": latent_width * frames,
                   "call_constants": {
            "conv_input_elements": output_width * (frames + 2),
            "conv_input_stride": frames + 2,
            "conv_weight_elements": latent_width * output_width * 3,
            "conv_bias_elements": latent_width,
            "conv_output_elements": latent_width * frames,
            "conv_output_stride": frames,
            "conv_scratch_elements": latent_width * frames,
            "conv_input_channels": output_width,
            "conv_output_channels": latent_width,
            "conv_input_frames": frames + 2,
            "conv_kernel_size": 3,
            "conv_stride": 1, "conv_padding": 0,
            "conv_output_frames": frames}},
    }, "channel_major")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec pre-convolution circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
