#!/usr/bin/env python3
"""Extend the generated codec through the first ConvNeXt causal depthwise convolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_first_upsample_circuit import build_circuit as first_upsample
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_depthwise_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8,
                  depthwise_kernel: int = 7) -> dict:
    if isinstance(depthwise_kernel, bool) or not isinstance(depthwise_kernel, int) or depthwise_kernel < 1:
        raise ValueError("depthwise_kernel must be positive")
    graph = first_upsample(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width, layers=layers)
    graph["name"] = "qwen3_tts_codec_depthwise_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_depthwise_bounded"
    graph["contract"]["runtime_invariants"]["first_convnext_causal_kernel"] = depthwise_kernel
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
    exports = graph["semantic_checkpoints"]["exports"]
    length = frames * 2
    padded = length + depthwise_kernel - 1
    for name, shape in (("depthwise_padded", [latent_width, padded]),
                        ("depthwise_output", [latent_width, length])):
        shapes[name] = {"shape": shape}
        bindings[name] = name
    pad = {
        "id": "depthwise_left_pad", "op": "audio_zero_pad1d_left_checked",
        "kernel": "audio_zero_pad1d_left_channel_major_f32_checked",
        "returns_status": True,
        "graph_slots": {"inputs": {"input": "first_upsample"},
                        "outputs": {"output": "depthwise_padded"}},
        "params": {"C": latent_width, "T": length, "U": padded,
                   "call_constants": {
            "zero_pad_input_elements": latent_width * length,
            "zero_pad_input_stride": length,
            "zero_pad_output_elements": latent_width * padded,
            "zero_pad_output_stride": padded,
            "zero_pad_channels": latent_width,
            "zero_pad_input_frames": length,
            "zero_pad_left_padding": depthwise_kernel - 1,
            "zero_pad_output_frames": padded}},
    }
    conv = {
        "id": "depthwise_output", "op": "audio_conv1d_depthwise_checked",
        "kernel": "audio_conv1d_depthwise_channel_major_f32_checked",
        "returns_status": True,
        "weight_refs": {
            "weight": "codec.decoder.upsample.0.1.dwconv.conv.weight",
            "bias": "codec.decoder.upsample.0.1.dwconv.conv.bias"},
        "graph_slots": {"inputs": {"input": "depthwise_padded"},
                        "outputs": {"output": "depthwise_output"}},
        "params": {"C": latent_width, "T": padded, "K": depthwise_kernel,
                   "U": length, "Q": latent_width * length,
                   "call_constants": {
            "conv_input_elements": latent_width * padded,
            "conv_input_stride": padded,
            "conv_weight_elements": latent_width * depthwise_kernel,
            "conv_bias_elements": latent_width,
            "conv_output_elements": latent_width * length,
            "conv_output_stride": length,
            "conv_scratch_elements": latent_width * length,
            "conv_channels": latent_width,
            "conv_input_frames": padded,
            "conv_kernel_size": depthwise_kernel,
            "conv_stride": 1,
            "conv_output_frames": length}},
    }
    for op in (pad, conv):
        ops.append(op)
        exports[op["id"]] = checkpoint(op, "channel_major")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec depthwise circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
