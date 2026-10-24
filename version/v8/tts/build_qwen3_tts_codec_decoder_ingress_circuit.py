#!/usr/bin/env python3
"""Connect the codec's channel-changing causal decoder ingress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_rvq_circuit import checkpoint
from build_qwen3_tts_codec_second_convnext_circuit import build_circuit as second_block_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_decoder_ingress_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8,
                  depthwise_kernel: int = 7, expansion: int = 4,
                  decoder_width: int = 1536,
                  ingress_kernel: int = 7) -> dict:
    for name, value in (("decoder_width", decoder_width),
                        ("ingress_kernel", ingress_kernel)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    graph = second_block_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width, layers=layers,
        depthwise_kernel=depthwise_kernel, expansion=expansion)
    graph["name"] = "qwen3_tts_codec_decoder_ingress_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_decoder_ingress_bounded"
    graph["contract"]["runtime_invariants"]["decoder_ingress_kernel"] = ingress_kernel
    length = frames * 4
    padded = length + ingress_kernel - 1
    for name, shape in (
        ("decoder_ingress_padded", [latent_width, padded]),
        ("decoder_ingress_output", [decoder_width, length]),
    ):
        graph["activation_buffers"][name] = {"shape": shape}
        graph["activation_bindings"][name] = name
    ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
    exports = graph["semantic_checkpoints"]["exports"]
    pad = {
        "id": "decoder_ingress_left_pad", "op": "audio_zero_pad1d_left_checked",
        "kernel": "audio_zero_pad1d_left_channel_major_f32_checked",
        "returns_status": True,
        "graph_slots": {"inputs": {"input": "second_convnext_output"},
                        "outputs": {"output": "decoder_ingress_padded"}},
        "params": {"C": latent_width, "T": length, "U": padded,
                   "call_constants": {
            "zero_pad_input_elements": latent_width * length,
            "zero_pad_input_stride": length,
            "zero_pad_output_elements": latent_width * padded,
            "zero_pad_output_stride": padded,
            "zero_pad_channels": latent_width,
            "zero_pad_input_frames": length,
            "zero_pad_left_padding": ingress_kernel - 1,
            "zero_pad_output_frames": padded}},
    }
    conv = {
        "id": "decoder_ingress_output", "op": "audio_conv1d_checked",
        "kernel": "audio_conv1d_checked_channel_major_f32",
        "returns_status": True,
        "weight_refs": {"weight": "codec.decoder.decoder.0.conv.weight",
                        "bias": "codec.decoder.decoder.0.conv.bias"},
        "graph_slots": {"inputs": {"input": "decoder_ingress_padded"},
                        "outputs": {"output": "decoder_ingress_output"}},
        "params": {"I": latent_width, "O": decoder_width,
                   "T": padded, "K": ingress_kernel, "U": length,
                   "Q": decoder_width * length,
                   "call_constants": {
            "conv_input_elements": latent_width * padded,
            "conv_input_stride": padded,
            "conv_weight_elements": decoder_width * latent_width * ingress_kernel,
            "conv_bias_elements": decoder_width,
            "conv_output_elements": decoder_width * length,
            "conv_output_stride": length,
            "conv_scratch_elements": decoder_width * length,
            "conv_input_channels": latent_width,
            "conv_output_channels": decoder_width,
            "conv_input_frames": padded,
            "conv_kernel_size": ingress_kernel,
            "conv_stride": 1, "conv_padding": 0,
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
            raise SystemExit("codec decoder-ingress circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
