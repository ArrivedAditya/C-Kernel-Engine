#!/usr/bin/env python3
"""Connect the first causal, right-cropped Qwen3-TTS decoder upsample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_snake_beta_circuit import build_circuit as snake_circuit
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_decoder_upsample_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8,
                  depthwise_kernel: int = 7, expansion: int = 4,
                  decoder_width: int = 1536, ingress_kernel: int = 7,
                  decoder_upsample_rate: int = 8) -> dict:
    if (isinstance(decoder_width, bool) or not isinstance(decoder_width, int)
            or decoder_width < 2 or decoder_width % 2):
        raise ValueError("decoder_width must be a positive even channel count")
    if (isinstance(decoder_upsample_rate, bool)
            or not isinstance(decoder_upsample_rate, int)
            or decoder_upsample_rate < 1):
        raise ValueError("decoder_upsample_rate must be a positive integer")
    graph = snake_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width, layers=layers,
        depthwise_kernel=depthwise_kernel, expansion=expansion,
        decoder_width=decoder_width, ingress_kernel=ingress_kernel)
    graph["name"] = "qwen3_tts_codec_decoder_upsample_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_decoder_upsample_bounded"
    graph["contract"]["runtime_invariants"]["decoder_upsample_rate"] = decoder_upsample_rate
    input_frames = frames * 4
    output_frames = input_frames * decoder_upsample_rate
    out_channels = decoder_width // 2
    kernel = decoder_upsample_rate * 2
    name = "decoder_upsample_output"
    graph["activation_buffers"][name] = {"shape": [out_channels, output_frames]}
    graph["activation_bindings"][name] = name
    op = {
        "id": name, "op": "audio_conv_transpose1d_dense_crop_right_checked",
        "kernel": "audio_conv_transpose1d_dense_crop_right_channel_major_f32_checked",
        "returns_status": True,
        "weight_refs": {
            "weight": "codec.decoder.decoder.1.block.1.conv.weight",
            "bias": "codec.decoder.decoder.1.block.1.conv.bias"},
        "graph_slots": {"inputs": {"input": "decoder_snake_beta_output"},
                        "outputs": {"output": name}},
        "params": {"I": decoder_width, "O": out_channels, "T": input_frames,
                   "K": kernel, "U": output_frames,
                   "Q": out_channels * output_frames,
                   "call_constants": {
            "cropped_deconv_input_elements": decoder_width * input_frames,
            "cropped_deconv_input_stride": input_frames,
            "cropped_deconv_weight_elements": decoder_width * out_channels * kernel,
            "cropped_deconv_bias_elements": out_channels,
            "cropped_deconv_output_elements": out_channels * output_frames,
            "cropped_deconv_output_stride": output_frames,
            "cropped_deconv_scratch_elements": out_channels * output_frames,
            "cropped_deconv_input_channels": decoder_width,
            "cropped_deconv_output_channels": out_channels,
            "cropped_deconv_input_frames": input_frames,
            "cropped_deconv_kernel_size": kernel,
            "cropped_deconv_stride": decoder_upsample_rate,
            "cropped_deconv_padding": 0,
            "cropped_deconv_output_padding": 0,
            "cropped_deconv_right_crop": decoder_upsample_rate,
            "cropped_deconv_output_frames": output_frames}}}
    graph["block_types"]["codec_semantic"]["body"]["ops"].append(op)
    graph["semantic_checkpoints"]["exports"][name] = checkpoint(op, "channel_major")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec decoder-upsample circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
