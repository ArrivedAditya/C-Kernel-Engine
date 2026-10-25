#!/usr/bin/env python3
"""Connect Qwen3-TTS codec's second upsample and three residual units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_first_residual_pool_circuit import (
    _snake, _unit, build_circuit as first_pool_circuit,
)
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_second_decoder_block_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8,
                  depthwise_kernel: int = 7, expansion: int = 4,
                  decoder_width: int = 1536, ingress_kernel: int = 7,
                  decoder_upsample_rate: int = 8,
                  second_decoder_upsample_rate: int = 5) -> dict:
    if (isinstance(second_decoder_upsample_rate, bool)
            or not isinstance(second_decoder_upsample_rate, int)
            or second_decoder_upsample_rate < 1):
        raise ValueError("second decoder upsample rate must be positive")
    if decoder_width % 4:
        raise ValueError("decoder width must be divisible by four")
    graph = first_pool_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width, layers=layers,
        depthwise_kernel=depthwise_kernel, expansion=expansion,
        decoder_width=decoder_width, ingress_kernel=ingress_kernel,
        decoder_upsample_rate=decoder_upsample_rate)
    graph["name"] = "qwen3_tts_codec_second_decoder_block_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_second_decoder_block_bounded"
    graph["contract"]["runtime_invariants"]["second_decoder_upsample_rate"] = second_decoder_upsample_rate
    input_channels = decoder_width // 2
    output_channels = decoder_width // 4
    input_frames = frames * 4 * decoder_upsample_rate
    output_frames = input_frames * second_decoder_upsample_rate
    kernel = 2 * second_decoder_upsample_rate
    activation = "decoder_second_snake_beta_output"
    upsample = "decoder_second_upsample_output"
    source = "decoder_residual_2_output"
    op = _snake(source, activation, input_channels, input_frames,
                "codec.decoder.decoder.2.block.0")
    graph["activation_buffers"][activation] = {"shape": [input_channels, input_frames]}
    graph["activation_bindings"][activation] = activation
    graph["semantic_checkpoints"]["exports"][activation] = checkpoint(op, "channel_major")
    graph["block_types"]["codec_semantic"]["body"]["ops"].append(op)
    op = {
        "id": upsample, "op": "audio_conv_transpose1d_dense_crop_right_checked",
        "kernel": "audio_conv_transpose1d_dense_crop_right_channel_major_f32_checked",
        "returns_status": True,
        "weight_refs": {
            "weight": "codec.decoder.decoder.2.block.1.conv.weight",
            "bias": "codec.decoder.decoder.2.block.1.conv.bias"},
        "graph_slots": {"inputs": {"input": activation},
                        "outputs": {"output": upsample}},
        "params": {"I": input_channels, "O": output_channels,
                   "T": input_frames, "K": kernel, "U": output_frames,
                   "Q": output_channels * output_frames,
                   "call_constants": {
            "cropped_deconv_input_elements": input_channels * input_frames,
            "cropped_deconv_input_stride": input_frames,
            "cropped_deconv_weight_elements": input_channels * output_channels * kernel,
            "cropped_deconv_bias_elements": output_channels,
            "cropped_deconv_output_elements": output_channels * output_frames,
            "cropped_deconv_output_stride": output_frames,
            "cropped_deconv_scratch_elements": output_channels * output_frames,
            "cropped_deconv_input_channels": input_channels,
            "cropped_deconv_output_channels": output_channels,
            "cropped_deconv_input_frames": input_frames,
            "cropped_deconv_kernel_size": kernel,
            "cropped_deconv_stride": second_decoder_upsample_rate,
            "cropped_deconv_padding": 0,
            "cropped_deconv_output_padding": 0,
            "cropped_deconv_right_crop": second_decoder_upsample_rate,
            "cropped_deconv_output_frames": output_frames}},
    }
    graph["activation_buffers"][upsample] = {"shape": [output_channels, output_frames]}
    graph["activation_bindings"][upsample] = upsample
    graph["semantic_checkpoints"]["exports"][upsample] = checkpoint(op, "channel_major")
    graph["block_types"]["codec_semantic"]["body"]["ops"].append(op)
    source = upsample
    for unit, dilation in enumerate((1, 3, 9)):
        source = _unit(graph, unit, dilation, source,
                       output_channels, output_frames,
                       decoder_block=2, stem_prefix="decoder_second_residual")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec second-decoder circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
