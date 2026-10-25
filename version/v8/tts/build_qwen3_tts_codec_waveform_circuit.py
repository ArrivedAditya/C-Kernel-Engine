#!/usr/bin/env python3
"""Connect the final Qwen3-TTS codec decoder stage and bounded PCM output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_decoder_stage import append_decoder_stage
from build_qwen3_tts_codec_first_residual_pool_circuit import _snake
from build_qwen3_tts_codec_rvq_circuit import checkpoint
from build_qwen3_tts_codec_third_decoder_block_circuit import build_circuit as third_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_waveform_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8,
                  depthwise_kernel: int = 7, expansion: int = 4,
                  decoder_width: int = 1536, ingress_kernel: int = 7,
                  decoder_upsample_rate: int = 8,
                  second_decoder_upsample_rate: int = 5,
                  third_decoder_upsample_rate: int = 4,
                  fourth_decoder_upsample_rate: int = 3) -> dict:
    if (isinstance(fourth_decoder_upsample_rate, bool)
            or not isinstance(fourth_decoder_upsample_rate, int)
            or fourth_decoder_upsample_rate < 1):
        raise ValueError("fourth decoder upsample rate must be positive")
    if decoder_width % 16:
        raise ValueError("decoder width must be divisible by sixteen")
    graph = third_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width, layers=layers,
        depthwise_kernel=depthwise_kernel, expansion=expansion,
        decoder_width=decoder_width, ingress_kernel=ingress_kernel,
        decoder_upsample_rate=decoder_upsample_rate,
        second_decoder_upsample_rate=second_decoder_upsample_rate,
        third_decoder_upsample_rate=third_decoder_upsample_rate)
    graph["name"] = "qwen3_tts_codec_waveform_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_waveform_bounded"
    graph["contract"]["runtime_invariants"]["fourth_decoder_upsample_rate"] = fourth_decoder_upsample_rate
    input_frames = (frames * 4 * decoder_upsample_rate *
                    second_decoder_upsample_rate * third_decoder_upsample_rate)
    output_frames = input_frames * fourth_decoder_upsample_rate
    output_channels = decoder_width // 16
    previous = append_decoder_stage(
        graph, stage_name="fourth", decoder_block=4,
        input_name="decoder_third_residual_2_output",
        input_channels=decoder_width // 8,
        input_frames=input_frames, rate=fourth_decoder_upsample_rate)
    ops = graph["block_types"]["codec_semantic"]["body"]["ops"]

    def add(op: dict, name: str, shape: list[int]) -> None:
        graph["activation_buffers"][name] = {"shape": shape}
        graph["activation_bindings"][name] = name
        graph["semantic_checkpoints"]["exports"][name] = checkpoint(op, "channel_major")
        ops.append(op)

    name = "decoder_final_snake_beta_output"
    add(_snake(previous, name, output_channels, output_frames,
               "codec.decoder.decoder.5"), name, [output_channels, output_frames])
    padded = "decoder_final_padded_output"
    pad = 6
    add({
        "id": padded, "op": "audio_zero_pad1d_left_checked",
        "kernel": "audio_zero_pad1d_left_channel_major_f32_checked",
        "returns_status": True,
        "graph_slots": {"inputs": {"input": name}, "outputs": {"output": padded}},
        "params": {"C": output_channels, "T": output_frames,
                   "U": output_frames + pad, "call_constants": {
            "zero_pad_input_elements": output_channels * output_frames,
            "zero_pad_input_stride": output_frames,
            "zero_pad_output_elements": output_channels * (output_frames + pad),
            "zero_pad_output_stride": output_frames + pad,
            "zero_pad_channels": output_channels,
            "zero_pad_input_frames": output_frames,
            "zero_pad_left_padding": pad,
            "zero_pad_output_frames": output_frames + pad}},
    }, padded, [output_channels, output_frames + pad])
    preclip = "decoder_waveform_preclip_output"
    add({
        "id": preclip, "op": "audio_conv1d_dilated_checked",
        "kernel": "audio_conv1d_dilated_checked_channel_major_f32",
        "returns_status": True,
        "weight_refs": {"weight": "codec.decoder.decoder.6.conv.weight",
                        "bias": "codec.decoder.decoder.6.conv.bias"},
        "graph_slots": {"inputs": {"input": padded},
                        "outputs": {"output": preclip}},
        "params": {"I": output_channels, "O": 1, "T": output_frames + pad,
                   "K": 7, "U": output_frames, "Q": output_frames,
                   "call_constants": {
            "conv_input_elements": output_channels * (output_frames + pad),
            "conv_input_stride": output_frames + pad,
            "conv_weight_elements": output_channels * 7,
            "conv_bias_elements": 1,
            "conv_output_elements": output_frames,
            "conv_output_stride": output_frames,
            "conv_scratch_elements": output_frames,
            "conv_input_channels": output_channels,
            "conv_output_channels": 1,
            "conv_input_frames": output_frames + pad,
            "conv_kernel_size": 7,
            "conv_stride": 1, "conv_padding": 0,
            "conv_dilation": 1,
            "conv_output_frames": output_frames}},
    }, preclip, [1, output_frames])
    output = "waveform_output"
    add({
        "id": output, "op": "clip_strided_checked",
        "kernel": "clip_strided_f32_checked", "returns_status": True,
        "graph_slots": {"inputs": {"input": preclip},
                        "outputs": {"output": output}},
        "params": {"R": 1, "C": output_frames, "minimum": -1.0,
                   "maximum": 1.0, "call_constants": {
            "clip_input_elements": output_frames,
            "clip_input_stride": output_frames,
            "clip_output_elements": output_frames,
            "clip_output_stride": output_frames,
            "clip_rows": 1, "clip_columns": output_frames}},
    }, output, [1, output_frames])
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec waveform circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
