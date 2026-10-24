#!/usr/bin/env python3
"""Connect all three residual units after Qwen3-TTS's first decoder upsample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_decoder_upsample_circuit import build_circuit as upsample_circuit
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_first_residual_pool_bounded.json"


def _snake(input_name: str, output_name: str,
           channels: int, frames: int, weights: str) -> dict:
    return {
        "id": output_name, "op": "audio_snake_beta_strided_checked",
        "kernel": "audio_snake_beta_strided_f32_checked", "returns_status": True,
        "weight_refs": {"alpha": f"{weights}.alpha", "beta": f"{weights}.beta"},
        "graph_slots": {"inputs": {"input": input_name},
                        "outputs": {"output": output_name}},
        "params": {"C": channels, "T": frames, "call_constants": {
            "snake_beta_input_elements": channels * frames,
            "snake_beta_input_stride": frames,
            "snake_beta_alpha_elements": channels,
            "snake_beta_beta_elements": channels,
            "snake_beta_output_elements": channels * frames,
            "snake_beta_output_stride": frames,
            "snake_beta_channels": channels,
            "snake_beta_frames": frames}},
    }


def _conv(input_name: str, output_name: str,
          channels: int, input_frames: int, output_frames: int,
          kernel: int, dilation: int, weights: str) -> dict:
    return {
        "id": output_name, "op": "audio_conv1d_dilated_checked",
        "kernel": "audio_conv1d_dilated_checked_channel_major_f32",
        "returns_status": True,
        "weight_refs": {"weight": f"{weights}.weight", "bias": f"{weights}.bias"},
        "graph_slots": {"inputs": {"input": input_name},
                        "outputs": {"output": output_name}},
        "params": {"I": channels, "O": channels, "T": input_frames,
                   "K": kernel, "U": output_frames,
                   "Q": channels * output_frames, "call_constants": {
            "conv_input_elements": channels * input_frames,
            "conv_input_stride": input_frames,
            "conv_weight_elements": channels * channels * kernel,
            "conv_bias_elements": channels,
            "conv_output_elements": channels * output_frames,
            "conv_output_stride": output_frames,
            "conv_scratch_elements": channels * output_frames,
            "conv_input_channels": channels,
            "conv_output_channels": channels,
            "conv_input_frames": input_frames,
            "conv_kernel_size": kernel,
            "conv_stride": 1, "conv_padding": 0,
            "conv_dilation": dilation,
            "conv_output_frames": output_frames}},
    }


def _unit(graph: dict, unit: int, dilation: int,
          input_name: str, channels: int, frames: int) -> str:
    stem = f"decoder_residual_{unit}"
    weights = f"codec.decoder.decoder.1.block.{unit + 2}"
    pad = 6 * dilation
    names = {part: f"{stem}_{part}" for part in
             ("act1", "padded", "conv1", "act2", "conv2", "output")}
    ops = [
        _snake(input_name, names["act1"], channels, frames,
               f"{weights}.act1"),
        {
            "id": names["padded"], "op": "audio_zero_pad1d_left_checked",
            "kernel": "audio_zero_pad1d_left_channel_major_f32_checked",
            "returns_status": True,
            "graph_slots": {"inputs": {"input": names["act1"]},
                            "outputs": {"output": names["padded"]}},
            "params": {"C": channels, "T": frames, "U": frames + pad,
                       "call_constants": {
                "zero_pad_input_elements": channels * frames,
                "zero_pad_input_stride": frames,
                "zero_pad_output_elements": channels * (frames + pad),
                "zero_pad_output_stride": frames + pad,
                "zero_pad_channels": channels,
                "zero_pad_input_frames": frames,
                "zero_pad_left_padding": pad,
                "zero_pad_output_frames": frames + pad}},
        },
        _conv(names["padded"], names["conv1"], channels,
              frames + pad, frames, 7, dilation,
              f"{weights}.conv1.conv"),
        _snake(names["conv1"], names["act2"], channels, frames,
               f"{weights}.act2"),
        _conv(names["act2"], names["conv2"], channels,
              frames, frames, 1, 1, f"{weights}.conv2.conv"),
        {
            "id": names["output"], "op": "audio_scaled_sum_strided_checked",
            "kernel": "audio_scaled_sum_strided_f32_checked",
            "returns_status": True,
            "graph_slots": {"inputs": {"left": names["conv2"],
                                      "right": input_name},
                            "outputs": {"output": names["output"]}},
            "params": {"R": channels, "C": frames, "scale": 1.0,
                       "call_constants": {
                "sum_left_elements": channels * frames,
                "sum_left_stride": frames,
                "sum_right_elements": channels * frames,
                "sum_right_stride": frames,
                "sum_output_elements": channels * frames,
                "sum_output_stride": frames,
                "sum_rows": channels,
                "sum_columns": frames}},
        },
    ]
    for op in ops:
        name = op["id"]
        width = frames + pad if name == names["padded"] else frames
        graph["activation_buffers"][name] = {"shape": [channels, width]}
        graph["activation_bindings"][name] = name
        graph["semantic_checkpoints"]["exports"][name] = checkpoint(op, "channel_major")
    graph["block_types"]["codec_semantic"]["body"]["ops"].extend(ops)
    return names["output"]


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8,
                  depthwise_kernel: int = 7, expansion: int = 4,
                  decoder_width: int = 1536, ingress_kernel: int = 7,
                  decoder_upsample_rate: int = 8) -> dict:
    graph = upsample_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width, layers=layers,
        depthwise_kernel=depthwise_kernel, expansion=expansion,
        decoder_width=decoder_width, ingress_kernel=ingress_kernel,
        decoder_upsample_rate=decoder_upsample_rate)
    graph["name"] = "qwen3_tts_codec_first_residual_pool_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_first_residual_pool_bounded"
    graph["contract"]["runtime_invariants"]["first_residual_pool_units"] = 3
    channels = decoder_width // 2
    output_frames = frames * 4 * decoder_upsample_rate
    previous = "decoder_upsample_output"
    for unit, dilation in enumerate((1, 3, 9)):
        previous = _unit(graph, unit, dilation, previous, channels, output_frames)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec first-residual-pool circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
