#!/usr/bin/env python3
"""Connect the first complete ConvNeXt block of the pinned codec decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_depthwise_circuit import build_circuit as depthwise_circuit
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_convnext_complete_bounded.json"
PREFIX = "codec.decoder.upsample.0.1"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8,
                  depthwise_kernel: int = 7,
                  expansion: int = 4) -> dict:
    if isinstance(expansion, bool) or not isinstance(expansion, int) or expansion < 1:
        raise ValueError("ConvNeXt expansion must be positive")
    graph = depthwise_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width, layers=layers,
        depthwise_kernel=depthwise_kernel)
    graph["name"] = "qwen3_tts_codec_convnext_complete_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_convnext_complete_bounded"
    graph["contract"]["runtime_invariants"]["first_convnext_complete"] = True
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
    exports = graph["semantic_checkpoints"]["exports"]
    length = frames * 2
    wide = latent_width * expansion
    for name, shape in (
        ("convnext_depthwise_token", [length, latent_width]),
        ("convnext_norm", [length, latent_width]),
        ("convnext_pw1", [length, wide]),
        ("convnext_gelu", [length, wide]),
        ("convnext_pw2", [length, latent_width]),
        ("convnext_scaled", [length, latent_width]),
        ("convnext_scaled_channel", [latent_width, length]),
        ("convnext_output", [latent_width, length]),
    ):
        shapes[name] = {"shape": shape}
        bindings[name] = name

    def append(op: dict, layout: str = "token_major") -> None:
        ops.append(op)
        exports[op["id"]] = checkpoint(op, layout)

    def transpose(name: str, source: str, rows: int, columns: int,
                  layout: str) -> None:
        append({
            "id": name, "op": "transpose_strided_checked",
            "kernel": "transpose_strided_f32_checked", "returns_status": True,
            "graph_slots": {"inputs": {"input": source},
                            "outputs": {"output": name}},
            "params": {"R": rows, "C": columns, "call_constants": {
                "transpose_input_elements": rows * columns,
                "transpose_input_stride": columns,
                "transpose_output_elements": rows * columns,
                "transpose_output_stride": rows,
                "transpose_rows": rows, "transpose_columns": columns}},
        }, layout)

    transpose("convnext_depthwise_token", "depthwise_output",
              latent_width, length, "token_major")
    append({
        "id": "convnext_norm", "op": "layernorm_rows_param_checked",
        "kernel": "layernorm_rows_checked_param_epsilon_f32",
        "returns_status": True,
        "weight_refs": {"gamma": f"{PREFIX}.norm.weight",
                        "beta": f"{PREFIX}.norm.bias"},
        "graph_slots": {"inputs": {"input": "convnext_depthwise_token"},
                        "outputs": {"output": "convnext_norm"}},
        "params": {"M": length, "C": latent_width,
                   "Q": length * (latent_width + 2), "epsilon": 1e-6,
                   "call_constants": {
            "norm_input_elements": length * latent_width,
            "norm_input_stride": latent_width,
            "norm_gamma_elements": latent_width,
            "norm_beta_elements": latent_width,
            "norm_output_elements": length * latent_width,
            "norm_output_stride": latent_width,
            "norm_scratch_elements": length * (latent_width + 2),
            "norm_rows": length, "norm_channels": latent_width}},
    })

    def linear(name: str, source: str, input_width: int,
               output_width: int, weight_name: str) -> None:
        append({
            "id": name, "op": "linear_rows_checked",
            "kernel": "linear_rows_checked_f32", "returns_status": True,
            "weight_refs": {"weight": f"{PREFIX}.{weight_name}.weight",
                            "bias": f"{PREFIX}.{weight_name}.bias"},
            "graph_slots": {"inputs": {"input": source},
                            "outputs": {"output": name}},
            "params": {"M": length, "K": input_width, "N": output_width,
                       "call_constants": {
                "linear_input_elements": length * input_width,
                "linear_input_stride": input_width,
                "linear_weight_elements": output_width * input_width,
                "linear_weight_stride": input_width,
                "linear_bias_elements": output_width,
                "linear_output_elements": length * output_width,
                "linear_output_stride": output_width,
                "linear_rows": length,
                "linear_input_channels": input_width,
                "linear_output_channels": output_width}},
        })

    linear("convnext_pw1", "convnext_norm", latent_width, wide, "pwconv1")
    append({
        "id": "convnext_gelu", "op": "gelu",
        "kernel": "gelu_rows_erf_checked_f32", "returns_status": True,
        "graph_slots": {"inputs": {"input": "convnext_pw1"},
                        "outputs": {"output": "convnext_gelu"}},
        "params": {"M": length, "C": wide, "Q": length * wide,
                   "call_constants": {
            "gelu_input_elements": length * wide,
            "gelu_input_stride": wide,
            "gelu_output_elements": length * wide,
            "gelu_output_stride": wide,
            "gelu_scratch_elements": length * wide,
            "gelu_rows": length, "gelu_channels": wide}},
    })
    linear("convnext_pw2", "convnext_gelu", wide, latent_width, "pwconv2")
    append({
        "id": "convnext_scaled", "op": "channel_scale_rows_checked",
        "kernel": "channel_scale_rows_checked_f32", "returns_status": True,
        "weight_refs": {"scale": f"{PREFIX}.gamma"},
        "graph_slots": {"inputs": {"input": "convnext_pw2"},
                        "outputs": {"output": "convnext_scaled"}},
        "params": {"R": length, "C": latent_width,
                   "call_constants": {
            "input_elements": length * latent_width,
            "input_stride": latent_width,
            "scale_elements": latent_width,
            "output_elements": length * latent_width,
            "output_stride": latent_width,
            "rows": length, "channels": latent_width}},
    })
    transpose("convnext_scaled_channel", "convnext_scaled",
              length, latent_width, "channel_major")
    count = length * latent_width
    append({
        "id": "convnext_output", "op": "add_rows_checked",
        "kernel": "add_rows_checked_f32", "returns_status": True,
        "graph_slots": {"inputs": {"left": "first_upsample",
                                   "right": "convnext_scaled_channel"},
                        "outputs": {"output": "convnext_output"}},
        "params": {"N": count, "call_constants": {
            "left_elements": count, "right_elements": count,
            "output_elements": count, "elements": count}},
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
            raise SystemExit("codec ConvNeXt circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
