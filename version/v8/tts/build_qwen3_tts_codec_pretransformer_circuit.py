#!/usr/bin/env python3
"""Author the complete codec pre-transformer from reusable declared blocks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_qwen3_tts_codec_first_block_circuit import build_circuit as first_block_circuit
from build_qwen3_tts_codec_qkv_circuit import PREFIX
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_pretransformer_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8) -> dict:
    if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= 8:
        raise ValueError("codec pre-transformer layers must be from 1 to 8")
    graph = first_block_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width)
    graph["name"] = "qwen3_tts_codec_pretransformer_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_pretransformer_bounded"
    graph["contract"]["runtime_invariants"]["codec_transformer_layers"] = layers
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
    exports = graph["semantic_checkpoints"]["exports"]
    first_ops = [copy.deepcopy(op) for op in ops if op["id"].startswith("first_")]
    first_shapes = {name: copy.deepcopy(shape) for name, shape in shapes.items()
                    if name.startswith("first_")}
    first_exports = {name: copy.deepcopy(contract) for name, contract in exports.items()
                     if name.startswith("first_")}

    def rename(value, layer, source):
        if isinstance(value, str):
            return (value.replace(f"{PREFIX}.layers.0.", f"{PREFIX}.layers.{layer}.")
                    .replace("first_", f"layer_{layer}_")
                    .replace("transformer_input", source))
        if isinstance(value, list):
            return [rename(item, layer, source) for item in value]
        if isinstance(value, dict):
            return {rename(key, layer, source): rename(item, layer, source)
                    for key, item in value.items()}
        return value

    for layer in range(1, layers):
        source = "first_block_output" if layer == 1 else f"layer_{layer - 1}_block_output"
        for name, shape in first_shapes.items():
            new_name = rename(name, layer, source)
            shapes[new_name] = rename(shape, layer, source)
            bindings[new_name] = new_name
        for op in first_ops:
            ops.append(rename(op, layer, source))
        for name, contract in first_exports.items():
            exports[rename(name, layer, source)] = rename(contract, layer, source)

    source = "first_block_output" if layers == 1 else f"layer_{layers - 1}_block_output"
    shapes["decoder_final_norm"] = {"shape": [frames, hidden_width]}
    shapes["decoder_output"] = {"shape": [frames, latent_width]}
    bindings["decoder_final_norm"] = "decoder_final_norm"
    bindings["decoder_output"] = "decoder_output"
    norm = {
        "id": "decoder_final_norm", "op": "rmsnorm_rows_checked",
        "kernel": "rmsnorm_rows_checked_f32", "returns_status": True,
        "weight_refs": {"gamma": f"{PREFIX}.norm.weight"},
        "graph_slots": {"inputs": {"input": source},
                        "outputs": {"output": "decoder_final_norm"}},
        "params": {"R": frames, "C": hidden_width, "eps": 1e-5,
                   "call_constants": {
            "input_elements": frames * hidden_width,
            "input_stride": hidden_width, "gamma_elements": hidden_width,
            "output_elements": frames * hidden_width,
            "output_stride": hidden_width, "rows": frames, "width": hidden_width}},
    }
    projection = {
        "id": "decoder_output", "op": "linear_rows_checked",
        "kernel": "linear_rows_checked_f32", "returns_status": True,
        "weight_refs": {"weight": f"{PREFIX}.output_proj.weight",
                        "bias": f"{PREFIX}.output_proj.bias"},
        "graph_slots": {"inputs": {"input": "decoder_final_norm"},
                        "outputs": {"output": "decoder_output"}},
        "params": {"M": frames, "K": hidden_width, "N": latent_width,
                   "call_constants": {
            "linear_input_elements": frames * hidden_width,
            "linear_input_stride": hidden_width,
            "linear_weight_elements": latent_width * hidden_width,
            "linear_weight_stride": hidden_width,
            "linear_bias_elements": latent_width,
            "linear_output_elements": frames * latent_width,
            "linear_output_stride": latent_width,
            "linear_rows": frames, "linear_input_channels": hidden_width,
            "linear_output_channels": latent_width}},
    }
    for op in (norm, projection):
        ops.append(op)
        exports[op["id"]] = checkpoint(op, "token_major")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec pre-transformer circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
