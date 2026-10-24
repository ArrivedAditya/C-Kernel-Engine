#!/usr/bin/env python3
"""Connect the codec decoder's first two-parameter SnakeBeta activation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_decoder_ingress_circuit import build_circuit as ingress_circuit
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_snake_beta_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8,
                  depthwise_kernel: int = 7, expansion: int = 4,
                  decoder_width: int = 1536, ingress_kernel: int = 7) -> dict:
    graph = ingress_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width, layers=layers,
        depthwise_kernel=depthwise_kernel, expansion=expansion,
        decoder_width=decoder_width, ingress_kernel=ingress_kernel)
    graph["name"] = "qwen3_tts_codec_snake_beta_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_snake_beta_bounded"
    graph["contract"]["runtime_invariants"]["decoder_snake_beta"] = True
    length = frames * 4
    name = "decoder_snake_beta_output"
    graph["activation_buffers"][name] = {"shape": [decoder_width, length]}
    graph["activation_bindings"][name] = name
    op = {
        "id": name, "op": "audio_snake_beta_strided_checked",
        "kernel": "audio_snake_beta_strided_f32_checked",
        "returns_status": True,
        "weight_refs": {
            "alpha": "codec.decoder.decoder.1.block.0.alpha",
            "beta": "codec.decoder.decoder.1.block.0.beta",
        },
        "graph_slots": {"inputs": {"input": "decoder_ingress_output"},
                        "outputs": {"output": name}},
        "params": {"C": decoder_width, "T": length,
                   "call_constants": {
            "snake_beta_input_elements": decoder_width * length,
            "snake_beta_input_stride": length,
            "snake_beta_alpha_elements": decoder_width,
            "snake_beta_beta_elements": decoder_width,
            "snake_beta_output_elements": decoder_width * length,
            "snake_beta_output_stride": length,
            "snake_beta_channels": decoder_width,
            "snake_beta_frames": length,
        }},
    }
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
            raise SystemExit("codec SnakeBeta circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
