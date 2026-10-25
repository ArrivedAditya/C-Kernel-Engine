#!/usr/bin/env python3
"""Connect Qwen3-TTS codec's second upsample and three residual units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_first_residual_pool_circuit import (
    build_circuit as first_pool_circuit,
)
from build_qwen3_tts_codec_decoder_stage import append_decoder_stage

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
    append_decoder_stage(
        graph, stage_name="second", decoder_block=2,
        input_name="decoder_residual_2_output",
        input_channels=decoder_width // 2,
        input_frames=frames * 4 * decoder_upsample_rate,
        rate=second_decoder_upsample_rate)
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
