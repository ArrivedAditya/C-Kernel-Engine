#!/usr/bin/env python3
"""Connect the second codec ConvNeXt block using the canonical first-block declaration."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_qwen3_tts_codec_convnext_complete_circuit import build_circuit as first_block_circuit
from build_qwen3_tts_codec_rvq_circuit import checkpoint
from build_qwen3_tts_codec_second_upsample_circuit import build_circuit as second_upsample_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_second_convnext_bounded.json"
FIRST_PREFIX = "codec.decoder.upsample.0.1"
SECOND_PREFIX = "codec.decoder.upsample.1.1"
FIRST_OPS = (
    "depthwise_left_pad", "depthwise_output", "convnext_depthwise_token",
    "convnext_norm", "convnext_pw1", "convnext_gelu", "convnext_pw2",
    "convnext_scaled", "convnext_scaled_channel", "convnext_output",
)
FIRST_BUFFERS = (
    "depthwise_padded", "depthwise_output", "convnext_depthwise_token",
    "convnext_norm", "convnext_pw1", "convnext_gelu", "convnext_pw2",
    "convnext_scaled", "convnext_scaled_channel", "convnext_output",
)


def _renamed(value, names):
    if isinstance(value, str):
        return names.get(value, value.replace(FIRST_PREFIX, SECOND_PREFIX))
    if isinstance(value, list):
        return [_renamed(item, names) for item in value]
    if isinstance(value, dict):
        return {key: _renamed(item, names) for key, item in value.items()}
    return value


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8,
                  depthwise_kernel: int = 7, expansion: int = 4) -> dict:
    args = dict(
        vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width, layers=layers,
        depthwise_kernel=depthwise_kernel, expansion=expansion)
    graph = second_upsample_circuit(frames=frames, **args)
    # The second block has the same width and operations as the first. Expand
    # its canonical declaration at the doubled frame length, then bind its
    # ports and weights to the second stage. This is authoring only: the
    # committed circuit contains explicit edges and normal codegen executes it.
    template = first_block_circuit(frames=frames * 2, **args)
    graph["name"] = "qwen3_tts_codec_second_convnext_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_second_convnext_bounded"
    graph["contract"]["runtime_invariants"]["second_convnext_complete"] = True
    names = {name: f"second_{name}" for name in set(FIRST_OPS + FIRST_BUFFERS)}
    names["first_upsample"] = "second_upsample"
    for original in FIRST_BUFFERS:
        name = names[original]
        graph["activation_buffers"][name] = copy.deepcopy(
            template["activation_buffers"][original])
        graph["activation_bindings"][name] = name
    template_ops = template["block_types"]["codec_semantic"]["body"]["ops"]
    selected = template_ops[-len(FIRST_OPS):]
    if tuple(op["id"] for op in selected) != FIRST_OPS:
        raise ValueError("canonical first ConvNeXt declaration changed")
    ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
    exports = graph["semantic_checkpoints"]["exports"]
    for original in selected:
        op = _renamed(original, names)
        ops.append(op)
        layout = template["semantic_checkpoints"]["exports"][original["id"]][
            "checkpoints"][0]["logical_layout"]
        exports[op["id"]] = checkpoint(op, layout)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec second ConvNeXt circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
