#!/usr/bin/env python3
"""Extend generated codec Q/K/V through checked causal sliding attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_qkv_circuit import build_circuit as qkv_circuit
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_attention_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72) -> dict:
    if (isinstance(sliding_window, bool) or
            not isinstance(sliding_window, int) or
            not 1 <= sliding_window <= 8000):
        raise ValueError("sliding_window must be an integer from 1 to 8000")
    graph = qkv_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width,
        heads=heads, head_dim=head_dim)
    graph["name"] = "qwen3_tts_codec_attention_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_attention_bounded"
    graph["contract"]["runtime_invariants"]["sliding_window"] = sliding_window
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]
    for name in ("first_attention_work", "first_attention_head"):
        shapes[name] = {"shape": [heads, frames, head_dim]}
        bindings[name] = name
    op = {
        "id": "first_attention_head", "op": "attention_sliding_bounded",
        "kernel": "attention_sliding_head_major_bounded_f32",
        "returns_status": True,
        "graph_slots": {
            "inputs": {"q": "first_q_head", "k": "first_k_head",
                       "v": "first_v_head"},
            "outputs": {"work": "first_attention_work",
                        "out": "first_attention_head"}},
        "params": {"H": heads, "KV": heads, "T": frames,
                   "D": head_dim, "W": sliding_window,
                   "call_constants": {
            "q_elements": heads * frames * head_dim,
            "k_elements": heads * frames * head_dim,
            "v_elements": heads * frames * head_dim,
            "work_elements": heads * frames * head_dim,
            "output_elements": heads * frames * head_dim,
            "heads": heads, "kv_heads": heads,
            "tokens": frames, "head_dim": head_dim,
            "sliding_window": sliding_window}},
    }
    graph["block_types"]["codec_semantic"]["body"]["ops"].append(op)
    contract = checkpoint(op, "head_major")
    contract["checkpoints"][0]["tensor"] = "first_attention_head"
    contract["checkpoints"][0]["axis_names"] = ["head", "frame", "channel"]
    graph["semantic_checkpoints"]["exports"][op["id"]] = contract
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec attention circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
