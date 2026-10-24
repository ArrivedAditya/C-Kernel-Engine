#!/usr/bin/env python3
"""Capture direct full-model first codec transformer block checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_qwen3_tts_customvoice_oracle import MODEL_REVISION, REFERENCE_REVISION
from capture_qwen3_tts_codec_attention import (
    FIXTURES, load_pinned_transformer,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(model_dir: Path, output_dir: Path) -> None:
    import torch

    transformer, inputs, config, source_dir = load_pinned_transformer(model_dir)
    layer = transformer.layers[0]
    values = {}
    handles = []

    def record(name, *, before=False):
        def save(_module, args, output=None):
            tensor = args[0] if before else output
            if name in values:
                raise RuntimeError(f"duplicate first-block checkpoint: {name}")
            values[name] = np.ascontiguousarray(
                tensor[0].detach().cpu().numpy().astype(np.float32))
        return save

    for name, module, before in (
        ("first_attention_token", layer.self_attn.o_proj, True),
        ("first_attention_proj", layer.self_attn.o_proj, False),
        ("first_attention_scaled", layer.self_attn_layer_scale, False),
        ("first_post_attention", layer.post_attention_layernorm, True),
        ("first_post_attention_norm", layer.post_attention_layernorm, False),
        ("first_gate", layer.mlp.gate_proj, False),
        ("first_up", layer.mlp.up_proj, False),
        ("first_gated_mlp", layer.mlp.down_proj, True),
        ("first_down", layer.mlp.down_proj, False),
        ("first_mlp_scaled", layer.mlp_layer_scale, False),
        ("first_block_output", layer, False),
    ):
        handles.append(module.register_forward_pre_hook(record(name, before=True))
                       if before else module.register_forward_hook(record(name)))
    try:
        with torch.inference_mode():
            transformer(inputs_embeds=inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(values) != 11:
        raise RuntimeError("incomplete first-block direct checkpoint capture")
    for name, value in values.items():
        expected_width = (config.num_attention_heads * config.head_dim
                          if name == "first_attention_token" else
                          config.intermediate_size
                          if name in ("first_gate", "first_up", "first_gated_mlp")
                          else config.hidden_size)
        if value.shape != (inputs.shape[1], expected_width) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {name} checkpoint: {value.shape}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "qwen3_tts_codec_first_block.npz"
    np.savez_compressed(output, **values)
    manifest = {
        "schema": 1,
        "scope": "direct first codec transformer block checkpoints; reference codec codes; waveform NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "codec_source_sha256": digest(
            source_dir / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"),
        "source_preconv_sha256": digest(FIXTURES / "qwen3_tts_codec_preconv.npz"),
        "source_attention_sha256": digest(FIXTURES / "qwen3_tts_codec_attention.npz"),
        "npz_sha256": digest(output),
        "frames": inputs.shape[1],
        "hidden_width": config.hidden_size,
        "intermediate_width": config.intermediate_size,
        "sliding_window": config.sliding_window,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "attention_implementation": "eager",
    }
    (output_dir / "qwen3_tts_codec_first_block.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"npz_sha256": manifest["npz_sha256"],
                      "checkpoints": list(values)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=FIXTURES)
    args = parser.parse_args()
    capture(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
