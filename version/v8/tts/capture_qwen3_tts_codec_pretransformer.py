#!/usr/bin/env python3
"""Capture every codec pre-transformer layer from the pinned full-model decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_qwen3_tts_customvoice_oracle import MODEL_REVISION, REFERENCE_REVISION
from capture_qwen3_tts_codec_attention import FIXTURES, load_pinned_transformer


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(model_dir: Path, output_dir: Path) -> None:
    import torch

    transformer, inputs, config, source_dir = load_pinned_transformer(model_dir)
    values: dict[str, np.ndarray] = {}
    handles = []

    def record(name):
        def save(_module, _args, output):
            if name in values:
                raise RuntimeError(f"duplicate codec checkpoint: {name}")
            values[name] = np.ascontiguousarray(
                output[0].detach().cpu().numpy().astype(np.float32))
        return save

    for layer, module in enumerate(transformer.layers):
        name = "first_block_output" if layer == 0 else f"layer_{layer}_block_output"
        handles.append(module.register_forward_hook(record(name)))
    handles.append(transformer.norm.register_forward_hook(record("decoder_final_norm")))
    handles.append(transformer.output_proj.register_forward_hook(record("decoder_output")))
    try:
        with torch.inference_mode():
            transformer(inputs_embeds=inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(values) != config.num_hidden_layers + 2:
        raise RuntimeError("incomplete codec pre-transformer checkpoint capture")
    for name, value in values.items():
        width = config.latent_dim if name == "decoder_output" else config.hidden_size
        if value.shape != (inputs.shape[1], width) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {name} checkpoint: {value.shape}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "qwen3_tts_codec_pretransformer.npz"
    np.savez_compressed(output, **values)
    manifest = {
        "schema": 1,
        "scope": "direct full codec pre-transformer checkpoints from captured reference codes; vocoder and waveform NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "codec_source_sha256": digest(source_dir / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"),
        "source_preconv_sha256": digest(FIXTURES / "qwen3_tts_codec_preconv.npz"),
        "source_first_block_sha256": digest(FIXTURES / "qwen3_tts_codec_first_block.npz"),
        "npz_sha256": digest(output),
        "frames": inputs.shape[1],
        "hidden_width": config.hidden_size,
        "latent_width": config.latent_dim,
        "layers": config.num_hidden_layers,
        "sliding_window": config.sliding_window,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "attention_implementation": "eager",
    }
    (output_dir / "qwen3_tts_codec_pretransformer.json").write_text(
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
