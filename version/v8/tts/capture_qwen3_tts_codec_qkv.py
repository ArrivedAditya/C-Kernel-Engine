#!/usr/bin/env python3
"""Capture official codec pre-transformer ingress and first Q/K/V RoPE operations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_qwen3_tts_customvoice_oracle import (
    MODEL_REVISION, REFERENCE_REVISION, verify_model_dir,
    verify_reference_code,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/tts/qwen3_tts_0b6_customvoice"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(model_dir: Path, output_dir: Path) -> None:
    verify_model_dir(model_dir)
    import qwen_tts
    import torch
    from safetensors import safe_open
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderConfig,
    )
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderAttention,
        Qwen3TTSTokenizerV2DecoderRMSNorm,
        Qwen3TTSTokenizerV2DecoderRotatoryEmbedding,
        apply_rotary_pos_emb,
    )

    source_dir = Path(qwen_tts.__file__).resolve().parent
    verify_reference_code(source_dir)
    torch.set_num_threads(4)
    decoder_config = json.loads(
        (model_dir / "speech_tokenizer/config.json").read_text())["decoder_config"]
    config = Qwen3TTSTokenizerV2DecoderConfig(**decoder_config)
    prior = FIXTURES / "qwen3_tts_codec_preconv.npz"
    with np.load(prior) as values:
        source = torch.from_numpy(values["preconv_output"].T.copy()).unsqueeze(0)
    projection = torch.nn.Linear(config.latent_dim, config.hidden_size).eval()
    norm = Qwen3TTSTokenizerV2DecoderRMSNorm(
        config.hidden_size, config.rms_norm_eps).eval()
    attention = Qwen3TTSTokenizerV2DecoderAttention(config, 0).eval()
    rotary = Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(config).eval()
    with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                   framework="pt", device="cpu") as weights:
        prefix = "decoder.pre_transformer."
        with torch.no_grad():
            projection.weight.copy_(weights.get_tensor(prefix + "input_proj.weight"))
            projection.bias.copy_(weights.get_tensor(prefix + "input_proj.bias"))
            norm.weight.copy_(weights.get_tensor(
                prefix + "layers.0.input_layernorm.weight"))
            for name in ("q", "k", "v"):
                getattr(attention, f"{name}_proj").weight.copy_(
                    weights.get_tensor(
                        prefix + f"layers.0.self_attn.{name}_proj.weight"))
    with torch.inference_mode():
        projected = projection(source)
        normalized = norm(projected)
        q, k, v = (getattr(attention, f"{name}_proj")(normalized)
                   for name in ("q", "k", "v"))
        frames = source.shape[1]
        heads = config.num_attention_heads
        head_dim = config.head_dim
        q_head = q.view(1, frames, heads, head_dim).transpose(1, 2)
        k_head = k.view(1, frames, heads, head_dim).transpose(1, 2)
        v_head = v.view(1, frames, heads, head_dim).transpose(1, 2)
        positions = torch.arange(frames).unsqueeze(0)
        cos, sin = rotary(projected, positions)
        rope_q, rope_k = apply_rotary_pos_emb(q_head, k_head, cos, sin)
    tensors = {
        "transformer_input": projected[0],
        "first_input_norm": normalized[0],
        "first_q": q[0], "first_k": k[0], "first_v": v[0],
        "first_v_head": v_head[0],
        "first_rope_q": rope_q[0], "first_rope_k": rope_k[0],
    }
    arrays = {name: np.ascontiguousarray(value.numpy().astype(np.float32))
              for name, value in tensors.items()}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("nonfinite reference Q/K/V checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "qwen3_tts_codec_qkv.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "schema": 1,
        "scope": "official operation replay from captured pre-convolution; first attention output and waveform NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "codec_source_sha256": sha256(
            source_dir / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"),
        "source_preconv_sha256": sha256(prior),
        "npz_sha256": sha256(npz_path),
        "frames": frames,
        "heads": heads,
        "head_dim": head_dim,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
    }
    (output_dir / "qwen3_tts_codec_qkv.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(output_dir / "qwen3_tts_codec_qkv.json"),
                      "npz_sha256": manifest["npz_sha256"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=FIXTURES)
    args = parser.parse_args()
    capture(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
