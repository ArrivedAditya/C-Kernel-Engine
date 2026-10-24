#!/usr/bin/env python3
"""Capture direct pinned first codec pre-transformer sliding attention output."""

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
    from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as modeling

    source_dir = Path(qwen_tts.__file__).resolve().parent
    verify_reference_code(source_dir)
    torch.set_num_threads(4)
    decoder_config = json.loads(
        (model_dir / "speech_tokenizer/config.json").read_text())["decoder_config"]
    config = Qwen3TTSTokenizerV2DecoderConfig(**decoder_config)
    config._attn_implementation = "eager"
    transformer = modeling.Qwen3TTSTokenizerV2DecoderTransformerModel(config).eval()
    with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                   framework="pt", device="cpu") as weights:
        prefix = "decoder.pre_transformer."
        state = {name[len(prefix):]: weights.get_tensor(name)
                 for name in weights.keys() if name.startswith(prefix)}
    transformer.load_state_dict(state, strict=True)
    prior = FIXTURES / "qwen3_tts_codec_preconv.npz"
    with np.load(prior) as values:
        inputs = torch.from_numpy(values["preconv_output"].T.copy()).unsqueeze(0)
    captured = []
    original = modeling.eager_attention_forward

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        if not captured:
            # The official eager function returns [batch, frame, head, channel].
            captured.append(result[0][0].transpose(0, 1).contiguous()
                              .detach().cpu().numpy().astype(np.float32))
        return result

    modeling.eager_attention_forward = record
    try:
        with torch.inference_mode():
            transformer(inputs_embeds=inputs, use_cache=False)
    finally:
        modeling.eager_attention_forward = original
    if len(captured) != 1 or captured[0].shape != (16, inputs.shape[1], 64):
        raise RuntimeError("first pinned attention checkpoint was not captured")
    if not np.isfinite(captured[0]).all():
        raise RuntimeError("nonfinite reference attention output")
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "qwen3_tts_codec_attention.npz"
    np.savez_compressed(npz_path, first_attention_head=captured[0])
    manifest = {
        "schema": 1,
        "scope": "direct pinned full pre-transformer first sliding attention output from captured pre-convolution input; waveform NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "codec_source_sha256": sha256(
            source_dir / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"),
        "source_preconv_sha256": sha256(prior),
        "source_qkv_sha256": sha256(FIXTURES / "qwen3_tts_codec_qkv.npz"),
        "npz_sha256": sha256(npz_path),
        "frames": inputs.shape[1],
        "sliding_window": config.sliding_window,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "attention_implementation": "eager",
    }
    (output_dir / "qwen3_tts_codec_attention.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(output_dir / "qwen3_tts_codec_attention.json"),
                      "npz_sha256": manifest["npz_sha256"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=FIXTURES)
    args = parser.parse_args()
    capture(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
