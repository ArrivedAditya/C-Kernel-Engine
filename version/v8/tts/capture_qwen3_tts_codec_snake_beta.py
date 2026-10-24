#!/usr/bin/env python3
"""Capture the first decoder SnakeBeta from the pinned full codec."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_qwen3_tts_customvoice_oracle import (
    MODEL_REVISION, REFERENCE_REVISION, verify_model_dir, verify_reference_code,
)
from capture_qwen3_tts_codec_attention import FIXTURES


def digest(path: Path) -> str:
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
        Qwen3TTSTokenizerV2Decoder,
    )

    source_dir = Path(qwen_tts.__file__).resolve().parent
    verify_reference_code(source_dir)
    torch.set_num_threads(4)
    config_values = json.loads(
        (model_dir / "speech_tokenizer/config.json").read_text())["decoder_config"]
    config = Qwen3TTSTokenizerV2DecoderConfig(**config_values)
    config._attn_implementation = "eager"
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                   framework="pt", device="cpu") as source:
        state = {name[len("decoder."):]: source.get_tensor(name)
                 for name in source.keys() if name.startswith("decoder.")}
    decoder.load_state_dict(state, strict=True)
    with np.load(FIXTURES / "qwen3_tts_customvoice_reference.npz") as fixture:
        codes = np.ascontiguousarray(fixture["codec_tokens"].T[None], dtype=np.int64)
    values: dict[str, np.ndarray] = {}

    class CaptureDone(Exception):
        pass

    def finish(_module, args, output):
        values["decoder_ingress_output"] = np.ascontiguousarray(
            args[0][0].detach().cpu().numpy().astype(np.float32))
        values["decoder_snake_beta_output"] = np.ascontiguousarray(
            output[0].detach().cpu().numpy().astype(np.float32))
        raise CaptureDone

    handle = decoder.decoder[1].block[0].register_forward_hook(finish)
    try:
        with torch.inference_mode():
            decoder(torch.from_numpy(codes))
    except CaptureDone:
        pass
    finally:
        handle.remove()
    frames = codes.shape[-1] * 4
    shape = (config.decoder_dim, frames)
    if set(values) != {"decoder_ingress_output", "decoder_snake_beta_output"} or any(
            value.shape != shape or not np.isfinite(value).all()
            for value in values.values()):
        raise RuntimeError("invalid full-model SnakeBeta capture")
    prior_path = FIXTURES / "qwen3_tts_codec_decoder_ingress.npz"
    with np.load(prior_path) as prior:
        source_error = float(np.max(np.abs(
            values["decoder_ingress_output"] - prior["decoder_ingress_output"])))
    if source_error > 1e-7:
        raise RuntimeError(f"prior codec checkpoint drift: {source_error}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "qwen3_tts_codec_snake_beta.npz"
    np.savez_compressed(output, **values)
    manifest = {
        "schema": 1,
        "scope": "direct pinned full decoder through first SnakeBeta; upsampling and waveform NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "codec_source_sha256": digest(
            source_dir / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"),
        "source_decoder_ingress_sha256": digest(prior_path),
        "source_decoder_ingress_max_abs": source_error,
        "npz_sha256": digest(output),
        "frames": frames,
        "channels": config.decoder_dim,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "attention_implementation": "eager",
    }
    (output_dir / "qwen3_tts_codec_snake_beta.json").write_text(
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
