#!/usr/bin/env python3
"""Capture the second ConvNeXt block from the pinned full codec decoder."""

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
        prefix = "decoder."
        state = {name[len(prefix):]: source.get_tensor(name)
                 for name in source.keys() if name.startswith(prefix)}
    decoder.load_state_dict(state, strict=True)
    with np.load(FIXTURES / "qwen3_tts_customvoice_reference.npz") as fixture:
        codes = np.ascontiguousarray(fixture["codec_tokens"].T[None], dtype=np.int64)
    values: dict[str, np.ndarray] = {}

    class CaptureDone(Exception):
        pass

    def record(name: str):
        def hook(_module, _args, output):
            values[name] = np.ascontiguousarray(
                output[0].detach().cpu().numpy().astype(np.float32))
        return hook

    block = decoder.upsample[1][1]
    handles = [getattr(block, module).register_forward_hook(record(name))
               for module, name in (
                   ("dwconv", "second_depthwise_output"),
                   ("norm", "second_convnext_norm"),
                   ("pwconv1", "second_convnext_pw1"),
                   ("act", "second_convnext_gelu"),
                   ("pwconv2", "second_convnext_pw2"),
               )]

    def finish(_module, args, output):
        values["second_upsample"] = np.ascontiguousarray(
            args[0][0].detach().cpu().numpy().astype(np.float32))
        values["second_convnext_output"] = np.ascontiguousarray(
            output[0].detach().cpu().numpy().astype(np.float32))
        raise CaptureDone

    handles.append(block.register_forward_hook(finish))
    try:
        with torch.inference_mode():
            decoder(torch.from_numpy(codes))
    except CaptureDone:
        pass
    finally:
        for handle in handles:
            handle.remove()
    frames = codes.shape[-1] * 4
    narrow = (config.latent_dim, frames)
    wide = (frames, 4 * config.latent_dim)
    shapes = {
        "second_upsample": narrow, "second_depthwise_output": narrow,
        "second_convnext_norm": (frames, config.latent_dim),
        "second_convnext_pw1": wide, "second_convnext_gelu": wide,
        "second_convnext_pw2": (frames, config.latent_dim),
        "second_convnext_output": narrow,
    }
    if set(values) != set(shapes) or any(
            values[name].shape != shape or not np.isfinite(values[name]).all()
            for name, shape in shapes.items()):
        raise RuntimeError("invalid direct full-model second ConvNeXt checkpoint")
    prior_path = FIXTURES / "qwen3_tts_codec_second_upsample.npz"
    with np.load(prior_path) as prior:
        source_error = float(np.max(np.abs(
            values["second_upsample"] - prior["second_upsample"])))
    if source_error > 1e-7:
        raise RuntimeError(f"prior codec checkpoint drift: {source_error}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "qwen3_tts_codec_second_convnext.npz"
    np.savez_compressed(output, **values)
    manifest = {
        "schema": 1,
        "scope": "direct pinned full decoder through second complete ConvNeXt; remaining decoder and waveform NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "codec_source_sha256": digest(
            source_dir / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"),
        "source_second_upsample_sha256": digest(prior_path),
        "source_second_upsample_max_abs": source_error,
        "npz_sha256": digest(output),
        "frames": frames, "latent_width": config.latent_dim,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "attention_implementation": "eager",
    }
    (output_dir / "qwen3_tts_codec_second_convnext.json").write_text(
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
