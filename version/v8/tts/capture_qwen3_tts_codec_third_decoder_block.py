#!/usr/bin/env python3
"""Capture pinned Qwen3-TTS third decoder block at short and full code lengths."""

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

SHORT_FRAMES = 8


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
    source_codes_path = FIXTURES / "qwen3_tts_customvoice_reference.npz"
    with np.load(source_codes_path) as fixture:
        all_codes = np.ascontiguousarray(fixture["codec_tokens"].T[None], dtype=np.int64)
    if all_codes.shape[-1] != 46:
        raise RuntimeError("pinned codec sequence changed")

    class CaptureDone(Exception):
        pass

    def run(code_count: int) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}

        def capture_stage(stage: int):
            def hook(_module, args, output):
                if stage == 0:
                    values["decoder_second_residual_2_output"] = np.ascontiguousarray(
                        args[0][0].detach().cpu().numpy().astype(np.float32))
                name = ("decoder_third_snake_beta_output" if stage == 0 else
                        "decoder_third_upsample_output" if stage == 1 else
                        f"decoder_third_residual_{stage - 2}_output")
                values[name] = np.ascontiguousarray(
                    output[0].detach().cpu().numpy().astype(np.float32))
                if stage == 4:
                    raise CaptureDone
            return hook

        handles = [decoder.decoder[3].block[stage].register_forward_hook(
            capture_stage(stage)) for stage in range(5)]
        try:
            with torch.inference_mode():
                decoder(torch.from_numpy(all_codes[..., :code_count].copy()))
        except CaptureDone:
            pass
        finally:
            for handle in handles:
                handle.remove()
        input_frames = code_count * 4 * config.upsample_rates[0] * config.upsample_rates[1]
        output_frames = input_frames * config.upsample_rates[2]
        required = {"decoder_second_residual_2_output",
                    "decoder_third_snake_beta_output",
                    "decoder_third_upsample_output", *(
                        f"decoder_third_residual_{unit}_output" for unit in range(3))}
        if set(values) != required:
            raise RuntimeError("missing full-model third decoder checkpoints")
        for name, value in values.items():
            shape = ((config.decoder_dim // 4, input_frames)
                     if name in ("decoder_second_residual_2_output",
                                 "decoder_third_snake_beta_output")
                     else (config.decoder_dim // 8, output_frames))
            if value.shape != shape or not np.isfinite(value).all():
                raise RuntimeError(f"invalid full-model checkpoint {name}: {value.shape}")
        return values

    short = run(SHORT_FRAMES)
    full = run(all_codes.shape[-1])
    prior_path = FIXTURES / "qwen3_tts_codec_second_decoder_block.npz"
    with np.load(prior_path) as prior:
        source_error = float(np.max(np.abs(
            full["decoder_second_residual_2_output"] -
            prior["decoder_second_residual_2_output"])))
    if source_error > 1e-7:
        raise RuntimeError(f"prior codec checkpoint drift: {source_error}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "qwen3_tts_codec_third_decoder_block.npz"
    # Store all short boundaries and only the final full-length output. The
    # preceding fixture already owns the full-length input checkpoint.
    np.savez_compressed(output, **short,
                        full_decoder_third_residual_2_output=full[
                            "decoder_third_residual_2_output"])
    manifest = {
        "schema": 1,
        "scope": "reference-code-fed full model through third decoder block; later decoder and waveform NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "codec_source_sha256": digest(
            source_dir / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"),
        "source_second_decoder_block_sha256": digest(prior_path),
        "source_second_decoder_block_max_abs": source_error,
        "source_codec_fixture_sha256": digest(source_codes_path),
        "short_codec_ids_sha256": hashlib.sha256(
            all_codes[..., :SHORT_FRAMES].tobytes()).hexdigest(),
        "full_codec_ids_sha256": hashlib.sha256(all_codes.tobytes()).hexdigest(),
        "npz_sha256": digest(output),
        "short_code_frames": SHORT_FRAMES,
        "full_code_frames": int(all_codes.shape[-1]),
        "short_input_frames": int(short["decoder_second_residual_2_output"].shape[-1]),
        "short_output_frames": int(short["decoder_third_residual_2_output"].shape[-1]),
        "full_output_frames": int(full["decoder_third_residual_2_output"].shape[-1]),
        "output_channels": config.decoder_dim // 8,
        "dilations": [1, 3, 9],
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "attention_implementation": "eager",
    }
    (output_dir / "qwen3_tts_codec_third_decoder_block.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"npz_sha256": manifest["npz_sha256"],
                      "checkpoints": list(short) + ["full_decoder_third_residual_2_output"]},
                     sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=FIXTURES)
    args = parser.parse_args()
    capture(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
