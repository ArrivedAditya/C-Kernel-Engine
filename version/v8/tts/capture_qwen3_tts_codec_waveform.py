#!/usr/bin/env python3
"""Capture direct pinned-model codec waveform checkpoints for 8 and 46 codes."""

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
    config_values = json.loads((model_dir / "speech_tokenizer/config.json").read_text())["decoder_config"]
    config = Qwen3TTSTokenizerV2DecoderConfig(**config_values)
    config._attn_implementation = "eager"
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                   framework="pt", device="cpu") as source:
        state = {name[len("decoder."):]: source.get_tensor(name)
                 for name in source.keys() if name.startswith("decoder.")}
    decoder.load_state_dict(state, strict=True)
    source_codes = FIXTURES / "qwen3_tts_customvoice_reference.npz"
    with np.load(source_codes) as fixture:
        codes = np.ascontiguousarray(fixture["codec_tokens"].T[None], dtype=np.int64)
    if codes.shape[-1] != 46:
        raise RuntimeError("pinned codec sequence changed")

    def run(count: int) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        handles = []

        def hook(name: str, input_name: str | None = None):
            def save(_module, args, output):
                if input_name is not None:
                    values[input_name] = np.ascontiguousarray(
                        args[0][0].detach().cpu().numpy(), dtype=np.float32)
                values[name] = np.ascontiguousarray(
                    output[0].detach().cpu().numpy(), dtype=np.float32)
            return save

        for stage in range(5):
            name = ("decoder_fourth_snake_beta_output" if stage == 0 else
                    "decoder_fourth_upsample_output" if stage == 1 else
                    f"decoder_fourth_residual_{stage - 2}_output")
            handles.append(decoder.decoder[4].block[stage].register_forward_hook(
                hook(name, "decoder_third_residual_2_output" if stage == 0 else None)))
        handles.append(decoder.decoder[5].register_forward_hook(
            hook("decoder_final_snake_beta_output")))
        handles.append(decoder.decoder[6].register_forward_hook(
            hook("decoder_waveform_preclip_output")))
        try:
            with torch.inference_mode():
                waveform = decoder(torch.from_numpy(codes[..., :count].copy()))
            values["waveform_output"] = np.ascontiguousarray(
                waveform[0].detach().cpu().numpy(), dtype=np.float32)
        finally:
            for handle in handles:
                handle.remove()
        expected = count * 4 * 8 * 5 * 4 * 3
        if values["waveform_output"].shape != (1, expected):
            raise RuntimeError("reference sample count changed")
        if not all(np.isfinite(value).all() for value in values.values()):
            raise RuntimeError("nonfinite reference checkpoint")
        return values

    short, full = run(8), run(46)
    prior = FIXTURES / "qwen3_tts_codec_third_decoder_block.npz"
    with np.load(prior) as previous:
        short_error = float(np.max(np.abs(
            short["decoder_third_residual_2_output"] -
            previous["decoder_third_residual_2_output"])))
        full_error = float(np.max(np.abs(
            full["decoder_third_residual_2_output"] -
            previous["full_decoder_third_residual_2_output"])))
    if max(short_error, full_error) > 1e-7:
        raise RuntimeError("prior codec checkpoint drift")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "qwen3_tts_codec_waveform.npz"
    # Full-length PCM is retained; short checkpoints diagnose each new stage.
    np.savez_compressed(output, **short,
                        full_waveform_output=full["waveform_output"])
    manifest = {
        "schema": 1,
        "scope": "reference-code-fed pinned full-model codec waveform; autonomous talker NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "codec_source_sha256": digest(source_dir / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"),
        "source_third_decoder_block_sha256": digest(prior),
        "source_third_decoder_short_max_abs": short_error,
        "source_third_decoder_full_max_abs": full_error,
        "source_codec_fixture_sha256": digest(source_codes),
        "short_codec_ids_sha256": hashlib.sha256(codes[..., :8].tobytes()).hexdigest(),
        "full_codec_ids_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
        "npz_sha256": digest(output),
        "short_code_frames": 8,
        "full_code_frames": 46,
        "short_samples": int(short["waveform_output"].size),
        "full_samples": int(full["waveform_output"].size),
        "sample_rate_hz": 24000,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "attention_implementation": "eager",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"npz_sha256": manifest["npz_sha256"],
                      "samples": [manifest["short_samples"], manifest["full_samples"]]},
                     sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=FIXTURES)
    args = parser.parse_args()
    capture(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
