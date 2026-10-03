#!/usr/bin/env python3
"""Capture one pinned Kokoro v1.0 reference utterance from local assets.

The script is oracle tooling only. It uses upstream Python at export time and
performs no network access. Runtime TTS must use native CKE operations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import wave

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np


HERE = Path(__file__).resolve().parent
PIN = json.loads((HERE / "kokoro_v1_reference.json").read_text())
CAPTURED_REFERENCE = json.loads((HERE / "fixture_manifest.json").read_text())
FINE_PREDICTOR_HOOKS = tuple(
    f"predictor.text_encoder.lstms.{index}" for index in range(6)
)
PROSODY_BRANCH_HOOKS = tuple(
    f"predictor.{branch}.{index}"
    for branch in ("F0", "N") for index in range(3)
)
DECODER_HOOKS = (
    "decoder.F0_conv", "decoder.N_conv", "decoder.asr_res",
    "decoder.encode.norm1", "decoder.encode.conv1",
    "decoder.encode.norm2", "decoder.encode.conv2",
    "decoder.encode.conv1x1", "decoder.encode",
    *(f"decoder.decode.{index}.{part}" for index in range(4)
      for part in ("norm1", "conv1", "norm2", "conv2", "conv1x1")),
    "decoder.decode.3.pool", "decoder.decode.3.upsample",
    *(f"decoder.decode.{index}" for index in range(4)),
)
GENERATOR_HOOKS = (
    "decoder.generator.f0_upsamp",
    "decoder.generator.m_source",
    "decoder.generator.m_source.l_sin_gen",
    "decoder.generator.noise_convs.0",
    "decoder.generator.noise_res.0",
    "decoder.generator.ups.0",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_tensor(value, stem: str, out_dir: Path, records: dict) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        path = out_dir / f"{stem}.npy"
        np.save(path, array, allow_pickle=False)
        records[stem] = {
            "file": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256(path),
        }
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            capture_tensor(item, f"{stem}_{index}", out_dir, records)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Local pinned HF snapshot containing config, weights and voice")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", help="Additional oracle utterance; same pinned assets and reference versions")
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    fixture = PIN["fixture"]
    model_dir = args.model_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    weight_path = model_dir / "kokoro-v1_0.pth"
    voice_path = model_dir / "voices" / "af_heart.pt"
    required = [config_path] if args.preprocess_only else [config_path, weight_path, voice_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing pinned local asset(s): " + ", ".join(missing))
    for path in required:
        name = str(path.relative_to(model_dir))
        expected_hash = CAPTURED_REFERENCE["assets"][name]
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            parser.error(f"{name}: SHA-256 {actual_hash} != pinned {expected_hash}")

    import torch
    from kokoro import KModel, KPipeline

    for package, expected_version in CAPTURED_REFERENCE["environment"]["packages"].items():
        actual_version = package_version(package)
        if actual_version != expected_version:
            parser.error(f"{package} {actual_version} != pinned {expected_version}")

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.manual_seed(fixture["torch_seed"])
    pipeline = KPipeline(lang_code="a", repo_id=PIN["model"]["repository"], model=False, trf=False)
    input_text = args.text if args.text is not None else fixture["text"]
    segments = list(pipeline(input_text, model=False))
    if len(segments) != 1 or not segments[0].phonemes:
        raise RuntimeError(f"expected one nonempty Kokoro segment, got {len(segments)}")
    phonemes = segments[0].phonemes
    config = json.loads(config_path.read_text())
    ids = [0, *(config["vocab"].get(phone) for phone in phonemes), 0]
    if any(token is None for token in ids):
        raise RuntimeError("phoneme outside pinned config vocabulary")
    if len(phonemes) > 510:
        raise RuntimeError("fixture exceeds Kokoro's 510-phoneme segment limit")

    record = {
        "schema_version": 1,
        "status": "preprocessing_captured" if args.preprocess_only else "full_oracle_captured",
        "pin": PIN,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {name: package_version(name) for name in (
                "kokoro", "misaki", "torch", "transformers", "spacy",
                "num2words", "phonemizer-fork", "espeakng-loader")},
        },
        "assets": {str(path.relative_to(model_dir)): sha256(path) for path in required},
        "input_text": input_text,
        "input_text_override": args.text is not None,
        "graphemes": segments[0].graphemes,
        "phonemes": phonemes,
        "phoneme_codepoints": [f"U+{ord(char):04X}" for char in phonemes],
        "input_ids": ids,
        "tensors": {},
        "evidence": {
            "upstream_preprocessing": "CAPTURED",
            "upstream_waveform": "NOT_TESTED" if args.preprocess_only else "CAPTURED",
            "native_primitive_parity": "NOT_TESTED",
            "native_full_waveform_parity": "NOT_TESTED",
            "human_listening": "NOT_TESTED",
            "application_playback": "NOT_TESTED",
        },
    }
    if args.preprocess_only:
        (out_dir / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return 0

    voice_pack = torch.load(voice_path, map_location="cpu", weights_only=True)
    voice_row = voice_pack[len(phonemes) - 1]
    if voice_row.ndim == 1:
        voice_row = voice_row.unsqueeze(0)
    if tuple(voice_row.shape) != (1, 256):
        raise RuntimeError(f"unexpected voice style shape {tuple(voice_row.shape)}")
    capture_tensor(voice_row, "voice_style", out_dir, record["tensors"])
    capture_tensor(voice_row[:, :128], "decoder_style", out_dir, record["tensors"])
    capture_tensor(voice_row[:, 128:], "predictor_style", out_dir, record["tensors"])
    record["voice_row_index"] = len(phonemes) - 1

    model = KModel(repo_id=PIN["model"]["repository"],
                   config=str(config_path), model=str(weight_path)).eval()
    module_names = (
        "bert", "bert_encoder", "predictor.text_encoder", "predictor.lstm",
        "predictor.duration_proj", "predictor.shared", "predictor.F0_proj",
        "predictor.N_proj", "text_encoder", "decoder", "decoder.generator",
        "decoder.generator.conv_post",
    ) + FINE_PREDICTOR_HOOKS + PROSODY_BRANCH_HOOKS + DECODER_HOOKS + GENERATOR_HOOKS
    modules = dict(model.named_modules())
    hooks = []
    for name in module_names:
        module = modules.get(name)
        if module is None:
            if name in FINE_PREDICTOR_HOOKS + PROSODY_BRANCH_HOOKS + DECODER_HOOKS + GENERATOR_HOOKS:
                raise RuntimeError(f"pinned predictor checkpoint module missing: {name}")
            record.setdefault("unavailable_hooks", []).append(name)
            continue
        def on_output(_module, _inputs, output, label=name):
            stem = label.replace(".", "_")
            if label == "decoder":
                capture_tensor(_inputs, "decoder_input", out_dir, record["tensors"])
            if label.startswith("decoder.decode.") and label.count(".") == 2:
                capture_tensor(_inputs, f"{stem}_input", out_dir,
                               record["tensors"])
            if label in FINE_PREDICTOR_HOOKS:
                capture_tensor(_inputs, f"{stem}_input", out_dir, record["tensors"])
                stem += "_output"
            capture_tensor(output, stem, out_dir, record["tensors"])
        hooks.append(module.register_forward_hook(on_output))
    stft = model.decoder.generator.stft
    capture_tensor(stft.window, "generator_stft_window", out_dir,
                   record["tensors"])
    original_transform = stft.transform
    original_inverse = stft.inverse
    original_rand = torch.rand
    original_randn_like = torch.randn_like
    def capture_rand(*shape, **kwargs):
        value = original_rand(*shape, **kwargs)
        if tuple(value.shape) == (1, 9):
            capture_tensor(value, "generator_initial_phase_random", out_dir,
                           record["tensors"])
        return value
    def capture_randn_like(input_value, *random_args, **random_kwargs):
        value = original_randn_like(input_value, *random_args, **random_kwargs)
        if value.ndim == 3 and value.shape[0] == 1:
            if value.shape[-1] == 9:
                label = "generator_harmonic_gaussian"
            elif value.shape[-1] == 1:
                label = "generator_source_gaussian"
            else:
                return value
            capture_tensor(value, label, out_dir, record["tensors"])
        return value
    def capture_transform(samples):
        capture_tensor(samples, "generator_source_samples", out_dir,
                       record["tensors"])
        magnitude, phase = original_transform(samples)
        capture_tensor(magnitude, "generator_source_magnitude", out_dir,
                       record["tensors"])
        capture_tensor(phase, "generator_source_phase", out_dir,
                       record["tensors"])
        return magnitude, phase
    def capture_inverse(spec, phase):
        capture_tensor(spec, "istft_magnitude", out_dir, record["tensors"])
        capture_tensor(phase, "istft_phase", out_dir, record["tensors"])
        return original_inverse(spec, phase)
    stft.transform = capture_transform
    stft.inverse = capture_inverse
    torch.rand = capture_rand
    torch.randn_like = capture_randn_like
    try:
        output = model(phonemes, voice_row, speed=fixture["speed"], return_output=True)
    finally:
        stft.transform = original_transform
        stft.inverse = original_inverse
        torch.rand = original_rand
        torch.randn_like = original_randn_like
        for hook in hooks:
            hook.remove()
    for name in ("generator_initial_phase_random", "generator_harmonic_gaussian",
                 "generator_source_gaussian", "generator_stft_window",
                 "generator_source_samples",
                 "generator_source_magnitude", "generator_source_phase"):
        if name not in record["tensors"]:
            raise RuntimeError(f"pinned source checkpoint not reached: {name}")
    for name in FINE_PREDICTOR_HOOKS:
        stem = name.replace(".", "_")
        if not any(key.startswith(f"{stem}_input") for key in record["tensors"]):
            raise RuntimeError(f"pinned predictor input checkpoint not reached: {name}")
        if not any(key.startswith(f"{stem}_output") for key in record["tensors"]):
            raise RuntimeError(f"pinned predictor output checkpoint not reached: {name}")
    for name in PROSODY_BRANCH_HOOKS:
        if name.replace(".", "_") not in record["tensors"]:
            raise RuntimeError(f"pinned prosody block checkpoint not reached: {name}")
    for name in DECODER_HOOKS:
        if name.replace(".", "_") not in record["tensors"]:
            raise RuntimeError(f"pinned decoder checkpoint not reached: {name}")
    for name in GENERATOR_HOOKS:
        if not any(key.startswith(name.replace(".", "_"))
                   for key in record["tensors"]):
            raise RuntimeError(f"pinned generator checkpoint not reached: {name}")
    record["fine_predictor_hooks"] = list(FINE_PREDICTOR_HOOKS)
    record["prosody_branch_hooks"] = list(PROSODY_BRANCH_HOOKS)
    record["decoder_hooks"] = list(DECODER_HOOKS)
    record["generator_hooks"] = list(GENERATOR_HOOKS)
    capture_tensor(output.pred_dur, "predicted_duration", out_dir, record["tensors"])
    durations = output.pred_dur.detach().cpu().reshape(-1).to(torch.int64)
    if len(durations) != len(ids) or (durations < 1).any():
        raise RuntimeError("oracle duration shape or lower bound mismatch")
    frame_to_token = torch.repeat_interleave(torch.arange(len(ids)), durations)
    capture_tensor(frame_to_token, "frame_to_token", out_dir, record["tensors"])
    capture_tensor(output.audio, "waveform_f32", out_dir, record["tensors"])
    audio = output.audio.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    if not np.isfinite(audio).all() or len(audio) == 0:
        raise RuntimeError("oracle produced empty or nonfinite waveform")
    wav_path = out_dir / "waveform_pcm16.wav"
    write_wav(wav_path, audio, fixture["sample_rate_hz"])
    record["waveform"] = {
        "frames": len(audio), "seconds": len(audio) / fixture["sample_rate_hz"],
        "min": float(audio.min()), "max": float(audio.max()),
        "pcm16_file": wav_path.name, "pcm16_sha256": sha256(wav_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
