#!/usr/bin/env python3
"""Compare a generated Whisper long-form feature window with PyTorch."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import wave

import numpy as np


class CKAudioWavInfo(ctypes.Structure):
    _fields_ = [
        ("format_tag", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("bits_per_sample", ctypes.c_int),
        ("frames", ctypes.c_int),
        ("data_offset", ctypes.c_size_t),
        ("data_bytes", ctypes.c_size_t),
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _read_pcm16(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("Whisper frontend oracle requires PCM16 WAV")
        rate = source.getframerate()
        channels = source.getnchannels()
        frames = source.getnframes()
        raw = source.readframes(frames)
    values = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    values = values.reshape(-1, channels).mean(axis=1)
    return values / np.float32(32768.0), rate, frames


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {actual.shape}")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    worst = int(np.argmax(absolute))
    return {
        "finite": bool(np.all(np.isfinite(actual))),
        "byte_exact": bool(np.array_equal(reference, actual)),
        "max_abs": float(np.max(absolute)),
        "mean_abs": float(np.mean(absolute)),
        "rmse": float(math.sqrt(float(np.mean(difference * difference)))),
        "exact_ratio": float(np.mean(reference == actual)),
        "worst_coordinate": [
            int(value) for value in np.unravel_index(worst, reference.shape)
        ],
    }


def _resolved_execution(operation: dict[str, object]) -> dict[str, str | None]:
    resolved = operation.get("resolved_contract")
    contract = resolved if isinstance(resolved, dict) else {}

    def scalar(*values: object) -> str | None:
        for value in values:
            if isinstance(value, (str, int, float)):
                return str(value)
        return None

    return {
        "kernel_id": scalar(
            operation.get("kernel_id"),
            contract.get("kernel_id"),
        ),
        "function": scalar(
            operation.get("function"),
            contract.get("function"),
        ),
        "resolved_contract_id": scalar(
            operation.get("resolved_contract_id"),
            contract.get("resolved_contract_id"),
            contract.get("contract_id"),
        ),
    }


def _pytorch_window(
    checkpoint: Path,
    samples: np.ndarray,
    sample_rate: int,
    start_frame: int,
    output_frames: int,
    hop_length: int,
) -> tuple[np.ndarray, str, str]:
    import torch
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained(
        checkpoint, local_files_only=True
    )
    complete = extractor(
        samples,
        sampling_rate=sample_rate,
        truncation=False,
        padding="longest",
        return_tensors="np",
    ).input_features[0].astype(np.float32, copy=False)
    seek = start_frame // hop_length
    output = np.zeros((complete.shape[0], output_frames), dtype=np.float32)
    copied = min(output_frames, max(0, complete.shape[1] - seek))
    output[:, :copied] = complete[:, seek : seek + copied]
    return output, torch.__version__, __import__("transformers").__version__


def _generated_window(
    run_dir: Path,
    wav_path: Path,
    start_frame: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, CKAudioWavInfo]:
    library_path = run_dir / "libmodel.so"
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    u8_pointer = ctypes.POINTER(ctypes.c_uint8)
    library.ck_model_init_with_manifest.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    library.ck_model_init_with_manifest.restype = ctypes.c_int
    library.ck_model_prepare_audio_wav_window.argtypes = [
        u8_pointer,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(CKAudioWavInfo),
    ]
    library.ck_model_prepare_audio_wav_window.restype = ctypes.c_int
    library.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
    library.ck_model_get_named_activation_ptr.restype = ctypes.c_void_p
    library.ck_model_get_named_activation_nbytes.argtypes = [ctypes.c_char_p]
    library.ck_model_get_named_activation_nbytes.restype = ctypes.c_ssize_t
    library.ck_model_free.argtypes = []

    status = int(
        library.ck_model_init_with_manifest(
            str(run_dir / "weights.bump").encode(),
            str(run_dir / "weights_manifest.map").encode(),
        )
    )
    if status != 0:
        raise RuntimeError(f"generated frontend init failed with code {status}")
    try:
        wav = np.frombuffer(wav_path.read_bytes(), dtype=np.uint8)
        info = CKAudioWavInfo()
        status = int(
            library.ck_model_prepare_audio_wav_window(
                wav.ctypes.data_as(u8_pointer),
                wav.size,
                start_frame,
                ctypes.byref(info),
            )
        )
        if status != 0:
            raise RuntimeError(
                f"generated frontend failed with code {status}"
            )
        count = math.prod(shape)
        expected_bytes = count * np.dtype(np.float32).itemsize
        pointer = int(
            library.ck_model_get_named_activation_ptr(b"audio_features") or 0
        )
        nbytes = int(
            library.ck_model_get_named_activation_nbytes(b"audio_features")
        )
        if pointer == 0 or nbytes != expected_bytes:
            raise RuntimeError(
                f"audio_features ABI mismatch: ptr={pointer} "
                f"nbytes={nbytes} expected={expected_bytes}"
            )
        raw = np.ctypeslib.as_array(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float)),
            shape=(count,),
        )
        return raw.copy().reshape(shape), info
    finally:
        library.ck_model_free()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-abs", type=float, default=1.1e-3)
    parser.add_argument("--rmse", type=float, default=9.0e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    wav_path = args.wav.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    sample_rate = int(config["audio_sample_rate"])
    hop_length = int(config["audio_hop_length"])
    shape = (
        int(config["audio_feature_channels"]),
        int(config["audio_feature_frames"]),
    )
    if args.start_frame < 0 or args.start_frame % hop_length:
        raise ValueError("start frame must be nonnegative and hop-aligned")
    samples, wav_rate, source_frames = _read_pcm16(wav_path)
    if wav_rate != sample_rate:
        raise ValueError(
            "long-form frontend parity requires identity resampling"
        )
    if source_frames <= int(config["audio_sample_extent"]):
        raise ValueError("frontend window X-ray requires audio longer than one window")
    os.environ["CK_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    reference, torch_version, transformers_version = _pytorch_window(
        checkpoint,
        samples,
        wav_rate,
        args.start_frame,
        shape[1],
        hop_length,
    )
    actual, info = _generated_window(
        run_dir, wav_path, args.start_frame, shape
    )
    metrics = _metrics(reference, actual)
    material = (
        not bool(metrics["finite"])
        or float(metrics["max_abs"]) > args.max_abs
        or float(metrics["rmse"]) > args.rmse
    )
    call = json.loads((run_dir / "call.json").read_text(encoding="utf-8"))
    op_idx, operation = next(
        (index, row)
        for index, row in enumerate(call["operations"])
        if row.get("op") == "audio_feature_window"
    )
    report = {
        "schema": "cke.whisper_frontend_pytorch_xray",
        "schema_version": 1,
        "status": "fail" if material else "pass",
        "first_divergence": (
            "audio.frontend.feature_window.output" if material else None
        ),
        "comparisons": [
            {
                "sequence_index": 0,
                "checkpoint_id": "audio.frontend.feature_window.output",
                "op_idx": op_idx,
                "phase": "prefill",
                "subject_backend": "cke",
                "oracle_backend": "pytorch",
                "shape": list(shape),
                "observed_dtype": "fp32",
                "reference_sha256": _sha256_array(reference),
                "actual_sha256": _sha256_array(actual),
                "metrics": metrics,
                "material_divergence": material,
                "resolved_execution": _resolved_execution(operation),
            }
        ],
        "thresholds": {"max_abs": args.max_abs, "rmse": args.rmse},
        "window": {
            "source_frame_start": args.start_frame,
            "source_frame_count": source_frames,
            "sample_rate": wav_rate,
            "feature_frame_start": args.start_frame // hop_length,
            "feature_frame_count": shape[1],
        },
        "provenance": {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "wav": str(wav_path),
            "wav_sha256": _sha256_file(wav_path),
            "runtime_sha256": _sha256_file(run_dir / "libmodel.so"),
            "weights_sha256": _sha256_file(run_dir / "weights.bump"),
            "call_ir_sha256": _sha256_file(run_dir / "call.json"),
            "python": platform.python_version(),
            "pytorch": torch_version,
            "transformers": transformers_version,
            "threads": args.threads,
            "wav_metadata": {
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "frames": info.frames,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"status={report['status']} max_abs={metrics['max_abs']:.9g} "
        f"rmse={metrics['rmse']:.9g} report={args.output}"
    )
    return 1 if material else 0


if __name__ == "__main__":
    raise SystemExit(main())
