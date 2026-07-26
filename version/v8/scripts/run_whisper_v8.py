#!/usr/bin/env python3
"""Run generated CKE Whisper encoder and decoder artifacts on a PCM16 WAV."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


SAMPLE_RATE = 16000
N_SAMPLES = 480000
N_FFT = 400
HOP_LENGTH = 160
POWER_BINS = N_FFT // 2 + 1
N_MELS = 80
N_FRAMES = N_SAMPLES // HOP_LENGTH
_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_U8_P = ctypes.POINTER(ctypes.c_uint8)


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


def _fptr(values: np.ndarray) -> _FLOAT_P:
    return values.ctypes.data_as(_FLOAT_P)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_artifact(run_dir: Path) -> None:
    for name in (
        "libckernel_engine.so",
        "libmodel.so",
        "weights.bump",
        "weights_manifest.map",
        "config.json",
    ):
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    values = np.asarray(frequencies, dtype=np.float64)
    result = values / (200.0 / 3.0)
    logarithmic = values >= 1000.0
    result[logarithmic] = 15.0 + np.log(values[logarithmic] / 1000.0) / (
        math.log(6.4) / 27.0
    )
    return result


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    values = np.asarray(mels, dtype=np.float64)
    result = (200.0 / 3.0) * values
    logarithmic = values >= 15.0
    result[logarithmic] = 1000.0 * np.exp(
        (math.log(6.4) / 27.0) * (values[logarithmic] - 15.0)
    )
    return result


def whisper_mel_filters() -> np.ndarray:
    """Return Whisper's fixed Slaney 80x201 FP32 filter bank."""
    fft_frequencies = np.linspace(0.0, 8000.0, POWER_BINS, dtype=np.float64)
    mel_edges = np.linspace(
        _hz_to_mel(np.array([0.0]))[0],
        _hz_to_mel(np.array([8000.0]))[0],
        N_MELS + 2,
        dtype=np.float64,
    )
    filter_frequencies = _mel_to_hz(mel_edges)
    differences = np.diff(filter_frequencies)
    ramps = filter_frequencies[:, None] - fft_frequencies[None, :]
    lower = -ramps[:-2] / differences[:-1, None]
    upper = ramps[2:] / differences[1:, None]
    filters = np.maximum(0.0, np.minimum(lower, upper))
    filters *= (
        2.0 / (filter_frequencies[2:] - filter_frequencies[:-2])
    )[:, None]
    return np.ascontiguousarray(filters.astype(np.float32))


def _load_audio_api(engine_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)
    lib.audio_wav_parse_memory.argtypes = [
        _U8_P,
        ctypes.c_size_t,
        ctypes.POINTER(CKAudioWavInfo),
    ]
    lib.audio_wav_parse_memory.restype = ctypes.c_int
    lib.audio_wav_decode_pcm16_mono_f32.argtypes = [
        _U8_P,
        ctypes.c_size_t,
        ctypes.POINTER(CKAudioWavInfo),
        _FLOAT_P,
        ctypes.c_int,
    ]
    lib.audio_wav_decode_pcm16_mono_f32.restype = ctypes.c_int
    lib.audio_resampled_frame_count.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.audio_resampled_frame_count.restype = ctypes.c_int
    lib.audio_resample_windowed_sinc_f32.argtypes = [
        _FLOAT_P,
        ctypes.c_int,
        ctypes.c_int,
        _FLOAT_P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.audio_resample_windowed_sinc_f32.restype = ctypes.c_int
    lib.audio_stft_precompute_tables_f32.argtypes = [
        ctypes.c_int,
        _FLOAT_P,
        _FLOAT_P,
        _FLOAT_P,
    ]
    lib.audio_stft_precompute_tables_f32.restype = ctypes.c_int
    lib.audio_stft_power_fft400_f32.argtypes = [
        _FLOAT_P,
        ctypes.c_int,
        _FLOAT_P,
        _FLOAT_P,
        _FLOAT_P,
        ctypes.c_int,
        _FLOAT_P,
        ctypes.c_int,
        _FLOAT_P,
    ]
    lib.audio_stft_power_fft400_f32.restype = ctypes.c_int
    lib.audio_whisper_log_mel_from_power_reference_f32.argtypes = [
        _FLOAT_P,
        _FLOAT_P,
        ctypes.c_int,
        ctypes.c_int,
        _FLOAT_P,
    ]
    lib.audio_whisper_log_mel_from_power_reference_f32.restype = ctypes.c_int
    return lib


def _wav_features(lib: ctypes.CDLL, wav_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    wav = np.frombuffer(wav_path.read_bytes(), dtype=np.uint8)
    info = CKAudioWavInfo()
    status = int(
        lib.audio_wav_parse_memory(
            wav.ctypes.data_as(_U8_P), wav.size, ctypes.byref(info)
        )
    )
    if status != 0:
        raise RuntimeError(f"audio_wav_parse_memory failed with code {status}")
    if info.format_tag != 1 or info.bits_per_sample != 16:
        raise ValueError("the initial Whisper runner accepts PCM16 WAV input")

    mono = np.empty(info.frames, dtype=np.float32)
    decoded = int(
        lib.audio_wav_decode_pcm16_mono_f32(
            wav.ctypes.data_as(_U8_P),
            wav.size,
            ctypes.byref(info),
            _fptr(mono),
            mono.size,
        )
    )
    if decoded != info.frames:
        raise RuntimeError(f"PCM decode returned {decoded}, expected {info.frames}")

    if info.sample_rate != SAMPLE_RATE:
        output_frames = int(
            lib.audio_resampled_frame_count(
                mono.size, info.sample_rate, SAMPLE_RATE
            )
        )
        resampled = np.empty(output_frames, dtype=np.float32)
        status = int(
            lib.audio_resample_windowed_sinc_f32(
                _fptr(mono),
                mono.size,
                info.sample_rate,
                _fptr(resampled),
                resampled.size,
                SAMPLE_RATE,
                16,
            )
        )
        if status != 0:
            raise RuntimeError(
                f"audio_resample_windowed_sinc_f32 failed with code {status}"
            )
        mono = resampled

    samples = np.zeros(N_SAMPLES, dtype=np.float32)
    copied = min(samples.size, mono.size)
    samples[:copied] = mono[:copied]
    window = np.empty(N_FFT, dtype=np.float32)
    cosine = np.empty((POWER_BINS, N_FFT), dtype=np.float32)
    sine = np.empty_like(cosine)
    status = int(
        lib.audio_stft_precompute_tables_f32(
            N_FFT, _fptr(window), _fptr(cosine), _fptr(sine)
        )
    )
    if status != 0:
        raise RuntimeError(
            f"audio_stft_precompute_tables_f32 failed with code {status}"
        )
    power = np.empty((N_FRAMES, POWER_BINS), dtype=np.float32)
    fft_scratch = np.empty(N_FFT * 2, dtype=np.float32)
    status = int(
        lib.audio_stft_power_fft400_f32(
            _fptr(samples),
            samples.size,
            _fptr(window),
            _fptr(cosine),
            _fptr(sine),
            HOP_LENGTH,
            _fptr(power),
            N_FRAMES,
            _fptr(fft_scratch),
        )
    )
    if status != 0:
        raise RuntimeError(f"audio_stft_power_fft400_f32 failed with code {status}")
    filters = whisper_mel_filters()
    features = np.empty((N_MELS, N_FRAMES), dtype=np.float32)
    status = int(
        lib.audio_whisper_log_mel_from_power_reference_f32(
            _fptr(power), _fptr(filters), N_MELS, N_FRAMES, _fptr(features)
        )
    )
    if status != 0:
        raise RuntimeError(
            "audio_whisper_log_mel_from_power_reference_f32 "
            f"failed with code {status}"
        )
    return features, {
        "source_sample_rate": info.sample_rate,
        "source_channels": info.channels,
        "source_frames": info.frames,
        "resampled_frames": int(mono.size),
        "consumed_frames": copied,
        "truncated": mono.size > samples.size,
    }


def _load_generated_model(run_dir: Path) -> ctypes.CDLL:
    ctypes.CDLL(str(run_dir / "libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL)
    model = ctypes.CDLL(str(run_dir / "libmodel.so"))
    model.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    model.ck_model_init_with_manifest.restype = ctypes.c_int
    model.ck_model_free.argtypes = []
    model.ck_model_free.restype = None
    return model


def _encoder_worker(args: argparse.Namespace) -> int:
    run_dir = args.encoder_run_dir.resolve()
    _require_artifact(run_dir)
    audio = _load_audio_api(run_dir / "libckernel_engine.so")
    started = time.perf_counter()
    features, audio_metadata = _wav_features(audio, args.wav.resolve())
    frontend_seconds = time.perf_counter() - started

    model = _load_generated_model(run_dir)
    model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
    model.ck_model_get_named_activation_ptr.restype = ctypes.c_void_p
    model.ck_model_get_named_activation_nbytes.argtypes = [ctypes.c_char_p]
    model.ck_model_get_named_activation_nbytes.restype = ctypes.c_ssize_t
    model.ck_model_run_encoder.argtypes = []
    model.ck_model_run_encoder.restype = ctypes.c_int
    status = int(
        model.ck_model_init_with_manifest(
            str(run_dir / "weights.bump").encode(),
            str(run_dir / "weights_manifest.map").encode(),
        )
    )
    if status != 0:
        raise RuntimeError(f"encoder initialization failed with code {status}")
    try:
        input_ptr = int(
            model.ck_model_get_named_activation_ptr(b"audio_features") or 0
        )
        input_bytes = int(
            model.ck_model_get_named_activation_nbytes(b"audio_features")
        )
        if input_ptr == 0 or input_bytes != features.nbytes:
            raise RuntimeError(
                "audio feature ABI mismatch: "
                f"ptr={input_ptr} bytes={input_bytes} expected={features.nbytes}"
            )
        ctypes.memmove(input_ptr, features.ctypes.data, features.nbytes)
        started = time.perf_counter()
        status = int(model.ck_model_run_encoder())
        encoder_seconds = time.perf_counter() - started
        if status != 0:
            raise RuntimeError(f"encoder execution failed with code {status}")
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        tokens = int(config["context_length"])
        embed = int(config["embed_dim"])
        output_ptr = int(
            model.ck_model_get_named_activation_ptr(b"embedded_input") or 0
        )
        output_bytes = int(
            model.ck_model_get_named_activation_nbytes(b"embedded_input")
        )
        required = tokens * embed * np.dtype(np.float32).itemsize
        if output_ptr == 0 or output_bytes < required:
            raise RuntimeError(
                "encoder output ABI mismatch: "
                f"ptr={output_ptr} bytes={output_bytes} required={required}"
            )
        output = np.ctypeslib.as_array(
            ctypes.cast(output_ptr, _FLOAT_P), shape=(tokens * embed,)
        ).copy().reshape(tokens, embed)
    finally:
        model.ck_model_free()

    np.save(args.encoder_output, output)
    args.worker_report.write_text(
        json.dumps(
            {
                "audio": audio_metadata,
                "features_shape": list(features.shape),
                "encoder_shape": list(output.shape),
                "frontend_seconds": frontend_seconds,
                "encoder_seconds": encoder_seconds,
                "feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                "encoder_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def forced_decoder_prefix(
    generation: dict[str, Any], language: str, task: str
) -> list[int]:
    start = int(generation["decoder_start_token_id"])
    language_token = generation.get("lang_to_id", {}).get(f"<|{language}|>")
    task_token = generation.get("task_to_id", {}).get(task)
    no_timestamps = generation.get("no_timestamps_token_id")
    if language_token is None:
        raise ValueError(f"unsupported Whisper language: {language}")
    if task_token is None:
        raise ValueError(f"unsupported Whisper task: {task}")
    if no_timestamps is None:
        raise ValueError("generation_config.json has no no_timestamps_token_id")
    return [start, int(language_token), int(task_token), int(no_timestamps)]


def _decoder_worker(args: argparse.Namespace) -> int:
    run_dir = args.decoder_run_dir.resolve()
    _require_artifact(run_dir)
    generation_path = run_dir / "generation_config.json"
    tokenizer_path = run_dir / "tokenizer.json"
    for path in (generation_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    encoder_memory = np.load(args.encoder_output).astype(np.float32, copy=False)

    model = _load_generated_model(run_dir)
    model.ck_model_set_encoder_memory.argtypes = [
        _FLOAT_P,
        ctypes.c_int,
        ctypes.c_int,
    ]
    model.ck_model_set_encoder_memory.restype = ctypes.c_int
    model.ck_model_embed_tokens.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    model.ck_model_embed_tokens.restype = ctypes.c_int
    model.ck_model_decode.argtypes = [ctypes.c_int32, _FLOAT_P]
    model.ck_model_decode.restype = ctypes.c_int
    model.ck_model_get_logits.argtypes = []
    model.ck_model_get_logits.restype = _FLOAT_P
    model.ck_model_get_vocab_size.argtypes = []
    model.ck_model_get_vocab_size.restype = ctypes.c_int

    status = int(
        model.ck_model_init_with_manifest(
            str(run_dir / "weights.bump").encode(),
            str(run_dir / "weights_manifest.map").encode(),
        )
    )
    if status != 0:
        raise RuntimeError(f"decoder initialization failed with code {status}")
    try:
        status = int(
            model.ck_model_set_encoder_memory(
                _fptr(encoder_memory),
                encoder_memory.shape[0],
                encoder_memory.shape[1],
            )
        )
        if status != 0:
            raise RuntimeError(f"encoder-memory binding failed with code {status}")
        prefix = forced_decoder_prefix(generation, args.language, args.task)
        prefix_array = (ctypes.c_int32 * len(prefix))(*prefix)
        started = time.perf_counter()
        status = int(model.ck_model_embed_tokens(prefix_array, len(prefix)))
        prefill_seconds = time.perf_counter() - started
        if status != 0:
            raise RuntimeError(f"decoder prefill failed with code {status}")

        vocab_size = int(model.ck_model_get_vocab_size())
        suppress = np.asarray(generation.get("suppress_tokens", []), dtype=np.int64)
        begin_suppress = np.asarray(
            generation.get("begin_suppress_tokens", []), dtype=np.int64
        )
        no_timestamps = int(generation["no_timestamps_token_id"])
        eos = int(generation["eos_token_id"])
        tokens: list[int] = []
        decode_started = time.perf_counter()
        stop = "max_tokens"
        for step in range(args.max_tokens):
            logits = np.ctypeslib.as_array(
                model.ck_model_get_logits(), shape=(vocab_size,)
            ).copy()
            logits[suppress] = -np.inf
            if step == 0:
                logits[begin_suppress] = -np.inf
            logits[no_timestamps:] = -np.inf
            token = int(np.argmax(logits))
            if token == eos:
                stop = "eos"
                break
            tokens.append(token)
            status = int(model.ck_model_decode(token, None))
            if status != 0:
                raise RuntimeError(
                    f"decoder step {step} failed with code {status}"
                )
        decode_seconds = time.perf_counter() - decode_started
    finally:
        model.ck_model_free()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    args.worker_report.write_text(
        json.dumps(
            {
                "forced_prefix": prefix,
                "generated_tokens": tokens,
                "generated_count": len(tokens),
                "stop": stop,
                "text": text,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _run_parent(args: argparse.Namespace) -> int:
    encoder_dir = args.encoder_run_dir.resolve()
    decoder_dir = args.decoder_run_dir.resolve()
    wav_path = args.wav.resolve()
    _require_artifact(encoder_dir)
    _require_artifact(decoder_dir)
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)

    with tempfile.TemporaryDirectory(prefix="cke-whisper-") as temp_text:
        temp = Path(temp_text)
        encoder_output = temp / "encoder.npy"
        encoder_report = temp / "encoder.json"
        decoder_report = temp / "decoder.json"
        common = [sys.executable, str(Path(__file__).resolve())]
        subprocess.run(
            [
                *common,
                "_encoder",
                "--encoder-run-dir",
                str(encoder_dir),
                "--wav",
                str(wav_path),
                "--encoder-output",
                str(encoder_output),
                "--worker-report",
                str(encoder_report),
            ],
            check=True,
        )
        subprocess.run(
            [
                *common,
                "_decoder",
                "--decoder-run-dir",
                str(decoder_dir),
                "--encoder-output",
                str(encoder_output),
                "--language",
                args.language,
                "--task",
                args.task,
                "--max-tokens",
                str(args.max_tokens),
                "--worker-report",
                str(decoder_report),
            ],
            check=True,
        )
        encoder = json.loads(encoder_report.read_text(encoding="utf-8"))
        decoder = json.loads(decoder_report.read_text(encoding="utf-8"))

    report = {
        "schema": "cke.whisper_e2e",
        "schema_version": 1,
        "status": "ok",
        "wav": str(wav_path),
        "wav_sha256": _sha256(wav_path),
        "encoder_run_dir": str(encoder_dir),
        "decoder_run_dir": str(decoder_dir),
        "encoder_runtime_sha256": _sha256(encoder_dir / "libmodel.so"),
        "decoder_runtime_sha256": _sha256(decoder_dir / "libmodel.so"),
        "language": args.language,
        "task": args.task,
        "encoder": encoder,
        "decoder": decoder,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(decoder["text"])
    print(
        "frontend={:.3f}s encoder={:.3f}s prefill={:.3f}s "
        "decode={:.3f}s tokens={} stop={}".format(
            encoder["frontend_seconds"],
            encoder["encoder_seconds"],
            decoder["prefill_seconds"],
            decoder["decode_seconds"],
            decoder["generated_count"],
            decoder["stop"],
        ),
        file=sys.stderr,
    )
    if args.output:
        print(f"report={args.output}", file=sys.stderr)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--encoder-run-dir", type=Path, required=True)
    run.add_argument("--decoder-run-dir", type=Path, required=True)
    run.add_argument("--wav", type=Path, required=True)
    run.add_argument("--language", default="en")
    run.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument("--output", type=Path)

    encoder = subparsers.add_parser("_encoder", help=argparse.SUPPRESS)
    encoder.add_argument("--encoder-run-dir", type=Path, required=True)
    encoder.add_argument("--wav", type=Path, required=True)
    encoder.add_argument("--encoder-output", type=Path, required=True)
    encoder.add_argument("--worker-report", type=Path, required=True)

    decoder = subparsers.add_parser("_decoder", help=argparse.SUPPRESS)
    decoder.add_argument("--decoder-run-dir", type=Path, required=True)
    decoder.add_argument("--encoder-output", type=Path, required=True)
    decoder.add_argument("--language", required=True)
    decoder.add_argument("--task", required=True)
    decoder.add_argument("--max-tokens", type=int, required=True)
    decoder.add_argument("--worker-report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "_encoder":
        return _encoder_worker(args)
    if args.command == "_decoder":
        return _decoder_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
