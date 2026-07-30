#!/usr/bin/env python3
"""Whisper-compatible audio frontend parity tests."""

from __future__ import annotations

import ctypes
import hashlib
import math
import struct

import numpy as np
import torch

from lib_loader import load_lib


N_FFT = 400
HOP_LENGTH = 160
POWER_BINS = N_FFT // 2 + 1
N_MELS = 80

lib = load_lib("libckernel_audio.so", "libckernel_engine.so")
_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_U8_P = ctypes.POINTER(ctypes.c_uint8)

lib.audio_whisper_stft_power_reference_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, _FLOAT_P, ctypes.c_int,
]
lib.audio_whisper_stft_power_reference_f32.restype = ctypes.c_int
lib.audio_whisper_log_mel_from_power_reference_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, ctypes.c_int, ctypes.c_int, _FLOAT_P,
]
lib.audio_whisper_log_mel_from_power_reference_f32.restype = ctypes.c_int
lib.audio_whisper_log_mel_reference_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, _FLOAT_P, ctypes.c_int, _FLOAT_P, _FLOAT_P,
    ctypes.c_int,
]
lib.audio_whisper_log_mel_reference_f32.restype = ctypes.c_int
lib.audio_whisper_mel_filters_slaney_f32.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, _FLOAT_P,
]
lib.audio_whisper_mel_filters_slaney_f32.restype = ctypes.c_int
lib.audio_stft_precompute_tables_f32.argtypes = [
    ctypes.c_int, _FLOAT_P, _FLOAT_P, _FLOAT_P,
]
lib.audio_stft_precompute_tables_f32.restype = ctypes.c_int
lib.audio_whisper_log_mel_window_wav_pcm16_f32.argtypes = [
    _U8_P, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
    _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, _FLOAT_P,
]
lib.audio_whisper_log_mel_window_wav_pcm16_f32.restype = ctypes.c_int


def _ptr(array: np.ndarray) -> _FLOAT_P:
    return array.ctypes.data_as(_FLOAT_P)


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    f_sp = 200.0 / 3.0
    mels = frequencies / f_sp
    logarithmic = frequencies >= 1000.0
    mels[logarithmic] = 15.0 + np.log(frequencies[logarithmic] / 1000.0) / (
        math.log(6.4) / 27.0
    )
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    frequencies = (200.0 / 3.0) * mels
    logarithmic = mels >= 15.0
    frequencies[logarithmic] = 1000.0 * np.exp(
        (math.log(6.4) / 27.0) * (mels[logarithmic] - 15.0)
    )
    return frequencies


def _whisper_mel_filters() -> np.ndarray:
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
    filters *= (2.0 / (filter_frequencies[2:] - filter_frequencies[:-2]))[:, None]
    return np.ascontiguousarray(filters.astype(np.float32))


def _torch_reference(samples: np.ndarray, filters: np.ndarray):
    audio = torch.from_numpy(samples)
    window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float32)
    spectrum = torch.stft(
        audio,
        N_FFT,
        HOP_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    power = spectrum[:, :-1].abs().square().transpose(0, 1).contiguous()
    mel = torch.from_numpy(filters) @ power.transpose(0, 1)
    log_mel = torch.clamp(mel, min=1.0e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    return power.numpy(), log_mel.numpy()


def _signal(kind: str, n_samples: int) -> np.ndarray:
    if kind == "tones":
        t = np.arange(n_samples, dtype=np.float32) / np.float32(16000.0)
        signal = (
            0.35 * np.sin(2.0 * np.pi * 440.0 * t)
            + 0.20 * np.sin(2.0 * np.pi * 1000.0 * t)
            + 0.08 * np.cos(2.0 * np.pi * 3125.0 * t)
        )
        return np.ascontiguousarray(signal.astype(np.float32))
    if kind == "noise":
        return np.ascontiguousarray(
            np.random.default_rng(20260720).normal(0.0, 0.12, n_samples).astype(np.float32)
        )
    if kind == "impulse":
        signal = np.zeros(n_samples, dtype=np.float32)
        signal[0] = 0.75
        signal[n_samples // 2] = -0.5
        signal[-1] = 0.25
        return signal
    raise ValueError(kind)


def _check_case(kind: str) -> tuple[float, float]:
    samples = _signal(kind, 3200)
    filters = _whisper_mel_filters()
    n_frames = samples.size // HOP_LENGTH
    expected_power, expected_log_mel = _torch_reference(samples, filters)
    actual_power = np.empty((n_frames, POWER_BINS), dtype=np.float32)
    actual_log_mel = np.empty((N_MELS, n_frames), dtype=np.float32)

    status = lib.audio_whisper_log_mel_reference_f32(
        _ptr(samples), samples.size, _ptr(filters), N_MELS,
        _ptr(actual_power), _ptr(actual_log_mel), n_frames,
    )
    assert status == 0, f"{kind}: kernel status {status}"

    power_scale = max(1.0, float(np.max(np.abs(expected_power))))
    power_error = float(np.max(np.abs(actual_power - expected_power))) / power_scale
    mel_error = float(np.max(np.abs(actual_log_mel - expected_log_mel)))
    mel_rmse = float(np.sqrt(np.mean((actual_log_mel - expected_log_mel) ** 2)))
    assert power_error <= 2.0e-5, f"{kind}: STFT relative max error {power_error}"
    assert mel_error <= 1.1e-3, f"{kind}: log-Mel max error {mel_error}"
    assert mel_rmse <= 9.0e-5, f"{kind}: log-Mel RMSE {mel_rmse}"
    print(
        f"whisper_log_mel_{kind} max_diff={mel_error:.8e} tol=1.1e-03 [PASS] "
        f"power_rel_max={power_error:.8e} log_mel_rmse={mel_rmse:.8e} "
        "rmse_tol=9.0e-05"
    )
    return power_error, mel_error


def _check_stage_composition() -> None:
    samples = _signal("tones", 3200)
    filters = _whisper_mel_filters()
    n_frames = samples.size // HOP_LENGTH
    composed_power = np.empty((n_frames, POWER_BINS), dtype=np.float32)
    composed_log_mel = np.empty((N_MELS, n_frames), dtype=np.float32)
    staged_power = np.empty_like(composed_power)
    staged_log_mel = np.empty_like(composed_log_mel)
    assert lib.audio_whisper_log_mel_reference_f32(
        _ptr(samples), samples.size, _ptr(filters), N_MELS,
        _ptr(composed_power), _ptr(composed_log_mel), n_frames,
    ) == 0
    assert lib.audio_whisper_stft_power_reference_f32(
        _ptr(samples), samples.size, _ptr(staged_power), n_frames,
    ) == 0
    assert lib.audio_whisper_log_mel_from_power_reference_f32(
        _ptr(staged_power), _ptr(filters), N_MELS, n_frames, _ptr(staged_log_mel),
    ) == 0
    assert np.array_equal(staged_power, composed_power)
    assert np.array_equal(staged_log_mel, composed_log_mel)
    print("whisper_log_mel_stage_composition max_diff=0 tol=0 [PASS]")


def _check_invalid_shapes() -> None:
    samples = np.zeros(400, dtype=np.float32)
    power = np.zeros((2, POWER_BINS), dtype=np.float32)
    assert lib.audio_whisper_stft_power_reference_f32(
        _ptr(samples), samples.size, _ptr(power), 3,
    ) == -3
    assert lib.audio_whisper_stft_power_reference_f32(
        _ptr(samples[:200]), 200, _ptr(power), 1,
    ) == -2
    print("whisper_log_mel_shape_contract max_diff=0 tol=0 [PASS]")


def _check_filter_identity() -> None:
    filters = _whisper_mel_filters()
    generated = np.empty_like(filters)
    assert lib.audio_whisper_mel_filters_slaney_f32(
        16000, N_FFT, N_MELS, _ptr(generated)
    ) == 0
    assert np.array_equal(generated, filters)
    digest = hashlib.sha256(filters.tobytes()).hexdigest()
    assert digest == "2150c30cbbeb6029f52002ffa666c1c72d83dbf53f463cb8462052055806e891"
    assert filters.shape == (80, 201)
    assert int(np.count_nonzero(filters)) == 391
    print("whisper_slaney_80_filter_identity max_diff=0 tol=0 [PASS]")


def _check_production_shape_silence() -> None:
    samples = np.zeros(480000, dtype=np.float32)
    filters = _whisper_mel_filters()
    power = np.empty((3000, POWER_BINS), dtype=np.float32)
    log_mel = np.empty((N_MELS, 3000), dtype=np.float32)
    status = lib.audio_whisper_log_mel_reference_f32(
        _ptr(samples), samples.size, _ptr(filters), N_MELS,
        _ptr(power), _ptr(log_mel), 3000,
    )
    assert status == 0
    assert np.all(np.isfinite(log_mel))
    assert np.array_equal(log_mel, np.full_like(log_mel, -1.5))
    print("whisper_log_mel_30s_silence(shape=80x3000) max_diff=0 tol=0 [PASS]")


def _check_global_feature_window() -> None:
    source = _signal("tones", 5280)
    pcm = np.rint(source * 32768.0).clip(-32768, 32767).astype("<i2")
    decoded = pcm.astype(np.float32) / np.float32(32768.0)
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    chunks = (
        b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", pcm.nbytes) + pcm.tobytes()
    )
    wav = np.frombuffer(
        b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks,
        dtype=np.uint8,
    )
    filters = _whisper_mel_filters()
    _, complete = _torch_reference(decoded, filters)
    window = np.empty(N_FFT, dtype=np.float32)
    cos_table = np.empty((POWER_BINS, N_FFT), dtype=np.float32)
    sin_table = np.empty((POWER_BINS, N_FFT), dtype=np.float32)
    assert lib.audio_stft_precompute_tables_f32(
        N_FFT, _ptr(window), _ptr(cos_table), _ptr(sin_table)
    ) == 0
    actual = np.full((N_MELS, 20), 99.0, dtype=np.float32)
    valid = lib.audio_whisper_log_mel_window_wav_pcm16_f32(
        wav.ctypes.data_as(_U8_P), wav.size, 20 * HOP_LENGTH, 16000,
        _ptr(window), _ptr(cos_table), _ptr(sin_table), _ptr(filters),
        N_MELS, actual.shape[1], _ptr(actual)
    )
    expected = np.zeros_like(actual)
    expected[:, :13] = complete[:, 20:33]
    difference = np.abs(actual - expected)
    max_diff = float(np.max(difference))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    assert valid == 13
    assert np.array_equal(actual[:, 13:], expected[:, 13:])
    assert max_diff <= 1.1e-3, max_diff
    assert rmse <= 9.0e-5, rmse
    print(
        f"whisper_global_feature_window max_diff={max_diff:.8e} "
        f"tol=1.1e-03 [PASS] rmse={rmse:.8e} rmse_tol=9.0e-05 "
        "tail_zero=1"
    )


def main() -> None:
    torch.set_num_threads(1)
    for kind in ("tones", "noise", "impulse"):
        _check_case(kind)
    _check_stage_composition()
    _check_invalid_shapes()
    _check_filter_identity()
    _check_production_shape_silence()
    _check_global_feature_window()
    print("ALL TESTS PASSED (8/8)")


if __name__ == "__main__":
    main()
