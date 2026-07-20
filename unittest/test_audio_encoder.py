#!/usr/bin/env python3
"""Oracle coverage for reusable audio-transformer primitive kernels."""

from __future__ import annotations

import ctypes
import math

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib


lib = load_lib("libckernel_audio.so", "libckernel_engine.so")
attention_lib = load_lib("libckernel_attention.so", "libckernel_engine.so")
_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_I16_P = ctypes.POINTER(ctypes.c_int16)


def _fptr(array: np.ndarray) -> _FLOAT_P:
    return array.ctypes.data_as(_FLOAT_P)


def _i16ptr(array: np.ndarray) -> _I16_P:
    return array.ctypes.data_as(_I16_P)


lib.audio_pcm_s16_to_mono_f32.argtypes = [
    _I16_P, ctypes.c_int, ctypes.c_int, _FLOAT_P,
]
lib.audio_pcm_s16_to_mono_f32.restype = ctypes.c_int
lib.audio_resampled_frame_count.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.audio_resampled_frame_count.restype = ctypes.c_int
lib.audio_resample_linear_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, ctypes.c_int, _FLOAT_P, ctypes.c_int, ctypes.c_int,
]
lib.audio_resample_linear_f32.restype = ctypes.c_int
lib.audio_stft_precompute_tables_f32.argtypes = [
    ctypes.c_int, _FLOAT_P, _FLOAT_P, _FLOAT_P,
]
lib.audio_stft_precompute_tables_f32.restype = ctypes.c_int
lib.audio_stft_power_precomputed_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, _FLOAT_P, ctypes.c_int,
]
lib.audio_stft_power_precomputed_f32.restype = ctypes.c_int
lib.audio_whisper_stft_power_reference_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, _FLOAT_P, ctypes.c_int,
]
lib.audio_whisper_stft_power_reference_f32.restype = ctypes.c_int
lib.audio_conv1d_channel_major_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.audio_conv1d_channel_major_f32.restype = ctypes.c_int
lib.audio_transpose_channel_to_token_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, ctypes.c_int, ctypes.c_int,
]
lib.audio_transpose_channel_to_token_f32.restype = ctypes.c_int
attention_lib.attention_forward_query_key_head_major_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
attention_lib.attention_forward_query_key_head_major_f32.restype = ctypes.c_int


def check_pcm() -> None:
    stereo = np.array(
        [[-32768, -32768], [32767, 32767], [-1000, 1000], [1234, 5678]],
        dtype=np.int16,
    )
    actual = np.empty(stereo.shape[0], dtype=np.float32)
    assert lib.audio_pcm_s16_to_mono_f32(
        _i16ptr(stereo), stereo.shape[0], stereo.shape[1], _fptr(actual)
    ) == 0
    expected = np.array([-1.0, 32767.0 / 32768.0, 0.0, 3456.0 / 32768.0], dtype=np.float32)
    assert np.array_equal(actual, expected)
    print("audio_pcm_s16_stereo_to_mono max_diff=0 tol=0 [PASS]")


def check_resample() -> None:
    source = np.random.default_rng(20260720).normal(0.0, 0.2, 97).astype(np.float32)
    output_frames = lib.audio_resampled_frame_count(source.size, 48000, 16000)
    assert output_frames == 33
    actual = np.empty(output_frames, dtype=np.float32)
    assert lib.audio_resample_linear_f32(
        _fptr(source), source.size, 48000, _fptr(actual), output_frames, 16000
    ) == 0
    expected = source[np.arange(output_frames, dtype=np.int64) * 3]
    assert np.array_equal(actual, expected)
    invalid = np.empty(output_frames + 1, dtype=np.float32)
    assert lib.audio_resample_linear_f32(
        _fptr(source), source.size, 48000, _fptr(invalid), invalid.size, 16000
    ) == -2
    print("audio_resample_linear_48k_to_16k max_diff=0 tol=0 [PASS]")

    source = np.random.default_rng(44100).normal(0.0, 0.2, 127).astype(np.float32)
    output_frames = lib.audio_resampled_frame_count(source.size, 44100, 16000)
    actual = np.empty(output_frames, dtype=np.float32)
    assert lib.audio_resample_linear_f32(
        _fptr(source), source.size, 44100, _fptr(actual), output_frames, 16000
    ) == 0
    expected = np.empty_like(actual)
    for output_index in range(output_frames):
        numerator = output_index * 44100
        left = numerator // 16000
        remainder = numerator % 16000
        right = min(left + 1, source.size - 1)
        fraction = np.float32(remainder) / np.float32(16000)
        expected[output_index] = np.float32(
            source[left] + fraction * np.float32(source[right] - source[left])
        )
    max_diff = float(np.max(np.abs(actual - expected)))
    assert max_diff <= 1.0e-7, max_diff
    print(
        f"audio_resample_linear_44k1_to_16k max_diff={max_diff:.8e} "
        "tol=1.0e-07 [PASS]"
    )


def check_precomputed_stft() -> None:
    n_fft = 400
    hop = 160
    bins = n_fft // 2 + 1
    samples = np.random.default_rng(73).normal(0.0, 0.1, 3200).astype(np.float32)
    frames = samples.size // hop
    window = np.empty(n_fft, dtype=np.float32)
    cos_table = np.empty((bins, n_fft), dtype=np.float32)
    sin_table = np.empty_like(cos_table)
    assert lib.audio_stft_precompute_tables_f32(
        n_fft, _fptr(window), _fptr(cos_table), _fptr(sin_table)
    ) == 0
    direct = np.empty((frames, bins), dtype=np.float32)
    table = np.empty_like(direct)
    assert lib.audio_whisper_stft_power_reference_f32(
        _fptr(samples), samples.size, _fptr(direct), frames
    ) == 0
    assert lib.audio_stft_power_precomputed_f32(
        _fptr(samples), samples.size, _fptr(window), _fptr(cos_table),
        _fptr(sin_table), n_fft, hop, _fptr(table), frames
    ) == 0
    assert np.array_equal(table, direct)
    print("audio_stft_precomputed_vs_direct max_diff=0 tol=0 [PASS]")


def _check_conv(name: str, cin: int, cout: int, frames: int, stride: int) -> None:
    rng = np.random.default_rng(cin * 1000 + cout + frames + stride)
    source = rng.normal(0.0, 0.15, (cin, frames)).astype(np.float32)
    weight = rng.normal(0.0, 0.08, (cout, cin, 3)).astype(np.float32)
    bias = rng.normal(0.0, 0.03, cout).astype(np.float32)
    output_frames = (frames + 2 - 3) // stride + 1
    actual = np.empty((cout, output_frames), dtype=np.float32)
    assert lib.audio_conv1d_channel_major_f32(
        _fptr(source), _fptr(weight), _fptr(bias), _fptr(actual),
        cin, cout, frames, 3, stride, 1, output_frames,
    ) == 0
    expected = F.conv1d(
        torch.from_numpy(source)[None], torch.from_numpy(weight),
        torch.from_numpy(bias), stride=stride, padding=1,
    )[0].numpy()
    max_diff = float(np.max(np.abs(actual - expected)))
    rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
    assert max_diff <= 2.0e-5, (name, max_diff)
    assert rmse <= 2.0e-6, (name, rmse)
    print(
        f"{name} max_diff={max_diff:.8e} tol=2.0e-05 [PASS] "
        f"rmse={rmse:.8e} rmse_tol=2.0e-06"
    )


def check_transpose() -> None:
    source = np.arange(7 * 13, dtype=np.float32).reshape(7, 13)
    actual = np.empty((13, 7), dtype=np.float32)
    assert lib.audio_transpose_channel_to_token_f32(
        _fptr(source), _fptr(actual), 7, 13
    ) == 0
    assert np.array_equal(actual, source.T)
    print("audio_channel_to_token_transpose max_diff=0 tol=0 [PASS]")


def _check_cross_attention(name: str, heads: int, query_tokens: int, key_tokens: int, dim: int) -> None:
    rng = np.random.default_rng(heads * 100000 + query_tokens * 1000 + key_tokens + dim)
    query = rng.normal(0.0, 0.12, (heads, query_tokens, dim)).astype(np.float32)
    key = rng.normal(0.0, 0.12, (heads, key_tokens, dim)).astype(np.float32)
    value = rng.normal(0.0, 0.12, (heads, key_tokens, dim)).astype(np.float32)
    actual = np.empty_like(query)
    scratch = np.empty(key_tokens, dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(dim))
    assert attention_lib.attention_forward_query_key_head_major_f32(
        _fptr(query), _fptr(key), _fptr(value), _fptr(actual), _fptr(scratch),
        heads, query_tokens, key_tokens, dim, float(scale),
    ) == 0
    tq = torch.from_numpy(query)
    tk = torch.from_numpy(key)
    tv = torch.from_numpy(value)
    expected = (torch.softmax((tq @ tk.transpose(-1, -2)) * float(scale), dim=-1) @ tv).numpy()
    max_diff = float(np.max(np.abs(actual - expected)))
    rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
    assert max_diff <= 2.0e-6, (name, max_diff)
    assert rmse <= 3.0e-7, (name, rmse)
    print(
        f"{name} max_diff={max_diff:.8e} tol=2.0e-06 [PASS] "
        f"rmse={rmse:.8e} rmse_tol=3.0e-07"
    )


def main() -> None:
    torch.set_num_threads(1)
    check_pcm()
    check_resample()
    check_precomputed_stft()
    _check_conv("audio_conv1d_whisper_stem1", 80, 384, 16, 1)
    _check_conv("audio_conv1d_whisper_stem2", 384, 384, 16, 2)
    check_transpose()
    _check_cross_attention("audio_encoder_self_attention_equal", 6, 11, 11, 64)
    _check_cross_attention("audio_cross_attention_unequal_small", 3, 5, 17, 8)
    _check_cross_attention("audio_cross_attention_whisper_decode", 6, 1, 1500, 64)
    print("ALL TESTS PASSED (10/10)")


if __name__ == "__main__":
    main()
