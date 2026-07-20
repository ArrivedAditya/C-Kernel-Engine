#!/usr/bin/env python3
"""Oracle coverage for reusable audio-transformer primitive kernels."""

from __future__ import annotations

import ctypes
import math
import struct

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib


lib = load_lib("libckernel_audio.so", "libckernel_engine.so")
attention_lib = load_lib("libckernel_attention.so", "libckernel_engine.so")
_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_I16_P = ctypes.POINTER(ctypes.c_int16)
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
lib.audio_wav_parse_memory.argtypes = [
    _U8_P, ctypes.c_size_t, ctypes.POINTER(CKAudioWavInfo),
]
lib.audio_wav_parse_memory.restype = ctypes.c_int
lib.audio_wav_decode_pcm16_mono_f32.argtypes = [
    _U8_P, ctypes.c_size_t, ctypes.POINTER(CKAudioWavInfo), _FLOAT_P, ctypes.c_int,
]
lib.audio_wav_decode_pcm16_mono_f32.restype = ctypes.c_int
lib.audio_resample_windowed_sinc_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, ctypes.c_int, _FLOAT_P, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
]
lib.audio_resample_windowed_sinc_f32.restype = ctypes.c_int
lib.audio_stft_power_fft400_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, _FLOAT_P, ctypes.c_int, _FLOAT_P,
]
lib.audio_stft_power_fft400_f32.restype = ctypes.c_int


def check_wav_pcm16() -> None:
    pcm = np.array(
        [[-32768, -32768], [32767, 32767], [-1000, 1000], [1234, 5678]],
        dtype="<i2",
    )
    fmt = struct.pack("<HHIIHH", 1, 2, 48000, 48000 * 4, 4, 16)
    junk = b"abc"
    chunks = (
        b"JUNK" + struct.pack("<I", len(junk)) + junk + b"\0"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", pcm.nbytes) + pcm.tobytes()
    )
    wav = np.frombuffer(
        b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks,
        dtype=np.uint8,
    )
    info = CKAudioWavInfo()
    assert lib.audio_wav_parse_memory(
        wav.ctypes.data_as(_U8_P), wav.size, ctypes.byref(info)
    ) == 0
    assert (info.channels, info.sample_rate, info.bits_per_sample, info.frames) == (
        2, 48000, 16, 4,
    )
    actual = np.empty(info.frames, dtype=np.float32)
    assert lib.audio_wav_decode_pcm16_mono_f32(
        wav.ctypes.data_as(_U8_P), wav.size, ctypes.byref(info),
        _fptr(actual), actual.size,
    ) == info.frames
    expected = np.array(
        [-1.0, 32767.0 / 32768.0, 0.0, 3456.0 / 32768.0], dtype=np.float32,
    )
    assert np.array_equal(actual, expected)
    truncated = wav[:-1].copy()
    assert lib.audio_wav_parse_memory(
        truncated.ctypes.data_as(_U8_P), truncated.size, ctypes.byref(info)
    ) == -3
    print("audio_wav_pcm16_chunked_decode max_diff=0 tol=0 [PASS]")


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


def _windowed_sinc_reference(
    source: np.ndarray, input_rate: int, output_rate: int, radius: int,
) -> np.ndarray:
    output_frames = 1 + ((source.size - 1) * output_rate) // input_rate
    output = np.empty(output_frames, dtype=np.float32)
    cutoff = min(1.0, output_rate / input_rate)
    for frame in range(output_frames):
        coordinate = frame * input_rate / output_rate
        center = math.floor(coordinate)
        weighted = 0.0
        weight_sum = 0.0
        for tap in range(center - radius + 1, center + radius + 1):
            if tap < 0 or tap >= source.size:
                continue
            distance = coordinate - tap
            scaled = cutoff * distance
            sinc = 1.0 if abs(scaled) < 1.0e-12 else math.sin(math.pi * scaled) / (math.pi * scaled)
            window_x = distance / radius
            if abs(window_x) >= 1.0:
                continue
            weight = cutoff * sinc * (0.5 * (1.0 + math.cos(math.pi * window_x)))
            weighted += float(source[tap]) * weight
            weight_sum += weight
        output[frame] = weighted / weight_sum if weight_sum else 0.0
    return output


def check_bandlimited_resample() -> None:
    input_rate = 44100
    output_rate = 16000
    radius = 16
    source = np.random.default_rng(16000).normal(0.0, 0.2, 257).astype(np.float32)
    expected = _windowed_sinc_reference(source, input_rate, output_rate, radius)
    actual = np.empty_like(expected)
    assert lib.audio_resample_windowed_sinc_f32(
        _fptr(source), source.size, input_rate, _fptr(actual), actual.size,
        output_rate, radius,
    ) == 0
    max_diff = float(np.max(np.abs(actual - expected)))
    assert max_diff <= 3.0e-8, max_diff

    time = np.arange(4800, dtype=np.float64) / 48000.0
    alias_source = np.sin(2.0 * math.pi * 12000.0 * time).astype(np.float32)
    alias_frames = lib.audio_resampled_frame_count(alias_source.size, 48000, 16000)
    linear = np.empty(alias_frames, dtype=np.float32)
    filtered = np.empty(alias_frames, dtype=np.float32)
    assert lib.audio_resample_linear_f32(
        _fptr(alias_source), alias_source.size, 48000, _fptr(linear), alias_frames, 16000
    ) == 0
    assert lib.audio_resample_windowed_sinc_f32(
        _fptr(alias_source), alias_source.size, 48000, _fptr(filtered), alias_frames,
        16000, radius,
    ) == 0
    trim = radius
    linear_rms = float(np.sqrt(np.mean(linear[trim:-trim] ** 2)))
    filtered_rms = float(np.sqrt(np.mean(filtered[trim:-trim] ** 2)))
    assert filtered_rms < linear_rms * 0.05, (linear_rms, filtered_rms)
    print(
        f"audio_resample_windowed_sinc max_diff={max_diff:.8e} tol=3.0e-08 [PASS] "
        f"alias_rejection={linear_rms / max(filtered_rms, 1.0e-20):.2f}x"
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

    fft_power = np.empty_like(direct)
    fft_scratch = np.empty(n_fft * 2, dtype=np.float32)
    assert lib.audio_stft_power_fft400_f32(
        _fptr(samples), samples.size, _fptr(window), _fptr(cos_table),
        _fptr(sin_table), hop, _fptr(fft_power), frames, _fptr(fft_scratch),
    ) == 0
    max_diff = float(np.max(np.abs(fft_power - direct)))
    rmse = float(np.sqrt(np.mean((fft_power - direct) ** 2)))
    assert max_diff <= 4.0e-4, max_diff
    assert rmse <= 5.0e-5, rmse
    print(
        f"audio_stft_fft400_vs_direct max_diff={max_diff:.8e} tol=4.0e-04 [PASS] "
        f"rmse={rmse:.8e} rmse_tol=5.0e-05"
    )


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
    check_wav_pcm16()
    check_pcm()
    check_resample()
    check_bandlimited_resample()
    check_precomputed_stft()
    _check_conv("audio_conv1d_whisper_stem1", 80, 384, 16, 1)
    _check_conv("audio_conv1d_whisper_stem2", 384, 384, 16, 2)
    check_transpose()
    _check_cross_attention("audio_encoder_self_attention_equal", 6, 11, 11, 64)
    _check_cross_attention("audio_cross_attention_unequal_small", 3, 5, 17, 8)
    _check_cross_attention("audio_cross_attention_whisper_decode", 6, 1, 1500, 64)
    print("ALL TESTS PASSED (13/13)")


if __name__ == "__main__":
    main()
