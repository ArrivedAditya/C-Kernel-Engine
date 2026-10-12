"""Captured Gemma attention arithmetic at the first ordinary/staged split.

The small operand fixture contains post-RoPE Q at queries 8/9 and logical
K/V rows 0..9 from a single identified generated runtime. It intentionally
does not make a whole-model parity or mixed-batching claim.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import platform
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/gemma3_attention_boundary_20261012.json"
FLOAT = ctypes.c_float
FLOAT_PTR = ctypes.POINTER(FLOAT)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _softmax64(scores: list[float]) -> list[float]:
    weights = [math.exp(score - max(scores)) for score in scores]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _reduce_eight(lanes: list[float]) -> float:
    halves = [_f32(lanes[i] + lanes[i + 4]) for i in range(4)]
    pairs = [_f32(halves[i] + halves[i + 2]) for i in range(2)]
    return _f32(pairs[0] + pairs[1])


def _phase_dot(a: list[float], b: list[float], phase: str) -> float:
    """Independent scalar emulation of the declared AVX2 accumulation tree."""
    assert len(a) == len(b) and len(a) % 32 == 0
    count = 1 if phase == "ordinary" else 4
    accumulators = [[0.0] * 8 for _ in range(count)]
    for offset in range(0, len(a), count * 8):
        for group in range(count):
            for lane in range(8):
                index = offset + group * 8 + lane
                product = _f16(a[index]) * _f16(b[index])
                accumulators[group][lane] = _f32(
                    accumulators[group][lane] + product
                )
    if count == 4:
        merged = [
            _f32(_f32(accumulators[0][i] + accumulators[2][i]) +
                 _f32(accumulators[1][i] + accumulators[3][i]))
            for i in range(8)
        ]
    else:
        merged = accumulators[0]
    return _reduce_eight(merged)


def _fixture() -> tuple[dict, list[list[float]], list[list[float]], list[list[float]]]:
    meta = json.loads(FIXTURE.read_text())
    raw = (FIXTURE.parent / meta["operand_file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == meta["operand_sha256"]
    values = struct.unpack("<5632f", raw)
    rows = [list(values[i:i + 256]) for i in range(0, len(values), 256)]
    assert len(rows) == 22
    return meta, rows[:2], rows[2:12], rows[12:22]


def _phase_result(meta: dict, v: list[list[float]], position: int,
                  phase: str) -> list[float]:
    captured = meta["cases"][str(position)]["phases"][phase]
    padded = captured["padded_tokens"]
    probs = captured["probabilities"] + [0.0] * (padded - position - 1)
    out = []
    for channel in range(256):
        column = [row[channel] for row in v[:position + 1]]
        column += [0.0] * (padded - len(column))
        out.append(_phase_dot(probs, column, phase))
    return out


def test_captured_scores_and_value_reductions_follow_declared_phase_contract():
    meta, queries, keys, values = _fixture()
    for position, q in ((8, queries[0]), (9, queries[1])):
        exact_scores = [math.fsum(_f16(a) * _f16(b) for a, b in zip(q, key))
                        * 0.0625 for key in keys[:position + 1]]
        exact_probabilities = _softmax64(exact_scores)
        unit_roundoff = 2 ** -24
        gamma_256 = 256 * unit_roundoff / (1 - 256 * unit_roundoff)
        for phase in ("ordinary", "staged"):
            captured = meta["cases"][str(position)]["phases"][phase]
            scores = [_f32(_phase_dot(q, key, phase) * 0.0625)
                      for key in keys[:position + 1]]
            assert scores == captured["scores"]
            for observed, exact, key in zip(scores, exact_scores, keys):
                absolute_product_sum = math.fsum(
                    abs(_f16(a) * _f16(b)) for a, b in zip(q, key)) * 0.0625
                assert abs(observed - exact) <= gamma_256 * absolute_product_sum
            out = _phase_result(meta, values, position, phase)
            raw = struct.pack("<256f", *out)
            assert hashlib.sha256(raw).hexdigest() == captured["output_sha256"]
            for channel, observed in enumerate(out):
                rounded_products = [
                    _f16(captured["probabilities"][token]) *
                    _f16(values[token][channel])
                    for token in range(position + 1)
                ]
                exact_output = math.fsum(
                    exact_probabilities[token] * _f16(values[token][channel])
                    for token in range(position + 1))
                probability_conversion = math.fsum(
                    abs(_f16(captured["probabilities"][token]) -
                        exact_probabilities[token]) * abs(_f16(values[token][channel]))
                    for token in range(position + 1))
                accumulation = gamma_256 * math.fsum(map(abs, rounded_products))
                assert abs(observed - exact_output) <= (
                    probability_conversion + accumulation)

    # Query 8 is the preceding control. At query 9 only head 2's position-7
    # probability crosses a half-precision midpoint after FP32 softmax.
    control = meta["cases"]["8"]["phases"]
    assert control["ordinary"]["output_sha256"] == control["staged"]["output_sha256"]
    assert _f16(control["ordinary"]["probabilities"][7]) == _f16(
        control["staged"]["probabilities"][7])
    failing = meta["cases"]["9"]["phases"]
    assert failing["ordinary"]["output_sha256"] != failing["staged"]["output_sha256"]
    p_ordinary = failing["ordinary"]["probabilities"][7]
    p_staged = failing["staged"]["probabilities"][7]
    low, high = _f16(p_ordinary), _f16(p_staged)
    assert low < high
    midpoint = (low + high) / 2
    assert p_ordinary < midpoint < p_staged
    # The score reduction alone puts the two phases on opposite sides. The
    # common FP32 softmax changes the distances but not that classification.
    from_ordinary_scores = _softmax64(failing["ordinary"]["scores"])[7]
    from_staged_scores = _softmax64(failing["staged"]["scores"])[7]
    assert from_ordinary_scores < midpoint < from_staged_scores
    exact_scores = [math.fsum(_f16(a) * _f16(b) for a, b in zip(queries[1], key))
                    * 0.0625 for key in keys]
    high_precision_p = _softmax64(exact_scores)[7]
    assert high_precision_p > midpoint
    assert abs(p_staged - high_precision_p) < abs(p_ordinary - high_precision_p)


def test_captured_boundary_detects_perturbed_operand_and_position():
    meta, queries, keys, _ = _fixture()
    captured = meta["cases"]["9"]["phases"]["ordinary"]["scores"]
    wrong_position = [_f32(_phase_dot(queries[0], key, "ordinary") * 0.0625)
                      for key in keys]
    assert wrong_position != captured
    wrong_key_order = keys.copy()
    wrong_key_order[7], wrong_key_order[8] = wrong_key_order[8], wrong_key_order[7]
    assert [_f32(_phase_dot(queries[1], key, "ordinary") * 0.0625)
            for key in wrong_key_order] != captured

    perturbed = queries[1].copy()
    assert _f16(perturbed[0]) == -0.364501953125
    perturbed[0] = -0.36474609375  # Adjacent representable FP16 operand.
    changed_scores = [_f32(_phase_dot(perturbed, key, "ordinary") * 0.0625)
                      for key in keys]
    assert changed_scores != captured
    exact_scores = [math.fsum(_f16(a) * _f16(b) for a, b in zip(perturbed, key))
                    * 0.0625 for key in keys]
    probability = _softmax64(exact_scores)[7]
    failing = meta["cases"]["9"]["phases"]
    midpoint = (_f16(failing["ordinary"]["probabilities"][7]) +
                _f16(failing["staged"]["probabilities"][7])) / 2
    assert probability < midpoint


def _avx2_available() -> bool:
    if (platform.machine().lower() not in {"x86_64", "amd64"} or
            not Path("/proc/cpuinfo").exists()):
        return False
    flags = Path("/proc/cpuinfo").read_text().lower().split()
    return all(flag in flags for flag in ("avx2", "fma", "f16c"))


@pytest.mark.skipif(not _avx2_available(), reason="captured AVX2 arithmetic requires AVX2/FMA/F16C")
def test_native_attention_replays_captured_operands_in_both_phases():
    meta, queries, keys, values = _fixture()
    with tempfile.TemporaryDirectory(prefix="cke-gemma3-attention-") as directory:
        library = Path(directory) / "libattention.so"
        subprocess.run([
            "gcc", "-std=gnu11", "-O2", "-shared", "-fPIC", "-mavx2",
            "-mfma", "-mf16c", "-I", str(ROOT / "include"),
            *[str(ROOT / source) for source in (
                "src/kernels/attention_kernels.c",
                "src/kernels/attention_kernels_sliding.c",
                "src/kernels/attention_flash_true.c",
                "src/kernels/softmax_kernels.c",
                "src/kernels/gemm_kernels_bf16.c",
                "src/ckernel_strict.c", "src/ck_threadpool.c")],
            "-lm", "-lpthread", "-o", str(library),
        ], check=True)
        native = ctypes.CDLL(str(library))
        prefill = getattr(
            native,
            "attention_forward_causal_head_major_gqa_llama_regular_strided_sliding_workspace",
        )
        prefill.argtypes = [FLOAT_PTR] * 4 + [ctypes.c_int] * 7 + [
            FLOAT_PTR, ctypes.c_size_t] * 3
        decode = native.attention_forward_decode_head_major_gqa_llama_regular_sliding_workspace
        decode.argtypes = [FLOAT_PTR] * 4 + [ctypes.c_int] * 7 + [
            FLOAT_PTR, ctypes.c_size_t] * 3

        def floats(items: list[float]):
            return (FLOAT * len(items))(*items)

        for position, query in ((8, queries[0]), (9, queries[1])):
            count = position + 1
            k = floats([item for row in keys for item in row] +
                       [0.0] * ((128 - 10) * 256))
            v = floats([item for row in values for item in row] +
                       [0.0] * ((128 - 10) * 256))
            for heads in (1, 4):
                selected_head = 0 if heads == 1 else 2
                q_rows = [0.0] * (heads * count * 256)
                q_rows[(selected_head * count + position) * 256:
                       (selected_head * count + position + 1) * 256] = query
                q_decode_rows = [0.0] * (heads * 256)
                q_decode_rows[selected_head * 256:(selected_head + 1) * 256] = query
                q = floats(q_rows)
                q_decode = floats(q_decode_rows)
                output_prefill = floats([0.0] * (heads * count * 256))
                output_decode = floats([0.0] * (heads * 256))
                for phase, function, q_arg, output, padded in (
                    ("ordinary", prefill, q, output_prefill, 256),
                    ("staged", decode, q_decode, output_decode, 128),
                ):
                    scores = floats([0.0] * (padded * 2))
                    columns = floats([0.0] * (256 * padded))
                    scaled = floats([0.0] * (padded * 2))
                    if phase == "ordinary":
                        function(q_arg, k, v, output, heads, 1, count, 256, 256,
                                 128, 512, scores, ctypes.sizeof(scores), columns,
                                 ctypes.sizeof(columns), scaled, ctypes.sizeof(scaled))
                        start = (selected_head * count + position) * 256
                    else:
                        function(q_arg, k, v, output, heads, 1, count, 128, 256,
                                 256, 512, scores, ctypes.sizeof(scores), columns,
                                 ctypes.sizeof(columns), scaled, ctypes.sizeof(scaled))
                        start = selected_head * 256
                    result = list(output)[start:start + 256]
                    assert hashlib.sha256(struct.pack("<256f", *result)).hexdigest() == (
                        meta["cases"][str(position)]["phases"][phase]["output_sha256"])


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
