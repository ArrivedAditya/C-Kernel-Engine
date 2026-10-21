"""Contracts for llama.cpp-compatible regular Gemma attention."""

import ctypes
import math

import numpy as np

from lib_loader import load_lib


FLOAT_P = ctypes.POINTER(ctypes.c_float)


def _ptr(values: np.ndarray) -> FLOAT_P:
    return values.ctypes.data_as(FLOAT_P)


def _reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    sliding_window: int,
) -> np.ndarray:
    heads, tokens, dim = q.shape
    kv_heads = k.shape[0]
    output = np.zeros_like(q)
    scale = np.float32(1.0 / math.sqrt(dim))
    q_f16 = q.astype(np.float16).astype(np.float32)
    k_f16 = k.astype(np.float16).astype(np.float32)
    v_f16 = v.astype(np.float16).astype(np.float32)
    for head in range(heads):
        kv_head = head * kv_heads // heads
        for query in range(tokens):
            first = max(0, query - sliding_window + 1)
            scores = np.asarray(
                [
                    np.dot(q_f16[head, query], k_f16[kv_head, key])
                    for key in range(first, query + 1)
                ],
                dtype=np.float32,
            )
            scores *= scale
            probabilities = np.exp(scores - np.max(scores), dtype=np.float32)
            probabilities /= np.sum(probabilities, dtype=np.float32)
            probabilities_f16 = probabilities.astype(np.float16).astype(np.float32)
            output[head, query] = probabilities_f16 @ v_f16[kv_head, first : query + 1]
    return output


def test_prefill_uses_declared_scratch_capacity_with_compact_kv_stride() -> None:
    lib = load_lib("libckernel_engine.so")
    kernel = lib.attention_forward_causal_head_major_gqa_llama_regular_strided_sliding_workspace
    kernel.argtypes = [
        FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        FLOAT_P, ctypes.c_size_t, FLOAT_P, ctypes.c_size_t,
        FLOAT_P, ctypes.c_size_t,
    ]
    kernel.restype = None

    rng = np.random.default_rng(20260905)
    heads, kv_heads, tokens, dim = 4, 2, 7, 8
    padded_tokens = 256
    sliding_window = 4
    q = rng.standard_normal((heads, tokens, dim), dtype=np.float32)
    k = rng.standard_normal((kv_heads, tokens, dim), dtype=np.float32)
    v = rng.standard_normal((kv_heads, tokens, dim), dtype=np.float32)
    # Keep the KV heads observably different so an incorrect head stride
    # cannot pass by aliasing equivalent data.
    k[1] += np.float32(3.0)
    v[1] -= np.float32(5.0)
    output = np.zeros_like(q)
    scores = np.full(padded_tokens, np.nan, dtype=np.float32)
    scaled_scores = np.full(padded_tokens, np.nan, dtype=np.float32)
    value_columns = np.full((dim, padded_tokens), np.nan, dtype=np.float32)

    kernel(
        _ptr(q), _ptr(k), _ptr(v), _ptr(output),
        heads, kv_heads, tokens, dim, dim, tokens, sliding_window,
        _ptr(scores), scores.nbytes,
        _ptr(value_columns), value_columns.nbytes,
        _ptr(scaled_scores), scaled_scores.nbytes,
    )

    zeros = np.zeros(padded_tokens - tokens, dtype=np.float32)
    assert np.array_equal(scores[tokens:].view(np.uint32), zeros.view(np.uint32))
    assert np.all(np.isneginf(scaled_scores[tokens:]))
    assert np.all(value_columns[:, tokens:] == 0.0)
    np.testing.assert_allclose(
        output, _reference(q, k, v, sliding_window), rtol=2e-5, atol=2e-5
    )


def test_global_attention_does_not_inherit_512_token_window() -> None:
    lib = load_lib("libckernel_engine.so")
    kernel = lib.attention_forward_causal_head_major_gqa_llama_regular_strided_sliding_workspace
    kernel.argtypes = [
        FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        FLOAT_P, ctypes.c_size_t, FLOAT_P, ctypes.c_size_t,
        FLOAT_P, ctypes.c_size_t,
    ]
    kernel.restype = None

    heads = kv_heads = 1
    tokens, dim, padded_tokens = 520, 8, 768
    q = np.zeros((heads, tokens, dim), dtype=np.float32)
    k = np.zeros((kv_heads, tokens, dim), dtype=np.float32)
    v = np.zeros((kv_heads, tokens, dim), dtype=np.float32)
    v[0, 0, 0] = np.float32(1.0)
    output = np.zeros_like(q)
    scores = np.empty(padded_tokens, dtype=np.float32)
    scaled_scores = np.empty(padded_tokens, dtype=np.float32)
    value_columns = np.empty((dim, padded_tokens), dtype=np.float32)

    kernel(
        _ptr(q), _ptr(k), _ptr(v), _ptr(output),
        heads, kv_heads, tokens, dim, dim, tokens, 0,
        _ptr(scores), scores.nbytes,
        _ptr(value_columns), value_columns.nbytes,
        _ptr(scaled_scores), scaled_scores.nbytes,
    )

    # All causal scores are equal, so the first value contributes 1 / 520.
    # A stale 512-token mask would exclude token zero and produce zero.
    expected = np.float32(np.float16(1.0 / tokens))
    np.testing.assert_allclose(output[0, -1, 0], expected, rtol=0.0, atol=0.0)


def test_parallel_prefill_is_bit_exact_with_multiple_kv_heads() -> None:
    lib = load_lib("libckernel_engine.so")
    kernel = lib.attention_forward_causal_head_major_gqa_llama_regular_strided_sliding_workspace
    kernel.argtypes = [
        FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        FLOAT_P, ctypes.c_size_t, FLOAT_P, ctypes.c_size_t,
        FLOAT_P, ctypes.c_size_t,
    ]
    kernel.restype = None
    lib.ck_set_num_threads.argtypes = [ctypes.c_int]
    lib.ck_threadpool_global_destroy.argtypes = []

    rng = np.random.default_rng(20260909)
    heads, kv_heads = 8, 2
    lanes = 16

    try:
        cases = (
            (67, 32, 31),
            (255, 8, 64),
            (256, 8, 256),
            (257, 8, 256),
            (520, 8, 0),
        )
        for tokens, dim, sliding_window in cases:
            padded_tokens = 256 if tokens < 256 else math.ceil(tokens / 256) * 256
            q = rng.standard_normal((heads, tokens, dim), dtype=np.float32)
            k = rng.standard_normal((kv_heads, tokens, dim), dtype=np.float32)
            v = rng.standard_normal((kv_heads, tokens, dim), dtype=np.float32)
            k[1] += np.float32(2.0)
            v[1] -= np.float32(4.0)

            def run(threads: int) -> np.ndarray:
                lib.ck_threadpool_global_destroy()
                lib.ck_set_num_threads(threads)
                output = np.zeros_like(q)
                scores = np.empty((lanes, padded_tokens), dtype=np.float32)
                scaled_scores = np.empty((lanes, padded_tokens), dtype=np.float32)
                value_columns = np.empty(
                    (kv_heads, dim, padded_tokens), dtype=np.float32
                )
                kernel(
                    _ptr(q), _ptr(k), _ptr(v), _ptr(output),
                    heads, kv_heads, tokens, dim, dim, tokens, sliding_window,
                    _ptr(scores), scores.nbytes,
                    _ptr(value_columns), value_columns.nbytes,
                    _ptr(scaled_scores), scaled_scores.nbytes,
                )
                return output

            serial = run(1)
            threaded = run(8)
            threaded_repeat = run(8)
            assert np.array_equal(serial.view(np.uint32), threaded.view(np.uint32))
            assert np.array_equal(
                threaded.view(np.uint32), threaded_repeat.view(np.uint32)
            )
    finally:
        lib.ck_threadpool_global_destroy()
        lib.ck_set_num_threads(0)


def test_prefill_decode_boundary_uses_prior_kv_with_phase_numerics() -> None:
    """A chunk continuation must read old KV; phase reductions need not be bit-equal."""
    lib = load_lib("libckernel_engine.so")
    prefill = lib.attention_forward_causal_head_major_gqa_llama_regular_strided_sliding_workspace
    decode = lib.attention_forward_decode_head_major_gqa_llama_regular_sliding_workspace
    continuation = (
        lib.attention_forward_decode_head_major_gqa_llama_regular_sliding_prefill_phase_workspace
    )
    prefill.restype = decode.restype = continuation.restype = None
    lib.ck_set_num_threads(1)

    rng = np.random.default_rng(833)
    heads, tokens, dim = 2, 129, 256
    q = rng.normal(0.0, 0.1, (heads, tokens, dim)).astype(np.float32)
    k = rng.normal(0.0, 0.1, (1, tokens, dim)).astype(np.float32)
    v = rng.normal(0.0, 1.0, (1, tokens, dim)).astype(np.float32)
    # Make the oldest row material to the last output. A current-chunk-only
    # attention read or an off-by-one window must fail the oracle comparison.
    v[0, 0, 0] = 32.0
    scores = np.zeros((2, 256), dtype=np.float32)
    scaled = np.zeros_like(scores)
    columns = np.zeros((1, dim, 256), dtype=np.float32)
    batched = np.zeros_like(q)
    single = np.zeros((heads, dim), dtype=np.float32)

    prefill(
        _ptr(q), _ptr(k), _ptr(v), _ptr(batched),
        heads, 1, tokens, dim, dim, tokens, 0,
        _ptr(scores), scores.nbytes,
        _ptr(columns), columns.nbytes,
        _ptr(scaled), scaled.nbytes,
    )
    # The decode path sees exactly the same last Q and all 129 K/V rows.
    last_q = np.ascontiguousarray(q[:, -1, :])
    decode(
        _ptr(last_q), _ptr(k), _ptr(v), _ptr(single),
        heads, 1, tokens, tokens, dim, dim, 0,
        _ptr(scores), scores.nbytes,
        _ptr(columns), columns.nbytes,
        _ptr(scaled), scaled.nbytes,
    )
    phase_matched = np.zeros_like(single)
    continuation(
        _ptr(last_q), _ptr(k), _ptr(v), _ptr(phase_matched),
        heads, 1, tokens, tokens, dim, dim, 0,
        _ptr(scores), scores.nbytes,
        _ptr(columns), columns.nbytes,
        _ptr(scaled), scaled.nbytes,
    )

    # Independent arithmetic reference: convert the operands and normalized
    # probabilities at the FP16 boundaries declared by both provider maps,
    # then accumulate in float64. It does not copy either C reduction schedule.
    q16 = last_q.astype(np.float16).astype(np.float64)
    k16 = k[0].astype(np.float16).astype(np.float64)
    v16 = v[0].astype(np.float16).astype(np.float64)
    logits = (q16 @ k16.T) / math.sqrt(dim)
    probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    reference = probabilities.astype(np.float16).astype(np.float64) @ v16
    # Bound the declared FP32 reductions and FP16 probability storage rather
    # than requiring identical prefill/decode reduction orders. The conservative
    # envelope is still far below the old row's 0.25 output contribution.
    u32 = np.finfo(np.float32).eps / 2
    u16 = np.finfo(np.float16).eps / 2
    gamma_scores = dim * u32 / (1 - dim * u32)
    gamma_values = 256 * u32 / (1 - 256 * u32)
    max_score_error = gamma_scores * np.max(
        (np.abs(q16) @ np.abs(k16).T) / math.sqrt(dim)
    )
    weighted_values = np.max(np.sum(
        np.abs(probabilities[:, :, None] * v16[None, :, :]), axis=1
    ))
    arithmetic_bound = (
        2 * max_score_error * np.max(np.abs(v16))
        + (u16 + gamma_values) * weighted_values
    )
    # Both paths use the prefill reduction contract; minor FP32 operation
    # scheduling differences on a supported ISA remain within its FP32 bound.
    np.testing.assert_allclose(
        phase_matched, batched[:, -1, :], rtol=0,
        atol=2 * gamma_values * weighted_values,
    )
    np.testing.assert_allclose(batched[:, -1, :], reference, rtol=0, atol=arithmetic_bound)
    np.testing.assert_allclose(single, reference, rtol=0, atol=arithmetic_bound)
    truncated_probabilities = probabilities[:, 1:] / np.sum(
        probabilities[:, 1:], axis=1, keepdims=True
    )
    missing_history = truncated_probabilities.astype(np.float16).astype(np.float64) @ v16[1:]
    assert np.max(np.abs(reference - missing_history)) > 0.2
    assert np.max(np.abs(single - missing_history)) > 0.2

    corrupted_v = v.copy()
    corrupted_v[0, 0, 0] = 0.0
    wrong_owner = np.zeros_like(single)
    decode(
        _ptr(last_q), _ptr(k), _ptr(corrupted_v), _ptr(wrong_owner),
        heads, 1, tokens, tokens, dim, dim, 0,
        _ptr(scores), scores.nbytes,
        _ptr(columns), columns.nbytes,
        _ptr(scaled), scaled.nbytes,
    )
    assert np.max(np.abs(wrong_owner - reference)) > 0.2
    wrong_position = np.zeros_like(single)
    decode(
        _ptr(last_q), _ptr(k), _ptr(v), _ptr(wrong_position),
        heads, 1, tokens - 1, tokens, dim, dim, 0,
        _ptr(scores), scores.nbytes,
        _ptr(columns), columns.nbytes,
        _ptr(scaled), scaled.nbytes,
    )
    assert np.max(np.abs(wrong_position - reference)) > 0.01
    lib.ck_set_num_threads(0)


if __name__ == "__main__":
    test_prefill_uses_declared_scratch_capacity_with_compact_kv_stride()
    test_global_attention_does_not_inherit_512_token_window()
    test_parallel_prefill_is_bit_exact_with_multiple_kv_heads()
    test_prefill_decode_boundary_uses_prior_kv_with_phase_numerics()
    print("llama regular attention contracts: PASS")
