#!/usr/bin/env python3
"""Direct parity tests for Gemma4 assistant / q-only shared-KV kernels."""

import ctypes
import math
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libckernel_engine.so"
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)


def load_lib():
    if not LIB.exists():
        raise unittest.SkipTest(f"missing shared library: {LIB}")
    return ctypes.CDLL(str(LIB))


def as_f32(values):
    return np.ascontiguousarray(values, dtype=np.float32)


def ptr(array):
    return array.ctypes.data_as(F32P)


def iptr(array):
    return array.ctypes.data_as(I32P)


def rmsnorm_rows(x, gamma, eps):
    y = np.empty_like(x)
    flat = x.reshape(-1, x.shape[-1])
    out = y.reshape(-1, x.shape[-1])
    for i, row in enumerate(flat):
        rstd = 1.0 / math.sqrt(float(np.mean(row * row)) + eps)
        out[i] = row * gamma * rstd
    return y


def causal_shared_q_attention(q):
    """Reference for Gemma4 shared-KV prefill: q is also k/v, scale=1.0."""
    heads, tokens, dim = q.shape
    out = np.zeros_like(q)
    for h in range(heads):
        for t in range(tokens):
            scores = np.array(
                [float(np.dot(q[h, t], q[h, j])) for j in range(t + 1)],
                dtype=np.float64,
            )
            scores -= float(np.max(scores))
            weights = np.exp(scores)
            weights /= float(np.sum(weights))
            acc = np.zeros(dim, dtype=np.float64)
            for j, w in enumerate(weights):
                acc += w * q[h, j].astype(np.float64)
            out[h, t] = acc.astype(np.float32)
    return out


def decode_attention(q_token, k_cache, v_cache, kv_tokens):
    """Reference for Gemma4 decode attention: head-major cache, scale=1.0."""
    heads, dim = q_token.shape
    out = np.zeros_like(q_token)
    for h in range(heads):
        scores = np.array(
            [float(np.dot(q_token[h], k_cache[h, j])) for j in range(kv_tokens)],
            dtype=np.float64,
        )
        scores -= float(np.max(scores))
        weights = np.exp(scores)
        weights /= float(np.sum(weights))
        acc = np.zeros(dim, dtype=np.float64)
        for j, w in enumerate(weights):
            acc += w * v_cache[h, j].astype(np.float64)
        out[h] = acc.astype(np.float32)
    return out


class TestGemma4AssistantKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_lib()
        cls.lib.assistant_layer_scale_forward.argtypes = [
            F32P,
            F32P,
            ctypes.c_int,
            ctypes.c_int,
        ]
        cls.lib.q_norm_forward.argtypes = [
            F32P,
            F32P,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        cls.lib.kv_cache_store_shared_q.argtypes = [
            F32P,
            F32P,
            F32P,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        cls.lib.attention_forward_causal_head_major_shared_kv_gemma4.argtypes = [
            F32P,
            F32P,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        cls.lib.attention_forward_decode_head_major_shared_kv_gemma4.argtypes = [
            F32P,
            F32P,
            F32P,
            F32P,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        cls.lib.speculative_verify_greedy_f32.argtypes = [
            F32P,
            ctypes.c_int,
            ctypes.c_int,
            I32P,
            I32P,
        ]
        cls.lib.speculative_commit_one_i32.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            I32P,
            I32P,
            ctypes.c_int,
            I32P,
            I32P,
            I32P,
            I32P,
        ]

    def test_assistant_layer_scale_forward_matches_numpy(self):
        hidden = as_f32(np.arange(15, dtype=np.float32).reshape(3, 5) - 4.0)
        scale = as_f32([0.375])
        expected = hidden * scale[0]
        actual = hidden.copy()

        self.lib.assistant_layer_scale_forward(ptr(actual), ptr(scale), 3, 5)

        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)

    def test_q_norm_forward_matches_rmsnorm_rows(self):
        rng = np.random.default_rng(7)
        q = as_f32(rng.normal(size=(3, 4, 5)))
        gamma = as_f32(rng.normal(size=(5,)))
        eps = 1e-6
        expected = rmsnorm_rows(q, gamma, eps)
        actual = q.copy()

        self.lib.q_norm_forward(ptr(actual), ptr(gamma), 3, 4, 5, ctypes.c_float(eps))

        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_kv_cache_store_shared_q_copies_q_to_k_and_v(self):
        q = as_f32(np.arange(2 * 4, dtype=np.float32).reshape(2, 4) / 10.0)
        k_cache = as_f32(np.full((2, 3, 4), -9.0, dtype=np.float32))
        v_cache = as_f32(np.full((2, 3, 4), -8.0, dtype=np.float32))

        self.lib.kv_cache_store_shared_q(
            ptr(k_cache), ptr(v_cache), ptr(q), 0, 1, 2, 4, 3
        )

        np.testing.assert_allclose(k_cache[:, 1, :], q, rtol=0, atol=0)
        np.testing.assert_allclose(v_cache[:, 1, :], q, rtol=0, atol=0)
        np.testing.assert_allclose(k_cache[:, 0, :], -9.0, rtol=0, atol=0)
        np.testing.assert_allclose(v_cache[:, 2, :], -8.0, rtol=0, atol=0)

    def test_causal_shared_kv_prefill_matches_reference(self):
        rng = np.random.default_rng(11)
        q = as_f32(rng.normal(scale=0.15, size=(2, 4, 6)))
        expected = causal_shared_q_attention(q)
        actual = np.zeros_like(q)

        self.lib.attention_forward_causal_head_major_shared_kv_gemma4(
            ptr(q), ptr(actual), 2, 4, 6, 6, 4
        )

        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_decode_shared_kv_matches_reference(self):
        rng = np.random.default_rng(13)
        q_token = as_f32(rng.normal(scale=0.12, size=(2, 5)))
        k_cache = as_f32(rng.normal(scale=0.10, size=(2, 4, 5)))
        v_cache = as_f32(rng.normal(scale=0.10, size=(2, 4, 5)))
        expected = decode_attention(q_token, k_cache, v_cache, kv_tokens=3)
        actual = np.zeros_like(q_token)

        self.lib.attention_forward_decode_head_major_shared_kv_gemma4(
            ptr(q_token), ptr(k_cache), ptr(v_cache), ptr(actual), 2, 3, 4, 5, 5
        )

        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_speculative_verify_accepts_matching_draft_token(self):
        logits = as_f32([-0.5, 0.25, 2.0, 1.25])
        accepted = np.array([-1], dtype=np.int32)
        verified = np.array([-1], dtype=np.int32)

        self.lib.speculative_verify_greedy_f32(ptr(logits), 4, 2, iptr(accepted), iptr(verified))

        self.assertEqual(int(accepted[0]), 1)
        self.assertEqual(int(verified[0]), 2)

    def test_speculative_verify_rejects_and_uses_target_argmax(self):
        logits = as_f32([-0.5, 0.25, 2.0, 1.25])
        accepted = np.array([-1], dtype=np.int32)
        verified = np.array([-1], dtype=np.int32)

        self.lib.speculative_verify_greedy_f32(ptr(logits), 4, 1, iptr(accepted), iptr(verified))

        self.assertEqual(int(accepted[0]), 0)
        self.assertEqual(int(verified[0]), 2)

    def test_speculative_commit_updates_buffer_positions_and_counters(self):
        token_buffer = np.full(4, -99, dtype=np.int32)
        token_count = np.array([1], dtype=np.int32)
        target_position = np.array([7], dtype=np.int32)
        draft_position = np.array([3], dtype=np.int32)
        accepted_count = np.array([2], dtype=np.int32)
        rejected_count = np.array([5], dtype=np.int32)

        self.lib.speculative_commit_one_i32(
            1,
            42,
            iptr(token_buffer),
            iptr(token_count),
            4,
            iptr(target_position),
            iptr(draft_position),
            iptr(accepted_count),
            iptr(rejected_count),
        )

        self.assertEqual(token_buffer.tolist(), [-99, 42, -99, -99])
        self.assertEqual(int(token_count[0]), 2)
        self.assertEqual(int(target_position[0]), 8)
        self.assertEqual(int(draft_position[0]), 8)
        self.assertEqual(int(accepted_count[0]), 3)
        self.assertEqual(int(rejected_count[0]), 5)

    def test_speculative_commit_reject_counter_and_full_buffer_guard(self):
        token_buffer = np.array([7, 8], dtype=np.int32)
        token_count = np.array([2], dtype=np.int32)
        target_position = np.array([4], dtype=np.int32)
        draft_position = np.array([4], dtype=np.int32)
        accepted_count = np.array([0], dtype=np.int32)
        rejected_count = np.array([0], dtype=np.int32)

        self.lib.speculative_commit_one_i32(
            0,
            11,
            iptr(token_buffer),
            iptr(token_count),
            2,
            iptr(target_position),
            iptr(draft_position),
            iptr(accepted_count),
            iptr(rejected_count),
        )

        self.assertEqual(token_buffer.tolist(), [7, 8])
        self.assertEqual(int(token_count[0]), 2)
        self.assertEqual(int(target_position[0]), 5)
        self.assertEqual(int(draft_position[0]), 5)
        self.assertEqual(int(accepted_count[0]), 0)
        self.assertEqual(int(rejected_count[0]), 1)


if __name__ == "__main__":
    unittest.main()
