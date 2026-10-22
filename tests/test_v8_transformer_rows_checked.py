"""Asset-free nonzero arithmetic and fail-before-write checks for talker providers."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F = ctypes.POINTER(ctypes.c_float)
Z = ctypes.c_size_t


def pointer(array):
    return array.ctypes.data_as(F)


def independent_norm(value, gamma):
    return (value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True)
                            + np.float32(1e-6)) * gamma).astype(np.float32)


def independent_linear(value, weight):
    return (value.astype(np.float64) @ weight.T.astype(np.float64)).astype(np.float32)


class TransformerRowsCheckedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "libchecked_talker.so"
        subprocess.run([
            "cc", "-std=gnu11", "-O2", "-w", "-shared", "-fPIC",
            "-fopenmp", str(ROOT / "src/kernels/transformer_rows_checked.c"),
            str(ROOT / "src/kernels/linear_rows_checked.c"),
            str(ROOT / "src/kernels/rope_kernels.c"),
            str(ROOT / "src/kernels/attention_kernels.c"),
            str(ROOT / "src/kernels/softmax_kernels.c"),
            str(ROOT / "src/kernels/attention_flash_true.c"),
            str(ROOT / "src/ck_threadpool.c"),
            str(ROOT / "src/ckernel_strict.c"),
            "-I", str(ROOT / "include"), "-lm", "-ldl", "-lpthread",
            "-o", str(library),
        ], check=True, capture_output=True, text=True)
        lib = ctypes.CDLL(str(library))
        cls.norm = lib.rmsnorm_rows_checked_f32
        cls.norm.argtypes = [F, Z, Z, F, Z, F, Z, Z, Z, Z, ctypes.c_float]
        cls.norm.restype = ctypes.c_int
        cls.linear = lib.linear_rows_checked_f32
        cls.linear.argtypes = [F, Z, Z, F, Z, Z, F, Z, F, Z, Z, Z, Z, Z]
        cls.linear.restype = ctypes.c_int
        cls.layout = lib.token_head_layout_checked_f32
        cls.layout.argtypes = [F, Z, F, Z, Z, Z, Z, ctypes.c_int]
        cls.layout.restype = ctypes.c_int
        cls.rope = lib.rope_qk_bounded_checked_f32
        cls.rope.argtypes = [F, Z, F, Z, Z, Z, Z, Z, Z, ctypes.c_float]
        cls.rope.restype = ctypes.c_int
        cls.attention = lib.attention_causal_bounded_checked_f32
        cls.attention.argtypes = [F, Z, F, Z, F, Z, F, Z, F, Z, Z, Z, Z, Z]
        cls.attention.restype = ctypes.c_int
        cls.add = lib.add_rows_checked_f32
        cls.add.argtypes = [F, Z, F, Z, F, Z, Z]
        cls.add.restype = ctypes.c_int
        cls.silu_mul = lib.silu_mul_rows_checked_f32
        cls.silu_mul.argtypes = [F, Z, F, Z, F, Z, Z]
        cls.silu_mul.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_rope_rejects_overflow_before_touching_either_buffer(self):
        q = np.linspace(-0.2, 0.2, 2 * 2 * 128, dtype=np.float32)
        k = np.linspace(0.3, -0.3, 1 * 2 * 128, dtype=np.float32)
        original_q, original_k = q.copy(), k.copy()
        for tokens, dim, position, base in (
            (2, 128, 0, 1e-40),
            (2, 128, 2**31 - 1, 10000.0),
            (2, 127, 0, 10000.0),
        ):
            with self.subTest(tokens=tokens, dim=dim, position=position,
                              base=base):
                self.assertNotEqual(self.rope(pointer(q), q.size, pointer(k), k.size,
                                              2, 1, tokens, dim, position, base), 0)
                np.testing.assert_array_equal(q, original_q)
                np.testing.assert_array_equal(k, original_k)

    def test_other_checked_providers_preserve_output_on_rejection(self):
        source = np.ones((2, 4), dtype=np.float32)
        gamma = np.ones(4, dtype=np.float32)
        output = np.full(8, -777.0, dtype=np.float32)
        gamma[3] = np.nan
        self.assertNotEqual(self.norm(pointer(source), source.size, 4,
                                      pointer(gamma), gamma.size, pointer(output),
                                      output.size, 4, 2, 4, 1e-6), 0)
        self.assertTrue(np.all(output == -777.0))
        self.assertNotEqual(self.layout(pointer(source), source.size,
                                        pointer(output), output.size - 1,
                                        2, 2, 2, 0), 0)
        self.assertTrue(np.all(output == -777.0))
        self.assertNotEqual(self.add(pointer(source), source.size - 1,
                                     pointer(source), source.size, pointer(output),
                                     output.size, source.size), 0)
        self.assertTrue(np.all(output == -777.0))
        bad_gate = source.copy()
        bad_gate[1, 2] = np.nan
        self.assertNotEqual(self.silu_mul(pointer(bad_gate), bad_gate.size,
                                          pointer(source), source.size,
                                          pointer(output), output.size,
                                          output.size), 0)
        self.assertTrue(np.all(output == -777.0))
        q = np.ones(2 * 2 * 2, dtype=np.float32)
        k = np.ones(1 * 2 * 2, dtype=np.float32)
        work = np.full_like(q, -555.0)
        self.assertNotEqual(self.attention(pointer(q), q.size, pointer(k), k.size,
                                           pointer(k), k.size, pointer(work),
                                           work.size - 1, pointer(output), output.size,
                                           2, 1, 2, 2), 0)
        self.assertTrue(np.all(output == -777.0))
        self.assertTrue(np.all(work == -555.0))

    def test_nonzero_talker_provider_composition(self):
        rng = np.random.default_rng(20261022)
        tokens, hidden, heads, kv_heads, dim, intermediate = 3, 4, 2, 1, 2, 6
        x = rng.normal(size=(tokens, hidden)).astype(np.float32) * 0.15
        g1 = rng.uniform(0.7, 1.2, hidden).astype(np.float32)
        g2 = rng.uniform(0.7, 1.2, hidden).astype(np.float32)
        gq = rng.uniform(0.7, 1.2, dim).astype(np.float32)
        gk = rng.uniform(0.7, 1.2, dim).astype(np.float32)
        weights = {
            name: (rng.normal(size=(out, width)) * 0.2).astype(np.float32)
            for name, out, width in (
                ("q", heads * dim, hidden), ("k", kv_heads * dim, hidden),
                ("v", kv_heads * dim, hidden), ("o", hidden, heads * dim),
                ("gate", intermediate, hidden), ("up", intermediate, hidden),
                ("down", hidden, intermediate))
        }

        def native_norm(value, gamma):
            rows, width = value.shape
            out = np.empty_like(value)
            self.assertEqual(self.norm(pointer(value), value.size, width,
                                       pointer(gamma), gamma.size, pointer(out),
                                       out.size, width, rows, width, 1e-6), 0)
            return out

        def native_linear(value, weight):
            rows, width = value.shape
            channels = weight.shape[0]
            out = np.empty((rows, channels), dtype=np.float32)
            self.assertEqual(self.linear(pointer(value), value.size, width,
                                         pointer(weight), weight.size, width,
                                         None, 0, pointer(out), out.size,
                                         channels, rows, width, channels), 0)
            return out

        def native_layout(value, nheads, reverse=False):
            out = np.empty_like(value)
            self.assertEqual(self.layout(pointer(value), value.size, pointer(out),
                                         out.size, tokens, nheads, dim, int(reverse)), 0)
            return out

        def native_add(left, right):
            out = np.empty_like(left)
            self.assertEqual(self.add(pointer(left), left.size, pointer(right),
                                      right.size, pointer(out), out.size,
                                      left.size), 0)
            return out

        xn = native_norm(x, g1)
        q = native_norm(native_linear(xn, weights["q"]).reshape(-1, dim), gq)
        k = native_norm(native_linear(xn, weights["k"]).reshape(-1, dim), gk)
        v = native_linear(xn, weights["v"])
        qh = native_layout(q, heads)
        kh = native_layout(k, kv_heads)
        vh = native_layout(v, kv_heads)
        self.assertEqual(self.rope(pointer(qh), qh.size, pointer(kh), kh.size,
                                   heads, kv_heads, tokens, dim, 0, 10000.0), 0)
        work = np.empty_like(qh)
        ah = np.empty_like(qh)
        self.assertEqual(self.attention(pointer(qh), qh.size, pointer(kh), kh.size,
                                        pointer(vh), vh.size, pointer(work), work.size,
                                        pointer(ah), ah.size, heads, kv_heads,
                                        tokens, dim), 0)
        attn = native_linear(native_layout(ah, heads, True).reshape(tokens, -1),
                             weights["o"])
        post = native_add(x, attn)
        yn = native_norm(post, g2)
        gate = native_linear(yn, weights["gate"])
        up = native_linear(yn, weights["up"])
        activated = np.empty_like(gate)
        self.assertEqual(self.silu_mul(pointer(gate), gate.size, pointer(up),
                                       up.size, pointer(activated), activated.size,
                                       gate.size), 0)
        actual = native_add(post, native_linear(activated, weights["down"]))

        reference_norm = independent_norm(x, g1)
        qr = independent_norm(independent_linear(reference_norm,
                                                weights["q"]).reshape(-1, dim),
                              gq).reshape(tokens, heads, dim).transpose(1, 0, 2)
        kr = independent_norm(independent_linear(reference_norm,
                                                weights["k"]).reshape(-1, dim),
                              gk).reshape(tokens, kv_heads, dim).transpose(1, 0, 2)
        vr = independent_linear(reference_norm, weights["v"]).reshape(
            tokens, kv_heads, dim).transpose(1, 0, 2)
        for t in range(tokens):
            for value in (qr, kr):
                source = value[:, t].copy()
                angle = np.float32(t)
                cosine, sine = np.cos(angle), np.sin(angle)
                value[:, t, 0] = source[:, 0] * cosine - source[:, 1] * sine
                value[:, t, 1] = source[:, 1] * cosine + source[:, 0] * sine
        np.testing.assert_allclose(qh.reshape(heads, tokens, dim), qr,
                                   rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(kh.reshape(kv_heads, tokens, dim), kr,
                                   rtol=2e-6, atol=2e-6)
        ar = np.empty((heads, tokens, dim), dtype=np.float32)
        for h in range(heads):
            for t in range(tokens):
                scores = np.array([
                    np.dot(qr[h, t].astype(np.float64),
                           kr[h // (heads // kv_heads), j].astype(np.float64))
                    / math.sqrt(dim) for j in range(t + 1)], dtype=np.float64)
                probability = np.exp(scores - scores.max())
                probability /= probability.sum()
                ar[h, t] = sum(probability[j] * vr[h // (heads // kv_heads), j]
                               for j in range(t + 1))
        np.testing.assert_allclose(ah.reshape(heads, tokens, dim), ar,
                                   rtol=2e-5, atol=2e-5)
        reference_attn = independent_linear(
            ar.transpose(1, 0, 2).reshape(tokens, -1), weights["o"])
        reference_post = x + reference_attn
        reference_yn = independent_norm(reference_post, g2)
        reference_gate = independent_linear(reference_yn, weights["gate"])
        reference_up = independent_linear(reference_yn, weights["up"])
        reference_activated = (reference_gate / (1 + np.exp(-reference_gate))
                               * reference_up)
        expected = reference_post + independent_linear(reference_activated,
                                                       weights["down"])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
