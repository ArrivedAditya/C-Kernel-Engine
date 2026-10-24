"""Independent scalar oracle and rejection controls for checked sliding attention."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F = ctypes.POINTER(ctypes.c_float)
Z = ctypes.c_size_t


def pointer(value: np.ndarray):
    return value.ctypes.data_as(F)


def reference(q, k, v, window):
    heads, tokens, dim = q.shape
    kv_heads = k.shape[0]
    output = np.empty_like(q)
    for head in range(heads):
        kv_head = head // (heads // kv_heads)
        for row in range(tokens):
            begin = max(0, row + 1 - window)
            scores = (k[kv_head, begin:row + 1].astype(np.float64)
                      @ q[head, row].astype(np.float64)) / np.sqrt(dim)
            weights = np.exp(scores - np.max(scores))
            weights /= np.sum(weights)
            output[head, row] = weights @ v[kv_head, begin:row + 1]
    return output


class SlidingAttentionOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "libsliding.so"
        subprocess.run([
            "cc", "-std=gnu11", "-O2", "-shared", "-fPIC", "-fopenmp",
            str(ROOT / "src/kernels/transformer_rows_checked.c"),
            str(ROOT / "src/kernels/rope_kernels.c"),
            str(ROOT / "src/kernels/attention_kernels.c"),
            str(ROOT / "src/kernels/softmax_kernels.c"),
            str(ROOT / "src/kernels/attention_flash_true.c"),
            str(ROOT / "src/ck_threadpool.c"),
            str(ROOT / "src/ckernel_strict.c"),
            "-I", str(ROOT / "include"), "-lm", "-ldl", "-lpthread",
            "-o", str(library),
        ], check=True, capture_output=True, text=True)
        cls.call = ctypes.CDLL(str(library)).attention_sliding_bounded_checked_f32
        cls.call.argtypes = [F, Z, F, Z, F, Z, F, Z, F, Z, Z, Z, Z, Z, Z]
        cls.call.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, q, k, v, work, output, window, *, output_elements=None):
        return self.call(
            pointer(q), q.size, pointer(k), k.size, pointer(v), v.size,
            pointer(work), work.size, pointer(output),
            output.size if output_elements is None else output_elements,
            q.shape[0], k.shape[0], q.shape[1], q.shape[2], window)

    def test_lengths_windows_and_grouped_heads(self):
        rng = np.random.default_rng(20261024)
        for tokens, window in ((1, 1), (4, 1), (5, 3), (73, 72), (73, 73)):
            with self.subTest(tokens=tokens, window=window):
                q = rng.normal(size=(4, tokens, 4)).astype(np.float32)
                k = rng.normal(size=(2, tokens, 4)).astype(np.float32)
                v = rng.normal(size=(2, tokens, 4)).astype(np.float32)
                work, output = np.empty_like(q), np.full_like(q, -777)
                self.assertEqual(self.invoke(q, k, v, work, output, window), 0)
                np.testing.assert_allclose(
                    output, reference(q, k, v, window), atol=2e-6, rtol=2e-6)
                if tokens == 1:
                    np.testing.assert_array_equal(output[0, 0], v[0, 0])

    def test_window_excludes_old_key_and_future_is_causal(self):
        tokens = 73
        q = np.ones((1, tokens, 2), dtype=np.float32)
        k = np.zeros_like(q)
        v = np.zeros_like(q)
        v[0, 0] = 1000
        v[0, -1] = 2
        work, output = np.empty_like(q), np.empty_like(q)
        self.assertEqual(self.invoke(q, k, v, work, output, 72), 0)
        np.testing.assert_allclose(output[0, -1], [2 / 72, 2 / 72], atol=2e-6)
        baseline = output.copy()
        v[0, -1] = 100
        self.assertEqual(self.invoke(q, k, v, work, output, 72), 0)
        np.testing.assert_array_equal(output[:, :-1], baseline[:, :-1])
        self.assertGreater(float(np.max(np.abs(output[:, -1] - baseline[:, -1]))),
                           1.0)

    def test_rejection_preserves_output_and_recovery(self):
        q = np.array([[[1e10, 1e10], [1, 0]]], dtype=np.float32)
        k = np.array([[[1e10, 1e10], [0, 1]]], dtype=np.float32)
        v = np.array([[[2, 3], [4, 5]]], dtype=np.float32)
        work, output = np.empty_like(q), np.full_like(q, -123)
        self.assertNotEqual(self.invoke(q, k, v, work, output, 0), 0)
        np.testing.assert_array_equal(output, -123)
        self.assertNotEqual(self.invoke(
            q, k, v, work, output, 2, output_elements=output.size - 1), 0)
        np.testing.assert_array_equal(output, -123)
        q[0, 0, 0] = np.nan
        self.assertNotEqual(self.invoke(q, k, v, work, output, 2), 0)
        np.testing.assert_array_equal(output, -123)
        q[0, 0, 0] = 1e10
        self.assertEqual(self.invoke(q, k, v, work, output, 2), 0)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(output[0, 0], v[0, 0])
        output[:] = -123
        q[0, 0, 0] = np.finfo(np.float32).max
        self.assertNotEqual(self.invoke(q, k, v, work, output, 2), 0)
        np.testing.assert_array_equal(output, -123)


if __name__ == "__main__":
    unittest.main()
