"""Independent NumPy and rejection controls for checked FP32 decode attention."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
F32 = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class CheckedDecodeAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "decode.so"
        subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-shared", "-fPIC",
            str(ROOT / "src/kernels/attention_decode_append_checked.c"),
            "-lm", "-o", str(library),
        ], check=True, capture_output=True, text=True)
        loaded = ctypes.CDLL(str(library))
        cls.seed = loaded.ck_kv_cache_seed_checked_f32
        cls.seed.argtypes = [F32, SIZE, F32, SIZE, F32, SIZE, F32, SIZE,
                             SIZE, SIZE, SIZE, SIZE]
        cls.seed.restype = ctypes.c_int
        cls.decode = loaded.ck_attention_decode_append_checked_f32
        cls.decode.argtypes = [F32, SIZE, F32, SIZE, F32, SIZE,
                               F32, SIZE, F32, SIZE, F32, SIZE,
                               F32, SIZE, F32, SIZE,
                               SIZE, SIZE, SIZE, SIZE, SIZE]
        cls.decode.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def pointer(array):
        return array.ctypes.data_as(F32)

    def invoke_seed(self, k, v, kc, vc, tokens, capacity):
        return self.seed(self.pointer(k), k.size, self.pointer(v), v.size,
                         self.pointer(kc), kc.size, self.pointer(vc), vc.size,
                         k.shape[0], tokens, capacity, k.shape[-1])

    def invoke_decode(self, q, kn, vn, kc, vc, scores, staged, out, position):
        return self.decode(
            self.pointer(q), q.size, self.pointer(kn), kn.size,
            self.pointer(vn), vn.size, self.pointer(kc), kc.size,
            self.pointer(vc), vc.size, self.pointer(scores), scores.size,
            self.pointer(staged), staged.size, self.pointer(out), out.size,
            q.shape[0], kn.shape[0], kc.shape[1], q.shape[1], position)

    def test_numpy_grouped_attention_seed_append_and_reset(self):
        rng = np.random.default_rng(20261025)
        h, kv, capacity, dim, tokens = 4, 2, 5, 4, 2
        k = rng.normal(size=(kv, tokens, dim)).astype(np.float32)
        v = rng.normal(size=(kv, tokens, dim)).astype(np.float32)
        kc = np.full((kv, capacity, dim), -91, np.float32)
        vc = np.full_like(kc, -91)
        self.assertEqual(self.invoke_seed(k, v, kc, vc, tokens, capacity), 0)
        np.testing.assert_array_equal(kc[:, :tokens], k)
        np.testing.assert_array_equal(vc[:, :tokens], v)
        self.assertTrue(np.all(kc[:, tokens:] == -91))
        for position in (2, 3):
            q = rng.normal(size=(h, dim)).astype(np.float32)
            kn = rng.normal(size=(kv, dim)).astype(np.float32)
            vn = rng.normal(size=(kv, dim)).astype(np.float32)
            prefix_k = kc.copy()
            prefix_v = vc.copy()
            scores = np.full((h, capacity), -77, np.float32)
            staged = np.full((h, dim), -77, np.float32)
            out = np.full((h, dim), -77, np.float32)
            expected = np.zeros((h, dim), dtype=np.float64)
            for head in range(h):
                owner = head // (h // kv)
                keys = np.concatenate((prefix_k[owner, :position],
                                       kn[owner:owner + 1]), axis=0).astype(np.float64)
                values = np.concatenate((prefix_v[owner, :position],
                                         vn[owner:owner + 1]), axis=0).astype(np.float64)
                logits = keys @ q[head].astype(np.float64) / np.sqrt(dim)
                weights = np.exp(logits - logits.max())
                expected[head] = (weights / weights.sum()) @ values
            self.assertEqual(self.invoke_decode(q, kn, vn, kc, vc,
                                                scores, staged, out, position), 0)
            np.testing.assert_allclose(out, expected, atol=2e-7, rtol=2e-7)
            np.testing.assert_array_equal(kc[:, :position], prefix_k[:, :position])
            np.testing.assert_array_equal(vc[:, :position], prefix_v[:, :position])
            np.testing.assert_array_equal(kc[:, position], kn)
            np.testing.assert_array_equal(vc[:, position], vn)
            self.assertTrue(np.all(kc[:, position + 1:] == -91))

        self.assertEqual(self.invoke_seed(k, v, kc, vc, tokens, capacity), 0)
        np.testing.assert_array_equal(kc[:, :tokens], k)

    def test_rejection_preserves_output_and_cache(self):
        q = np.ones((2, 2), np.float32)
        kn = np.ones((1, 2), np.float32)
        vn = np.ones((1, 2), np.float32)
        kc = np.full((1, 3, 2), 3, np.float32)
        vc = np.full_like(kc, 4)
        scores = np.zeros((2, 3), np.float32)
        staged = np.zeros((2, 2), np.float32)
        out = np.full((2, 2), -77, np.float32)
        for position in (3, 2**63):
            self.assertNotEqual(self.invoke_decode(q, kn, vn, kc, vc,
                                                   scores, staged, out, position), 0)
            self.assertTrue(np.all(out == -77))
            self.assertTrue(np.all(kc == 3))
            self.assertTrue(np.all(vc == 4))
        kn[0, 0] = np.nan
        self.assertNotEqual(self.invoke_decode(q, kn, vn, kc, vc,
                                               scores, staged, out, 1), 0)
        self.assertTrue(np.all(out == -77))
        self.assertTrue(np.all(kc == 3))
        kn[0, 0] = 1
        self.assertEqual(self.invoke_decode(q, kn, vn, kc, vc,
                                            scores, staged, out, 1), 0)
        self.assertTrue(np.isfinite(out).all())

    def test_large_finite_score_and_seed_rejection(self):
        q = np.array([[1e10]], np.float32)
        kn = np.array([[1e10]], np.float32)
        vn = np.array([[2]], np.float32)
        kc = np.full((1, 1, 1), -91, np.float32)
        vc = kc.copy()
        scores = np.zeros((1, 1), np.float32)
        staged = np.zeros((1, 1), np.float32)
        out = np.full((1, 1), -77, np.float32)
        self.assertEqual(self.invoke_decode(q, kn, vn, kc, vc,
                                            scores, staged, out, 0), 0)
        np.testing.assert_array_equal(out, vn)
        old = kc.copy()
        bad = kn.copy()
        bad[0, 0] = np.inf
        self.assertNotEqual(self.invoke_seed(bad[:, None], vn[:, None],
                                             kc, vc, 1, 1), 0)
        np.testing.assert_array_equal(kc, old)

    def test_finite_score_overflow_short_capacity_and_partial_overlap(self):
        q = np.array([[1e30, 1e30]], np.float32)
        kn = np.array([[1e30, 1e30]], np.float32)
        vn = np.array([[2, 3]], np.float32)
        kc = np.full((1, 2, 2), -91, np.float32)
        vc = kc.copy()
        scores = np.full((1, 2), -77, np.float32)
        staged = np.full((1, 2), -77, np.float32)
        out = np.full((1, 2), -77, np.float32)
        self.assertNotEqual(self.invoke_decode(q, kn, vn, kc, vc,
                                               scores, staged, out, 0), 0)
        self.assertTrue(np.all(out == -77))
        self.assertTrue(np.all(kc == -91))
        self.assertTrue(np.all(vc == -91))
        q[:] = 1
        kn[:] = 1
        self.assertNotEqual(self.decode(
            self.pointer(q), q.size, self.pointer(kn), kn.size,
            self.pointer(vn), vn.size, self.pointer(kc), kc.size,
            self.pointer(vc), vc.size, self.pointer(scores), scores.size,
            self.pointer(staged), staged.size, self.pointer(out), out.size - 1,
            1, 1, 2, 2, 0), 0)
        self.assertTrue(np.all(out == -77))
        self.assertTrue(np.all(kc == -91))
        pool = np.ones(3, np.float32)
        q_overlap = pool[:2].reshape(1, 2)
        out_overlap = pool[1:].reshape(1, 2)
        self.assertNotEqual(self.invoke_decode(q_overlap, kn, vn, kc, vc,
                                               scores, staged, out_overlap, 0), 0)
        np.testing.assert_array_equal(pool, np.ones(3, np.float32))
        self.assertEqual(self.invoke_decode(q, kn, vn, kc, vc,
                                            scores, staged, out, 0), 0)
        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
