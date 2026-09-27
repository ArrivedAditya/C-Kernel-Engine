"""Independent pinned PyTorch ALBERT embedding boundary and safety tests."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tts/bert_embedding_pinned.npz"
FLOAT_P = ctypes.POINTER(ctypes.c_float)
INT_P = ctypes.POINTER(ctypes.c_int32)
SIZE = ctypes.c_size_t


class BertEmbeddingOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "libbert_embedding.so"
        subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", "-I", str(ROOT / "include"),
            str(ROOT / "src/kernels/embedding_three_table_layer_norm.c"),
            "-lm", "-o", str(library),
        ], check=True)
        cls.fn = ctypes.CDLL(str(library)).embedding_three_table_layer_norm_f32
        cls.fn.argtypes = [INT_P, INT_P, SIZE, FLOAT_P, SIZE, SIZE,
                           FLOAT_P, SIZE, SIZE, FLOAT_P, SIZE, SIZE,
                           FLOAT_P, FLOAT_P, SIZE, FLOAT_P, SIZE, SIZE,
                           SIZE, SIZE, SIZE, ctypes.c_float]
        cls.fn.restype = ctypes.c_int
        cls.manifest = json.loads(FIXTURE.with_suffix(".json").read_text())
        cls.data = dict(np.load(FIXTURE))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, ids=None, type_ids=None, output_capacity=None,
               weight_stride=None, output_stride=None, epsilon=None):
        data = self.data
        ids = np.ascontiguousarray(data["ids"] if ids is None else ids, dtype=np.int32)
        type_ids = np.ascontiguousarray(data["type_ids"] if type_ids is None else type_ids,
                                        dtype=np.int32)
        tokens = len(ids)
        channels = data["word"].shape[1]
        weight_stride = channels if weight_stride is None else weight_stride
        output_stride = channels if output_stride is None else output_stride
        capacity = tokens * output_stride if output_capacity is None else output_capacity
        output = np.full(capacity, -777.0, dtype=np.float32)
        ptr = lambda a, ty: a.ctypes.data_as(ty)
        result = self.fn(
            ptr(ids, INT_P), ptr(type_ids, INT_P), len(ids),
            ptr(data["word"], FLOAT_P), data["word"].size, data["word"].shape[0],
            ptr(data["position"], FLOAT_P), data["position"].size, data["position"].shape[0],
            ptr(data["token_type"], FLOAT_P), data["token_type"].size,
            data["token_type"].shape[0],
            ptr(data["gamma"], FLOAT_P), ptr(data["beta"], FLOAT_P), channels,
            ptr(output, FLOAT_P), capacity, tokens, channels, weight_stride,
            output_stride, self.manifest["epsilon"] if epsilon is None else epsilon)
        return result, output

    def test_pinned_kokoro_embedding_boundary(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                         self.manifest["fixture_sha256"])
        self.assertEqual(self.manifest["input_ids"], self.data["ids"].tolist())
        for _ in range(2):
            status, output = self.invoke(output_stride=136)
            self.assertEqual(status, 0)
            actual = output.reshape(-1, 136)[:, :128]
            expected = self.data["expected"]
            self.assertTrue(np.isfinite(actual).all())
            error = np.abs(actual - expected)
            worst = np.unravel_index(np.argmax(error), error.shape)
            self.assertLessEqual(float(error[worst]), 2e-5,
                                 f"worst token/channel {worst}: {actual[worst]} vs {expected[worst]}")
            self.assertTrue(np.all(output.reshape(-1, 136)[:, 128:] == -777.0))

    def test_invalid_ids_and_capacity_preserve_output(self):
        bad_ids = self.data["ids"].copy()
        bad_ids[-1] = self.data["word"].shape[0]
        cases = [dict(ids=bad_ids), dict(output_capacity=len(bad_ids) * 128 - 1),
                 dict(type_ids=np.full(len(bad_ids), -1, dtype=np.int32)),
                 dict(weight_stride=127), dict(epsilon=0.0)]
        for case in cases:
            with self.subTest(case=list(case)):
                status, output = self.invoke(**case)
                self.assertNotEqual(status, 0)
                self.assertTrue(np.all(output == -777.0))

    def test_small_tail_type_and_physical_strides(self):
        ids = np.array([2, 0, 1], dtype=np.int32)
        types = np.array([1, 0, 1], dtype=np.int32)
        channels, stride, out_stride = 5, 8, 7
        word = np.zeros((3, stride), dtype=np.float32)
        position = np.zeros((3, stride), dtype=np.float32)
        token_type = np.zeros((2, stride), dtype=np.float32)
        word[:, :channels] = np.arange(15, dtype=np.float32).reshape(3, 5) / 7
        position[:, :channels] = np.arange(15, dtype=np.float32).reshape(3, 5) / 13
        token_type[1, :channels] = np.array([.2, -.1, .3, .4, -.5], dtype=np.float32)
        gamma = np.array([1., .8, 1.1, -.3, .7], dtype=np.float32)
        beta = np.array([.1, -.2, .3, .4, -.5], dtype=np.float32)
        output = np.full((3, out_stride), -777., dtype=np.float32)
        ptr = lambda a, ty: a.ctypes.data_as(ty)
        status = self.fn(ptr(ids, INT_P), ptr(types, INT_P), 3,
                         ptr(word, FLOAT_P), word.size, 3,
                         ptr(position, FLOAT_P), position.size, 3,
                         ptr(token_type, FLOAT_P), token_type.size, 2,
                         ptr(gamma, FLOAT_P), ptr(beta, FLOAT_P), 5,
                         ptr(output, FLOAT_P), output.size, 3, 5, stride,
                         out_stride, ctypes.c_float(1e-5))
        self.assertEqual(status, 0)
        summed = word[ids, :5] + token_type[types, :5]
        summed = summed + position[:, :5]
        expected = ((summed - summed.mean(axis=1, keepdims=True)) /
                    np.sqrt(summed.var(axis=1, keepdims=True) + 1e-5))
        expected = expected * gamma + beta
        np.testing.assert_allclose(output[:, :5], expected, atol=1e-6, rtol=0)
        self.assertTrue(np.all(output[:, 5:] == -777.))

    def test_nonfinite_weights_and_hostile_stride_preserve_output(self):
        data = self.data
        original = data["word"].copy()
        try:
            data["word"][int(data["ids"][0]), 0] = np.nan
            status, output = self.invoke()
            self.assertNotEqual(status, 0)
            self.assertTrue(np.all(output == -777.0))
        finally:
            data["word"][:] = original
        ids = data["ids"]
        types = data["type_ids"]
        output = np.full(1, -777., dtype=np.float32)
        ptr = lambda a, ty: a.ctypes.data_as(ty)
        status = self.fn(
            ptr(ids, INT_P), ptr(types, INT_P), len(ids),
            ptr(data["word"], FLOAT_P), data["word"].size, data["word"].shape[0],
            ptr(data["position"], FLOAT_P), data["position"].size,
            data["position"].shape[0],
            ptr(data["token_type"], FLOAT_P), data["token_type"].size,
            data["token_type"].shape[0],
            ptr(data["gamma"], FLOAT_P), ptr(data["beta"], FLOAT_P), 128,
            ptr(output, FLOAT_P), 1, len(ids), 128, 128, SIZE(-1).value,
            self.manifest["epsilon"])
        self.assertNotEqual(status, 0)
        self.assertTrue(np.all(output == -777.0))


if __name__ == "__main__":
    unittest.main()
