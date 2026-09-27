"""Independent numerical and bounds checks for full token-major attention."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


def numpy_oracle(query, key, value, heads):
    tokens, width = query.shape
    depth = width // heads
    q, k, v = (x.reshape(tokens, heads, depth).transpose(1, 0, 2)
               for x in (query, key, value))
    scores = np.matmul(q, k.transpose(0, 2, 1)) * np.float32(depth ** -0.5)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probability = np.exp(scores)
    probability /= np.sum(probability, axis=-1, keepdims=True)
    return np.matmul(probability, v).transpose(1, 0, 2).copy().reshape(tokens, width)


class FullTokenMajorAttentionOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "attention.so"
        subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", str(ROOT / "src/kernels/attention_full_token_major.c"),
            "-I", str(ROOT / "include"), "-lm", "-o", str(library),
        ], check=True, capture_output=True, text=True)
        cls.function = ctypes.CDLL(str(library)).attention_full_token_major_f32_checked
        cls.function.argtypes = [FLOAT_PTR, ctypes.c_size_t, FLOAT_PTR, ctypes.c_size_t,
                                 FLOAT_PTR, ctypes.c_size_t, FLOAT_PTR, ctypes.c_size_t,
                                 FLOAT_PTR, ctypes.c_size_t, ctypes.c_size_t,
                                 ctypes.c_size_t, ctypes.c_size_t]
        cls.function.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_kernel(self, query, key, value, heads, *, query_elements=None,
                   key_elements=None, value_elements=None, output_elements=None,
                   scratch_bytes=None, tokens=None, head_dim=None):
        query, key, value = (np.ascontiguousarray(x, dtype=np.float32)
                             for x in (query, key, value))
        count = query.shape[0] if tokens is None else tokens
        depth = query.shape[1] // heads if head_dim is None else head_dim
        output = np.full(query.shape, -777.0, dtype=np.float32)
        scratch = np.zeros(max(query.shape[0], 1), dtype=np.float32)
        status = self.function(
            query.ctypes.data_as(FLOAT_PTR), query.size if query_elements is None else query_elements,
            key.ctypes.data_as(FLOAT_PTR), key.size if key_elements is None else key_elements,
            value.ctypes.data_as(FLOAT_PTR), value.size if value_elements is None else value_elements,
            output.ctypes.data_as(FLOAT_PTR), output.size if output_elements is None else output_elements,
            scratch.ctypes.data_as(FLOAT_PTR), scratch.nbytes if scratch_bytes is None else scratch_bytes,
            count, heads, depth)
        return status, output

    def test_small_shapes_and_simd_tails_against_independent_numpy(self):
        rng = np.random.default_rng(2671)
        for tokens, heads, depth in ((1, 1, 1), (3, 2, 5), (5, 3, 7)):
            with self.subTest(tokens=tokens, heads=heads, depth=depth):
                arrays = [rng.normal(size=(tokens, heads * depth)).astype(np.float32)
                          for _ in range(3)]
                status, actual = self.run_kernel(*arrays, heads)
                self.assertEqual(status, 0)
                expected = numpy_oracle(*arrays, heads)
                self.assertTrue(np.isfinite(actual).all())
                error = np.abs(actual - expected)
                self.assertLessEqual(float(error.max()), 2e-6,
                                     f"worst sample {np.unravel_index(error.argmax(), error.shape)}")
                again, repeated = self.run_kernel(*arrays, heads)
                self.assertEqual(again, 0)
                np.testing.assert_array_equal(actual, repeated)

    def test_pinned_kokoro_py_torch_capture(self):
        path = ROOT / "tests/fixtures/tts/kokoro_attention_context_pinned.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), meta["fixture_sha256"])
        fixture = dict(np.load(path))
        status, actual = self.run_kernel(*(fixture[name] for name in
                                           ("query", "key", "value")), 12)
        self.assertEqual(status, 0)
        expected = fixture["context_expected"]
        self.assertTrue(np.isfinite(actual).all())
        error = np.abs(actual - expected)
        worst = np.unravel_index(error.argmax(), error.shape)
        self.assertLessEqual(float(error.max()), 1e-5, f"worst sample {worst}")
        print("TTS_ATTENTION_PRIMITIVE_EVIDENCE " + json.dumps({
            "status": "PASS", "oracle": meta["oracle"],
            "max_abs_error": float(error.max()), "worst_sample": list(map(int, worst)),
            "backward": "NOT_TESTED"}, sort_keys=True))

    def test_live_pytorch_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable; live attention oracle NOT_TESTED")
        torch.set_num_threads(1)
        fixture = dict(np.load(ROOT / "tests/fixtures/tts/kokoro_attention_context_pinned.npz"))
        arrays = [fixture[name] for name in ("query", "key", "value")]
        q, k, v = (torch.from_numpy(x).reshape(36, 12, 64).transpose(0, 1)
                   for x in arrays)
        with torch.no_grad():
            expected = torch.matmul(torch.nn.functional.softmax(
                torch.matmul(q, k.transpose(1, 2)) * 0.125, dim=-1),
                v).transpose(0, 1).contiguous().reshape(36, 768).numpy()
        status, actual = self.run_kernel(*arrays, 12)
        self.assertEqual(status, 0)
        self.assertTrue(np.isfinite(expected).all())
        error = np.abs(actual - expected)
        self.assertLessEqual(float(error.max()), 1e-5,
                             f"PyTorch {torch.__version__}; worst sample "
                             f"{np.unravel_index(error.argmax(), error.shape)}")

    def test_rejections_leave_output_unchanged(self):
        data = np.ones((3, 10), dtype=np.float32)
        bad = data.copy()
        bad[0, 0] = np.nan
        infinite = data.copy()
        infinite[0, 0] = np.inf
        huge = data.copy()
        huge[0, 0] = np.finfo(np.float32).max
        cases = [
            (data, data, data, {"query_elements": 29}),
            (data, data, data, {"key_elements": 29}),
            (data, data, data, {"value_elements": 29}),
            (data, data, data, {"output_elements": 29}),
            (data, data, data, {"scratch_bytes": 2 * 4}),
            (data, data, data, {"tokens": 0}),
            (data, data, data, {"head_dim": 0}),
            (data, data, data, {"head_dim": 6}),
            (data, data, data, {"tokens": ctypes.c_size_t(-1).value}),
            (bad, data, data, {}),
            (data, bad, data, {}),
            (data, data, bad, {}),
            (infinite, data, data, {}),
            (huge, huge, data, {}),
        ]
        for query, key, value, options in cases:
            with self.subTest(options=options, nonfinite=not np.isfinite(query).all()):
                status, output = self.run_kernel(query, key, value, 2, **options)
                self.assertNotEqual(status, 0)
                self.assertTrue(np.all(output == -777.0))


if __name__ == "__main__":
    unittest.main()
