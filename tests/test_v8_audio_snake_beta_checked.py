"""Independent checked SnakeBeta arithmetic and adversarial buffer contracts."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOAT = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class AudioSnakeBetaCheckedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.functions = []
        for flag in ("-O0", "-O3"):
            library = Path(cls.temporary.name) / f"snake_beta_{flag}.so"
            subprocess.run([
                "cc", "-std=c11", flag, "-Wall", "-Wextra", "-Werror",
                "-pedantic", "-shared", "-fPIC", "-I", str(ROOT / "include"),
                str(ROOT / "src/kernels/audio_snake_checked.c"),
                "-lm", "-o", str(library),
            ], check=True, capture_output=True)
            fn = ctypes.CDLL(str(library)).audio_snake_beta_strided_f32_checked
            fn.argtypes = [FLOAT, SIZE, SIZE, FLOAT, SIZE, FLOAT, SIZE,
                           FLOAT, SIZE, SIZE, SIZE, SIZE]
            fn.restype = ctypes.c_int
            cls.functions.append(fn)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def pointer(value):
        return value.ctypes.data_as(FLOAT)

    def call(self, fn, source, alpha, beta, output, frames,
             input_elements=None, alpha_elements=None,
             beta_elements=None, output_elements=None):
        return fn(self.pointer(source),
                  source.size if input_elements is None else input_elements,
                  source.shape[1], self.pointer(alpha),
                  alpha.size if alpha_elements is None else alpha_elements,
                  self.pointer(beta),
                  beta.size if beta_elements is None else beta_elements,
                  self.pointer(output),
                  output.size if output_elements is None else output_elements,
                  output.shape[1], source.shape[0], frames)

    def test_independent_numpy_oracle_tail_padding_and_repeat(self):
        rng = np.random.default_rng(20261024)
        for channels, frames in ((1, 1), (3, 7), (5, 19)):
            source = np.full((channels, frames + 3), np.nan, np.float32)
            source[:, :frames] = rng.normal(size=(channels, frames)).astype(np.float32)
            alpha = rng.uniform(-1, 1, channels).astype(np.float32)
            beta = rng.uniform(-1, 1, channels).astype(np.float32)
            frequency = np.exp(alpha)[:, None]
            magnitude = np.exp(beta)[:, None]
            x = source[:, :frames]
            expected = x + (1 / (magnitude + np.float32(1e-9))) * np.sin(
                x * frequency) ** 2
            results = []
            for fn in self.functions:
                output = np.full((channels, frames + 5), -91, np.float32)
                self.assertEqual(self.call(fn, source, alpha, beta, output, frames), 0)
                np.testing.assert_allclose(output[:, :frames], expected,
                                           atol=3e-6, rtol=3e-6)
                np.testing.assert_array_equal(output[:, frames:], -91)
                results.append(output.copy())
            np.testing.assert_array_equal(results[0], results[1])

    def test_rejection_preserves_output_and_recovery(self):
        source = np.full((3, 9), np.nan, np.float32)
        source[:, :7] = np.arange(21, dtype=np.float32).reshape(3, 7) / 10
        alpha = np.array([.2, -.1, .4], np.float32)
        beta = np.array([-.3, .5, .1], np.float32)
        for fn in self.functions:
            output = np.full((3, 10), -91, np.float32)
            self.assertEqual(self.call(fn, source, alpha, beta, output, 7), 0)
            baseline = output.copy()
            for failure in ("input_short", "alpha_short", "beta_short",
                            "output_short", "nan_input", "nan_alpha",
                            "nan_beta", "exp_overflow", "product_overflow",
                            "output_overflow"):
                data, a, b = source.copy(), alpha.copy(), beta.copy()
                capacities = {}
                if failure == "input_short": capacities["input_elements"] = 24
                if failure == "alpha_short": capacities["alpha_elements"] = 2
                if failure == "beta_short": capacities["beta_elements"] = 2
                if failure == "output_short": capacities["output_elements"] = 26
                if failure == "nan_input": data[-1, 6] = np.nan
                if failure == "nan_alpha": a[-1] = np.nan
                if failure == "nan_beta": b[-1] = np.nan
                if failure == "exp_overflow": a[-1] = 100
                if failure == "product_overflow":
                    data[-1, 6] = np.float32(3e38)
                    a[-1] = 2
                if failure == "output_overflow":
                    data[-1, 6] = np.float32(3e38)
                    b[-1] = -80
                output[:] = -91
                self.assertNotEqual(self.call(fn, data, a, b, output, 7,
                                              **capacities), 0, failure)
                np.testing.assert_array_equal(output, -91)
            self.assertNotEqual(self.call(fn, source, alpha, beta,
                                          source, 7), 0)
            output[:] = -91
            self.assertEqual(self.call(fn, source, alpha, beta, output, 7), 0)
            np.testing.assert_array_equal(output, baseline)


if __name__ == "__main__":
    unittest.main()
