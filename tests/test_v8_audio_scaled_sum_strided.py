"""Independent numerical and bounds checks for a checked strided scaled sum."""
import ctypes
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POINTER = ctypes.POINTER(ctypes.c_float)


class ScaledSumTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.functions = []
        for optimization in ('-O0', '-O3'):
            library = Path(cls.temp.name) / f'scaled_sum_{optimization}.so'
            subprocess.run(['cc', '-std=c11', optimization, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-ffp-contract=off', '-shared', '-fPIC',
                str(ROOT / 'src/kernels/audio_scaled_sum_strided.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            function = ctypes.CDLL(str(library)).audio_scaled_sum_strided_f32_checked
            function.argtypes = [POINTER, ctypes.c_size_t, ctypes.c_size_t,
                POINTER, ctypes.c_size_t, ctypes.c_size_t, POINTER,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_float]
            function.restype = ctypes.c_int
            cls.functions.append(function)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def call(function, left, right, output, columns, scale,
             left_elements=None, right_elements=None, output_elements=None):
        pointer = lambda value: value.ctypes.data_as(POINTER)
        return function(pointer(left), left.size if left_elements is None else left_elements,
            left.shape[1], pointer(right), right.size if right_elements is None else right_elements,
            right.shape[1], pointer(output),
            output.size if output_elements is None else output_elements,
            output.shape[1], left.shape[0], columns, scale)

    def test_numpy_oracle_strides_tails_and_repeated_lengths(self):
        rng = np.random.default_rng(42)
        worst = (0.0, None)
        for rows, columns in ((1, 1), (3, 7), (8, 36), (512, 103)):
            left = np.full((rows, columns + 3), np.nan, np.float32)
            right = np.full((rows, columns + 5), np.nan, np.float32)
            output = np.full((rows, columns + 11), -999., np.float32)
            left[:, :columns] = rng.normal(size=(rows, columns)).astype(np.float32)
            right[:, :columns] = rng.normal(size=(rows, columns)).astype(np.float32)
            scale = np.float32(1 / math.sqrt(2))
            expected = np.multiply(np.add(left[:, :columns],
                right[:, :columns], dtype=np.float32), scale, dtype=np.float32)
            for function in self.functions:
                for valid in (columns, max(1, columns // 2), columns):
                    output.fill(-999.)
                    self.assertEqual(self.call(function, left, right, output,
                        valid, scale), 0)
                    error = np.abs(output[:, :valid] - expected[:, :valid])
                    self.assertTrue(np.isfinite(output[:, :valid]).all())
                    point = np.unravel_index(np.argmax(error), error.shape)
                    if float(error[point]) > worst[0]:
                        worst = (float(error[point]), (rows, valid, *point))
                    self.assertLessEqual(float(error[point]), 1e-7)
                    self.assertTrue(np.all(output[:, valid:] == -999.))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'tts.scaled-sum-strided.native-vs-numpy',
            'name': 'checked strided scaled sum native versus NumPy',
            'provider': 'audio_scaled_sum_strided_f32_checked',
            'dtype': 'fp32', 'direction': 'inference', 'oracle': 'NumPy FP32',
            'status': 'pass', 'max_diff': worst[0], 'worst_index': worst[1],
            'tolerance': 1e-7, 'configuration': 'rows 1/3/8/512; padded strides',
            'reproduction_command': 'python3 -m unittest tests.test_v8_audio_scaled_sum_strided',
        }))

    def test_rejection_preserves_output(self):
        left = np.ones((3, 8), np.float32)
        right = np.ones((3, 9), np.float32)
        output = np.full((3, 10), -999., np.float32)
        for function in self.functions:
            for capacities in ((0, None, None), (None, 0, None),
                               (None, None, 0)):
                self.assertEqual(self.call(function, left, right, output,
                    7, 1., *capacities), -2)
                self.assertTrue(np.all(output == -999.))
            for value in (math.nan, math.inf):
                self.assertEqual(self.call(function, left, right, output,
                    7, value), -1)
                self.assertTrue(np.all(output == -999.))
            left[2, 6] = math.inf
            self.assertEqual(self.call(function, left, right, output, 7, 1.), -3)
            self.assertTrue(np.all(output == -999.))
            left[2, 6] = 1.
            left[0, 0] = np.finfo(np.float32).max
            right[0, 0] = np.finfo(np.float32).max
            self.assertEqual(self.call(function, left, right, output, 7, 1.), -3)
            self.assertTrue(np.all(output == -999.))
            left[0, 0] = right[0, 0] = 1.
            self.assertEqual(self.call(function, left, right, output,
                0, 1.), -2)
            self.assertTrue(np.all(output == -999.))
            self.assertEqual(self.call(function, left, right, output,
                ctypes.c_size_t(-1).value, 1.), -2)
            self.assertTrue(np.all(output == -999.))
            before = left.copy()
            self.assertEqual(self.call(function, left, right, left,
                7, 1.), -1)
            np.testing.assert_array_equal(left, before)
            before = right.copy()
            self.assertEqual(self.call(function, left, right, right,
                7, 1.), -1)
            np.testing.assert_array_equal(right, before)
            left[0, 0] = np.finfo(np.float32).max / 2
            right[0, 0] = 0.
            self.assertEqual(self.call(function, left, right, output,
                7, 4.), -3)
            self.assertTrue(np.all(output == -999.))
            left[0, 0] = right[0, 0] = 1.


if __name__ == '__main__':
    unittest.main()
