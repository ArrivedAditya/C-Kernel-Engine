"""Checked strided tanh against independent NumPy and optional live PyTorch."""

import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FLOAT = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class CheckedTanhOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.functions = []
        for flag in ('-O0', '-O3'):
            library = Path(cls.temporary.name) / f'tanh_{flag}.so'
            subprocess.run(['cc', '-std=c11', flag, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC', '-I', str(ROOT / 'include'),
                str(ROOT / 'src/kernels/strided_unary_checked.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            function = ctypes.CDLL(str(library)).tanh_strided_f32_checked
            function.argtypes = [FLOAT, SIZE, SIZE, FLOAT, SIZE, SIZE, SIZE, SIZE]
            function.restype = ctypes.c_int
            cls.functions.append(function)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def call(function, source, result, rows, columns, in_stride, out_stride,
             input_capacity=None, output_capacity=None):
        return function(source.ctypes.data_as(FLOAT),
            source.size if input_capacity is None else input_capacity, in_stride,
            result.ctypes.data_as(FLOAT),
            result.size if output_capacity is None else output_capacity,
            out_stride, rows, columns)

    def test_numpy_and_live_pytorch_with_padding_and_tails(self):
        rows, columns, in_stride, out_stride = 7, 9, 12, 13
        source = np.full((rows, in_stride), np.nan, np.float32)
        values = np.random.default_rng(914).uniform(-20, 20,
            size=(rows, columns)).astype(np.float32)
        values[0, :5] = [-100., -1., 0., 1., 100.]
        source[:, :columns] = values
        expected = np.tanh(values)
        outputs = []
        for variant, function in enumerate(self.functions):
            output = np.full((rows, out_stride), -91., np.float32)
            self.assertEqual(self.call(function, source, output,
                rows, columns, in_stride, out_stride), 0)
            self.assertTrue(np.isfinite(output[:, :columns]).all())
            self.assertTrue(np.all(output[:, columns:] == -91.))
            error = np.abs(output[:, :columns] - expected)
            index = np.unravel_index(np.argmax(error), error.shape)
            self.assertLessEqual(float(error[index]), 2e-7,
                (variant, index, float(output[index]), float(expected[index])))
            outputs.append(output.copy())
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'audio.tanh-strided.numpy.{("o0", "o3")[variant]}',
                'name': 'checked strided tanh versus NumPy',
                'provider': 'tanh_strided_f32_checked', 'status': 'pass',
                'oracle': 'NumPy tanh', 'max_diff': float(error[index]),
                'tolerance': 2e-7, 'worst_index': list(map(int,index)),
                'configuration': 'rows7 columns9 input_stride12 output_stride13'}))
        np.testing.assert_array_equal(outputs[0], outputs[1])
        try:
            import torch
        except ImportError:
            self.skipTest('live PyTorch unavailable; NumPy comparison passed')
        torch_expected = torch.tanh(torch.from_numpy(values)).numpy()
        self.assertLessEqual(float(np.max(np.abs(outputs[1][:, :columns] -
            torch_expected))), 3e-7)

    def test_rejections_preserve_output_and_recovery(self):
        rows, columns, stride = 2, 3, 5
        source = np.zeros((rows, stride), np.float32)
        output = np.full((rows, stride), -91., np.float32)
        source[:, :columns] = [[-1, 0, 1], [2, -2, 3]]
        for function in self.functions:
            output.fill(-91.)
            for input_capacity, output_capacity in ((7, output.size),
                                                    (source.size, 7)):
                self.assertNotEqual(self.call(function, source, output,
                    rows, columns, stride, stride, input_capacity,
                    output_capacity), 0)
                self.assertTrue(np.all(output == -91.))
            source[1, 0] = np.inf
            self.assertNotEqual(self.call(function, source, output,
                rows, columns, stride, stride), 0)
            self.assertTrue(np.all(output == -91.))
            source[1, 0] = 2.
            self.assertNotEqual(self.call(function, source, source,
                rows, columns, stride, stride), 0)
            self.assertEqual(self.call(function, source, output,
                rows, columns, stride, stride), 0)
            np.testing.assert_allclose(output[:, :columns],
                np.tanh(source[:, :columns]), rtol=0, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
