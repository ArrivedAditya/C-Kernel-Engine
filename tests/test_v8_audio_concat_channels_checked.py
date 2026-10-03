"""Independent channel-major concatenation, stride and rejection oracle."""

import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POINTER = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class AudioConcatChannelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.functions = []
        for optimization in ('-O0', '-O3'):
            library = Path(cls.temp.name) / f'concat_{optimization}.so'
            subprocess.run(['cc', '-std=c11', optimization, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC', '-I',
                str(ROOT / 'include'), '-I', str(ROOT / 'src/kernels'),
                str(ROOT / 'src/kernels/audio_concat_channels_checked.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            function = ctypes.CDLL(str(library)).audio_concat_channels_checked_f32
            function.argtypes = [POINTER, SIZE, SIZE, POINTER, SIZE, SIZE,
                                 POINTER, SIZE, SIZE, SIZE, SIZE, SIZE, SIZE]
            function.restype = ctypes.c_int
            cls.functions.append(function)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def invoke(function, left, right, output, frames, left_elements=None,
               right_elements=None, output_elements=None):
        ptr = lambda array: array.ctypes.data_as(POINTER)
        return function(ptr(left), left.size if left_elements is None else left_elements,
            left.shape[1], ptr(right), right.size if right_elements is None else right_elements,
            right.shape[1], ptr(output), output.size if output_elements is None else output_elements,
            output.shape[1], left.shape[0], right.shape[0], output.shape[0], frames)

    def test_numpy_oracle_and_padding(self):
        rng = np.random.default_rng(19)
        for left_channels, right_channels, capacity in (
                (1, 1, 1), (3, 2, 7), (512, 1, 128), (513, 1, 128)):
            left = np.full((left_channels, capacity + 2), np.nan, np.float32)
            right = np.full((right_channels, capacity + 3), np.nan, np.float32)
            output = np.full((left_channels + right_channels, capacity + 5),
                             -999., np.float32)
            left[:, :capacity] = rng.normal(size=(left_channels, capacity))
            right[:, :capacity] = rng.normal(size=(right_channels, capacity))
            for function in self.functions:
                for valid in (1, capacity, max(1, capacity // 2)):
                    output.fill(-999.)
                    self.assertEqual(self.invoke(function, left, right, output,
                                                 valid), 0)
                    expected = np.concatenate((left[:, :valid],
                                               right[:, :valid]), axis=0)
                    np.testing.assert_array_equal(output[:, :valid], expected)
                    self.assertTrue(np.all(output[:, valid:] == -999.))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'tts.audio-concat-channels.native-vs-numpy',
            'name': 'channel-major checked concat native versus NumPy',
            'provider': 'audio_concat_channels_checked_f32', 'dtype': 'fp32',
            'direction': 'inference', 'oracle': 'NumPy concatenate axis=0',
            'status': 'pass', 'max_diff': 0., 'tolerance': 0.,
            'configuration': 'O0/O3, 1/7/128 frames, padded strides, 512+1 and 513+1 channels',
            'reproduction_command': 'python3 -m unittest tests.test_v8_audio_concat_channels_checked'}))

    def test_rejection_preserves_output(self):
        left = np.ones((2, 8), np.float32)
        right = np.ones((1, 9), np.float32)
        output = np.full((3, 10), -999., np.float32)
        for function in self.functions:
            for capacities in ((0, None, None), (None, 0, None),
                               (None, None, 0)):
                self.assertEqual(self.invoke(function, left, right, output,
                    7, *capacities), -2)
                self.assertTrue(np.all(output == -999.))
            self.assertEqual(self.invoke(function, left, right, output, 0), -1)
            self.assertTrue(np.all(output == -999.))
            self.assertEqual(self.invoke(function, left, right, output,
                SIZE(-1).value), -3)
            self.assertTrue(np.all(output == -999.))
            self.assertEqual(self.invoke(function, left, right, output, 10), -3)
            self.assertTrue(np.all(output == -999.))
            ptr = lambda array: array.ctypes.data_as(POINTER)
            self.assertEqual(function(ptr(left), left.size, left.shape[1],
                ptr(right), right.size, right.shape[1], ptr(output),
                output.size, output.shape[1], SIZE(-1).value, 1, 3, 7), -3)
            self.assertTrue(np.all(output == -999.))
            self.assertEqual(function(ptr(left), left.size, left.shape[1],
                ptr(right), right.size, right.shape[1], ptr(output),
                output.size, output.shape[1], 2, 1, 4, 7), -1)
            self.assertTrue(np.all(output == -999.))
            right[0, 3] = np.nan
            self.assertEqual(self.invoke(function, left, right, output, 7), -4)
            self.assertTrue(np.all(output == -999.))
            right[0, 3] = 1.
            before = left.copy()
            self.assertEqual(self.invoke(function, left, right, left, 7), -1)
            np.testing.assert_array_equal(left, before)
            before = right.copy()
            self.assertEqual(self.invoke(function, left, right, right, 7), -1)
            np.testing.assert_array_equal(right, before)


if __name__ == '__main__':
    unittest.main()
