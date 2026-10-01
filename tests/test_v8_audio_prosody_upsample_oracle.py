"""Independent pinned PyTorch nearest and depthwise transposed-conv oracles."""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/tts/kokoro_prosody_upsample_torch28.npz'
POINTER = ctypes.POINTER(ctypes.c_float)


class ProsodyUpsampleOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.meta = json.loads(FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != cls.meta['fixture_sha256']:
            raise RuntimeError('prosody upsampling fixture hash mismatch')
        with np.load(FIXTURE) as archive:
            cls.arrays = {name: archive[name].copy() for name in archive.files}
        for name, value in cls.arrays.items():
            if hashlib.sha256(value.tobytes()).hexdigest() != cls.meta['array_sha256'][name]:
                raise RuntimeError(f'prosody upsampling tensor mismatch: {name}')
        cls.functions = []
        for optimization in ('-O0', '-O3'):
            library = Path(cls.temp.name) / f'upsample_{optimization}.so'
            subprocess.run(['cc', '-std=c11', optimization, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC',
                '-I', str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_prosody_upsample_checked.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            loaded = ctypes.CDLL(str(library))
            nearest = loaded.audio_upsample_nearest_channel_major_f32_checked
            nearest.argtypes = [POINTER, ctypes.c_size_t, ctypes.c_size_t,
                POINTER, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
            nearest.restype = ctypes.c_int
            transpose = loaded.audio_conv_transpose1d_depthwise_channel_major_f32_checked
            transpose.argtypes = [POINTER, ctypes.c_size_t, ctypes.c_size_t,
                POINTER, ctypes.c_size_t, POINTER, ctypes.c_size_t,
                POINTER, ctypes.c_size_t, ctypes.c_size_t,
                POINTER, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
            transpose.restype = ctypes.c_int
            cls.functions.append((nearest, transpose))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def ptr(value):
        return value.ctypes.data_as(POINTER)

    def test_pinned_pytorch_oracle_and_padding(self):
        worst = (0., None)
        for index, (channels, frames) in enumerate(self.meta['shapes']):
            source = np.full((channels, frames + 3), np.nan, np.float32)
            source[:, :frames] = self.arrays[f'case{index}_input']
            nearest_output = np.full((channels, 2 * frames + 5), -99., np.float32)
            transposed_output = np.full((channels, 2 * frames + 7), -99., np.float32)
            scratch = np.full((channels * 2 * frames,), -11., np.float32)
            weight = self.arrays[f'case{index}_weight']
            bias = self.arrays[f'case{index}_bias']
            for nearest, transpose in self.functions:
                nearest_output.fill(-99.)
                transposed_output.fill(-99.)
                self.assertEqual(nearest(self.ptr(source), source.size,
                    source.shape[1], self.ptr(nearest_output),
                    nearest_output.size, nearest_output.shape[1],
                    channels, frames, 2, 2 * frames), 0)
                np.testing.assert_array_equal(nearest_output[:, :2 * frames],
                    self.arrays[f'case{index}_nearest'])
                self.assertTrue(np.all(nearest_output[:, 2 * frames:] == -99.))
                self.assertEqual(transpose(self.ptr(source), source.size,
                    source.shape[1], self.ptr(weight), weight.size,
                    self.ptr(bias), bias.size, self.ptr(transposed_output),
                    transposed_output.size, transposed_output.shape[1],
                    self.ptr(scratch), scratch.size, channels, frames,
                    3, 2, 1, 1, 2 * frames), 0)
                actual = transposed_output[:, :2 * frames]
                expected = self.arrays[f'case{index}_transposed']
                self.assertTrue(np.isfinite(actual).all())
                error = np.abs(actual - expected)
                point = np.unravel_index(np.argmax(error), error.shape)
                if float(error[point]) > worst[0]:
                    worst = (float(error[point]), (index, *(int(x) for x in point)))
                self.assertLessEqual(float(error[point]), 2e-6,
                    (index, point, float(error[point])))
                self.assertTrue(np.all(transposed_output[:, 2 * frames:] == -99.))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'tts.prosody-upsampling.native-vs-pytorch28',
            'name': 'nearest and depthwise transposed Conv1D versus pinned PyTorch 2.8',
            'provider': 'audio_prosody_upsample_checked', 'dtype': 'fp32',
            'direction': 'inference', 'oracle': 'committed-pytorch',
            'backend_version': self.meta['oracle_version'], 'status': 'pass',
            'max_diff': worst[0], 'worst_index': worst[1],
            'tolerance': 2e-6,
            'configuration': str(self.meta['shapes']),
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_rejection_preserves_output(self):
        source = np.ones((3, 8), np.float32)
        weight = np.ones((3, 1, 3), np.float32)
        bias = np.zeros(3, np.float32)
        output = np.full((3, 15), -99., np.float32)
        scratch = np.zeros(30, np.float32)
        for nearest, transpose in self.functions:
            base_nearest = [self.ptr(source), source.size, 8,
                self.ptr(output), output.size, 15, 3, 5, 2, 10]
            for index, value in ((1, 0), (4, 0), (6, 0),
                                 (7, 0), (8, 0), (9, 11),
                                 (7, ctypes.c_size_t(-1).value)):
                with self.subTest(nearest_index=index):
                    args = base_nearest.copy(); args[index] = value
                    self.assertNotEqual(nearest(*args), 0)
                    self.assertTrue(np.all(output == -99.))
            source[2, 4] = math.nan
            self.assertEqual(nearest(*base_nearest), -3)
            self.assertTrue(np.all(output == -99.))
            source[2, 4] = 1.
            base_transpose = [self.ptr(source), source.size, 8,
                self.ptr(weight), weight.size, self.ptr(bias), bias.size,
                self.ptr(output), output.size, 15,
                self.ptr(scratch), scratch.size, 3, 5, 3, 2, 1, 1, 10]
            for index, value in ((1, 0), (4, 0), (6, 0), (8, 0),
                                 (11, 0), (13, 0), (14, 0), (15, 0),
                                 (18, 11), (13, ctypes.c_size_t(-1).value)):
                with self.subTest(transpose_index=index):
                    args = base_transpose.copy(); args[index] = value
                    self.assertNotEqual(transpose(*args), 0)
                    self.assertTrue(np.all(output == -99.))
            for target in (source, weight, bias):
                target.flat[0] = math.inf
                self.assertEqual(transpose(*base_transpose), -3)
                self.assertTrue(np.all(output == -99.))
                target.flat[0] = 1. if target is not bias else 0.


if __name__ == '__main__':
    unittest.main()
