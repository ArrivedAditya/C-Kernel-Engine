"""Checked dense transposed Conv1D versus independent PyTorch fixtures."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/tts/audio_dense_deconv_torch_reference.npz'
POINTER = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class AudioDenseDeconvOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.meta = json.loads(FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != cls.meta['fixture_sha256']:
            raise RuntimeError('dense transposed-convolution fixture hash mismatch')
        with np.load(FIXTURE) as archive:
            cls.arrays = {name: archive[name].copy() for name in archive.files}
        for name, value in cls.arrays.items():
            if hashlib.sha256(value.tobytes()).hexdigest() != cls.meta['array_sha256'][name]:
                raise RuntimeError(f'dense transposed-convolution tensor mismatch: {name}')
        cls.functions = []
        for optimization in ('-O0', '-O3'):
            library = Path(cls.temp.name) / f'dense_deconv_{optimization}.so'
            subprocess.run(['cc', '-std=c11', optimization, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC',
                '-I', str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_conv_transpose_dense_checked.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            loaded = ctypes.CDLL(str(library))
            run = loaded.audio_conv_transpose1d_dense_channel_major_f32_checked
            run.argtypes = [POINTER, SIZE, SIZE, POINTER, SIZE,
                            POINTER, SIZE, POINTER, SIZE, SIZE,
                            POINTER] + [SIZE] * 9
            run.restype = ctypes.c_int
            workspace = loaded.audio_conv_transpose1d_dense_f32_workspace
            workspace.argtypes = [SIZE, SIZE, ctypes.POINTER(SIZE)]
            workspace.restype = ctypes.c_int
            cls.functions.append((run, workspace))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def ptr(value):
        return value.ctypes.data_as(POINTER)

    def call(self, function, index):
        inputs, outputs, frames, kernel, stride, padding, output_padding = \
            self.meta['cases'][index]
        expected = self.arrays[f'case{index}_output']
        out_frames = expected.shape[1]
        source = np.full((inputs, frames + 3), np.nan, np.float32)
        source[:, :frames] = self.arrays[f'case{index}_input']
        output = np.full((outputs, out_frames + 4), -91., np.float32)
        weight = self.arrays[f'case{index}_weight']
        bias = self.arrays[f'case{index}_bias']
        scratch = np.full(outputs * out_frames, -37., np.float32)
        args = [self.ptr(source), source.size, source.shape[1],
                self.ptr(weight), weight.size, self.ptr(bias), bias.size,
                self.ptr(output), output.size, output.shape[1],
                self.ptr(scratch), scratch.size,
                inputs, outputs, frames, kernel, stride, padding,
                output_padding, out_frames]
        return function(*args), output, expected, args, (source, weight, bias, scratch)

    def test_committed_pytorch_oracle_strides_tails_and_threads(self):
        worst = (0., None)
        for index, geometry in enumerate(self.meta['cases']):
            outputs = []
            for function, workspace in self.functions:
                required = SIZE(0)
                self.assertEqual(workspace(geometry[1],
                    self.arrays[f'case{index}_output'].shape[1],
                    ctypes.byref(required)), 0)
                status, actual, expected, _, _ = self.call(function, index)
                self.assertEqual(status, 0)
                self.assertEqual(required.value, expected.size)
                self.assertTrue(np.isfinite(actual[:, :expected.shape[1]]).all())
                delta = np.abs(actual[:, :expected.shape[1]] - expected)
                point = np.unravel_index(np.argmax(delta), delta.shape)
                if float(delta[point]) > worst[0]:
                    worst = (float(delta[point]), [index, *map(int, point)])
                self.assertLessEqual(float(delta[point]), 3e-6,
                                     (index, point, float(delta[point])))
                self.assertTrue(np.all(actual[:, expected.shape[1]:] == -91.))
                outputs.append(actual.copy())
            np.testing.assert_array_equal(outputs[0], outputs[1])
        function = self.functions[1][0]
        with ThreadPoolExecutor(max_workers=3) as pool:
            parallel = list(pool.map(lambda _: self.call(function, 2)[1], range(3)))
        for actual in parallel[1:]:
            np.testing.assert_array_equal(actual, parallel[0])
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'tts.dense-deconv.native-vs-pytorch-committed',
            'name': 'checked dense ConvTranspose1D versus committed PyTorch',
            'provider': 'audio_conv_transpose1d_dense_channel_major_f32_checked',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'committed-pytorch',
            'backend_version': self.meta['oracle_version'],
            'status': 'pass', 'max_diff': worst[0],
            'worst_index': worst[1], 'tolerance': 3e-6,
            'configuration': str(self.meta['cases']),
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_rejection_preserves_output(self):
        for function, workspace in self.functions:
            status, output, _, args, buffers = self.call(function, 1)
            self.assertEqual(status, 0)
            for index, value in ((1, 0), (4, 0), (6, 0), (8, 0), (11, 0),
                                 (12, 0), (13, 0), (14, 0), (15, 0),
                                 (16, 0), (18, 2), (19, 10),
                                 (2, 1), (9, 1),
                                 (14, SIZE(-1).value)):
                with self.subTest(index=index, value=value):
                    output.fill(-91.)
                    bad = args.copy(); bad[index] = value
                    self.assertNotEqual(function(*bad), 0)
                    self.assertTrue(np.all(output == -91.))
            source, weight, bias, _ = buffers
            for candidate in (source, weight, bias):
                original = float(candidate.flat[0])
                candidate.flat[0] = math.inf
                output.fill(-91.)
                self.assertEqual(function(*args), -3)
                self.assertTrue(np.all(output == -91.))
                candidate.flat[0] = original
            output.fill(-91.)
            alias = args.copy(); alias[10] = self.ptr(output)
            self.assertNotEqual(function(*alias), 0)
            self.assertTrue(np.all(output == -91.))
            required = SIZE(0)
            self.assertEqual(workspace(0, 3, ctypes.byref(required)), -4)
            self.assertEqual(workspace(SIZE(-1).value, 3,
                                       ctypes.byref(required)), -4)

    def test_live_pytorch_oracle_if_available(self):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest('live PyTorch unavailable; committed fixture still runs')
        rng = np.random.default_rng(333)
        x = (rng.standard_normal((16, 17)) * .1).astype(np.float32)
        w = (rng.standard_normal((16, 8, 20)) * .1).astype(np.float32)
        b = (rng.standard_normal(8) * .1).astype(np.float32)
        expected = functional.conv_transpose1d(
            torch.from_numpy(x)[None], torch.from_numpy(w),
            torch.from_numpy(b), stride=10, padding=5)[0].detach().numpy()
        actual = np.full((8, expected.shape[1]), -91., np.float32)
        scratch = np.empty(expected.size, np.float32)
        function = self.functions[1][0]
        self.assertEqual(function(self.ptr(x), x.size, x.shape[1],
            self.ptr(w), w.size, self.ptr(b), b.size,
            self.ptr(actual), actual.size, actual.shape[1],
            self.ptr(scratch), scratch.size, 16, 8, 17, 20, 10, 5, 0,
            expected.shape[1]), 0)
        self.assertTrue(np.isfinite(actual).all())
        error = np.abs(actual - expected)
        point = np.unravel_index(np.argmax(error), error.shape)
        self.assertLessEqual(float(error[point]), 1e-5,
                             (point, float(error[point])))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'tts.dense-deconv.native-vs-pytorch-live',
            'name': 'checked dense ConvTranspose1D versus live PyTorch',
            'provider': 'audio_conv_transpose1d_dense_channel_major_f32_checked',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'live-pytorch', 'backend_version': torch.__version__,
            'status': 'pass', 'max_diff': float(error[point]),
            'worst_index': list(map(int, point)), 'tolerance': 1e-5,
            'configuration': 'I=16 O=8 T=17 K=20 stride=10 padding=5',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_full_generator_shape_if_requested(self):
        if os.environ.get('CKE_TTS_GENERATOR_FULL_SHAPE') != '1':
            self.skipTest('full 512-to-256 generator shape reserved for asset-backed host')
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest('live PyTorch unavailable for full generator geometry')
        torch.set_num_threads(1)
        rng = np.random.default_rng(719)
        x = (rng.standard_normal((512, 206)) * .01).astype(np.float32)
        w = (rng.standard_normal((512, 256, 20)) * .01).astype(np.float32)
        b = (rng.standard_normal(256) * .01).astype(np.float32)
        expected = functional.conv_transpose1d(
            torch.from_numpy(x)[None], torch.from_numpy(w),
            torch.from_numpy(b), stride=10, padding=5)[0].detach().numpy()
        actual = np.full_like(expected, -91.)
        scratch = np.empty(expected.size, np.float32)
        function = self.functions[1][0]
        self.assertEqual(function(self.ptr(x), x.size, x.shape[1],
            self.ptr(w), w.size, self.ptr(b), b.size,
            self.ptr(actual), actual.size, actual.shape[1],
            self.ptr(scratch), scratch.size, 512, 256, 206, 20, 10, 5, 0,
            expected.shape[1]), 0)
        self.assertTrue(np.isfinite(actual).all())
        error = np.abs(actual - expected)
        point = np.unravel_index(np.argmax(error), error.shape)
        self.assertLessEqual(float(error[point]), 1e-5,
                             (point, float(error[point])))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'tts.dense-deconv.generator-shape-live',
            'name': 'dense ConvTranspose1D at first Kokoro generator geometry',
            'provider': 'audio_conv_transpose1d_dense_channel_major_f32_checked',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'live-pytorch', 'backend_version': torch.__version__,
            'status': 'pass', 'max_diff': float(error[point]),
            'worst_index': list(map(int, point)), 'tolerance': 1e-5,
            'configuration': 'I=512 O=256 T=206 K=20 stride=10 padding=5',
            'reproduction_command': 'CKE_TTS_GENERATOR_FULL_SHAPE=1 '
                'python3 -m unittest ' + self.id()}))


if __name__ == '__main__':
    unittest.main()
