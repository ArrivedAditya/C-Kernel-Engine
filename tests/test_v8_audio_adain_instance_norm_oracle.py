"""Independent PyTorch fixture and checked bounds for channelwise AdaIN."""
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
FIXTURE = ROOT / 'tests/fixtures/tts/adain_instance_norm_torch.npz'
FLOAT = ctypes.c_float
POINTER = ctypes.POINTER(FLOAT)


class AdaINInstanceNormOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        meta = json.loads(FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != meta['fixture_sha256']:
            raise RuntimeError('AdaIN fixture hash mismatch')
        cls.meta = meta
        with np.load(FIXTURE) as archive:
            cls.arrays = {name: archive[name].copy() for name in archive.files}
        for name, value in cls.arrays.items():
            if hashlib.sha256(value.tobytes()).hexdigest() != meta['array_sha256'][name]:
                raise RuntimeError(f'AdaIN array hash mismatch: {name}')
        cls.functions = []
        for optimization in ('-O0', '-O3'):
            library = Path(cls.temp.name) / f'adain_{optimization}.so'
            subprocess.run(['cc', '-std=c11', optimization, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-ffp-contract=off', '-shared', '-fPIC',
                '-I', str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_adain_instance_norm.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            function = ctypes.CDLL(str(library)).audio_adain_instance_norm_f32
            function.argtypes = [POINTER, ctypes.c_size_t, ctypes.c_size_t,
                POINTER, ctypes.c_size_t, POINTER, ctypes.c_size_t,
                POINTER, ctypes.c_size_t, POINTER, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, FLOAT]
            function.restype = ctypes.c_int
            cls.functions.append(function)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def case(self, index):
        channels, frames = self.meta['shapes'][index]
        values = {name: self.arrays[f'case{index}_{name}'].copy() for name in
                  ('input', 'norm_weight', 'norm_bias', 'style_affine', 'output')}
        stride = frames + (25 if channels == 512 else 3)
        source = np.full((channels, stride), -777., np.float32)
        source[:, :frames] = values['input']
        output = np.full((channels, stride), -999., np.float32)
        return values, source, output, stride

    @staticmethod
    def args(values, source, output, stride, frames, epsilon):
        pointer = lambda x: x.ctypes.data_as(POINTER)
        return [pointer(source), source.size, stride,
            pointer(values['norm_weight']), values['norm_weight'].size,
            pointer(values['norm_bias']), values['norm_bias'].size,
            pointer(values['style_affine']), values['style_affine'].size,
            pointer(output), output.size, stride,
            source.shape[0], frames, epsilon]

    def test_committed_independent_pytorch_fixture(self):
        worst = (0., None)
        for index, (_channels, frames) in enumerate(self.meta['shapes']):
            values, source, output, stride = self.case(index)
            arguments = self.args(values, source, output, stride, frames,
                                  self.meta['epsilon'])
            for function in self.functions:
                output[:] = -999.
                self.assertEqual(function(*arguments), 0)
                actual = output[:, :frames]
                self.assertTrue(np.isfinite(actual).all())
                error = np.abs(actual - values['output'])
                point = np.unravel_index(np.argmax(error), error.shape)
                if float(error[point]) > worst[0]:
                    worst = (float(error[point]), (index, *point))
                self.assertTrue(np.all(output[:, frames:] == -999.))
                self.assertLessEqual(float(error[point]), 1e-5,
                                     (index, point, float(error[point])))
            first = output.copy()
            self.assertEqual(self.functions[0](*arguments), 0)
            np.testing.assert_array_equal(output, first)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'tts.adain-instance-norm.native-vs-fixture',
            'name': 'channelwise AdaIN native versus committed PyTorch fixture',
            'provider': 'audio_adain_instance_norm_f32', 'dtype': 'fp32',
            'direction': 'inference', 'oracle': 'committed-pytorch',
            'backend_version': self.meta['backend_version'],
            'status': 'pass', 'max_diff': worst[0],
            'worst_index': [int(part) for part in worst[1]],
            'tolerance': 1e-5, 'configuration': str(self.meta['shapes']),
            'reproduction_command': 'python3 -m unittest ' + self.id(),
        }))

    def test_rejection_preserves_output(self):
        values, source, output, stride = self.case(2)
        basic = self.args(values, source, output, stride, 7,
                          self.meta['epsilon'])
        for index in (1, 4, 6, 8, 10):
            with self.subTest(short_capacity=index):
                arguments = basic.copy()
                arguments[index] = 0
                self.assertEqual(self.functions[0](*arguments), -2)
                self.assertTrue(np.all(output == -999.))
        for index, value in ((2, 6), (11, 6), (12, 0), (13, 0),
                             (14, 0.), (2, ctypes.c_size_t(-1).value)):
            with self.subTest(invalid=index):
                arguments = basic.copy()
                arguments[index] = value
                self.assertEqual(self.functions[0](*arguments), -1)
                self.assertTrue(np.all(output == -999.))
        for name in ('input', 'norm_weight', 'norm_bias', 'style_affine'):
            for nonfinite in (math.nan, math.inf):
                with self.subTest(nonfinite=name, value=nonfinite):
                    values, source, output, stride = self.case(2)
                    target = source if name == 'input' else values[name]
                    target.flat[0] = nonfinite
                    arguments = self.args(values, source, output, stride, 7,
                                          self.meta['epsilon'])
                    self.assertEqual(self.functions[0](*arguments), -3)
                    self.assertTrue(np.all(output == -999.))
        values, source, output, stride = self.case(2)
        values['style_affine'][0] = np.finfo(np.float32).max
        values['norm_weight'][0] = np.finfo(np.float32).max
        arguments = self.args(values, source, output, stride, 7,
                              self.meta['epsilon'])
        self.assertEqual(self.functions[0](*arguments), -3)
        self.assertTrue(np.all(output == -999.))

if __name__ == '__main__':
    unittest.main()
