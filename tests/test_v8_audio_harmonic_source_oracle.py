"""Checked harmonic source against direct pinned Kokoro model hooks."""

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/tts/kokoro_source_pinned.npz'
SMALL_FIXTURE = ROOT / 'tests/fixtures/tts/audio_harmonic_source_cases_pytorch28.npz'
FLOAT = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class HarmonicSourceOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.meta = json.loads(FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != cls.meta['fixture_sha256']:
            raise RuntimeError('Kokoro source fixture checksum mismatch')
        with np.load(FIXTURE) as archive:
            cls.arrays = {key: archive[key].copy() for key in archive.files}
        for name, value in cls.arrays.items():
            if hashlib.sha256(value.tobytes()).hexdigest() != cls.meta['array_sha256'][name]:
                raise RuntimeError(f'Kokoro source tensor checksum mismatch: {name}')
        cls.functions = []
        for optimization in ('-O0', '-O3'):
            library = Path(cls.temp.name) / f'harmonic_{optimization}.so'
            subprocess.run(['cc', '-std=c11', optimization, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC', '-I',
                str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_harmonic_source_checked.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            function = ctypes.CDLL(str(library)).audio_harmonic_source_checked_f32
            function.argtypes = [FLOAT, SIZE, FLOAT, SIZE, FLOAT, SIZE,
                FLOAT, SIZE, SIZE, SIZE, SIZE,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
            function.restype = ctypes.c_int
            cls.functions.append(function)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def pointer(array):
        return array.ctypes.data_as(FLOAT)

    def call(self, function, f0=None, gaussian=None, output=None, scratch=None,
             upsample=300, harmonics=9):
        f0 = (self.arrays['f0'].reshape(-1).copy() if f0 is None else f0)
        gaussian = (self.arrays['gaussian'].reshape(-1).copy()
                    if gaussian is None else gaussian)
        output = (np.full(gaussian.size, -91., np.float32)
                  if output is None else output)
        scratch = (np.full(f0.size * harmonics + gaussian.size, -37., np.float32)
                   if scratch is None else scratch)
        args = [self.pointer(f0), f0.size, self.pointer(gaussian), gaussian.size,
                self.pointer(output), output.size, self.pointer(scratch), scratch.size,
                f0.size, upsample, harmonics, 24000., 10., .1, .003]
        return function(*args), output, args, (f0, gaussian, scratch)

    def test_00_rejection_output_preservation_and_recovery(self):
        for function in self.functions:
            status, output, args, buffers = self.call(function)
            self.assertEqual(status, 0)
            f0, gaussian, scratch = buffers
            for index in (1, 3, 5, 7):
                with self.subTest(capacity=index):
                    bad = args.copy(); bad[index] -= 1
                    output.fill(-91.)
                    self.assertNotEqual(function(*bad), 0)
                    self.assertTrue(np.all(output == -91.))
            for index in (0, 2, 4, 6):
                with self.subTest(null_pointer=index):
                    bad = args.copy(); bad[index] = FLOAT()
                    output.fill(-91.)
                    self.assertNotEqual(function(*bad), 0)
                    self.assertTrue(np.all(output == -91.))
            bad = args.copy(); bad[6] = self.pointer(output)
            output.fill(-91.)
            self.assertNotEqual(function(*bad), 0)
            self.assertTrue(np.all(output == -91.))
            f0[0] = np.inf
            self.assertNotEqual(function(*args), 0)
            self.assertTrue(np.all(output == -91.))
            f0[0] = self.arrays['f0'].reshape(-1)[0]
            gaussian[0] = np.nan
            self.assertNotEqual(function(*args), 0)
            self.assertTrue(np.all(output == -91.))
            gaussian[0] = self.arrays['gaussian'].reshape(-1)[0]
            f0[0] = np.float32(3e38)
            self.assertNotEqual(function(*args), 0)
            self.assertTrue(np.all(output == -91.))
            f0[0] = self.arrays['f0'].reshape(-1)[0]
            self.assertEqual(function(*args), 0)
            self.assertTrue(np.isfinite(output).all())
            self.assertLessEqual(float(np.max(np.abs(output -
                self.arrays['sine_waves'].reshape(-1)))), 5e-4)
            for name, value in self.arrays.items():
                self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(),
                    self.meta['array_sha256'][name], name)

    def test_direct_pinned_model_harmonic_checkpoint(self):
        outputs = []
        for variant, function in enumerate(self.functions):
            status, output, _, _ = self.call(function)
            self.assertEqual(status, 0)
            expected = self.arrays['sine_waves'].reshape(-1)
            self.assertTrue(np.isfinite(output).all())
            error = np.abs(output - expected)
            point = int(np.argmax(error))
            self.assertLessEqual(float(error[point]), 5e-4,
                                 (point, float(output[point]), float(expected[point])))
            outputs.append(output.copy())
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.harmonic-source.native-vs-pinned-model.{("o0", "o3")[variant]}',
                'name': 'checked harmonic source versus direct full-model hook',
                'provider': 'audio_harmonic_source_checked_f32',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned-full-kmodel-pytorch28',
                'backend_version': self.meta['environment']['torch'],
                'status': 'pass', 'max_diff': float(error[point]),
                'worst_index': point, 'tolerance': 5e-4,
                'configuration': '206 F0 frames, 300x, 9 harmonics, captured Gaussian',
                'reproduction_command': 'python3 -m unittest ' + self.id()}))
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_pinned_voiced_unvoiced_lengths_and_upsampling(self):
        metadata = json.loads(SMALL_FIXTURE.with_suffix('.json').read_text())
        self.assertEqual(metadata['torch'], '2.8.0+cpu')
        self.assertEqual(hashlib.sha256(SMALL_FIXTURE.read_bytes()).hexdigest(),
                         metadata['fixture_sha256'])
        with np.load(SMALL_FIXTURE) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        for name, value in arrays.items():
            self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(),
                             metadata['array_sha256'][name], name)
        for case in metadata['cases']:
            name = case['name']
            f0 = arrays[f'{name}_f0'].reshape(-1)
            gaussian = arrays[f'{name}_gaussian'].reshape(-1)
            expected = arrays[f'{name}_output'].reshape(-1)
            outputs = []
            for function in self.functions:
                status, output, _, _ = self.call(function, f0.copy(),
                    gaussian.copy(), upsample=case['upsample'],
                    harmonics=case['harmonics'])
                self.assertEqual(status, 0, name)
                self.assertTrue(np.isfinite(output).all(), name)
                error = np.abs(output - expected)
                worst = int(np.argmax(error))
                self.assertLessEqual(float(error[worst]), 2e-7,
                    (name, worst, float(output[worst]), float(expected[worst])))
                outputs.append(output.copy())
            np.testing.assert_array_equal(outputs[0], outputs[1])
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'audio.harmonic-source.{name}.pytorch28',
                'name': f'harmonic source {name} versus pinned SineGen',
                'provider': 'audio_harmonic_source_checked_f32',
                'oracle': 'pinned-kokoro-sinegen-pytorch28',
                'status': 'pass', 'max_diff': float(error[worst]),
                'tolerance': 2e-7,
                'configuration': f"frames={case['frames']} upsample={case['upsample']} harmonics={case['harmonics']}"}))

    def test_small_geometry_and_invalid_bounds(self):
        for function in self.functions:
            f0 = np.array([100., 0., 220.], np.float32)
            gaussian = np.zeros(3 * 4 * 2, np.float32)
            output = np.full_like(gaussian, -91.)
            scratch = np.empty(f0.size * 2 + output.size, np.float32)
            status, result, args, _ = self.call(function, f0, gaussian,
                output, scratch, upsample=4, harmonics=2)
            self.assertEqual(status, 0)
            self.assertTrue(np.isfinite(result).all())
            for index, value in ((8, 0), (9, 0), (9, 1), (9, 2),
                                 (10, 0), (12, float('nan')),
                                 (8, SIZE(-1).value)):
                bad = args.copy(); bad[index] = value
                output.fill(-91.)
                self.assertNotEqual(function(*bad), 0)
                self.assertTrue(np.all(output == -91.))


if __name__ == '__main__':
    unittest.main()
