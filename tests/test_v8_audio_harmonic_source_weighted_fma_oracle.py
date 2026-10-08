"""Independent pinned PyTorch fixtures for weighted-FMA harmonic generation."""

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / 'tests/fixtures/tts/kokoro_source_pinned.npz'
SMALL = ROOT / 'tests/fixtures/tts/audio_harmonic_source_cases_pytorch28.npz'
FLOAT = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


def fixture(path):
    meta = json.loads(path.with_suffix('.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['fixture_sha256']:
        raise RuntimeError(f'fixture checksum mismatch: {path}')
    with np.load(path) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, array in arrays.items():
        if hashlib.sha256(array.tobytes()).hexdigest() != meta['array_sha256'][name]:
            raise RuntimeError(f'array checksum mismatch: {name}')
    return meta, arrays


class HarmonicWeightedFmaOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.meta, cls.pinned = fixture(PINNED)
        cls.small_meta, cls.small = fixture(SMALL)
        cls.functions = []
        for optimization in ('-O0', '-O3'):
            library = Path(cls.temp.name) / f'weighted_{optimization}.so'
            subprocess.run(['cc', '-std=c11', optimization, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC', '-I', str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_harmonic_source_checked.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            function = ctypes.CDLL(str(library)).audio_harmonic_source_weighted_fma_checked_f32
            function.argtypes = [FLOAT, SIZE, FLOAT, SIZE, FLOAT, SIZE,
                FLOAT, SIZE, SIZE, SIZE, SIZE,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
            function.restype = ctypes.c_int
            cls.functions.append(function)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def pointer(value):
        return value.ctypes.data_as(FLOAT)

    def run_case(self, function, f0, gaussian, upsample, harmonics,
                 output=None, scratch=None):
        f0 = np.ascontiguousarray(f0.reshape(-1), dtype=np.float32)
        gaussian = np.ascontiguousarray(gaussian.reshape(-1), dtype=np.float32)
        output = np.full_like(gaussian, -91.) if output is None else output
        scratch = (np.full(f0.size * harmonics + gaussian.size, -37., np.float32)
                   if scratch is None else scratch)
        args = [self.pointer(f0), f0.size, self.pointer(gaussian), gaussian.size,
            self.pointer(output), output.size, self.pointer(scratch), scratch.size,
            f0.size, upsample, harmonics, 24000., 10., .1, .003]
        return function(*args), output, scratch, args

    def test_full_pinned_source(self):
        expected = self.pinned['sine_waves'].reshape(-1)
        outputs = []
        for optimization, function in zip(('o0', 'o3'), self.functions):
            status, output, _, _ = self.run_case(function, self.pinned['f0'],
                self.pinned['gaussian'], 300, 9)
            self.assertEqual(status, 0)
            self.assertTrue(np.isfinite(output).all())
            error = np.abs(output - expected)
            worst = int(np.argmax(error))
            rmse = float(np.sqrt(np.mean(np.square(
                output.astype(np.float64) - expected.astype(np.float64)))))
            self.assertLessEqual(float(error[worst]), 2e-7,
                (worst, float(output[worst]), float(expected[worst])))
            outputs.append(output.copy())
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'kokoro.harmonic-source.weighted-fma.' + optimization,
                'name': 'weighted-FMA harmonic source versus pinned full-model SineGen',
                'provider': 'audio_harmonic_source_weighted_fma_checked_f32',
                'contract': 'audio_harmonic_source_wide_cumsum_weighted_fma_phase_fp32',
                'oracle': 'direct-pinned-full-model-pytorch28',
                'status': 'pass', 'gate': 'numerical', 'blocking': True,
                'max_diff': float(error[worst]), 'worst_index': worst,
                'actual': float(output[worst]), 'reference': float(expected[worst]),
                'rmse': rmse, 'tolerance': 2e-7,
                'configuration': '206 frames, 300x, 9 harmonics, captured Gaussian',
                'reproduction_command': 'python3 -m unittest ' + self.id()}))
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_small_shapes_and_repeated_execution(self):
        for case in self.small_meta['cases']:
            stem = case['name']
            f0 = self.small[f'{stem}_f0']
            gaussian = self.small[f'{stem}_gaussian']
            expected = self.small[f'{stem}_output'].reshape(-1)
            for function in self.functions:
                with self.subTest(case=stem):
                    status, output, _, _ = self.run_case(function, f0,
                        gaussian, case['upsample'], case['harmonics'])
                    self.assertEqual(status, 0)
                    self.assertTrue(np.isfinite(output).all())
                    self.assertLessEqual(float(np.max(np.abs(output - expected))), 2e-7)
                    first = output.copy()
                    output.fill(-91.)
                    self.assertEqual(self.run_case(function, f0, gaussian,
                        case['upsample'], case['harmonics'], output=output)[0], 0)
                    np.testing.assert_array_equal(output, first)

    def test_rejected_capacity_and_nonfinite_leave_output_untouched(self):
        f0 = self.pinned['f0'].reshape(-1).copy()
        gaussian = self.pinned['gaussian'].reshape(-1).copy()
        for function in self.functions:
            status, output, scratch, args = self.run_case(function, f0,
                gaussian, 300, 9)
            self.assertEqual(status, 0)
            for index in (1, 3, 5, 7):
                bad = args.copy(); bad[index] -= 1
                output.fill(-91.)
                self.assertNotEqual(function(*bad), 0)
                self.assertTrue(np.all(output == -91.))
            bad_f0 = f0.copy(); bad_f0[0] = np.nan
            bad = args.copy(); bad[0] = self.pointer(bad_f0)
            self.assertNotEqual(function(*bad), 0)
            self.assertTrue(np.all(output == -91.))
            self.assertEqual(function(*args), 0)
            self.assertLessEqual(float(np.max(np.abs(output -
                self.pinned['sine_waves'].reshape(-1)))), 2e-7)


if __name__ == '__main__':
    unittest.main()
