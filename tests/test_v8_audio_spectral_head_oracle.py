"""Pinned PyTorch 2.8 spectral-head oracle and rejection contracts."""

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
FIXTURE = ROOT / 'tests/fixtures/tts/spectral_head_torch28.json'
FloatPtr = ctypes.POINTER(ctypes.c_float)


class AudioSpectralHeadOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / 'libspectral_head.so'
        subprocess.run([
            'cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
            '-shared', '-fPIC', '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/audio_spectral_head.c'), '-lm',
            '-o', str(library)], check=True)
        cls.native = ctypes.CDLL(str(library))
        cls.fn = cls.native.audio_spectral_head_mag_phase_f32
        cls.fn.argtypes = [FloatPtr, ctypes.c_size_t, ctypes.c_size_t,
                           FloatPtr, ctypes.c_size_t, FloatPtr, ctypes.c_size_t,
                           ctypes.c_size_t, ctypes.c_size_t]
        cls.fn.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def arrays(self):
        fixture = self.fixture
        flat = [value for row in fixture['input'] for value in row]
        source = (ctypes.c_float * len(flat))(*flat)
        count = fixture['bins'] * fixture['frames']
        magnitude = (ctypes.c_float * (count + 3))(*([-91.] * (count + 3)))
        phase = (ctypes.c_float * (count + 3))(*([-91.] * (count + 3)))
        return source, magnitude, phase

    def call(self, source, magnitude, phase, *, frames=None,
             input_capacity=None, magnitude_capacity=None,
             phase_capacity=None):
        fixture = self.fixture
        return self.fn(source,
            len(source) if input_capacity is None else input_capacity,
            fixture['input_stride'], magnitude,
            len(magnitude) if magnitude_capacity is None else magnitude_capacity,
            phase, len(phase) if phase_capacity is None else phase_capacity,
            fixture['bins'], fixture['frames'] if frames is None else frames)

    def test_pinned_torch28_fixture_and_padding(self):
        source, magnitude, phase = self.arrays()
        self.assertEqual(self.call(source, magnitude, phase), 0)
        count = self.fixture['bins'] * self.fixture['frames']
        for label, output, reference in (
            ('magnitude', magnitude,
             self.fixture['magnitude_frame_major']),
            ('phase', phase, self.fixture['phase_frame_major'])):
            expected = [value for row in reference for value in row]
            errors = [abs(float(output[i]) - value)
                      for i, value in enumerate(expected)]
            worst = max(range(count), key=errors.__getitem__)
            self.assertTrue(all(math.isfinite(output[i]) for i in range(count)))
            self.assertLessEqual(errors[worst],
                self.fixture['max_abs_error'],
                f'{label} worst={worst} actual={output[worst]} '
                f'expected={expected[worst]}')
            self.assertTrue(all(value == -91. for value in output[count:]))

    def test_rejection_preserves_both_outputs_and_recovers(self):
        source, magnitude, phase = self.arrays()
        required = self.fixture['bins'] * self.fixture['frames']
        self.assertNotEqual(self.call(source, magnitude, phase,
            input_capacity=len(source)-1), 0)
        self.assertNotEqual(self.call(source, magnitude, phase,
            magnitude_capacity=required-1), 0)
        self.assertNotEqual(self.call(source, magnitude, phase,
            phase_capacity=required-1), 0)
        self.assertNotEqual(self.call(source, magnitude, phase, frames=11), 0)
        original = float(source[0])
        source[0] = float('nan')
        self.assertNotEqual(self.call(source, magnitude, phase), 0)
        source[0] = 100.
        self.assertNotEqual(self.call(source, magnitude, phase), 0)
        source[0] = original
        phase_index = self.fixture['bins'] * self.fixture['input_stride']
        original_phase = float(source[phase_index])
        source[phase_index] = float('inf')
        self.assertNotEqual(self.call(source, magnitude, phase), 0)
        source[phase_index] = original_phase
        self.assertNotEqual(self.fn(source, len(source),
            self.fixture['input_stride'], magnitude, len(magnitude),
            phase, len(phase), 1 << 63, self.fixture['frames']), 0)
        self.assertTrue(all(value == -91. for value in magnitude))
        self.assertTrue(all(value == -91. for value in phase))
        self.assertEqual(self.call(source, magnitude, phase), 0)
        self.assertNotEqual(self.fn(source, len(source),
            self.fixture['input_stride'], source, required,
            phase, len(phase), self.fixture['bins'],
            self.fixture['frames']), 0)
        partial = ctypes.cast(ctypes.byref(source, ctypes.sizeof(ctypes.c_float)),
                              FloatPtr)
        self.assertNotEqual(self.fn(source, len(source),
            self.fixture['input_stride'], partial, required,
            phase, len(phase), self.fixture['bins'],
            self.fixture['frames']), 0)

    def test_pinned_full_model_spectral_transform(self):
        stem = ROOT / 'tests/fixtures/tts/kokoro_spectral_head_pinned'
        data_path = stem.with_suffix('.npz')
        meta = json.loads(stem.with_suffix('.json').read_text())
        self.assertEqual(hashlib.sha256(data_path.read_bytes()).hexdigest(),
                         meta['fixture_sha256'])
        previous = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_pool_pinned.npz'
        self.assertEqual(hashlib.sha256(previous.read_bytes()).hexdigest(),
                         meta['previous_fixture_sha256'])
        with np.load(data_path, allow_pickle=False) as data:
            values = {name: np.ascontiguousarray(data[name], dtype=np.float32)
                      for name in ('conv_post', 'magnitude', 'phase')}
        for name, value in values.items():
            self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(),
                             meta['array_sha256'][name])
        source = values['conv_post'][0]
        frames, bins = source.shape[1], source.shape[0] // 2
        magnitude = np.full((frames, bins), -91., dtype=np.float32)
        phase = np.full((frames, bins), -91., dtype=np.float32)
        self.assertEqual(self.fn(
            source.ctypes.data_as(FloatPtr), source.size, frames,
            magnitude.ctypes.data_as(FloatPtr), magnitude.size,
            phase.ctypes.data_as(FloatPtr), phase.size, bins, frames), 0)
        for name, actual in (('magnitude', magnitude), ('phase', phase)):
            reference = values[name][0].T
            error = np.abs(actual - reference)
            point = np.unravel_index(np.argmax(error), error.shape)
            limit = 2e-6
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.spectral-head.{name}.direct-v1',
                'name': f'Kokoro spectral head {name} direct-input parity',
                'provider': 'audio_spectral_head_mag_phase_f32',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned-PyTorch-2.8-full-model-hook',
                'status': 'pass' if float(error[point]) <= limit else 'fail',
                'max_diff': float(error[point]), 'tolerance': limit,
                'worst_index': list(map(int, point)),
                'actual': float(actual[point]),
                'reference': float(reference[point]),
                'configuration': '11 bins, 12361 frames, af_heart',
                'reproduction_command': 'python3 -m unittest '
                    'tests.test_v8_audio_spectral_head_oracle.'
                    'AudioSpectralHeadOracleTest.test_pinned_full_model_spectral_transform'}))
            self.assertTrue(np.isfinite(actual).all())
            self.assertLessEqual(float(error[point]), limit)


if __name__ == '__main__':
    unittest.main()
