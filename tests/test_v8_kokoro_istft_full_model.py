"""Direct pinned full-model waveform comparison for the native inverse STFT."""

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FloatPtr = ctypes.POINTER(ctypes.c_float)


class KokoroIstftFullModelTest(unittest.TestCase):
    def test_direct_pinned_spectrum_to_waveform(self):
        stem = ROOT / 'tests/fixtures/tts/kokoro_spectral_head_pinned'
        data_path = stem.with_suffix('.npz')
        meta = json.loads(stem.with_suffix('.json').read_text())
        self.assertEqual(hashlib.sha256(data_path.read_bytes()).hexdigest(),
                         meta['fixture_sha256'])
        with np.load(data_path, allow_pickle=False) as data:
            magnitude = np.ascontiguousarray(data['magnitude'][0].T)
            phase = np.ascontiguousarray(data['phase'][0].T)
            reference = np.ascontiguousarray(data['waveform'][0, 0])
        self.assertEqual((magnitude.shape, phase.shape),
                         ((12361, 11), (12361, 11)))
        self.assertEqual(reference.size, 61800)
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / 'libistft.so'
            subprocess.run([
                'cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
                '-shared', '-fPIC', '-I', str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_istft_mag_phase.c'), '-lm',
                '-o', str(library)], check=True)
            native = ctypes.CDLL(str(library))
            fn = native.audio_istft_mag_phase_f32
            fn.argtypes = [FloatPtr, FloatPtr, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                FloatPtr, ctypes.c_size_t, FloatPtr, ctypes.c_size_t]
            fn.restype = ctypes.c_int
            output = np.full(reference.size + 3, -91., dtype=np.float32)
            scratch = np.zeros(2 * (reference.size + 20) + 20,
                               dtype=np.float32)
            self.assertEqual(fn(
                magnitude.ctypes.data_as(FloatPtr),
                phase.ctypes.data_as(FloatPtr), magnitude.size,
                12361, 20, 5, output.ctypes.data_as(FloatPtr),
                reference.size, scratch.ctypes.data_as(FloatPtr),
                scratch.size), 0)
        self.assertTrue(np.isfinite(output[:reference.size]).all())
        self.assertTrue(np.all(output[reference.size:] == -91.))
        error = np.abs(output[:reference.size] - reference)
        point = int(np.argmax(error))
        rmse = float(np.sqrt(np.mean(
            (output[:reference.size].astype(np.float64) - reference) ** 2)))
        limit = 2e-6
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.istft.direct-full-model-v1',
            'name': 'Kokoro pinned spectrum to native waveform',
            'provider': 'audio_istft_mag_phase_f32',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-PyTorch-2.8-full-model-waveform',
            'status': 'pass' if float(error[point]) <= limit else 'fail',
            'max_diff': float(error[point]), 'rmse': rmse,
            'tolerance': limit, 'worst_index': point,
            'actual': float(output[point]),
            'reference': float(reference[point]),
            'configuration': '20-point FFT, hop 5, 12361 frames, 61800 samples',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_istft_full_model'}))
        self.assertLessEqual(float(error[point]), limit)


if __name__ == '__main__':
    unittest.main()
