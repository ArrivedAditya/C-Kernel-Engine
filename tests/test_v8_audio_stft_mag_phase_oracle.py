"""Checked magnitude/phase STFT against independent NumPy and optional PyTorch."""

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/tts/audio_stft_mag_phase_torch_reference.npz'
FLOAT = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class AudioStftMagnitudePhaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.calls = []
        for flag in ('-O0', '-O3'):
            library = Path(cls.temp.name) / f'stft_{flag}.so'
            subprocess.run(['cc', '-std=c11', flag, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC', '-I',
                str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_stft_mag_phase_checked.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            native = ctypes.CDLL(str(library))
            run = native.audio_stft_mag_phase_checked_f32
            run.argtypes = [FLOAT, SIZE, FLOAT, SIZE, FLOAT, FLOAT, SIZE,
                FLOAT, SIZE, SIZE, FLOAT, SIZE, SIZE, SIZE, SIZE, SIZE]
            run.restype = ctypes.c_int
            plan = native.audio_stft_mag_phase_plan_f32
            plan.argtypes = [SIZE, SIZE, SIZE, ctypes.POINTER(SIZE),
                ctypes.POINTER(SIZE)]
            plan.restype = ctypes.c_int
            cls.calls.append((run, plan))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def pointer(value):
        return value.ctypes.data_as(FLOAT)

    @staticmethod
    def tables(nfft):
        tap = np.arange(nfft, dtype=np.float64)
        window = (0.5 - 0.5 * np.cos(2 * np.pi * tap / nfft)).astype(np.float32)
        angle = -2 * np.pi * np.arange(nfft // 2 + 1)[:, None] * tap / nfft
        return window, np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)

    def invoke(self, run, samples, nfft, hop, frames=None, stride_extra=3,
               output_short=0, scratch_short=0):
        count = len(samples) // hop + 1 if frames is None else frames
        bins = nfft // 2 + 1
        window, cosine, sine = self.tables(nfft)
        stride = count + stride_extra
        output = np.full((2 * bins, stride), -91, dtype=np.float32)
        scratch = np.full(2 * bins * count, -37, dtype=np.float32)
        args = [self.pointer(samples), len(samples), self.pointer(window), len(window),
            self.pointer(cosine), self.pointer(sine), cosine.size,
            self.pointer(output), (2 * bins - 1) * stride + count - output_short,
            stride,
            self.pointer(scratch), scratch.size - scratch_short,
            len(samples), nfft, hop, count]
        return run(*args), output, scratch, args, (window, cosine, sine)

    @staticmethod
    def reference(samples, nfft, hop):
        padded = np.pad(samples.astype(np.float64), nfft // 2, mode='reflect')
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)
        frames = len(samples) // hop + 1
        spectrum = np.stack([np.fft.rfft(padded[i * hop:i * hop + nfft] * window)
            for i in range(frames)], axis=1)
        return np.concatenate((np.abs(spectrum), np.angle(spectrum)), axis=0)

    def test_numpy_oracle_and_padded_stride(self):
        rng = np.random.default_rng(731)
        for samples, nfft, hop in ((rng.normal(size=59).astype(np.float32), 20, 5),
                                   (rng.normal(size=71).astype(np.float32), 32, 8),
                                   (np.ones(60, dtype=np.float32), 20, 5),
                                   (np.zeros(60, dtype=np.float32), 20, 5)):
            expected = self.reference(samples, nfft, hop)
            for run, plan in self.calls:
                frames, elements = SIZE(), SIZE()
                self.assertEqual(plan(len(samples), nfft, hop,
                    ctypes.byref(frames), ctypes.byref(elements)), 0)
                self.assertEqual((frames.value, elements.value),
                    (expected.shape[1], expected.size))
                status, output, _, _, _ = self.invoke(run, samples, nfft, hop)
                self.assertEqual(status, 0)
                actual = output[:, :frames.value]
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual[:nfft // 2 + 1],
                    expected[:nfft // 2 + 1], rtol=1e-5, atol=2e-5)
                # A phase at a zero-amplitude bin is undefined.
                mask = expected[:nfft // 2 + 1] > 1e-5
                delta = np.angle(np.exp(1j *
                    (actual[nfft // 2 + 1:] - expected[nfft // 2 + 1:])))
                self.assertLess(np.max(np.abs(delta[mask]), initial=0), 3e-5)
                self.assertTrue(np.all(output[:, frames.value:] == -91))

    def test_rejection_preserves_output_and_recovery(self):
        samples = np.arange(60, dtype=np.float32) / 60
        for run, plan in self.calls:
            for option in ('output', 'scratch', 'frames', 'nan', 'inf'):
                source = samples.copy()
                if option == 'nan': source[7] = np.nan
                if option == 'inf': source[7] = np.inf
                status, output, _, _, _ = self.invoke(run, source, 20, 5,
                    frames=12 if option == 'frames' else None,
                    output_short=1 if option == 'output' else 0,
                    scratch_short=1 if option == 'scratch' else 0)
                self.assertNotEqual(status, 0, option)
                self.assertTrue(np.all(output == -91), option)
            status, output, _, _, _ = self.invoke(run, samples, 20, 5)
            self.assertEqual(status, 0)
            self.assertTrue(np.isfinite(output[:, :13]).all())
            frames, elements = SIZE(99), SIZE(99)
            self.assertNotEqual(plan(60, SIZE(-2).value, 5,
                ctypes.byref(frames), ctypes.byref(elements)), 0)
            self.assertEqual((frames.value, elements.value), (99, 99))

    def test_invalid_geometry_tables_and_aliasing(self):
        samples = np.arange(60, dtype=np.float32) / 60
        for run, plan in self.calls:
            for nfft, hop in ((20, 0), (19, 5), (128, 5)):
                frames, elements = SIZE(17), SIZE(19)
                self.assertNotEqual(plan(len(samples), nfft, hop,
                    ctypes.byref(frames), ctypes.byref(elements)), 0)
                self.assertEqual((frames.value, elements.value), (17, 19))
            status, output, _, args, live = self.invoke(run, samples, 20, 5)
            self.assertEqual(status, 0)
            for argument in (0, 2, 4, 5, 7, 10):
                altered = list(args)
                altered[argument] = FLOAT()
                before = output.copy()
                self.assertNotEqual(run(*altered), 0)
                np.testing.assert_array_equal(output, before)
            # Valid metadata cannot authorize an output/scratch overlap.
            altered = list(args)
            altered[10] = altered[7]
            before = output.copy()
            self.assertNotEqual(run(*altered), 0)
            np.testing.assert_array_equal(output, before)
            window, cosine, sine = live
            cosine[3, 4] = np.nan
            before = output.copy()
            self.assertNotEqual(run(*args), 0)
            np.testing.assert_array_equal(output, before)
            extreme = np.full(60, 3e38, dtype=np.float32)
            status, rejected, _, _, _ = self.invoke(run, extreme, 20, 5)
            self.assertNotEqual(status, 0)
            self.assertTrue(np.all(rejected == -91))

    def test_full_source_geometry_live_pytorch(self):
        try:
            import torch
        except ImportError:
            self.skipTest('live PyTorch unavailable; smaller NumPy cases still run')
        # 206 F0 frames * 300 samples/frame => 12,361 source STFT frames.
        samples = np.random.default_rng(531).normal(0, .05, 206 * 300).astype(np.float32)
        run, _ = self.calls[-1]
        status, output, _, _, _ = self.invoke(run, samples, 20, 5, stride_extra=0)
        self.assertEqual(status, 0)
        spectrum = torch.stft(torch.from_numpy(samples), 20, 5, 20,
            window=torch.hann_window(20, periodic=True), return_complex=True)
        reference = spectrum.numpy()
        magnitude = np.abs(reference)
        phase = np.angle(reference)
        actual_magnitude = output[:11]
        actual_phase = output[11:]
        self.assertTrue(np.isfinite(output).all())
        self.assertLess(np.max(np.abs(actual_magnitude - magnitude)), 2e-5)
        angular = np.angle(np.exp(1j * (actual_phase - phase)))
        self.assertLess(np.max(np.abs(angular[magnitude > 1e-5]), initial=0), 3e-5)
        # These phase channels are later convolved, so raw interior values
        # matter in addition to circular agreement. Frame zero is reflection
        # symmetric and retains a small FFT branch-cut sign discrepancy.
        self.assertLess(np.max(np.abs(actual_phase[:, 1:] - phase[:, 1:])), 3e-5)

    def test_pinned_source_window_raw_phase_diagnostic(self):
        source_fixture = ROOT / 'tests/fixtures/tts/kokoro_source_pinned.npz'
        metadata = json.loads(source_fixture.with_suffix('.json').read_text())
        self.assertEqual(hashlib.sha256(source_fixture.read_bytes()).hexdigest(),
                         metadata['fixture_sha256'])
        with np.load(source_fixture) as archive:
            samples = archive['source'].reshape(-1).copy()
            window = archive['window'].copy()
            expected_magnitude = archive['magnitude'][0]
            expected_phase = archive['phase'][0]
        for run, _ in self.calls:
            status, output, _, arguments, _ = self.invoke(
                run, samples, 20, 5, stride_extra=0)
            self.assertEqual(status, 0)
            arguments[2] = self.pointer(window)
            self.assertEqual(run(*arguments), 0)
            self.assertTrue(np.isfinite(output).all())
            magnitude_error = np.abs(output[:11] - expected_magnitude)
            phase_error = np.abs(output[11:] - expected_phase)
            self.assertLessEqual(float(magnitude_error.max()), 2e-7)
            angular = np.abs(np.angle(np.exp(1j *
                (output[11:] - expected_phase))))
            self.assertLessEqual(float(angular.max()), 3e-4)
            mismatches = np.argwhere(phase_error > 1.)
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'kokoro.source-stft.raw-phase-diagnostic',
                'name': 'pinned-window raw phase before source convolution',
                'provider': 'audio_stft_mag_phase_checked_f32',
                'oracle': 'direct-pinned-kokoro-pytorch28',
                'status': 'not_tested',
                'configuration': '61,800 samples, PyTorch 2.8 Hann window, nfft20/hop5',
                'max_magnitude_error': float(magnitude_error.max()),
                'max_raw_phase_error': float(phase_error.max()),
                'phase_branch_cut_mismatches': mismatches.tolist(),
                'reason': 'source-convolution parity remains unresolved'}))

    def test_live_pytorch_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest('live PyTorch unavailable; NumPy oracle still runs')
        samples = np.random.default_rng(991).normal(size=60).astype(np.float32)
        for run, _ in self.calls:
            status, output, _, _, _ = self.invoke(run, samples, 20, 5)
            self.assertEqual(status, 0)
            spectrum = torch.stft(torch.from_numpy(samples), 20, 5, 20,
                window=torch.hann_window(20, periodic=True), return_complex=True)
            expected = torch.cat([torch.abs(spectrum), torch.angle(spectrum)], dim=0)
            actual = output[:, :13]
            reference = expected.numpy()
            np.testing.assert_allclose(actual[:11], reference[:11],
                rtol=1e-5, atol=3e-5)
            angular_error = np.angle(np.exp(1j * (actual[11:] - reference[11:])))
            mask = reference[:11] > 1e-5
            self.assertLess(np.max(np.abs(angular_error[mask]), initial=0), 3e-5)

    def test_committed_pinned_pytorch_fixture(self):
        metadata = json.loads(FIXTURE.with_suffix('.json').read_text())
        self.assertEqual(metadata['torch'], '2.8.0+cpu')
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            metadata['fixture_sha256'])
        with np.load(FIXTURE) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        for name, value in arrays.items():
            self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(),
                metadata['array_sha256'][name])
        for run, _ in self.calls:
            status, output, _, _, _ = self.invoke(run, arrays['samples'], 20, 5)
            self.assertEqual(status, 0)
            self.assertTrue(np.isfinite(output[:, :13]).all())
            np.testing.assert_allclose(output[:11, :13], arrays['magnitude'],
                rtol=1e-5, atol=3e-5)
            angular = np.angle(np.exp(1j *
                (output[11:, :13] - arrays['phase'])))
            mask = arrays['magnitude'] > 1e-5
            self.assertLess(np.max(np.abs(angular[mask]), initial=0), 3e-5)
            self.assertLess(np.max(np.abs(output[11:, 1:13] -
                arrays['phase'][:, 1:])), 3e-5)


if __name__ == '__main__':
    unittest.main()
