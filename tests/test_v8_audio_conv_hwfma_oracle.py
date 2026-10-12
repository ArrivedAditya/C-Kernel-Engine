"""Checked Conv1D hardware-FMA provider against pinned PyTorch and scalar C."""

import ctypes
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEXT_FIXTURE = ROOT / 'tests/fixtures/tts/kokoro_text_encoder_pinned.npz'
DILATED_FIXTURE = ROOT / 'tests/fixtures/tts/audio_dilated_conv_pytorch28_reference.npz'
F = ctypes.POINTER(ctypes.c_float)
Z = ctypes.c_size_t
ARGTYPES = [F, Z, Z, F, Z, F, Z, F, Z, Z, F, Z] + [Z] * 7


class CheckedConvHardwareFmaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library_path = Path(cls.temp.name) / 'audio_conv_hwfma.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-pedantic', '-ffp-contract=off', '-shared', '-fPIC',
            '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/audio_conv1d_checked.c'),
            '-lm', '-o', str(cls.library_path)],
            check=True, capture_output=True, text=True)
        cls.native = ctypes.CDLL(str(cls.library_path))
        cls.available = cls.native.audio_conv1d_checked_hardware_fma_available
        cls.available.argtypes = []
        cls.available.restype = ctypes.c_int
        cls.functions = {}
        for name in ('audio_conv1d_checked_channel_major_f32',
                     'audio_conv1d_checked_channel_major_f32_hwfma',
                     'audio_conv1d_dilated_checked_channel_major_f32',
                     'audio_conv1d_dilated_checked_channel_major_f32_hwfma'):
            function = getattr(cls.native, name)
            function.argtypes = ARGTYPES + ([Z] if 'dilated' in name else [])
            function.restype = ctypes.c_int
            cls.functions[name] = function
        text_metadata = json.loads(TEXT_FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(TEXT_FIXTURE.read_bytes()).hexdigest() != \
                text_metadata['fixture_sha256']:
            raise RuntimeError('text encoder fixture checksum mismatch')
        with np.load(TEXT_FIXTURE) as archive:
            cls.text = {key: archive[key].copy() for key in archive.files}
        metadata = json.loads(DILATED_FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(DILATED_FIXTURE.read_bytes()).hexdigest() != metadata['fixture_sha256']:
            raise RuntimeError('dilated Conv1D fixture checksum mismatch')
        cls.metadata = metadata
        with np.load(DILATED_FIXTURE) as archive:
            cls.dilated = {key: archive[key].copy() for key in archive.files}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def ptr(array):
        return array.ctypes.data_as(F)

    def call(self, function, source, weight, bias, output, scratch,
             frames, kernel, stride, padding, dilation=1,
             *, output_elements=None, scratch_elements=None):
        args = [self.ptr(source), source.size, source.shape[1],
                self.ptr(weight), weight.size, self.ptr(bias), bias.size,
                self.ptr(output), output.size if output_elements is None else output_elements,
                output.shape[1], self.ptr(scratch),
                scratch.size if scratch_elements is None else scratch_elements,
                source.shape[0], output.shape[0], frames, kernel, stride,
                padding]
        if 'dilated' in function.__name__:
            args.append(dilation)
        args.append((frames + 2 * padding - (1 + (kernel - 1) * dilation)) // stride + 1)
        return function(*args)

    def require_fma(self):
        if not self.available():
            self.skipTest('x86 hardware FMA unavailable; scalar provider remains available')

    def test_runtime_guard_and_actual_instruction(self):
        machine = platform.machine().lower()
        if machine not in ('x86_64', 'amd64', 'i386', 'i686'):
            self.assertEqual(self.available(), 0)
        if not self.available():
            source = np.array([[2.]], np.float32)
            weight = np.array([[[3.]]], np.float32)
            bias = np.array([0.], np.float32)
            output = np.array([[-91.]], np.float32)
            scratch = np.empty((1, 1), np.float32)
            self.assertEqual(self.call(self.functions[
                'audio_conv1d_checked_channel_major_f32_hwfma'],
                source, weight, bias, output, scratch, 1, 1, 1, 0), -5)
            np.testing.assert_array_equal(output, np.array([[-91.]], np.float32))
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'audio.conv1d.hwfma.pinned-text-v1',
                'name': 'checked Conv1D hardware FMA versus pinned PyTorch',
                'status': 'not_tested', 'max_diff': None,
                'tolerance': None, 'reason': 'x86 FMA unavailable',
                'configuration': platform.machine()}))
            return
        if shutil.which('objdump'):
            disassembly = subprocess.run(['objdump', '-d', str(self.library_path)],
                check=True, capture_output=True, text=True).stdout
            section = disassembly.split('<audio_conv1d_compute_hwfma>:', 1)[1].split('\n\n', 1)[0]
            self.assertIn('vfmadd', section)
            self.assertNotIn('fmaf@plt', section)

    def test_pinned_text_conv_matches_scalar_and_pytorch(self):
        self.require_fma()
        ref = self.text
        source = np.full((512, 40), np.nan, np.float32)
        source[:, :36] = ref['embedding']
        weight, bias = ref['conv0_weight'], ref['conv0_bias']
        outputs = []
        for name in ('audio_conv1d_checked_channel_major_f32',
                     'audio_conv1d_checked_channel_major_f32_hwfma'):
            output = np.full((512, 40), -91., np.float32)
            scratch = np.empty((512, 36), np.float32)
            self.assertEqual(self.call(self.functions[name], source, weight,
                bias, output, scratch, 36, 5, 1, 2), 0)
            self.assertTrue(np.all(output[:, 36:] == -91.))
            outputs.append(output[:, :36].copy())
        np.testing.assert_array_equal(outputs[0], outputs[1])
        error = np.abs(outputs[1] - ref['conv0_output'])
        self.assertTrue(np.isfinite(outputs[1]).all())
        self.assertLessEqual(float(error.max()), 1.3e-5)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'audio.conv1d.hwfma.pinned-text-v1',
            'name': 'checked Conv1D hardware FMA versus pinned PyTorch',
            'provider': 'audio_conv1d_checked_channel_major_f32_hwfma',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-pytorch-2.8.0+cpu plus scalar-C exact comparison',
            'status': 'pass', 'max_diff': float(error.max()),
            'tolerance': 1.3e-5, 'scalar_changed_elements': 0,
            'configuration': '512x36 to 512x36, kernel 5, padded stride 40',
            'reproduction_command': 'python3 -m unittest tests.test_v8_audio_conv_hwfma_oracle'}))

    def test_varied_shapes_strides_tails_and_capacity(self):
        self.require_fma()
        scalar = self.functions['audio_conv1d_checked_channel_major_f32']
        hw = self.functions['audio_conv1d_checked_channel_major_f32_hwfma']
        rng = np.random.default_rng(740)
        for ic, oc, frames, kernel, step, padding in (
                (1, 1, 1, 1, 1, 0), (3, 5, 7, 3, 2, 1),
                (7, 9, 13, 5, 1, 2), (2, 4, 8, 2, 2, 0)):
            out_frames = (frames + 2 * padding - kernel) // step + 1
            source = np.full((ic, frames + 2), np.nan, np.float32)
            source[:, :frames] = rng.normal(size=(ic, frames))
            weight = rng.normal(size=(oc, ic, kernel)).astype(np.float32)
            bias = rng.normal(size=oc).astype(np.float32)
            results = []
            for function in (scalar, hw):
                output = np.full((oc, out_frames + 3), -91., np.float32)
                scratch = np.empty((oc, out_frames), np.float32)
                if function is hw:
                    self.assertEqual(self.call(function, source, weight, bias,
                        output, scratch, frames, kernel, step, padding,
                        output_elements=(oc - 1) * output.shape[1] +
                            out_frames - 1), -2)
                    self.assertTrue(np.all(output == -91.))
                    self.assertEqual(self.call(function, source, weight, bias,
                        output, scratch, frames, 0, step, padding), -1)
                    self.assertTrue(np.all(output == -91.))
                self.assertEqual(self.call(function, source, weight, bias,
                    output, scratch, frames, kernel, step, padding), 0)
                self.assertTrue(np.all(output[:, out_frames:] == -91.))
                results.append(output[:, :out_frames].copy())
            np.testing.assert_array_equal(results[0], results[1])
            oracle = np.empty((oc, out_frames), np.float64)
            for out_channel in range(oc):
                for frame in range(out_frames):
                    oracle[out_channel, frame] = float(bias[out_channel]) + sum(
                        float(source[in_channel, frame * step + tap - padding]) *
                        float(weight[out_channel, in_channel, tap])
                        for in_channel in range(ic) for tap in range(kernel)
                        if 0 <= frame * step + tap - padding < frames)
            # The independent FP64 sum has a different reduction precision;
            # the two FP32 providers must still agree elementwise exactly.
            np.testing.assert_allclose(results[1], oracle, rtol=0, atol=5e-6)

    def test_dilated_pytorch_cases_rejection_and_recovery(self):
        self.require_fma()
        scalar = self.functions['audio_conv1d_dilated_checked_channel_major_f32']
        hw = self.functions['audio_conv1d_dilated_checked_channel_major_f32_hwfma']
        for index, geometry in enumerate(self.metadata['cases']):
            source = self.dilated[f'case{index}_input']
            weight = self.dilated[f'case{index}_weight']
            bias = self.dilated[f'case{index}_bias']
            expected = self.dilated[f'case{index}_output']
            frames = source.shape[1]
            padded = np.full((source.shape[0], frames + 3), np.nan, np.float32)
            padded[:, :frames] = source
            results = []
            for function in (scalar, hw):
                output = np.full((expected.shape[0], frames + 4), -91., np.float32)
                scratch = np.empty(expected.shape, np.float32)
                if function is hw:
                    self.assertEqual(self.call(function, padded, weight, bias,
                        output, scratch, frames, geometry['kernel'], 1,
                        geometry['padding'], geometry['dilation'],
                        scratch_elements=scratch.size - 1), -2)
                    self.assertTrue(np.all(output == -91.))
                self.assertEqual(self.call(function, padded, weight, bias,
                    output, scratch, frames, geometry['kernel'], 1,
                    geometry['padding'], geometry['dilation']), 0)
                self.assertTrue(np.all(output[:, frames:] == -91.))
                results.append(output[:, :frames].copy())
            np.testing.assert_array_equal(results[0], results[1])
            error = np.abs(results[1] - expected)
            self.assertTrue(np.isfinite(results[1]).all())
            self.assertLessEqual(float(error.max()), 3e-5)

    def test_overflow_rejection_and_independent_concurrent_calls(self):
        self.require_fma()
        function = self.functions['audio_conv1d_checked_channel_major_f32_hwfma']
        source = np.array([[1e30, 1e30, 1e30]], np.float32)
        weight = np.array([[[1e30]]], np.float32)
        bias = np.zeros(1, np.float32)
        def run(valid):
            local_source = source.copy()
            local_weight = weight.copy()
            if valid:
                local_source.fill(2.)
                local_weight.fill(3.)
            output = np.full((1, 3), -91., np.float32)
            scratch = np.empty((1, 3), np.float32)
            status = self.call(function, local_source, local_weight, bias,
                output, scratch, 3, 1, 1, 0)
            return status, output
        status, output = run(False)
        self.assertEqual(status, -4)
        np.testing.assert_array_equal(output, np.full((1, 3), -91., np.float32))
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = list(pool.map(run, (True, True)))
        for status, output in (a, b):
            self.assertEqual(status, 0)
            np.testing.assert_array_equal(output, np.full((1, 3), 6., np.float32))


if __name__ == '__main__':
    unittest.main()
