"""Independent PyTorch and rejection tests for checked dilated Conv1D."""

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/tts/audio_dilated_conv_pytorch28_reference.npz'
FLOAT = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class CheckedDilatedConvOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.functions = []
        for flag in ('-O0', '-O3'):
            library = Path(cls.temporary.name) / f'dilated_conv_{flag}.so'
            subprocess.run(['cc', '-std=c11', flag, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC', '-I',
                str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_conv1d_checked.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            function = ctypes.CDLL(str(library)).audio_conv1d_dilated_checked_channel_major_f32
            function.argtypes = [FLOAT, SIZE, SIZE, FLOAT, SIZE, FLOAT,
                SIZE, FLOAT, SIZE, SIZE, FLOAT, SIZE] + [SIZE] * 8
            function.restype = ctypes.c_int
            cls.functions.append(function)
        cls.metadata = json.loads(FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != cls.metadata['fixture_sha256']:
            raise RuntimeError('dilated Conv1D fixture checksum mismatch')
        with np.load(FIXTURE) as archive:
            cls.arrays = {key: archive[key].copy() for key in archive.files}
        for name, array in cls.arrays.items():
            if hashlib.sha256(array.tobytes()).hexdigest() != cls.metadata['array_sha256'][name]:
                raise RuntimeError(f'dilated Conv1D array checksum mismatch: {name}')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def pointer(array):
        return array.ctypes.data_as(FLOAT)

    def call(self, function, source, weight, bias, output, scratch,
             frames, kernel, dilation, padding, *,
             input_capacity=None, output_capacity=None,
             scratch_capacity=None):
        return function(self.pointer(source), source.size if input_capacity is None else input_capacity,
            source.shape[1], self.pointer(weight), weight.size,
            self.pointer(bias), bias.size, self.pointer(output),
            output.size if output_capacity is None else output_capacity,
            output.shape[1], self.pointer(scratch),
            scratch.size if scratch_capacity is None else scratch_capacity,
            source.shape[0], output.shape[0], frames, kernel, 1,
            padding, dilation, frames)

    def test_pinned_pytorch_cases_padding_and_optimization(self):
        self.assertEqual(self.metadata['torch'], '2.8.0+cpu')
        for index, geometry in enumerate(self.metadata['cases']):
            inputs = self.arrays[f'case{index}_input']
            weight = self.arrays[f'case{index}_weight']
            bias = self.arrays[f'case{index}_bias']
            expected = self.arrays[f'case{index}_output']
            channels, frames = inputs.shape
            source = np.full((channels, frames + 3), np.nan, np.float32)
            source[:, :frames] = inputs
            outputs = []
            for variant, function in enumerate(self.functions):
                output = np.full((expected.shape[0], frames + 4), -91., np.float32)
                scratch = np.empty(expected.shape, np.float32)
                self.assertEqual(self.call(function, source, weight, bias,
                    output, scratch, frames, geometry['kernel'],
                    geometry['dilation'], geometry['padding']), 0)
                self.assertTrue(np.isfinite(output[:, :frames]).all())
                self.assertTrue(np.all(output[:, frames:] == -91.))
                error = np.abs(output[:, :frames] - expected)
                worst = np.unravel_index(np.argmax(error), error.shape)
                maximum = float(error[worst])
                self.assertLessEqual(maximum, 3e-5,
                    (index, variant, worst, float(output[worst]),
                     float(expected[worst])))
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': f'audio.dilated-conv.pytorch28.case{index}.{("o0", "o3")[variant]}',
                    'name': 'checked dilated Conv1D versus committed PyTorch 2.8',
                    'provider': 'audio_conv1d_dilated_checked_channel_major_f32',
                    'oracle': 'committed-pytorch-' + self.metadata['torch'],
                    'status': 'pass', 'max_diff': maximum, 'tolerance': 3e-5,
                    'rmse': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                    'worst_index': list(map(int, worst)),
                    'actual': float(output[worst]),
                    'reference': float(expected[worst]),
                    'configuration': str(geometry) + '; padded input/output strides',
                    'reproduction_command': 'python3 -m unittest ' + self.id()}))
                outputs.append(output)
            np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_rejection_preserves_output_and_recovery(self):
        source = self.arrays['case0_input']
        weight = self.arrays['case0_weight']
        bias = self.arrays['case0_bias']
        expected = self.arrays['case0_output']
        frames = source.shape[1]
        for function in self.functions:
            output = np.full(expected.shape, -91., np.float32)
            scratch = np.empty_like(output)
            for kind in ('short_input', 'short_output', 'short_scratch',
                         'zero_dilation', 'wrong_padding', 'nonfinite_input',
                         'nonfinite_weight', 'overflow_result'):
                x, w = source.copy(), weight.copy()
                input_capacity = output_capacity = scratch_capacity = None
                dilation, padding = 2, 2
                if kind == 'short_input': input_capacity = 20
                if kind == 'short_output': output_capacity = output.size - 1
                if kind == 'short_scratch': scratch_capacity = scratch.size - 1
                if kind == 'zero_dilation': dilation = 0
                if kind == 'wrong_padding': padding = 0
                if kind == 'nonfinite_input': x[-1, -1] = np.nan
                if kind == 'nonfinite_weight': w[-1, -1, -1] = np.inf
                if kind == 'overflow_result':
                    x[:] = np.float32(3e38)
                    w[:] = np.float32(3e38)
                self.assertNotEqual(self.call(function, x, w, bias, output,
                    scratch, frames, 3, dilation, padding,
                    input_capacity=input_capacity,
                    output_capacity=output_capacity,
                    scratch_capacity=scratch_capacity), 0, kind)
                self.assertTrue(np.all(output == -91.), kind)
            self.assertEqual(self.call(function, source, weight, bias, output,
                scratch, frames, 3, 2, 2), 0)
            np.testing.assert_allclose(output, expected, rtol=0, atol=3e-5)

    def test_null_overlap_hostile_geometry_and_repeated_lengths(self):
        weight = self.arrays['case0_weight']
        bias = self.arrays['case0_bias']
        function = self.functions[1]
        source = np.full((3, 9), np.nan, np.float32)
        output = np.full((4, 9), -91., np.float32)
        scratch = np.empty((4, 7), np.float32)
        first = None
        for frames in (7, 5, 7):
            source[:, :frames] = np.arange(3 * frames,
                dtype=np.float32).reshape(3, frames) / 8 - 1
            output[:] = -91.
            self.assertEqual(self.call(function, source, weight, bias,
                output, scratch, frames, 3, 2, 2), 0)
            self.assertTrue(np.all(output[:, frames:] == -91.))
            expected = np.empty((4, frames), np.float32)
            for oc in range(4):
                for frame in range(frames):
                    expected[oc, frame] = bias[oc] + sum(
                        float(source[ic, frame + 2 * tap - 2]) *
                        float(weight[oc, ic, tap])
                        for ic in range(3) for tap in range(3)
                        if 0 <= frame + 2 * tap - 2 < frames)
            np.testing.assert_allclose(output[:, :frames], expected,
                                       rtol=0, atol=3e-5)
            if frames == 7:
                if first is None: first = output.copy()
                else: np.testing.assert_array_equal(output, first)
        output[:] = -91.
        args = (self.pointer(weight), weight.size, self.pointer(bias),
                bias.size, self.pointer(output), output.size, 9,
                self.pointer(scratch), scratch.size, 3, 4, 7, 3, 1, 2, 2, 7)
        self.assertNotEqual(function(FLOAT(), source.size, 9, *args), 0)
        self.assertTrue(np.all(output == -91.))
        self.assertNotEqual(function(self.pointer(source), source.size, 9,
            FLOAT(), weight.size, *args[2:]), 0)
        self.assertTrue(np.all(output == -91.))
        self.assertNotEqual(function(self.pointer(source), source.size, 9,
            *args[:-2], SIZE(-2).value, 7), 0)
        self.assertTrue(np.all(output == -91.))
        backing = np.arange(70, dtype=np.float32)
        overlapping_output = backing[1:37].reshape(4, 9)
        before = overlapping_output.copy()
        self.assertNotEqual(self.call(function, backing[:27].reshape(3, 9),
            weight, bias, overlapping_output, scratch, 7, 3, 2, 2), 0)
        np.testing.assert_array_equal(overlapping_output, before)
        self.assertNotEqual(function(self.pointer(source), source.size, 9,
            self.pointer(weight), weight.size, self.pointer(bias), bias.size,
            self.pointer(output), output.size, 9,
            self.pointer(output), output.size, 3, 4, 7, 3, 1, 2, 2, 7), 0)
        self.assertTrue(np.all(output == -91.))

    def test_live_pytorch_when_available(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'audio.dilated-conv.live-pytorch',
                'name': 'live PyTorch checked dilated Conv1D comparison',
                'provider': 'audio_conv1d_dilated_checked_channel_major_f32',
                'oracle': 'live-pytorch', 'status': 'not_tested',
                'reason': 'PyTorch unavailable'}))
            self.skipTest('live PyTorch unavailable; committed fixture runs')
        source = self.arrays['case1_input']
        weight = self.arrays['case1_weight']
        bias = self.arrays['case1_bias']
        expected = F.conv1d(torch.from_numpy(source[None]),
            torch.from_numpy(weight), torch.from_numpy(bias),
            padding=6, dilation=3)[0].numpy()
        output = np.empty_like(expected)
        scratch = np.empty_like(expected)
        self.assertEqual(self.call(self.functions[1], source, weight, bias,
            output, scratch, 31, 5, 3, 6), 0)
        error = float(np.max(np.abs(output - expected)))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'audio.dilated-conv.live-pytorch',
            'name': 'live PyTorch checked dilated Conv1D comparison',
            'provider': 'audio_conv1d_dilated_checked_channel_major_f32',
            'oracle': 'live-pytorch-' + torch.__version__,
            'status': 'pass' if error <= 3e-5 else 'fail',
            'max_diff': error, 'tolerance': 3e-5,
            'configuration': '17 input, 19 output channels; T31 K5 dilation3',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        self.assertLessEqual(error, 3e-5)

    def test_full_generator_geometry_live_pytorch_when_available(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'audio.dilated-conv.full-generator.live-pytorch',
                'name': 'full generator geometry checked dilated Conv1D',
                'provider': 'audio_conv1d_dilated_checked_channel_major_f32',
                'oracle': 'live-pytorch', 'status': 'not_tested',
                'reason': 'PyTorch unavailable'}))
            self.skipTest('live PyTorch unavailable; 256-channel committed fixture runs')
        torch.set_num_threads(1)
        rng = np.random.default_rng(51072)
        source = rng.normal(0, .05, (256, 2060)).astype(np.float32)
        weight = rng.normal(0, .01, (256, 256, 7)).astype(np.float32)
        bias = rng.normal(0, .01, 256).astype(np.float32)
        expected = F.conv1d(torch.from_numpy(source[None]),
            torch.from_numpy(weight), torch.from_numpy(bias),
            padding=15, dilation=5)[0].numpy()
        output = np.full((256, 2064), -91., np.float32)
        scratch = np.empty((256, 2060), np.float32)
        self.assertEqual(self.call(self.functions[1], source, weight, bias,
            output, scratch, 2060, 7, 5, 15), 0)
        self.assertTrue(np.isfinite(output[:, :2060]).all())
        self.assertTrue(np.all(output[:, 2060:] == -91.))
        error = np.abs(output[:, :2060] - expected)
        worst = np.unravel_index(np.argmax(error), error.shape)
        maximum = float(error[worst])
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'audio.dilated-conv.full-generator.live-pytorch',
            'name': 'full generator geometry checked dilated Conv1D',
            'provider': 'audio_conv1d_dilated_checked_channel_major_f32',
            'oracle': 'live-pytorch-' + torch.__version__,
            'status': 'pass' if maximum <= 3e-5 else 'fail',
            'max_diff': maximum, 'tolerance': 3e-5,
            'worst_index': list(map(int, worst)),
            'actual': float(output[worst]),
            'reference': float(expected[worst]),
            'configuration': '256 input/output channels; T2060 K7 dilation5 padding15',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        self.assertLessEqual(maximum, 3e-5)


if __name__ == '__main__':
    unittest.main()
