"""Explicit-pool Conv1D channel partitioning versus serial AVX2 and pinned PyTorch."""
import ctypes
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/tts/audio_dilated_conv_pytorch28_reference.npz'
F = ctypes.POINTER(ctypes.c_float)
Z = ctypes.c_size_t
BASE = [F, Z, Z, F, Z, F, Z, F, Z, Z, F, Z] + [Z] * 8


class ParallelConvOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / 'conv_parallel.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-ffp-contract=off', '-shared', '-fPIC', '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/audio_conv1d_checked.c'),
            str(ROOT / 'src/kernels/audio_conv1d_parallel.c'),
            str(ROOT / 'src/ck_threadpool.c'),
            str(ROOT / 'src/ckernel_strict.c'), '-lpthread', '-lm', '-ldl',
            '-o', str(library)], check=True, capture_output=True, text=True)
        cls.native = ctypes.CDLL(str(library))
        cls.native.ck_threadpool_create.argtypes = [ctypes.c_int]
        cls.native.ck_threadpool_create.restype = ctypes.c_void_p
        cls.native.ck_threadpool_destroy.argtypes = [ctypes.c_void_p]
        cls.native.audio_conv1d_checked_avx2_fma_available.restype = ctypes.c_int
        cls.serial = cls.native.audio_conv1d_dilated_checked_channel_major_f32_avx2_fma
        cls.serial.argtypes = BASE
        cls.serial.restype = ctypes.c_int
        cls.parallel = cls.native.audio_conv1d_dilated_checked_channel_major_f32_avx2_parallel
        cls.parallel.argtypes = BASE + [ctypes.c_void_p, Z]
        cls.parallel.restype = ctypes.c_int
        ordinary_abi = BASE[:18] + BASE[19:]
        cls.ordinary_serial = cls.native.audio_conv1d_checked_channel_major_f32_avx2_fma
        cls.ordinary_serial.argtypes = ordinary_abi
        cls.ordinary_serial.restype = ctypes.c_int
        cls.ordinary_parallel = cls.native.audio_conv1d_checked_channel_major_f32_avx2_parallel
        cls.ordinary_parallel.argtypes = ordinary_abi + [ctypes.c_void_p, Z]
        cls.ordinary_parallel.restype = ctypes.c_int
        metadata = json.loads(FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != metadata['fixture_sha256']:
            raise RuntimeError('pinned PyTorch fixture checksum mismatch')
        cls.metadata = metadata
        with np.load(FIXTURE) as archive:
            cls.arrays = {key: archive[key].copy() for key in archive.files}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def pointer(array):
        return array.ctypes.data_as(F)

    def call(self, fn, pool, workers, source, weight, bias, output, scratch,
             padding, dilation, output_capacity=None, scratch_capacity=None):
        frames = source.shape[1]
        args = [self.pointer(source), source.size, source.shape[1],
                self.pointer(weight), weight.size, self.pointer(bias), bias.size,
                self.pointer(output), output.size if output_capacity is None else output_capacity,
                output.shape[1], self.pointer(scratch),
                scratch.size if scratch_capacity is None else scratch_capacity,
                source.shape[0], output.shape[0], frames, weight.shape[2], 1,
                padding, dilation, frames]
        if fn is self.parallel:
            args += [pool, workers]
        return fn(*args)

    def fixture_case(self, index, padded=0):
        prefix = f'case{index}_'
        source = self.arrays[prefix + 'input']
        if padded:
            physical = np.full((source.shape[0], source.shape[1] + padded), 123.0, np.float32)
            physical[:, :source.shape[1]] = source
            source = physical
        weight = self.arrays[prefix + 'weight']
        bias = self.arrays[prefix + 'bias']
        output = np.full((bias.size, self.arrays[prefix + 'output'].shape[1] + padded), -77.0, np.float32)
        scratch = np.full((bias.size, output.shape[1] - padded), 55.0, np.float32)
        return source, weight, bias, output, scratch

    def test_pinned_oracle_and_widths(self):
        if not self.native.audio_conv1d_checked_avx2_fma_available():
            self.skipTest('AVX2/FMA unavailable; unsupported path tested separately')
        pool = self.native.ck_threadpool_create(4)
        self.assertTrue(pool)
        try:
            maximum_error = 0.0
            for index in (0, 1):
                case = self.metadata['cases'][index]
                source, weight, bias, serial, scratch = self.fixture_case(index)
                self.assertEqual(self.call(self.serial, None, 0, source, weight, bias,
                    serial, scratch, case['padding'], case['dilation']), 0)
                oracle = self.arrays[f'case{index}_output']
                self.assertTrue(np.isfinite(serial).all())
                case_error = float(np.max(np.abs(serial - oracle)))
                self.assertLessEqual(case_error, 3e-5)
                maximum_error = max(maximum_error, case_error)
                for workers in (1, 2, 4):
                    output = np.full_like(serial, -77.0)
                    staged = np.empty_like(scratch)
                    self.assertEqual(self.call(self.parallel, pool, workers, source,
                        weight, bias, output, staged, case['padding'], case['dilation']), 0)
                    np.testing.assert_array_equal(output, serial)
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'audio.conv1d.parallel.pinned-v1',
                'name': 'explicit-pool Conv1D versus pinned PyTorch and serial AVX2/FMA',
                'provider': 'audio_conv1d_dilated_checked_channel_major_f32_avx2_parallel',
                'oracle': 'pinned PyTorch 2.8 plus exact serial AVX2/FMA',
                'configuration': 'two pinned dilated geometries; 1/2/4 workers',
                'max_diff': maximum_error, 'tolerance': 3e-5,
                'workers': [1, 2, 4], 'status': 'pass',
                'reproduction_command': 'python3 -m unittest tests.test_v8_audio_conv_parallel_oracle'}))
        finally:
            self.native.ck_threadpool_destroy(pool)

    def test_padding_rejection_and_recovery(self):
        if not self.native.audio_conv1d_checked_avx2_fma_available():
            self.skipTest('AVX2/FMA unavailable')
        pool = self.native.ck_threadpool_create(4)
        try:
            case = self.metadata['cases'][1]
            source, weight, bias, output, scratch = self.fixture_case(1, padded=3)
            # The valid width is 31; physical strides are 34.
            source_frames = case['frames']
            args = [self.pointer(source), source.size, source.shape[1],
                    self.pointer(weight), weight.size, self.pointer(bias), bias.size,
                    self.pointer(output), output.size, output.shape[1],
                    self.pointer(scratch), scratch.size,
                    source.shape[0], output.shape[0], source_frames, weight.shape[2],
                    1, case['padding'], case['dilation'], source_frames, pool, 4]
            self.assertEqual(self.parallel(*args), 0)
            self.assertTrue(np.all(output[:, source_frames:] == -77.0))
            success = output.copy()
            output.fill(-77.0)
            args[8] = output.size - output.shape[1]  # undersized physical span
            self.assertEqual(self.parallel(*args), -2)
            self.assertTrue(np.all(output == -77.0))
            args[8] = output.size
            args[11] = scratch.size - 1
            self.assertEqual(self.parallel(*args), -2)
            self.assertTrue(np.all(output == -77.0))
            args[11] = scratch.size
            bad_source = source.copy()
            bad_source[:, :source_frames] = 1e30
            bad_weight = weight.copy()
            bad_weight.fill(1e30)
            args[0] = self.pointer(bad_source)
            args[3] = self.pointer(bad_weight)
            self.assertEqual(self.parallel(*args), -4)
            self.assertTrue(np.all(output == -77.0))
            args[0] = self.pointer(source)
            args[3] = self.pointer(weight)
            self.assertEqual(self.parallel(*args), 0)
            np.testing.assert_array_equal(output, success)
        finally:
            self.native.ck_threadpool_destroy(pool)

    def test_concurrent_independent_calls(self):
        if not self.native.audio_conv1d_checked_avx2_fma_available():
            self.skipTest('AVX2/FMA unavailable')
        pool = self.native.ck_threadpool_create(4)
        case = self.metadata['cases'][1]
        def run(_):
            source, weight, bias, output, scratch = self.fixture_case(1)
            status = self.call(self.parallel, pool, 4, source, weight, bias, output,
                               scratch, case['padding'], case['dilation'])
            return status, output
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(run, range(8)))
            for status, output in results:
                self.assertEqual(status, 0)
                np.testing.assert_array_equal(output, results[0][1])
        finally:
            self.native.ck_threadpool_destroy(pool)

    def test_ordinary_stride_two_and_padded_layout(self):
        if not self.native.audio_conv1d_checked_avx2_fma_available():
            self.skipTest('AVX2/FMA unavailable')
        rng = np.random.default_rng(641)
        input_channels, output_channels = 2, 8
        input_frames, output_frames, kernel_size = 35, 18, 3
        source = np.full((input_channels, input_frames + 2), 211.0, np.float32)
        source[:, :input_frames] = rng.uniform(-0.2, 0.2,
            (input_channels, input_frames)).astype(np.float32)
        weight = rng.uniform(-0.2, 0.2,
            (output_channels, input_channels, kernel_size)).astype(np.float32)
        bias = rng.uniform(-0.2, 0.2, output_channels).astype(np.float32)
        serial = np.full((output_channels, output_frames + 3), -77.0, np.float32)
        parallel = serial.copy()
        scratch = np.empty((output_channels, output_frames), np.float32)
        args = [self.pointer(source), source.size, source.shape[1],
                self.pointer(weight), weight.size, self.pointer(bias), bias.size,
                self.pointer(serial), serial.size, serial.shape[1],
                self.pointer(scratch), scratch.size, input_channels,
                output_channels, input_frames, kernel_size, 2, 1, output_frames]
        pool = self.native.ck_threadpool_create(4)
        self.assertTrue(pool)
        try:
            self.assertEqual(self.ordinary_serial(*args), 0)
            args[7] = self.pointer(parallel)
            self.assertEqual(self.ordinary_parallel(*args, pool, 4), 0)
            np.testing.assert_array_equal(parallel, serial)
            self.assertTrue(np.all(parallel[:, output_frames:] == -77.0))
            expected = np.empty((output_channels, output_frames), np.float32)
            for oc in range(output_channels):
                for frame in range(output_frames):
                    total = float(bias[oc])
                    for ic in range(input_channels):
                        for tap in range(kernel_size):
                            position = frame * 2 + tap - 1
                            if 0 <= position < input_frames:
                                total += float(source[ic, position]) * float(weight[oc, ic, tap])
                    expected[oc, frame] = total
            np.testing.assert_allclose(parallel[:, :output_frames], expected,
                                       atol=2e-7, rtol=2e-6)
        finally:
            self.native.ck_threadpool_destroy(pool)

    def test_unsupported_or_invalid_pool(self):
        source, weight, bias, output, scratch = self.fixture_case(0)
        result = self.call(self.parallel, None, 2, source, weight, bias, output,
                           scratch, 2, 2)
        self.assertEqual(result, -1 if self.native.audio_conv1d_checked_avx2_fma_available() else -5)
        self.assertTrue(np.all(output == -77.0))


if __name__ == '__main__':
    unittest.main()
