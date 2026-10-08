"""Independent geometry and rejection checks for checked left reflection."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class ReflectPadOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / 'reflect_pad.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-shared', '-fPIC',
                        '-Iinclude', '-Isrc/kernels',
                        'src/kernels/audio_reflect_pad_checked.c',
                        '-o', str(library)], cwd=ROOT, check=True)
        cls.loaded = ctypes.CDLL(str(library))
        cls.kernel = cls.loaded.audio_reflect_pad1d_left_channel_major_f32_checked
        pointer = ctypes.POINTER(ctypes.c_float)
        cls.kernel.argtypes = [pointer, ctypes.c_size_t, ctypes.c_size_t,
                            pointer, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_size_t]
        cls.kernel.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def call(self, source, target, channels, frames, pad, out_frames):
        pointer = ctypes.POINTER(ctypes.c_float)
        return self.kernel(source.ctypes.data_as(pointer), source.size,
                        source.shape[1], target.ctypes.data_as(pointer),
                        target.size, target.shape[1], channels, frames,
                        pad, out_frames)

    def test_numpy_reference_lengths_strides_and_padding(self):
        for frames, pad in ((2, 1), (3, 0), (3, 1), (7, 2), (12, 4)):
            with self.subTest(frames=frames, pad=pad):
                source = np.full((3, frames + 3), np.nan, np.float32)
                source[:, :frames] = np.arange(3 * frames,
                    dtype=np.float32).reshape(3, frames) / 7
                target = np.full((3, frames + pad + 5), -47., np.float32)
                self.assertEqual(self.call(source, target, 3, frames,
                                           pad, frames + pad), 0)
                expected = np.pad(source[:, :frames], ((0, 0), (pad, 0)),
                                  mode='reflect')
                np.testing.assert_array_equal(target[:, :frames + pad], expected)
                np.testing.assert_array_equal(target[:, frames + pad:], -47.)

    def test_rejections_preserve_output_and_recovery(self):
        source = np.array([[1., 2., 3., np.nan],
                           [4., 5., 6., np.nan]], np.float32)
        target = np.full((2, 6), -47., np.float32)
        for channels, frames, pad, out_frames, expected in (
                (0, 3, 1, 4, -1), (2, 1, 1, 2, -1),
                (2, 3, 3, 6, -1), (2, 3, 1, 5, -1),
                (2, 3, 1, 7, -1)):
            with self.subTest(channels=channels, frames=frames, pad=pad,
                              out_frames=out_frames):
                self.assertEqual(self.call(source, target, channels, frames,
                                           pad, out_frames), expected)
                self.assertTrue(np.all(target == -47.))
        source[0, 1] = np.nan
        self.assertEqual(self.call(source, target, 2, 3, 1, 4), -3)
        self.assertTrue(np.all(target == -47.))
        source[0, 1] = np.inf
        self.assertEqual(self.call(source, target, 2, 3, 1, 4), -3)
        self.assertTrue(np.all(target == -47.))
        source[0, 1] = 2.
        self.assertEqual(self.call(source, target, 2, 3, 1, 4), 0)
        np.testing.assert_array_equal(target[0, :4], [2., 1., 2., 3.])

    def test_second_generator_stage_geometry(self):
        channels, frames = 128, 12_360
        source = np.arange(channels * frames, dtype=np.float32).reshape(
            channels, frames) / 1000
        target = np.full((channels, frames + 4), -47., np.float32)
        self.assertEqual(self.call(source, target, channels, frames, 1,
                                   frames + 1), 0)
        np.testing.assert_array_equal(target[:, 0], source[:, 1])
        np.testing.assert_array_equal(target[:, 1:frames + 1], source)
        self.assertTrue(np.all(target[:, frames + 1:] == -47.))

    def test_capacity_overlap_null_and_hostile_geometry(self):
        source = np.arange(8, dtype=np.float32).reshape(2, 4)
        target = np.full((2, 5), -47., np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        src = source.ctypes.data_as(pointer)
        dst = target.ctypes.data_as(pointer)
        cases = [
            (src, 7, 4, dst, 10, 5, 2, 4, 1, 5, -2),
            (src, 8, 4, dst, 9, 5, 2, 4, 1, 5, -2),
            (src, 8, 4, src, 8, 4, 2, 4, 0, 4, -1),
            (pointer(), 8, 4, dst, 10, 5, 2, 4, 1, 5, -1),
            (src, 8, 4, dst, 10, 5, 2, 4, ctypes.c_size_t(-1).value, 5, -1),
            (src, 8, 4, dst, 10, 5, 2, ctypes.c_size_t(-1).value,
             1, 5, -4),
        ]
        for *args, expected in cases:
            self.assertEqual(self.kernel(*args), expected)
            self.assertTrue(np.all(target == -47.))
        shared = np.arange(32, dtype=np.float32)
        before = shared.copy()
        shifted = ctypes.cast(shared.ctypes.data + 4, pointer)
        self.assertEqual(self.kernel(shared.ctypes.data_as(pointer), 8, 4,
                                     shifted, 10, 5, 2, 4, 1, 5), -1)
        np.testing.assert_array_equal(shared, before)

    def test_live_pytorch_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest('live PyTorch unavailable')
        source = np.arange(18, dtype=np.float32).reshape(2, 9)
        target = np.full((2, 12), -47., np.float32)
        self.assertEqual(self.call(source, target, 2, 9, 2, 11), 0)
        expected = torch.nn.functional.pad(torch.from_numpy(source),
                                           (2, 0), mode='reflect').numpy()
        np.testing.assert_array_equal(target[:, :11], expected)


if __name__ == '__main__':
    unittest.main()
