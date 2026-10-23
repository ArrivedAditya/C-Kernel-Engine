"""Checked left zero padding against an independent NumPy reference."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ZeroPadOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "zero_pad.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        "-Iinclude", "-Isrc/kernels",
                        "src/kernels/audio_zero_pad_checked.c",
                        "-o", str(library)], cwd=ROOT, check=True)
        cls.loaded = ctypes.CDLL(str(library))
        cls.kernel = cls.loaded.audio_zero_pad1d_left_channel_major_f32_checked
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

    def test_reference_lengths_strides_and_padding(self):
        for frames, pad in ((1, 2), (3, 0), (3, 2), (7, 4)):
            with self.subTest(frames=frames, pad=pad):
                source = np.full((3, frames + 3), np.nan, np.float32)
                source[:, :frames] = np.arange(3 * frames,
                    dtype=np.float32).reshape(3, frames) / 7
                target = np.full((3, frames + pad + 5), -47., np.float32)
                self.assertEqual(self.call(source, target, 3, frames,
                                           pad, frames + pad), 0)
                expected = np.pad(source[:, :frames], ((0, 0), (pad, 0)),
                                  mode="constant")
                np.testing.assert_array_equal(target[:, :frames + pad], expected)
                np.testing.assert_array_equal(target[:, frames + pad:], -47.)

    def test_rejection_preserves_output_and_recovery(self):
        source = np.array([[1., 2., 3., np.nan],
                           [4., 5., 6., np.nan]], np.float32)
        target = np.full((2, 7), -47., np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        src = source.ctypes.data_as(pointer)
        dst = target.ctypes.data_as(pointer)
        for args, expected in (
            ((src, 8, 4, dst, 14, 7, 0, 3, 2, 5), -1),
            ((src, 8, 4, dst, 14, 7, 2, 3, 2, 6), -1),
            ((src, 6, 4, dst, 14, 7, 2, 3, 2, 5), -2),
            ((src, 8, 4, dst, 11, 7, 2, 3, 2, 5), -2),
            ((src, 8, 4, src, 14, 7, 2, 3, 2, 5), -1),
            ((pointer(), 8, 4, dst, 14, 7, 2, 3, 2, 5), -1),
            ((src, 8, 4, dst, 14, 7, 2, 3, ctypes.c_size_t(-1).value, 5), -4),
        ):
            with self.subTest(args=args):
                self.assertEqual(self.kernel(*args), expected)
                np.testing.assert_array_equal(target, -47.)
        source[0, 1] = np.nan
        self.assertEqual(self.call(source, target, 2, 3, 2, 5), -3)
        np.testing.assert_array_equal(target, -47.)
        source[0, 1] = 2.
        self.assertEqual(self.call(source, target, 2, 3, 2, 5), 0)
        np.testing.assert_array_equal(target[0, :5], [0., 0., 1., 2., 3.])

    def test_partial_overlap_rejected_without_mutation(self):
        shared = np.arange(32, dtype=np.float32)
        before = shared.copy()
        pointer = ctypes.POINTER(ctypes.c_float)
        shifted = ctypes.cast(shared.ctypes.data + 4, pointer)
        self.assertEqual(self.kernel(shared.ctypes.data_as(pointer), 8, 4,
                                     shifted, 14, 7, 2, 3, 2, 5), -1)
        np.testing.assert_array_equal(shared, before)


if __name__ == "__main__":
    unittest.main()
