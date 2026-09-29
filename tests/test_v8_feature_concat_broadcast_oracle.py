"""Independent numerical and safety controls for checked row broadcast concat."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOAT = ctypes.c_float
POINTER = ctypes.POINTER(FLOAT)


class FeatureConcatBroadcastOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        exports = root / 'exports.map'
        exports.write_text('{ global: feature_concat_broadcast_rows_f32; local: *; };\n')
        library = root / 'broadcast.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-shared', '-fPIC', '-ffunction-sections', '-fdata-sections',
            '-Wl,--gc-sections', '-Wl,--no-undefined', '-Wl,--version-script=' + str(exports),
            '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/feature_concat_broadcast_rows.c'),
            str(ROOT / 'src/kernels/vision_kernels.c'), '-lm', '-o', str(library)],
            check=True, capture_output=True, text=True)
        cls.library = ctypes.CDLL(str(library))
        cls.fn = cls.library.feature_concat_broadcast_rows_f32
        cls.fn.argtypes = [POINTER, ctypes.c_size_t, ctypes.c_size_t,
                           POINTER, ctypes.c_size_t, POINTER, ctypes.c_size_t,
                           ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.c_size_t, ctypes.c_size_t]
        cls.fn.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def cases(rows, channels, style_channels, extra_input=0, extra_output=0):
        in_stride = channels + extra_input
        out_stride = channels + style_channels + extra_output
        input_data = np.full((rows, in_stride), np.nan, np.float32)
        values = np.arange(rows * channels, dtype=np.float32).reshape(rows, channels)
        input_data[:, :channels] = (values - 17) / np.float32(13)
        style = (np.arange(style_channels, dtype=np.float32) - 9) / np.float32(11)
        output = np.full((rows, out_stride), -777., np.float32)
        return input_data, style, output

    def run_case(self, input_data, style, output, channels, out_channels=None,
                 input_elements=None, feature_elements=None, output_elements=None):
        rows, input_stride = input_data.shape
        output_stride = output.shape[1]
        width = channels + len(style)
        return self.fn(input_data.ctypes.data_as(POINTER),
                       input_data.size if input_elements is None else input_elements,
                       input_stride, style.ctypes.data_as(POINTER),
                       style.size if feature_elements is None else feature_elements,
                       output.ctypes.data_as(POINTER),
                       output.size if output_elements is None else output_elements,
                       output_stride, rows, channels, len(style),
                       width if out_channels is None else out_channels)

    def test_numpy_exact_copy_and_padded_tails(self):
        for rows, channels, style_channels, ip, op in (
                (1, 1, 1, 0, 0), (2, 5, 3, 2, 4), (3, 7, 2, 1, 3),
                (36, 512, 128, 7, 5)):
            with self.subTest(shape=(rows, channels, style_channels, ip, op)):
                source, style, output = self.cases(rows, channels, style_channels, ip, op)
                original = output.copy()
                expected = np.concatenate((source[:, :channels],
                    np.broadcast_to(style, (rows, style_channels))), axis=1)
                for _ in range(3):
                    self.assertEqual(self.run_case(source, style, output, channels), 0)
                    np.testing.assert_array_equal(output[:, :expected.shape[1]].view(np.uint32),
                                                  expected.view(np.uint32))
                    np.testing.assert_array_equal(output[:, expected.shape[1]:],
                                                  original[:, expected.shape[1]:])
                self.assertTrue(np.isfinite(output[:, :expected.shape[1]]).all())

    def test_live_pytorch_expand_and_cat(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f'live PyTorch unavailable: {exc}')
        source, style, output = self.cases(36, 512, 128, 3, 2)
        expected = torch.cat((torch.from_numpy(source[:, :512].copy()),
            torch.from_numpy(style.copy()).expand(36, -1)), dim=1).numpy()
        self.assertEqual(self.run_case(source, style, output, 512), 0)
        np.testing.assert_array_equal(output[:, :640].view(np.uint32), expected.view(np.uint32))

    def test_capacity_geometry_nonfinite_and_output_preservation(self):
        source, style, output = self.cases(2, 5, 3, 2, 4)
        baseline = output.copy()
        tests = (
            (dict(input_elements=source.size - 5), -2),
            (dict(feature_elements=2), -2),
            (dict(output_elements=output.size - 5), -2),
            (dict(out_channels=7), -2),
            (dict(out_channels=9), -2),
        )
        for kwargs, status in tests:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.run_case(source, style, output, 5, **kwargs), status)
                np.testing.assert_array_equal(output, baseline)
        for slot in ('input', 'style'):
            changed = source.copy()
            voice = style.copy()
            (changed[1, 2:3] if slot == 'input' else voice[:1])[:] = np.nan
            self.assertEqual(self.run_case(changed, voice, output, 5), -3)
            np.testing.assert_array_equal(output, baseline)
        self.assertEqual(self.run_case(source, style, output, 5), 0)

    def test_alias_misalignment_and_hostile_dimensions(self):
        source, style, output = self.cases(2, 5, 3)
        before = output.copy()
        ptr = output.ctypes.data_as(POINTER)
        self.assertEqual(self.fn(ptr, 16, 8, style.ctypes.data_as(POINTER), 3,
            ptr, 16, 8, 2, 5, 3, 8), -1)
        np.testing.assert_array_equal(output, before)
        raw = (ctypes.c_uint8 * (output.nbytes + 4))()
        unaligned = ctypes.cast(ctypes.addressof(raw) + 1, POINTER)
        self.assertEqual(self.fn(source.ctypes.data_as(POINTER), 10, 5,
            style.ctypes.data_as(POINTER), 3, unaligned, 16, 8, 2, 5, 3, 8), -1)
        np.testing.assert_array_equal(output, before)
        max_size = ctypes.c_size_t(-1).value
        self.assertEqual(self.fn(source.ctypes.data_as(POINTER), max_size, max_size,
            style.ctypes.data_as(POINTER), 3, ptr, max_size, max_size,
            2, 5, 3, 8), -2)
        np.testing.assert_array_equal(output, before)
        self.assertEqual(self.fn(source.ctypes.data_as(POINTER), 10, 5,
            style.ctypes.data_as(POINTER), 3, ptr, 16, 8,
            2, 2**31, 3, 2**31+3), -2)
        np.testing.assert_array_equal(output, before)


if __name__ == '__main__':
    unittest.main()
