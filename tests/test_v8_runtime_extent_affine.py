"""Checked affine valid lengths against Python integer arithmetic."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeExtentAffineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name) / 'extent.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-shared', '-fPIC', '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/runtime_extent.c'), '-o', str(path)],
            check=True, capture_output=True)
        cls.function = ctypes.CDLL(str(path)).ck_runtime_affine_i32_checked
        cls.function.argtypes = [ctypes.c_size_t] * 4 + [ctypes.POINTER(ctypes.c_int32)]
        cls.function.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_affine_bounds_and_unchanged_rejection(self):
        cases = [
            (0, 60, 1, 1, 0),
            (206, 60, 1, 12361, 0),
            (206, 60, 1, 12360, -2),
            (1, 0, 1, 100, -1),
            (2**31 - 1, 1, 1, 2**31 - 1, -3),
            (2**63, 60, 1, 2**31 - 1, -3),
            (1, 1, 0, 2**31, -3),
        ]
        for source, factor, offset, capacity, expected in cases:
            with self.subTest(source=source, factor=factor, offset=offset):
                output = ctypes.c_int32(-999)
                status = self.function(source, factor, offset, capacity,
                    ctypes.byref(output))
                self.assertEqual(status, expected)
                self.assertEqual(output.value,
                    source * factor + offset if status == 0 else -999)
        self.assertEqual(self.function(1, 2, 0, 2, None), -1)


if __name__ == '__main__':
    unittest.main()
