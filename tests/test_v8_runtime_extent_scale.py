"""Checked value-dependent extent multiplication, including hostile dimensions."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimeExtentScaleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name) / 'extent.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-shared', '-fPIC', '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/runtime_extent.c'), '-o', str(path)],
            check=True, capture_output=True)
        cls.function = ctypes.CDLL(str(path)).ck_runtime_scale_i32_checked
        cls.function.argtypes = [ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32)]
        cls.function.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_valid_and_rejected_extents(self):
        cases = [(0, 2, 0, 0), (36, 2, 72, 0), (103, 2, 206, 0),
                 (103, 2, 205, -2),
                 (5, 0, 100, -1), (2**30, 2, 2**31 - 1, -3),
                 (1, 2**63, 2**31 - 1, -3)]
        for source, factor, capacity, status in cases:
            with self.subTest(source=source, factor=factor, capacity=capacity):
                output = ctypes.c_int32(-999)
                self.assertEqual(self.function(source, factor, capacity,
                    ctypes.byref(output)), status)
                self.assertEqual(output.value,
                    source * factor if status == 0 else -999)


if __name__ == '__main__':
    unittest.main()
