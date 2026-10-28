"""Expected-step equality with rejection before checked output publication."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class EqualI32ScalarCheckedTest(unittest.TestCase):
    def test_match_mismatch_capacity_alias_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "equal.so"
            subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                            "-I", str(ROOT / "include"),
                            str(ROOT / "src/kernels/equal_i32_scalar_checked.c"),
                            "-o", str(library)], check=True, capture_output=True)
            fn = ctypes.CDLL(str(library)).ck_equal_i32_scalar_checked
            ptr = ctypes.POINTER(ctypes.c_int32)
            fn.argtypes = [ptr, ctypes.c_size_t, ptr, ctypes.c_size_t,
                           ptr, ctypes.c_size_t]
            fn.restype = ctypes.c_int
            requested = np.array([1], dtype=np.int32)
            expected = np.array([1], dtype=np.int32)
            checked = np.array([-901], dtype=np.int32)
            a = requested.ctypes.data_as(ptr)
            b = expected.ctypes.data_as(ptr)
            c = checked.ctypes.data_as(ptr)
            self.assertEqual(fn(a, 1, b, 1, c, 1), 0)
            self.assertEqual(int(checked[0]), 1)
            checked[0] = -901
            requested[0] = 0
            self.assertNotEqual(fn(a, 1, b, 1, c, 1), 0)
            self.assertEqual(int(checked[0]), -901)
            requested[0] = 1
            self.assertNotEqual(fn(a, 0, b, 1, c, 1), 0)
            self.assertEqual(int(checked[0]), -901)
            self.assertNotEqual(fn(a, 1, b, 1, a, 1), 0)
            self.assertEqual(int(requested[0]), 1)
            self.assertNotEqual(fn(ptr(), 1, b, 1, c, 1), 0)
            self.assertEqual(int(checked[0]), -901)
            self.assertEqual(fn(a, 1, b, 1, c, 1), 0)
            self.assertEqual(int(checked[0]), 1)


if __name__ == "__main__":
    unittest.main()
