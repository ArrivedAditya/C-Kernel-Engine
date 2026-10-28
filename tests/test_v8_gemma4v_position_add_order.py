"""Oracle-order and bounds checks for Gemma4V learned X/Y position additions."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


class Gemma4VPositionAddOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="gemma4v_position_add_")
        root = Path(cls.temp.name)
        exports = root / "exports.map"
        exports.write_text(
            "{ global: position_embeddings_add_gemma4v_xy; local: *; };\n",
            encoding="ascii",
        )
        library = root / "position_add.so"
        subprocess.run(
            [
                "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-shared", "-fPIC", "-ffunction-sections", "-fdata-sections",
                "-Wl,--gc-sections", "-Wl,--no-undefined",
                "-Wl,--version-script=" + str(exports),
                "-I", str(ROOT / "include"),
                str(ROOT / "src/kernels/vision_kernels.c"),
                "-lm", "-o", str(library),
            ],
            check=True, capture_output=True, text=True,
        )
        cls.library = ctypes.CDLL(str(library))
        cls.fn = cls.library.position_embeddings_add_gemma4v_xy
        cls.fn.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int]
        cls.fn.restype = None

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_separate_fp32_adds_and_grid_edges(self) -> None:
        height, width, dim, source_grid = 2, 3, 3, 2
        half_ulp = np.float32(2.0 ** -24)
        patches = np.ones((height, width, dim), dtype=np.float32)
        patches[0, 1, :] = np.float32(-1.0)
        tables = np.full((2, source_grid, dim), half_ulp, dtype=np.float32)
        tables[0, 1, :] = np.float32(-half_ulp)
        tables[1, 1, :] = np.float32(-half_ulp)
        expected = patches.copy()
        for row in range(height):
            for col in range(width):
                x = tables[0, min(col, source_grid - 1)]
                y = tables[1, min(row, source_grid - 1)]
                expected[row, col] = np.add(np.add(expected[row, col], x), y)
        grouped = np.add(patches[0, 0], np.add(tables[0, 0], tables[1, 0]))
        self.assertFalse(np.array_equal(grouped.view(np.uint32),
                                        expected[0, 0].view(np.uint32)))
        storage = np.concatenate((patches.ravel(), np.array([123.0], np.float32)))
        self.fn(storage.ctypes.data_as(FLOAT_PTR),
                tables.ctypes.data_as(FLOAT_PTR), height, width, dim, source_grid)
        np.testing.assert_array_equal(storage[:patches.size].view(np.uint32),
                                      expected.ravel().view(np.uint32))
        self.assertEqual(float(storage[-1]), 123.0)
        self.assertEqual(float(expected[0, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
