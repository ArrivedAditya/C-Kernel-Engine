from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class GeGLUParallelAliasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._old_num_threads = os.environ.get("CK_NUM_THREADS")
        os.environ["CK_NUM_THREADS"] = "8"
        subprocess.run(
            ["make", "--no-print-directory", "build/libckernel_engine.so"],
            cwd=ROOT,
            check=True,
        )
        cls._tmp = tempfile.TemporaryDirectory(prefix="ck_geglu_alias_")
        cls.library_path = Path(cls._tmp.name) / "libgeglu_parallel.so"
        subprocess.run(
            [
                os.environ.get("CC", "gcc"),
                "-shared",
                "-fPIC",
                "-O2",
                "-Iinclude",
                "-Iversion/v8/src",
                "-o",
                str(cls.library_path),
                "version/v8/src/ck_parallel_prefill_v8.c",
                "-Lbuild",
                "-lckernel_engine",
                "-lm",
                "-lpthread",
                f"-Wl,-rpath,{ROOT / 'build'}",
            ],
            cwd=ROOT,
            check=True,
        )
        ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL)
        cls.lib = ctypes.CDLL(str(cls.library_path))
        pointer = ctypes.POINTER(ctypes.c_float)
        cls.parallel = cls.lib.geglu_forward_exact_parallel_dispatch
        cls.parallel.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int]
        cls.serial = cls.lib.geglu_forward_exact
        cls.serial.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()
        if cls._old_num_threads is None:
            os.environ.pop("CK_NUM_THREADS", None)
        else:
            os.environ["CK_NUM_THREADS"] = cls._old_num_threads

    def test_in_place_parallel_dispatch_matches_serial_reference_repeatedly(self) -> None:
        tokens, dim = 19, 128
        rng = np.random.default_rng(7)
        source = rng.standard_normal((tokens, 2 * dim), dtype=np.float32)
        reference = np.empty((tokens, dim), dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        self.serial(
            source.ctypes.data_as(pointer),
            reference.ctypes.data_as(pointer),
            tokens,
            dim,
        )

        for _ in range(20):
            in_place = source.copy()
            self.parallel(
                in_place.ctypes.data_as(pointer),
                in_place.ctypes.data_as(pointer),
                tokens,
                dim,
            )
            np.testing.assert_array_equal(
                in_place.reshape(-1)[: tokens * dim],
                reference.reshape(-1),
            )


if __name__ == "__main__":
    unittest.main()
