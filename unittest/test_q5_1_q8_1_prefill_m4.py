#!/usr/bin/env python3
"""Exact-output coverage for the four-row Q5_1 prefill provider."""

from __future__ import annotations

import ctypes
import struct
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libckernel_engine.so"
QK = 32
BLOCK_BYTES = 24


def make_q5_1_weights(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = rows * (cols // QK)
    raw = bytearray(blocks * BLOCK_BYTES)
    for block in range(blocks):
        offset = block * BLOCK_BYTES
        struct.pack_into("<e", raw, offset, float(rng.uniform(0.001, 0.05)))
        struct.pack_into("<e", raw, offset + 2, float(rng.uniform(-0.1, 0.1)))
        raw[offset + 4 : offset + 8] = rng.integers(
            0, 256, size=4, dtype=np.uint8
        ).tobytes()
        raw[offset + 8 : offset + 24] = rng.integers(
            0, 256, size=16, dtype=np.uint8
        ).tobytes()
    return np.frombuffer(raw, dtype=np.uint8).copy()


@unittest.skipUnless(LIB_PATH.exists(), "build/libckernel_engine.so is required")
class Q51PrefillM4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(LIB_PATH))
        signature = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        cls.reference = cls.lib.gemm_nt_q5_1_q8_1
        cls.reference.argtypes = signature
        cls.reference.restype = None
        cls.candidate = cls.lib.gemm_nt_q5_1_q8_1_m4
        cls.candidate.argtypes = signature
        cls.candidate.restype = None

    def check_shape(self, rows: int, outputs: int, cols: int) -> None:
        rng = np.random.default_rng(1000 + rows * 100 + outputs)
        activations = rng.normal(0.0, 0.7, size=(rows, cols)).astype(np.float32)
        weights = make_q5_1_weights(outputs, cols, seed=2000 + outputs)
        bias = rng.normal(0.0, 0.1, size=outputs).astype(np.float32)
        expected = np.empty((rows, outputs), dtype=np.float32)
        actual = np.empty_like(expected)

        args = (
            activations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(weights.ctypes.data),
            bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        self.reference(
            *args,
            expected.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            outputs,
            cols,
        )
        self.candidate(
            *args,
            actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            outputs,
            cols,
        )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_exact_shapes_and_tails(self) -> None:
        for rows in (1, 3, 4, 5, 8):
            for outputs in (1, 7, 16):
                with self.subTest(rows=rows, outputs=outputs, cols=640):
                    self.check_shape(rows, outputs, 640)


if __name__ == "__main__":
    unittest.main()
