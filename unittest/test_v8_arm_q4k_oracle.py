"""Independent integer-dot oracle for the portable ARM Q4_K/Q8_K path."""

from __future__ import annotations

import ctypes
import platform
import struct
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"))
F32P = ctypes.POINTER(ctypes.c_float)
ARGS = [F32P, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]


def _q4_block(values: np.ndarray) -> bytes:
    packed = bytearray(144)
    struct.pack_into("<e", packed, 0, 1.0)
    # All eight scales are 1; all minimums are 0.
    packed[4:8] = b"\x01" * 4
    packed[12:16] = b"\x01" * 4
    for group in range(4):
        base = group * 64
        for i in range(32):
            packed[16 + group * 32 + i] = int(values[base + i]) | (int(values[base + 32 + i]) << 4)
    return bytes(packed)


def _q8_block(values: np.ndarray) -> bytes:
    packed = bytearray(292)
    struct.pack_into("<f", packed, 0, 0.125)
    packed[4:260] = values.astype(np.int8).tobytes()
    for i in range(16):
        struct.pack_into("<h", packed, 260 + 2 * i,
                         int(values[i * 16:(i + 1) * 16].sum()))
    return bytes(packed)


@unittest.skipUnless(platform.machine() in ("aarch64", "arm64"), "ARM portable path")
class ArmQ4KOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        for name in ("gemv_q4_k_q8_k", "gemv_q4_k_q8_k_ref"):
            getattr(LIB, name).argtypes = ARGS
        LIB.gemm_nt_q4_k_q8_k.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

    def test_gemv_gemm_and_row_tails(self) -> None:
        rng = np.random.default_rng(584)
        for rows, batch, k in ((1, 1, 256), (7, 3, 256), (3, 2, 512)):
            with self.subTest(rows=rows, batch=batch, k=k):
                weights = rng.integers(0, 16, (rows, k), dtype=np.int32)
                inputs = rng.integers(-16, 17, (batch, k), dtype=np.int32)
                wbuf = ctypes.create_string_buffer(b"".join(
                    _q4_block(weights[row, block:block + 256])
                    for row in range(rows) for block in range(0, k, 256)
                ))
                xbuf = ctypes.create_string_buffer(b"".join(
                    _q8_block(inputs[n, block:block + 256])
                    for n in range(batch) for block in range(0, k, 256)
                ))
                expected = (inputs.astype(np.int64) @ weights.astype(np.int64).T) * 0.125
                actual = np.empty((batch, rows), dtype=np.float32)
                for n in range(batch):
                    xptr = ctypes.byref(xbuf, n * (k // 256) * 292)
                    for name in ("gemv_q4_k_q8_k", "gemv_q4_k_q8_k_ref"):
                        getattr(LIB, name)(actual[n].ctypes.data_as(F32P), wbuf, xptr, rows, k)
                        np.testing.assert_array_equal(actual[n], expected[n])
                LIB.gemm_nt_q4_k_q8_k(xbuf, wbuf, None,
                                       actual.ctypes.data_as(F32P), batch, rows, k)
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
