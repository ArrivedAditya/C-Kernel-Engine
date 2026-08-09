#!/usr/bin/env python3
"""Exact contracts for reusable multimodal stitch providers."""

from __future__ import annotations

import ctypes
import os
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(os.environ.get("CK_ENGINE_SO", str(ROOT / "build/libckernel_engine.so")))
FPTR = ctypes.POINTER(ctypes.c_float)
I32PTR = ctypes.POINTER(ctypes.c_int32)

LIB.ck_multimodal_prefix_insert_f32.argtypes = [
    FPTR, I32PTR, FPTR,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
]
LIB.ck_multimodal_prefix_insert_f32.restype = ctypes.c_int
LIB.ck_multimodal_mrope_positions_2d.argtypes = [
    I32PTR,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
LIB.ck_multimodal_mrope_positions_2d.restype = ctypes.c_int


class MultimodalBridgeKernelTests(unittest.TestCase):
    def test_prefix_insert_preserves_rows_and_destination_tail(self) -> None:
        source = np.arange(18, dtype=np.float32).reshape(3, 6)
        token_ids = np.full(5, 77, dtype=np.int32)
        decoder = np.full((5, 8), -3.0, dtype=np.float32)
        copied = LIB.ck_multimodal_prefix_insert_f32(
            source.ctypes.data_as(FPTR),
            token_ids.ctypes.data_as(I32PTR),
            decoder.ctypes.data_as(FPTR),
            3, 6, 8, 6, 1, 5,
        )
        self.assertEqual(copied, 3)
        np.testing.assert_array_equal(decoder[1:4, :6], source)
        np.testing.assert_array_equal(decoder[1:4, 6:], -3.0)
        np.testing.assert_array_equal(token_ids, [77, 0, 0, 0, 77])

    def test_prefix_insert_rejects_short_source_stride(self) -> None:
        source = np.zeros((1, 4), dtype=np.float32)
        token_ids = np.zeros(1, dtype=np.int32)
        decoder = np.zeros((1, 8), dtype=np.float32)
        rc = LIB.ck_multimodal_prefix_insert_f32(
            source.ctypes.data_as(FPTR),
            token_ids.ctypes.data_as(I32PTR),
            decoder.ctypes.data_as(FPTR),
            1, 4, 8, 6, 0, 1,
        )
        self.assertEqual(rc, -2)

    def test_mrope_positions_match_text_grid_text_contract(self) -> None:
        total_tokens = 10
        positions = np.full((4, total_tokens), -1, dtype=np.int32)
        resolved_text_pos = LIB.ck_multimodal_mrope_positions_2d(
            positions.ctypes.data_as(I32PTR),
            total_tokens,
            2,
            2,
            6,
            3,
            2,
            0,
        )
        self.assertEqual(resolved_text_pos, 5)
        expected = np.array(
            [
                [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],
                [0, 1, 2, 2, 2, 3, 3, 3, 5, 6],
                [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(positions, expected)

    def test_mrope_positions_honors_explicit_text_position(self) -> None:
        positions = np.empty((4, 3), dtype=np.int32)
        resolved = LIB.ck_multimodal_mrope_positions_2d(
            positions.ctypes.data_as(I32PTR), 3, 0, 0, 2, 2, 1, 9
        )
        self.assertEqual(resolved, 9)
        self.assertEqual(int(positions[0, 2]), 9)


if __name__ == "__main__":
    unittest.main()
