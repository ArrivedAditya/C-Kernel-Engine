#!/usr/bin/env python3
"""Exactness and capacity checks for planner-owned mega-attention scratch."""

from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libckernel_engine.so"
F32P = ctypes.POINTER(ctypes.c_float)


def pointer(value: np.ndarray | None):
    if value is None:
        return None
    return value.ctypes.data_as(F32P)


class MegaFusedAttentionWorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not LIB.is_file():
            raise unittest.SkipTest(f"missing {LIB}; build libckernel_engine.so first")
        cls.lib = ctypes.CDLL(str(LIB))
        pointer_args = [F32P] * 16
        scalar_args = [ctypes.c_int] * 8 + [ctypes.c_float]
        cls.lib.mega_fused_attention_decode.argtypes = pointer_args + scalar_args
        cls.lib.mega_fused_attention_decode.restype = None
        cls.lib.mega_fused_attention_decode_workspace.argtypes = (
            pointer_args
            + scalar_args
            + [F32P, ctypes.c_size_t, F32P, ctypes.c_size_t]
        )
        cls.lib.mega_fused_attention_decode_workspace.restype = None

    def fixture(self):
        rng = np.random.default_rng(20260814)
        embed = 64
        heads = 4
        kv_heads = 2
        head_dim = 16
        capacity = 8

        def values(count: int, scale: float = 0.04) -> np.ndarray:
            return (rng.standard_normal(count) * scale).astype(np.float32)

        return {
            "embed": embed,
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "capacity": capacity,
            "input": values(embed),
            "residual": values(embed),
            "gamma": np.ones(embed, dtype=np.float32),
            "wq": values(heads * head_dim * embed),
            "wk": values(kv_heads * head_dim * embed),
            "wv": values(kv_heads * head_dim * embed),
            "wo": values(heads * embed * head_dim),
        }

    @staticmethod
    def common_args(case, output, cache_k, cache_v):
        return [
            pointer(output), pointer(case["input"]), pointer(case["residual"]),
            pointer(case["gamma"]), pointer(case["wq"]), None,
            pointer(case["wk"]), None, pointer(case["wv"]), None,
            pointer(case["wo"]), None, pointer(cache_k), pointer(cache_v),
            None, None, 0, case["embed"], case["embed"], case["heads"],
            case["kv_heads"], case["head_dim"], case["head_dim"],
            case["capacity"], ctypes.c_float(1e-5),
        ]

    def test_workspace_is_byte_exact_with_compatibility_entry_point(self):
        case = self.fixture()
        cache_count = case["kv_heads"] * case["capacity"] * case["head_dim"]
        legacy_output = np.zeros(case["embed"], dtype=np.float32)
        workspace_output = np.zeros_like(legacy_output)
        legacy_k = np.zeros(cache_count, dtype=np.float32)
        legacy_v = np.zeros(cache_count, dtype=np.float32)
        workspace_k = np.zeros_like(legacy_k)
        workspace_v = np.zeros_like(legacy_v)

        self.lib.mega_fused_attention_decode(
            *self.common_args(case, legacy_output, legacy_k, legacy_v)
        )
        q_output = np.empty(2 * case["heads"] * case["head_dim"], dtype=np.float32)
        kv = np.empty(2 * case["kv_heads"] * case["head_dim"], dtype=np.float32)
        self.lib.mega_fused_attention_decode_workspace(
            *self.common_args(case, workspace_output, workspace_k, workspace_v),
            pointer(q_output), q_output.nbytes, pointer(kv), kv.nbytes,
        )

        self.assertEqual(legacy_output.tobytes(), workspace_output.tobytes())
        self.assertEqual(legacy_k.tobytes(), workspace_k.tobytes())
        self.assertEqual(legacy_v.tobytes(), workspace_v.tobytes())

    def test_undersized_workspace_fails_before_mutating_outputs(self):
        case = self.fixture()
        cache_count = case["kv_heads"] * case["capacity"] * case["head_dim"]
        output = np.full(case["embed"], 17.0, dtype=np.float32)
        cache_k = np.full(cache_count, 19.0, dtype=np.float32)
        cache_v = np.full(cache_count, 23.0, dtype=np.float32)
        q_output = np.empty(2 * case["heads"] * case["head_dim"], dtype=np.float32)
        kv = np.empty(2 * case["kv_heads"] * case["head_dim"], dtype=np.float32)

        self.lib.mega_fused_attention_decode_workspace(
            *self.common_args(case, output, cache_k, cache_v),
            pointer(q_output), q_output.nbytes - 4, pointer(kv), kv.nbytes,
        )

        self.assertTrue(np.all(output == np.float32(17.0)))
        self.assertTrue(np.all(cache_k == np.float32(19.0)))
        self.assertTrue(np.all(cache_v == np.float32(23.0)))


if __name__ == "__main__":
    unittest.main()
