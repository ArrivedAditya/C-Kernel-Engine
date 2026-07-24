#!/usr/bin/env python3
"""Byte-exact PyTorch oracle for Qwen3-VL BF16 KV-cache storage."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(
    os.environ.get("CK_ENGINE_LIB", ROOT / "build" / "libckernel_engine.so")
)
LIB = ctypes.CDLL(str(LIB_PATH))
U16_P = ctypes.POINTER(ctypes.c_uint16)
FLOAT_P = ctypes.POINTER(ctypes.c_float)

STORE = LIB.kv_cache_store_bf16
STORE.argtypes = [
    U16_P, U16_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
STORE.restype = None

STORE_BATCH = LIB.kv_cache_store_batch_bf16
STORE_BATCH.argtypes = [
    U16_P, U16_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
STORE_BATCH.restype = None


def bf16_bits(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(values)).to(torch.bfloat16)
    return tensor.view(torch.uint16).numpy()


def assert_equal(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    if np.array_equal(actual, expected):
        return
    mismatch = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
    index = int(mismatch[0])
    raise AssertionError(
        f"{context}: {mismatch.size} byte-exact elements differ; "
        f"first={index} actual=0x{actual.reshape(-1)[index]:04x} "
        f"expected=0x{expected.reshape(-1)[index]:04x}"
    )


def main() -> int:
    rng = np.random.default_rng(20260723)
    heads, head_dim, capacity = 8, 128, 1609
    sentinel = np.uint16(0x7FC1)

    for pos in (0, 511, 512, 1058, 1307, 1608):
        k = rng.standard_normal((heads, head_dim), dtype=np.float32)
        v = rng.standard_normal((heads, head_dim), dtype=np.float32)
        k_cache = np.full((heads, capacity, head_dim), sentinel, dtype=np.uint16)
        v_cache = np.full_like(k_cache, sentinel)
        STORE(
            k_cache.ctypes.data_as(U16_P),
            v_cache.ctypes.data_as(U16_P),
            k.ctypes.data_as(FLOAT_P),
            v.ctypes.data_as(FLOAT_P),
            0, pos, heads, head_dim, capacity,
        )
        assert_equal(k_cache[:, pos], bf16_bits(k), f"single K pos={pos}")
        assert_equal(v_cache[:, pos], bf16_bits(v), f"single V pos={pos}")
        if pos:
            assert_equal(
                k_cache[:, pos - 1],
                np.full((heads, head_dim), sentinel, dtype=np.uint16),
                f"single previous row pos={pos}",
            )

    for start, tokens in ((0, 1), (33, 64), (511, 2), (1008, 266)):
        k = rng.standard_normal((heads, tokens, head_dim), dtype=np.float32)
        v = rng.standard_normal((heads, tokens, head_dim), dtype=np.float32)
        k_cache = np.full((heads, capacity, head_dim), sentinel, dtype=np.uint16)
        v_cache = np.full_like(k_cache, sentinel)
        STORE_BATCH(
            k_cache.ctypes.data_as(U16_P),
            v_cache.ctypes.data_as(U16_P),
            k.ctypes.data_as(FLOAT_P),
            v.ctypes.data_as(FLOAT_P),
            start, tokens, heads, head_dim, capacity,
        )
        assert_equal(
            k_cache[:, start:start + tokens],
            bf16_bits(k),
            f"batch K start={start} tokens={tokens}",
        )
        assert_equal(
            v_cache[:, start:start + tokens],
            bf16_bits(v),
            f"batch V start={start} tokens={tokens}",
        )

    print("BF16 KV-cache storage parity: 10/10 byte-exact PyTorch cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
