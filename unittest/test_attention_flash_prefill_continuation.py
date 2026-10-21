"""A history-aware query must retain the selected flash prefill reduction."""

import ctypes

import numpy as np

from lib_loader import load_lib


FLOAT_P = ctypes.POINTER(ctypes.c_float)


def _ptr(array):
    return array.ctypes.data_as(FLOAT_P)


def test_flash_prefill_continuation_matches_last_full_query():
    lib = load_lib("libckernel_engine.so")
    full = lib.attention_forward_causal_head_major_gqa_flash_strided
    full.argtypes = [FLOAT_P] * 4 + [ctypes.c_int] * 6
    full.restype = None
    continuation = lib.attention_forward_decode_head_major_gqa_flash_prefill_phase
    continuation.argtypes = [FLOAT_P] * 4 + [ctypes.c_int] * 6
    continuation.restype = None

    rng = np.random.default_rng(833)
    heads, kv_heads, tokens, dim, capacity = 4, 2, 129, 16, 137
    queries = rng.standard_normal((heads, tokens, dim), dtype=np.float32)
    keys = rng.standard_normal((kv_heads, capacity, dim), dtype=np.float32)
    values = rng.standard_normal((kv_heads, capacity, dim), dtype=np.float32)
    values[1] += np.float32(3.0)
    full_output = np.zeros_like(queries)
    full(_ptr(queries), _ptr(keys), _ptr(values), _ptr(full_output),
         heads, kv_heads, tokens, dim, dim, capacity)

    query = np.ascontiguousarray(queries[:, -1, :])
    output = np.zeros_like(query)
    continuation(_ptr(query), _ptr(keys), _ptr(values), _ptr(output),
                 heads, kv_heads, tokens, capacity, dim, dim)
    assert np.array_equal(output.view(np.uint32),
                          full_output[:, -1, :].view(np.uint32))

    wrong_position = np.zeros_like(query)
    continuation(_ptr(query), _ptr(keys), _ptr(values), _ptr(wrong_position),
                 heads, kv_heads, tokens - 1, capacity, dim, dim)
    assert not np.array_equal(wrong_position, output)
