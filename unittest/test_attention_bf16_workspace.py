"""Planner-owned workspace tests for segmented BF16 cache attention."""

import ctypes

import numpy as np

from lib_loader import load_lib
from test_utils import numpy_to_ptr


lib = load_lib("libckernel_engine.so", "libckernel_attention.so")
FLOAT_P = ctypes.POINTER(ctypes.c_float)
U16_P = ctypes.POINTER(ctypes.c_uint16)
BASE_ARGS = [
    FLOAT_P, U16_P, U16_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
compatibility = (
    lib.attention_forward_causal_head_major_gqa_prefill_append_bf16cache_pytorch_contract
)
compatibility.argtypes = BASE_ARGS
compatibility.restype = ctypes.c_int
workspace_kernel = (
    lib.attention_forward_causal_head_major_gqa_prefill_append_bf16cache_pytorch_contract_workspace
)
workspace_kernel.argtypes = [*BASE_ARGS, FLOAT_P, ctypes.c_size_t]
workspace_kernel.restype = ctypes.c_int

CK_ATTN_REDUCTION_BF16_PYTORCH_SDPA = 4
CK_ATTENTION_STATUS_OK = 0
CK_ATTENTION_STATUS_INVALID_ARGUMENT = -1


def _bf16_storage(values: np.ndarray) -> np.ndarray:
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def test_bf16_append_workspace_matches_compatibility_entry() -> None:
    rng = np.random.default_rng(20260814)
    heads, kv_heads, tokens, capacity, dim = 4, 2, 3, 7, 16
    q = rng.standard_normal((heads, tokens, dim), dtype=np.float32)
    k = _bf16_storage(rng.standard_normal((kv_heads, capacity, dim), dtype=np.float32))
    v = _bf16_storage(rng.standard_normal((kv_heads, capacity, dim), dtype=np.float32))
    expected = np.zeros_like(q)
    actual = np.zeros_like(q)
    args = (
        numpy_to_ptr(q),
        k.ctypes.data_as(U16_P),
        v.ctypes.data_as(U16_P),
        numpy_to_ptr(expected),
        heads, kv_heads, tokens, 0, capacity, dim, dim,
        CK_ATTN_REDUCTION_BF16_PYTORCH_SDPA,
    )
    expected_status = compatibility(*args)
    token_workspace = np.empty((2, heads, dim), dtype=np.float32)
    workspace_args = (*args[:3], numpy_to_ptr(actual), *args[4:])
    actual_status = workspace_kernel(
        *workspace_args, numpy_to_ptr(token_workspace), token_workspace.nbytes
    )
    assert actual_status == expected_status
    if actual_status == CK_ATTENTION_STATUS_OK:
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_bf16_append_workspace_rejects_undersized_storage() -> None:
    heads, kv_heads, tokens, capacity, dim = 2, 1, 1, 1, 8
    q = np.zeros((heads, tokens, dim), dtype=np.float32)
    k = np.zeros((kv_heads, capacity, dim), dtype=np.uint16)
    v = np.zeros_like(k)
    output = np.zeros_like(q)
    workspace = np.empty((2, heads, dim), dtype=np.float32)
    status = workspace_kernel(
        numpy_to_ptr(q), k.ctypes.data_as(U16_P), v.ctypes.data_as(U16_P),
        numpy_to_ptr(output), heads, kv_heads, tokens, 0, capacity, dim, dim,
        CK_ATTN_REDUCTION_BF16_PYTORCH_SDPA,
        numpy_to_ptr(workspace), workspace.nbytes - ctypes.sizeof(ctypes.c_float),
    )
    assert status == CK_ATTENTION_STATUS_INVALID_ARGUMENT
