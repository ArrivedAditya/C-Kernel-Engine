"""Thread-count exactness for causal FP16-rounded K/V attention."""

import ctypes
import unittest

import numpy as np

from lib_loader import load_lib
from test_utils import numpy_to_ptr


lib = load_lib("libckernel_engine.so", "libckernel_attention.so")
lib.ck_set_num_threads.argtypes = [ctypes.c_int]
lib.ck_set_num_threads.restype = None

_signature = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
_workspace_signature = _signature + [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
serial = lib.attention_forward_causal_head_major_gqa_flash_strided_f16kv_serial
serial.argtypes = _signature
serial.restype = None
parallel = lib.attention_forward_causal_head_major_gqa_flash_strided_f16kv_workspace
parallel.argtypes = _workspace_signature
parallel.restype = None


class CausalF16KVParallelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(lib.ck_set_num_threads, ctypes.c_int(20))

    def test_parallel_heads_are_byte_exact(self) -> None:
        rng = np.random.default_rng(20260811)
        for tokens in (1, 7, 32, 65):
            for threads in (1, 4, 20, 24):
                with self.subTest(tokens=tokens, threads=threads):
                    heads, kv_heads, dim = 20, 4, 128
                    stride = tokens + 5
                    q = rng.standard_normal((heads, tokens, dim), dtype=np.float32)
                    k = rng.standard_normal((kv_heads, stride, dim), dtype=np.float32)
                    v = rng.standard_normal((kv_heads, stride, dim), dtype=np.float32)
                    expected = np.empty_like(q)
                    actual = np.empty_like(q)
                    workspace = np.empty(2 * kv_heads * stride * dim, dtype=np.float32)
                    args = (
                        numpy_to_ptr(q), numpy_to_ptr(k), numpy_to_ptr(v),
                        ctypes.c_int(heads), ctypes.c_int(kv_heads),
                        ctypes.c_int(tokens), ctypes.c_int(dim),
                        ctypes.c_int(dim), ctypes.c_int(stride),
                    )
                    serial(args[0], args[1], args[2], numpy_to_ptr(expected), *args[3:])
                    lib.ck_set_num_threads(ctypes.c_int(threads))
                    parallel(
                        args[0], args[1], args[2], numpy_to_ptr(actual), *args[3:],
                        numpy_to_ptr(workspace), ctypes.c_size_t(workspace.nbytes),
                    )
                    np.testing.assert_array_equal(
                        actual.view(np.uint32), expected.view(np.uint32)
                    )

    def test_nanbeige_512_token_shape_is_byte_exact(self) -> None:
        rng = np.random.default_rng(20260812)
        heads, kv_heads, tokens, dim = 20, 4, 512, 128
        stride = tokens + 8
        q = rng.standard_normal((heads, tokens, dim), dtype=np.float32)
        k = rng.standard_normal((kv_heads, stride, dim), dtype=np.float32)
        v = rng.standard_normal((kv_heads, stride, dim), dtype=np.float32)
        expected = np.empty_like(q)
        actual = np.empty_like(q)
        workspace = np.empty(2 * kv_heads * stride * dim, dtype=np.float32)
        args = (
            numpy_to_ptr(q), numpy_to_ptr(k), numpy_to_ptr(v),
            ctypes.c_int(heads), ctypes.c_int(kv_heads),
            ctypes.c_int(tokens), ctypes.c_int(dim),
            ctypes.c_int(dim), ctypes.c_int(stride),
        )
        serial(args[0], args[1], args[2], numpy_to_ptr(expected), *args[3:])
        lib.ck_set_num_threads(ctypes.c_int(20))
        parallel(
            args[0], args[1], args[2], numpy_to_ptr(actual), *args[3:],
            numpy_to_ptr(workspace), ctypes.c_size_t(workspace.nbytes),
        )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


if __name__ == "__main__":
    unittest.main()
