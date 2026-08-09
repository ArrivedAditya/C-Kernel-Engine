import ctypes
import math
import subprocess
import tempfile
import unittest
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DirectLayoutAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="cke-direct-attention-")
        cls._lib_path = Path(cls._tmp.name) / "libattention.so"
        subprocess.run([
            "gcc", "-std=gnu11", "-O2", "-shared", "-fPIC", "-mavx2", "-mfma", "-mf16c",
            "-I", str(ROOT / "include"),
            str(ROOT / "src" / "kernels" / "attention_kernels.c"),
            str(ROOT / "src" / "kernels" / "attention_kernels_sliding.c"),
            str(ROOT / "src" / "kernels" / "attention_flash_true.c"),
            str(ROOT / "src" / "kernels" / "softmax_kernels.c"),
            str(ROOT / "src" / "kernels" / "gemm_kernels_bf16.c"),
            str(ROOT / "src" / "ckernel_strict.c"),
            str(ROOT / "src" / "ck_threadpool.c"),
            "-lm", "-lpthread", "-o", str(cls._lib_path),
        ], check=True)
        cls._lib = ctypes.CDLL(str(cls._lib_path))
        signature = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        cls._lib.attention_forward_causal_head_major_gqa_flash_strided.argtypes = signature
        cls._lib.attention_forward_causal_head_major_gqa_flash_strided_token_output.argtypes = signature
        mixed_signature = signature + [ctypes.c_int, ctypes.c_int]
        cls._lib.attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4.argtypes = mixed_signature
        cls._lib.attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4_token_output.argtypes = mixed_signature

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_token_major_output_is_bit_exact_with_transposed_head_major_output(self):
        heads, kv_heads, tokens, dim = 4, 2, 11, 16
        count = heads * tokens * dim
        kv_count = kv_heads * tokens * dim
        q = array("f", (math.sin(index * 0.013) for index in range(count)))
        k = array("f", (math.cos(index * 0.017) for index in range(kv_count)))
        v = array("f", (math.sin(index * 0.019 + 0.3) for index in range(kv_count)))
        head_output = array("f", [0.0]) * count
        token_output = array("f", [0.0]) * count

        def pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        args = (pointer(q), pointer(k), pointer(v))
        self._lib.attention_forward_causal_head_major_gqa_flash_strided(
            *args, pointer(head_output), heads, kv_heads, tokens, dim, dim, tokens
        )
        self._lib.attention_forward_causal_head_major_gqa_flash_strided_token_output(
            *args, pointer(token_output), heads, kv_heads, tokens, dim, dim, tokens
        )

        expected = array("f", [0.0]) * count
        for token in range(tokens):
            for head in range(heads):
                src = (head * tokens + token) * dim
                dst = (token * heads + head) * dim
                expected[dst:dst + dim] = head_output[src:src + dim]
        self.assertEqual(expected.tobytes(), token_output.tobytes())

    def test_mixed_visual_attention_preserves_direct_output_layout(self):
        heads, kv_heads, tokens, dim = 4, 2, 9, 64
        count = heads * tokens * dim
        kv_count = kv_heads * tokens * dim
        q = array("f", (math.sin(index * 0.011) for index in range(count)))
        k = array("f", (math.cos(index * 0.023) for index in range(kv_count)))
        v = array("f", (math.sin(index * 0.029 + 0.2) for index in range(kv_count)))
        head_output = array("f", [0.0]) * count
        token_output = array("f", [0.0]) * count

        def pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        common = (
            pointer(q), pointer(k), pointer(v), heads, kv_heads, tokens,
            dim, dim, tokens, 2, 4,
        )
        self._lib.attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4(
            common[0], common[1], common[2], pointer(head_output), *common[3:]
        )
        self._lib.attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4_token_output(
            common[0], common[1], common[2], pointer(token_output), *common[3:]
        )

        expected = array("f", [0.0]) * count
        for token in range(tokens):
            for head in range(heads):
                src = (head * tokens + token) * dim
                dst = (token * heads + head) * dim
                expected[dst:dst + dim] = head_output[src:src + dim]
        self.assertEqual(expected.tobytes(), token_output.tobytes())


if __name__ == "__main__":
    unittest.main()
