from __future__ import annotations

import ctypes
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class RoPESplitDirectParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._old_num_threads = os.environ.get("CK_NUM_THREADS")
        os.environ["CK_NUM_THREADS"] = "8"
        subprocess.run(
            ["make", "--no-print-directory", "build/libckernel_engine.so"],
            cwd=ROOT,
            check=True,
        )
        cls._tmp = tempfile.TemporaryDirectory(prefix="ck_rope_split_direct_")
        cls.library_path = Path(cls._tmp.name) / "librope_split_parallel.so"
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
        ctypes.CDLL(
            str(ROOT / "build/libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL
        )
        cls.lib = ctypes.CDLL(str(cls.library_path))
        pointer = ctypes.POINTER(ctypes.c_float)
        signature = [
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        cls.parallel = cls.lib.rope_forward_qk_split_direct_parallel_dispatch
        cls.parallel.argtypes = signature
        cls.serial = cls.lib.rope_forward_qk_split_direct_f32
        cls.serial.argtypes = signature

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()
        if cls._old_num_threads is None:
            os.environ.pop("CK_NUM_THREADS", None)
        else:
            os.environ["CK_NUM_THREADS"] = cls._old_num_threads

    def test_parallel_token_rows_match_scalar_reference_exactly(self) -> None:
        heads, kv_heads, tokens = 8, 2, 19
        head_dim, aligned_head_dim, rotary_dim = 112, 128, 96
        rng = np.random.default_rng(31)
        q_source = rng.standard_normal(
            (heads, tokens, aligned_head_dim), dtype=np.float32
        )
        k_source = rng.standard_normal(
            (kv_heads, tokens, aligned_head_dim), dtype=np.float32
        )
        factors = rng.uniform(0.75, 1.25, rotary_dim // 2).astype(np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)

        for use_factors in (0, 1):
            q_reference = q_source.copy()
            k_reference = k_source.copy()
            q_candidate = q_source.copy()
            k_candidate = k_source.copy()
            args = (
                factors.ctypes.data_as(pointer),
                use_factors,
                heads,
                kv_heads,
                tokens,
                head_dim,
                aligned_head_dim,
                37,
                rotary_dim,
                ctypes.c_float(1000000.0),
            )
            self.serial(
                q_reference.ctypes.data_as(pointer),
                k_reference.ctypes.data_as(pointer),
                *args,
            )
            self.parallel(
                q_candidate.ctypes.data_as(pointer),
                k_candidate.ctypes.data_as(pointer),
                *args,
            )
            np.testing.assert_array_equal(q_candidate, q_reference)
            np.testing.assert_array_equal(k_candidate, k_reference)

    def test_recursive_provider_matches_existing_llama_cache(self) -> None:
        p = ctypes.POINTER(ctypes.c_float)
        direct = self.lib.rope_forward_qk_split_llama_parallel_dispatch
        direct.argtypes = self.parallel.argtypes
        cache = self.lib.rope_precompute_cache_llama_cpu
        cache.argtypes = [p, p, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_float]
        cached = self.lib.rope_forward_qk_with_rotary_dim
        cached.argtypes = [p]*4 + [ctypes.c_int]*7
        rng = np.random.default_rng(47)
        for tokens, offset, base in ((1, 0, 10000.0), (26, 37, 10000.0),
                                     (19, 512, 1000000.0)):
            with self.subTest(tokens=tokens, offset=offset, base=base):
                q = rng.standard_normal((4, tokens, 256), dtype=np.float32)
                k = rng.standard_normal((1, tokens, 256), dtype=np.float32)
                qr, kr = q.copy(), k.copy()
                cos = np.zeros((offset + tokens, 128), np.float32)
                sin = np.zeros_like(cos)
                cache(cos.ctypes.data_as(p), sin.ctypes.data_as(p), offset+tokens,
                      256, base, 256, b"none", 1.0)
                cached(qr.ctypes.data_as(p), kr.ctypes.data_as(p), cos.ctypes.data_as(p),
                       sin.ctypes.data_as(p), 4, 1, tokens, 256, 256, offset, 256)
                direct(q.ctypes.data_as(p), k.ctypes.data_as(p), None, 0,
                       4, 1, tokens, 256, 256, offset, 256, base)
                np.testing.assert_array_equal(q.view(np.uint32), qr.view(np.uint32))
                np.testing.assert_array_equal(k.view(np.uint32), kr.view(np.uint32))

    def test_gemma4_selected_rope_matches_ggml_frequency_recurrence(self) -> None:
        circuit = json.loads((ROOT / "version/v8/circuits/gemma4.json").read_text())
        provider_id = circuit["kernels"]["rope_qk"]
        provider = json.loads((ROOT / "version/v8/kernel_maps" / f"{provider_id}.json").read_text())
        self.assertEqual(provider["operation_interface"], "rope.split_direct.llama_cpu.fp32.v1")
        selected = getattr(self.lib, provider["impl"]["function"])
        selected.argtypes = self.parallel.argtypes
        p = ctypes.POINTER(ctypes.c_float)
        cache = self.lib.rope_precompute_cache_llama_cpu
        cache.argtypes = [p, p, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_float]
        cached = self.lib.rope_forward_qk_with_rotary_dim
        cached.argtypes = [p] * 4 + [ctypes.c_int] * 7

        rng = np.random.default_rng(4104)
        for tokens, offset in ((91, 0), (14, 77)):
            with self.subTest(tokens=tokens, offset=offset):
                q_source = rng.standard_normal((8, tokens, 256), dtype=np.float32)
                k_source = rng.standard_normal((2, tokens, 256), dtype=np.float32)
                q_selected, k_selected = q_source.copy(), k_source.copy()
                q_expected, k_expected = q_source.copy(), k_source.copy()
                q_direct, k_direct = q_source.copy(), k_source.copy()
                cos = np.zeros((offset + tokens, 128), dtype=np.float32)
                sin = np.zeros_like(cos)
                cache(cos.ctypes.data_as(p), sin.ctypes.data_as(p),
                      offset + tokens, 256, 10000.0, 256, b"none", 1.0)
                cached(q_expected.ctypes.data_as(p), k_expected.ctypes.data_as(p),
                       cos.ctypes.data_as(p), sin.ctypes.data_as(p),
                       8, 2, tokens, 256, 256, offset, 256)
                args = (None, 0, 8, 2, tokens, 256, 256, offset, 256, 10000.0)
                selected(q_selected.ctypes.data_as(p), k_selected.ctypes.data_as(p), *args)
                self.serial(q_direct.ctypes.data_as(p), k_direct.ctypes.data_as(p), *args)
                np.testing.assert_array_equal(q_selected.view(np.uint32), q_expected.view(np.uint32))
                np.testing.assert_array_equal(k_selected.view(np.uint32), k_expected.view(np.uint32))
                self.assertTrue(np.any(q_direct.view(np.uint32) != q_expected.view(np.uint32)))

        q_only_id = circuit["kernels"]["rope_q"]
        q_only_provider = json.loads((ROOT / "version/v8/kernel_maps" / f"{q_only_id}.json").read_text())
        q_only = getattr(self.lib, q_only_provider["impl"]["function"])
        q_only.argtypes = [p, p] + [ctypes.c_int] * 7 + [ctypes.c_float]
        old_q_only = self.lib.rope_forward_q_split_direct_f32
        old_q_only.argtypes = q_only.argtypes
        source = rng.standard_normal((8, 14, 256), dtype=np.float32)
        selected_q, paired_q, direct_q = source.copy(), source.copy(), source.copy()
        paired_k = np.zeros((2, 14, 256), dtype=np.float32)
        q_only_args = (None, 0, 8, 14, 256, 256, 77, 256, 10000.0)
        q_only(selected_q.ctypes.data_as(p), *q_only_args)
        selected(paired_q.ctypes.data_as(p), paired_k.ctypes.data_as(p),
                 None, 0, 8, 2, 14, 256, 256, 77, 256, 10000.0)
        old_q_only(direct_q.ctypes.data_as(p), *q_only_args)
        np.testing.assert_array_equal(selected_q.view(np.uint32), paired_q.view(np.uint32))
        self.assertTrue(np.any(direct_q.view(np.uint32) != paired_q.view(np.uint32)))


if __name__ == "__main__":
    unittest.main()
