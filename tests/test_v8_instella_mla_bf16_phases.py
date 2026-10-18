"""Captured publisher BF16 MLA arithmetic at Instella decoder layer 11."""

import ctypes
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/instella_mla11"
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


class InstellaMlaBf16PhasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ["make", "--no-print-directory", "build/libckernel_engine.so"],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
        )
        cls.engine = ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"))
        cls.prefill_args = [FLOAT_PTR] * 4 + [ctypes.c_int] * 5 + [ctypes.c_float, FLOAT_PTR, ctypes.c_size_t]
        cls.decode_args = [FLOAT_PTR] * 4 + [ctypes.c_int] * 7 + [ctypes.c_float, FLOAT_PTR, ctypes.c_size_t]
        cls.prefill = cls.engine.deepseek_mla_attention_bf16_phases_workspace
        cls.prefill.argtypes = cls.prefill_args
        cls.prefill_parallel = cls.engine.deepseek_mla_attention_bf16_phases_parallel_dispatch
        cls.prefill_parallel.argtypes = cls.prefill_args
        cls.old_prefill = cls.engine.deepseek_mla_attention_f32_workspace
        cls.old_prefill.argtypes = cls.prefill_args
        cls.decode = cls.engine.deepseek_mla_attention_decode_bf16_phases_workspace
        cls.decode.argtypes = cls.decode_args

    def setUp(self):
        metadata = json.loads((FIXTURE / "provenance.json").read_text())
        fixture_bytes = (FIXTURE / "captured_head8.npz").read_bytes()
        self.assertEqual(hashlib.sha256(fixture_bytes).hexdigest(), metadata["fixture_sha256"])
        self.scale = metadata["scale"]
        with np.load(FIXTURE / "captured_head8.npz") as fixture:
            self.q = np.ascontiguousarray(fixture["q"][:, None, :])
            self.k = np.ascontiguousarray(fixture["k"][:, None, :])
            self.v = np.ascontiguousarray(fixture["v"][:, None, :])
            self.expected = np.ascontiguousarray(fixture["publisher_out"][:, None, :])

    @staticmethod
    def ptr(array):
        return array.ctypes.data_as(FLOAT_PTR)

    def run_prefill(self, function, q=None, k=None, v=None):
        q = self.q if q is None else q
        k = self.k if k is None else k
        v = self.v if v is None else v
        output = np.full_like(self.expected, np.nan)
        scores = np.empty(len(q), dtype=np.float32)
        function(self.ptr(q), self.ptr(k), self.ptr(v), self.ptr(output),
                 1, 1, len(q), q.shape[2], v.shape[2], self.scale,
                 self.ptr(scores), scores.nbytes)
        return output

    def test_captured_publisher_prefill_and_parallel_are_exact(self):
        np.testing.assert_array_equal(self.run_prefill(self.prefill), self.expected)
        np.testing.assert_array_equal(self.run_prefill(self.prefill_parallel), self.expected)
        old = self.run_prefill(self.old_prefill)
        self.assertGreater(float(np.max(np.abs(old[20] - self.expected[20]))), 0.01)

    def test_decode_matches_each_causal_prefill_prefix(self):
        length, _, dim = self.k.shape
        k_cache = np.ascontiguousarray(self.k.transpose(1, 0, 2))
        v_cache = np.ascontiguousarray(self.v.transpose(1, 0, 2))
        for pos in range(length):
            with self.subTest(pos=pos):
                output = np.full((1, dim), np.nan, dtype=np.float32)
                scores = np.empty(pos + 1, dtype=np.float32)
                self.decode(self.ptr(self.q[pos]), self.ptr(k_cache),
                            self.ptr(v_cache), self.ptr(output),
                            1, 1, pos + 1, dim, dim, length, dim,
                            self.scale, self.ptr(scores), scores.nbytes)
                np.testing.assert_array_equal(output, self.expected[pos])

    def test_cache_position_and_value_perturbations_are_detected(self):
        prefix = self.run_prefill(self.prefill)
        shifted = np.ascontiguousarray(np.roll(self.k, 1, axis=0))
        self.assertFalse(np.array_equal(self.run_prefill(self.prefill, k=shifted)[20], prefix[20]))
        perturbed = self.v.copy()
        perturbed[20, 0, 51] += 16.0
        self.assertFalse(np.array_equal(self.run_prefill(self.prefill, v=perturbed)[20], prefix[20]))

    def test_float_buffers_obey_bf16_operand_storage_boundary(self):
        q = self.q.copy()
        k = self.k.copy()
        v = self.v.copy()
        q[20, 0, 0] = np.nextafter(q[20, 0, 0], np.float32(np.inf))
        k[20, 0, 0] = np.nextafter(k[20, 0, 0], np.float32(np.inf))
        v[20, 0, 0] = np.nextafter(v[20, 0, 0], np.float32(np.inf))
        np.testing.assert_array_equal(self.run_prefill(self.prefill, q, k, v), self.expected)

    def test_instella_selects_declared_phase_providers(self):
        circuit = json.loads((ROOT / "version/v8/circuits/instella_moe.json").read_text())
        self.assertEqual(circuit["kernels"]["mla_attention_prefill"],
                         "deepseek_mla_attention_bf16_phases")
        self.assertEqual(circuit["kernels"]["mla_attention_decode"],
                         "deepseek_mla_attention_decode_bf16_phases")


if __name__ == "__main__":
    unittest.main()
