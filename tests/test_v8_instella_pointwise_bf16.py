"""Pinned Instella layer-0 pointwise storage boundaries from publisher BF16."""

import ctypes
import copy
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/instella_pointwise0"
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


class InstellaPointwiseBf16Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The AVX-512 provider needs PyTorch's SLEEF symbol; do not silently
        # certify a different exponential implementation on that host.
        if "avx512f" in Path("/proc/cpuinfo").read_text() and not os.environ.get("CK_SLEEF_LIBRARY"):
            import torch
            candidate = Path(torch.__file__).resolve().parent / "lib/libtorch_cpu.so"
            if not candidate.is_file():
                raise RuntimeError("AVX-512 BF16 parity requires CK_SLEEF_LIBRARY")
            os.environ["CK_SLEEF_LIBRARY"] = str(candidate)
        subprocess.run(
            ["make", "--no-print-directory", "build/libckernel_engine.so"],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
        )
        cls.engine = ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"))

    def setUp(self):
        provenance = json.loads((FIXTURE / "provenance.json").read_text())
        payload = (FIXTURE / "captured_layer0.npz").read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), provenance["fixture_sha256"])
        with np.load(FIXTURE / "captured_layer0.npz") as data:
            self.capture = {name: np.ascontiguousarray(data[name]) for name in data.files}

    @staticmethod
    def ptr(array):
        return array.ctypes.data_as(FLOAT_PTR)

    def test_attention_gate_stores_publisher_bf16_result(self):
        x = self.capture["context"]
        gate = self.capture["attention_gate"]
        expected = self.capture["publisher_gated_context"]
        actual = np.empty_like(expected)
        kernel = self.engine.attn_gate_sigmoid_mul_pytorch_bf16_storage
        kernel.argtypes = [FLOAT_PTR] * 3 + [ctypes.c_int] * 3
        kernel(self.ptr(x), self.ptr(gate), self.ptr(actual), 1, 16, 128)
        np.testing.assert_array_equal(actual, expected)

        fp32 = np.empty_like(expected)
        old = self.engine.attn_gate_sigmoid_mul_forward
        old.argtypes = kernel.argtypes
        old(self.ptr(x), self.ptr(gate), self.ptr(fp32), 1, 16, 128)
        self.assertFalse(np.array_equal(fp32, expected))

    def test_swiglu_stores_publisher_bf16_intermediate_and_product(self):
        gate = self.capture["mlp_gate"]
        up = self.capture["mlp_up"]
        packed = np.concatenate((gate, up)).astype(np.float32, copy=False)
        expected = self.capture["publisher_swiglu"]
        actual = np.empty_like(expected)
        kernel = self.engine.swiglu_forward_pytorch_bf16_storage
        kernel.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int, ctypes.c_int]
        kernel(self.ptr(packed), self.ptr(actual), 1, gate.size)
        np.testing.assert_array_equal(actual, expected)

        fp32 = np.empty_like(expected)
        old = self.engine.swiglu_forward
        old.argtypes = kernel.argtypes
        old(self.ptr(packed), self.ptr(fp32), 1, gate.size)
        self.assertFalse(np.array_equal(fp32, expected))

    def test_circuit_requires_both_numerical_contracts(self):
        circuit = json.loads((ROOT / "version/v8/circuits/instella_moe.json").read_text())
        expected = {
            "decoder.attn_gate_bf16_pytorch": (
                "attn_gate_sigmoid_mul", "attn_gate_sigmoid_mul_pytorch_bf16_storage",
                "attention_gate_pytorch_sleef_bf16_storage"),
            "decoder.swiglu_bf16_pytorch": (
                "silu_mul", "swiglu_forward_pytorch_bf16_storage",
                "swiglu_pytorch_sleef_bf16_intermediate_bf16_output"),
        }
        for name, (op, kernel, contract) in expected.items():
            with self.subTest(operation=name):
                self.assertEqual(circuit["kernels"][op], kernel)
                phases = circuit["required_numerical_contracts"][name]["phases"]
                self.assertEqual(phases["prefill"]["contract_id"], contract)
                self.assertEqual(phases["decode"]["contract_id"], contract)

    def test_phase_specific_attention_ports_fail_closed(self):
        sys.path.insert(0, str(ROOT / "version/v8/scripts"))
        from build_ir_v8 import _template_graph_slots

        circuit = json.loads((ROOT / "version/v8/circuits/instella_moe.json").read_text())
        ops = circuit["block_types"]["decoder"]["body"]["ops_by_kind"]["mla_gated_farskip_dense_mlp"]
        attention = next(
            op for op in ops
            if op["op"] == "mla_attention"
        )
        prefill = _template_graph_slots(attention, "prefill")
        decode = _template_graph_slots(attention, "decode")
        self.assertEqual(set(prefill["inputs"]), {"q", "k", "v"})
        self.assertEqual(set(decode["inputs"]), {"q", "k_cache", "v_cache"})
        self.assertEqual(decode["inputs"]["k_cache"], "external:kv_cache_k_layer")
        with self.assertRaisesRegex(RuntimeError, "missing 'unsupported' graph slots"):
            _template_graph_slots(attention, "unsupported")
        conflicting = copy.deepcopy(attention)
        conflicting["graph_slots"] = prefill
        with self.assertRaisesRegex(RuntimeError, "graph_slots conflicts"):
            _template_graph_slots(conflicting, "prefill")


if __name__ == "__main__":
    unittest.main()
