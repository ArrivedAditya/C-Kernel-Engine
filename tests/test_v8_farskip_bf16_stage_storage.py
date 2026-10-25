"""Independent BF16 storage reference for two-stream shared SwiGLU."""

import ctypes
import json
import os
import subprocess
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FPTR = ctypes.POINTER(ctypes.c_float)
U16PTR = ctypes.POINTER(ctypes.c_uint16)


def bf16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def bits(values):
    return (bf16(values).view(np.uint32) >> 16).astype(np.uint16)


def reference(hidden, routed, residual, gate, up, down):
    rows, width = hidden.shape
    main = np.empty((rows, width), np.float32)
    routed_free = np.empty_like(main)
    for row in range(rows):
        x = bf16(hidden[row])
        gate_value = bf16(gate @ x)
        up_value = bf16(up @ x)
        silu = bf16(gate_value / (1.0 + np.exp(-gate_value)))
        activation = bf16(silu * up_value)
        shared = bf16(down @ activation)
        mlp = bf16(bf16(routed[row]) + shared)
        routed_free[row] = bf16(bf16(residual[row]) + shared)
        main[row] = bf16(bf16(residual[row]) + mlp)
    return main, routed_free


class FarSkipBf16StageStorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "avx512f" in Path("/proc/cpuinfo").read_text() and not os.environ.get("CK_SLEEF_LIBRARY"):
            import torch
            lib = Path(torch.__file__).resolve().parent / "lib/libtorch_cpu.so"
            if not lib.is_file():
                raise RuntimeError("AVX-512 BF16 parity requires CK_SLEEF_LIBRARY")
            os.environ["CK_SLEEF_LIBRARY"] = str(lib)
        subprocess.run(
            ["make", "--no-print-directory", "build/libckernel_engine.so"],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
        )
        cls.engine = ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"))

    def test_shared_stages_and_two_residuals(self):
        rng = np.random.default_rng(922)
        rows, width, intermediate = 3, 7, 9
        hidden = np.ascontiguousarray(rng.normal(0.0, 0.8, (rows, width)), np.float32)
        routed = np.ascontiguousarray(rng.normal(0.0, 0.2, (rows, width)), np.float32)
        residual = np.ascontiguousarray(rng.normal(0.0, 0.7, (rows, width)), np.float32)
        gate = bf16(rng.normal(0.0, 0.5, (intermediate, width)))
        up = bf16(rng.normal(0.0, 0.5, (intermediate, width)))
        down = bf16(rng.normal(0.0, 0.4, (width, intermediate)))
        expected_main, expected_free = reference(
            hidden, routed, residual, gate, up, down)
        gate_bits, up_bits, down_bits = bits(gate), bits(up), bits(down)

        def call(name, alias_main=False):
            kernel = getattr(self.engine, name)
            kernel.argtypes = [FPTR, FPTR, FPTR, U16PTR, U16PTR, U16PTR,
                               FPTR, FPTR, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int]
            residual_arg = residual.copy() if alias_main else residual
            main = residual_arg if alias_main else np.empty_like(expected_main)
            routed_free = np.empty_like(expected_free)
            kernel(hidden.ctypes.data_as(FPTR), routed.ctypes.data_as(FPTR),
                   residual_arg.ctypes.data_as(FPTR), gate_bits.ctypes.data_as(U16PTR),
                   up_bits.ctypes.data_as(U16PTR), down_bits.ctypes.data_as(U16PTR),
                   main.ctypes.data_as(FPTR), routed_free.ctypes.data_as(FPTR),
                   rows, width, intermediate)
            return main, routed_free

        actual_main, actual_free = call(
            "farskip_swiglu_shared_combine_pytorch_bf16_storage")
        np.testing.assert_array_equal(actual_main, expected_main)
        np.testing.assert_array_equal(actual_free, expected_free)
        np.testing.assert_array_equal(actual_main, bf16(actual_main))
        np.testing.assert_array_equal(actual_free, bf16(actual_free))
        alias_main, alias_free = call(
            "farskip_swiglu_shared_combine_pytorch_bf16_storage",
            alias_main=True)
        np.testing.assert_array_equal(alias_main, expected_main)
        np.testing.assert_array_equal(alias_free, expected_free)

        # Selecting the old FP32-intermediate contract must trip this fixture.
        old_main, old_free = call("farskip_swiglu_shared_combine_bf16")
        self.assertTrue(np.any(old_main != expected_main))
        self.assertTrue(np.any(old_free != expected_free))

    def test_explicit_candidate_does_not_change_current_circuit(self):
        circuit = json.loads((ROOT / "version/v8/circuits/instella_moe.json").read_text())
        selected = circuit["kernels"]["farskip_routed_shared_combine"]
        self.assertEqual(selected, "farskip_swiglu_shared_combine_bf16")
        candidate = "farskip_swiglu_shared_combine_pytorch_bf16_storage"
        provider = json.loads((ROOT / "version/v8/kernel_maps" /
                               (candidate + ".json")).read_text())
        self.assertEqual(provider["selection"]["status"], "candidate")
        self.assertEqual(provider["numerical_contract"],
                         "farskip_shared_swiglu_pytorch_bf16_storage_two_stream")
        self.assertNotEqual(provider["numerical_contract"],
                            "instella_farskip_bf16_shared_swiglu_fp32_two_stream")


if __name__ == "__main__":
    unittest.main()
