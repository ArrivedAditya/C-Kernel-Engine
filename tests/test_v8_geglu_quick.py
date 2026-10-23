from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class GeGLUQuickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            ["make", "--no-print-directory", "build/libckernel_engine.so"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        library = ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"))
        cls.quick = library.geglu_forward_quick
        pointer = ctypes.POINTER(ctypes.c_float)
        cls.quick.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int]
        cls.quick.restype = None
        cls.quick_fp16_table = library.geglu_forward_quick_fp16_table
        cls.quick_fp16_table.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int]
        cls.quick_fp16_table.restype = None

    def test_quick_formula_and_in_place_compaction(self) -> None:
        pointer = ctypes.POINTER(ctypes.c_float)
        rng = np.random.default_rng(665)
        for tokens, dim in ((1, 7), (3, 17), (19, 128)):
            source = rng.uniform(-5.0, 5.0, (tokens, 2 * dim)).astype(np.float32)
            gate, up = np.split(source, 2, axis=1)
            reference = gate / (1.0 + np.exp(-1.702 * gate)) * up
            separate = np.full((tokens, dim + 3), 91.0, dtype=np.float32)
            output = np.empty((tokens, dim), dtype=np.float32)
            self.quick(source.ctypes.data_as(pointer), output.ctypes.data_as(pointer), tokens, dim)
            np.testing.assert_allclose(output, reference, rtol=2e-6, atol=2e-6)

            in_place = np.concatenate((source.reshape(-1), separate.reshape(-1)))
            self.quick(in_place.ctypes.data_as(pointer), in_place.ctypes.data_as(pointer), tokens, dim)
            np.testing.assert_allclose(in_place[: tokens * dim].reshape(tokens, dim), reference, rtol=2e-6, atol=2e-6)
            np.testing.assert_array_equal(
                in_place[tokens * dim : tokens * 2 * dim],
                source.reshape(-1)[tokens * dim :],
            )
            np.testing.assert_array_equal(in_place[tokens * 2 * dim :], separate.reshape(-1))

    def test_pinned_quick_fp16_table_oracle_and_alias(self) -> None:
        pointer = ctypes.POINTER(ctypes.c_float)
        # Direct Gemma vision layer-0 values from llama.cpp 19e28a27702117d8f2eb16b825b9a308111f67d9.
        source = np.array([[
            4.466717720031738, -1.013685703277588, -0.940589427947998,
            4.46875, 1.1520159244537354, -1.130437970161438,
        ]], dtype=np.float32)
        expected = np.array([[
            19.934814453125, -0.17662744224071503, 0.17842483520507812,
        ]], dtype=np.float32)
        output = np.empty((1, 3), dtype=np.float32)
        self.quick_fp16_table(
            source.ctypes.data_as(pointer), output.ctypes.data_as(pointer), 1, 3
        )
        np.testing.assert_allclose(output, expected, rtol=0, atol=2e-6)

        unrounded = np.empty_like(output)
        self.quick(source.ctypes.data_as(pointer), unrounded.ctypes.data_as(pointer), 1, 3)
        self.assertGreater(abs(float(unrounded[0, 0] - expected[0, 0])), 0.01)

        for tokens, dim in ((1, 3), (3, 7), (19, 17)):
            rows = np.tile(source, (tokens, (dim + 2) // 3))[:, :2 * dim].copy()
            separate = np.empty((tokens, dim), dtype=np.float32)
            self.quick_fp16_table(
                rows.ctypes.data_as(pointer), separate.ctypes.data_as(pointer), tokens, dim
            )
            guarded = np.concatenate((rows.reshape(-1), np.full(8, 91.0, dtype=np.float32)))
            self.quick_fp16_table(
                guarded.ctypes.data_as(pointer), guarded.ctypes.data_as(pointer), tokens, dim
            )
            np.testing.assert_array_equal(guarded[:tokens * dim], separate.reshape(-1))
            np.testing.assert_array_equal(
                guarded[tokens * dim : tokens * 2 * dim], rows.reshape(-1)[tokens * dim :]
            )
            np.testing.assert_array_equal(guarded[-8:], np.full(8, 91.0, dtype=np.float32))

    def test_gemma_vision_selects_quick_fp16_table_not_unrounded(self) -> None:
        circuit = json.loads((ROOT / "version/v8/circuits/gemma4_vision.json").read_text())
        self.assertEqual(circuit["kernels"]["geglu"], "geglu_forward_quick_fp16_table")
        provider = json.loads((ROOT / "version/v8/kernel_maps/geglu_forward_quick_fp16_table.json").read_text())
        self.assertEqual(provider["operation_interface"], "geglu.fp32_compacting.v1")
        self.assertEqual(provider["impl"]["function"], "geglu_forward_quick_fp16_table")
        self.assertEqual(
            provider["memory_contract"]["input_output_overlap"],
            "exact_input_output_alias_or_disjoint_only",
        )

        sys.path.insert(0, str(ROOT / "version/v8/scripts"))
        import build_ir_v8

        self.assertEqual(
            build_ir_v8.find_kernel(
                build_ir_v8.load_kernel_registry(),
                "geglu",
                {"activation": "fp32", "output": "fp32"},
                mode="prefill",
                prefer_q8_activation=False,
            ),
            "geglu_forward_exact",
        )


if __name__ == "__main__":
    unittest.main()
