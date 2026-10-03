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

    def test_gemma_vision_selects_quick_not_tanh(self) -> None:
        circuit = json.loads((ROOT / "version/v8/circuits/gemma4_vision.json").read_text())
        self.assertEqual(circuit["kernels"]["geglu"], "geglu_forward_quick")
        provider = json.loads((ROOT / "version/v8/kernel_maps/geglu_forward_quick.json").read_text())
        self.assertEqual(provider["operation_interface"], "geglu.fp32_compacting.v1")
        self.assertEqual(provider["impl"]["function"], "geglu_forward_quick")

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
