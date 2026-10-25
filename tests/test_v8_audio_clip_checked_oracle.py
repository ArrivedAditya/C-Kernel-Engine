"""Reusable checked clip: independent NumPy, padding and rejection contracts."""

import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


ROOT = Path(__file__).resolve().parents[1]
FLOAT = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class CheckedClipOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.functions = []
        for flag in ("-O0", "-O3"):
            library = Path(cls.temporary.name) / f"clip_{flag}.so"
            subprocess.run(["cc", "-std=c11", flag, "-Wall", "-Wextra",
                            "-Werror", "-pedantic", "-shared", "-fPIC",
                            "-I", str(ROOT / "include"),
                            str(ROOT / "src/kernels/strided_unary_checked.c"),
                            "-lm", "-o", str(library)],
                           check=True, capture_output=True)
            function = ctypes.CDLL(str(library)).clip_strided_f32_checked
            function.argtypes = [FLOAT, SIZE, SIZE, FLOAT, SIZE, SIZE,
                                 SIZE, SIZE, ctypes.c_float, ctypes.c_float]
            function.restype = ctypes.c_int
            cls.functions.append(function)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def call(function, source, output, *, rows=2, columns=3,
             input_stride=5, output_stride=6,
             input_capacity=None, output_capacity=None,
             minimum=-1.0, maximum=1.0):
        return function(source.ctypes.data_as(FLOAT),
                        source.size if input_capacity is None else input_capacity,
                        input_stride, output.ctypes.data_as(FLOAT),
                        output.size if output_capacity is None else output_capacity,
                        output_stride, rows, columns, minimum, maximum)

    def test_numpy_padding_and_compiler_variants(self):
        source = np.full((2, 5), np.nan, np.float32)
        source[:, :3] = [[-10., -1., .25], [.75, 1., 10.]]
        expected = np.clip(source[:, :3], -1, 1)
        outputs = []
        for function in self.functions:
            output = np.full((2, 6), -91., np.float32)
            self.assertEqual(self.call(function, source, output), 0)
            np.testing.assert_array_equal(output[:, :3], expected)
            np.testing.assert_array_equal(output[:, 3:], -91.)
            outputs.append(output)
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_rejection_preserves_output_then_recovers(self):
        source = np.zeros((2, 5), np.float32)
        source[:, :3] = [[-3, 0, 3], [2, -2, 1]]
        for function in self.functions:
            output = np.full((2, 6), -91., np.float32)
            for kwargs in ({"input_capacity": 7}, {"output_capacity": 8},
                           {"minimum": 2, "maximum": 1},
                           {"minimum": np.nan},
                           {"input_stride": 2}, {"output_stride": 2}):
                self.assertNotEqual(self.call(function, source, output, **kwargs), 0)
                np.testing.assert_array_equal(output, -91.)
            source[1, 0] = np.inf
            self.assertNotEqual(self.call(function, source, output), 0)
            np.testing.assert_array_equal(output, -91.)
            source[1, 0] = 2
            self.assertNotEqual(self.call(function, source, source,
                                          output_stride=5), 0)
            self.assertEqual(self.call(function, source, output), 0)
            np.testing.assert_array_equal(output[:, :3],
                                          np.clip(source[:, :3], -1, 1))

    def test_model_neutral_generated_graph(self):
        op = {
            "id": "clip", "op": "clip_strided_checked",
            "kernel": "clip_strided_f32_checked", "returns_status": True,
            "graph_slots": {"inputs": {"input": "external:input"},
                            "outputs": {"output": "output"}},
            "params": {"R": 2, "C": 3, "minimum": -1.0, "maximum": 1.0,
                       "call_constants": {
                "clip_input_elements": 10, "clip_input_stride": 5,
                "clip_output_elements": 12, "clip_output_stride": 6,
                "clip_rows": 2, "clip_columns": 3}},
        }
        buffers = {"input": {"shape": [2, 5]}, "output": {"shape": [2, 6]}}
        circuit = {
            "version": 3, "name": "synthetic_checked_clip",
            "family": "bounded_graph", "checked_native_entry": True,
            "contract": {"runtime_invariants": {"inference_only": True}},
            "activation_buffers": buffers,
            "activation_bindings": {name: name for name in buffers},
            "native_entry": {"function": "ck_synthetic_checked_clip",
                             "params": [{"c_type": "uint8_t *", "name": "arena"},
                                        {"c_type": "size_t", "name": "arena_bytes"}],
                             "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
            "sequence": ["component"], "block_types": {"component": {
                "sequence": ["header", "body", "footer"],
                "header": [], "body": {"type": "dense", "ops": [op]},
                "footer": []}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "circuit.json"
            path.write_text(json.dumps(circuit))
            source = {"config": {
                "model": circuit["name"], "arch": circuit["name"],
                "num_layers": 1, "embed_dim": 3, "num_heads": 1,
                "num_kv_heads": 1, "head_dim": 3,
                "intermediate_size": 6, "context_length": 3,
                "max_seq_len": 3, "vocab_size": 1},
                "entries": [], "quant_summary": {}, "template": circuit}
            layout, calls, _library, _loaded, function = compile_native_graph(
                root, source, path)
            self.assertEqual(calls["errors"], [])
            self.assertEqual(calls["operations"][-1]["function"],
                             "clip_strided_f32_checked")
            function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            locations = {e["name"]: e for e in
                         layout["memory"]["activations"]["buffers"]}
            input_value = np.ndarray((2, 5), np.float32, buffer=arena,
                                     offset=locations["input"]["abs_offset"])
            output = np.ndarray((2, 6), np.float32, buffer=arena,
                                offset=locations["output"]["abs_offset"])
            input_value[:] = np.nan
            input_value[:, :3] = [[-2, -.5, .5], [2, 1, 0]]
            output[:] = -91
            self.assertEqual(function(arena, size), 0)
            np.testing.assert_array_equal(output[:, :3],
                                          np.clip(input_value[:, :3], -1, 1))
            np.testing.assert_array_equal(output[:, 3:], -91.)
            output[:] = -91
            self.assertNotEqual(function(arena, size - 1), 0)
            np.testing.assert_array_equal(output, -91.)
            input_value[0, 1] = np.inf
            self.assertNotEqual(function(arena, size), 0)
            np.testing.assert_array_equal(output, -91.)


if __name__ == "__main__":
    unittest.main()
