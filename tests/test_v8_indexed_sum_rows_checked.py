"""Model-neutral checked row selection and sum controls."""

from __future__ import annotations

import ctypes
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import build_ir_v8


class IndexedSumRowsCheckedTest(unittest.TestCase):
    def test_generic_generated_circuit(self):
        shapes = {"left": [2, 3], "right": [2, 3],
                  "left_indices": [3], "right_indices": [3],
                  "output": [3, 3]}
        graph = {
            "version": 3, "name": "indexed_sum_rows_synthetic",
            "family": "bounded_graph", "checked_native_entry": True,
            "contract": {"runtime_invariants": {
                "inference_only": True,
                "production_kernel_heap_allocation": False}},
            "activation_buffers": {key: {"shape": shape}
                                   for key, shape in shapes.items()},
            "activation_bindings": {key: key for key in shapes},
            "native_entry": {
                "function": "ck_indexed_sum_rows_synthetic",
                "params": [{"c_type": "uint8_t *", "name": "arena"},
                           {"c_type": "size_t", "name": "arena_bytes"}],
                "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
            "sequence": ["rows"],
            "block_types": {"rows": {
                "sequence": ["header", "body", "footer"], "header": [],
                "body": {"type": "dense", "ops": [{
                    "id": "mix", "op": "indexed_sum_rows_checked",
                    "kernel": "indexed_sum_rows_checked_f32", "returns_status": True,
                    "graph_slots": {"inputs": {
                        key: f"external:{key}" for key in (
                            "left", "right", "left_indices", "right_indices")},
                        "outputs": {"output": "output"}},
                    "params": {"L": 2, "K": 2, "R": 3, "C": 3,
                               "call_constants": {
                                   "left_elements": 6, "left_stride": 3,
                                   "left_rows": 2, "right_elements": 6,
                                   "right_stride": 3, "right_rows": 2,
                                   "left_index_elements": 3,
                                   "right_index_elements": 3,
                                   "output_elements": 9, "output_stride": 3,
                                   "rows": 3, "channels": 3}},
                }]}, "footer": []}},
        }
        source = {"config": {
            "model": graph["name"], "arch": graph["name"],
            "num_layers": 1, "embed_dim": 3, "num_heads": 1,
            "num_kv_heads": 1, "head_dim": 3, "intermediate_size": 3,
            "context_length": 3, "max_seq_len": 3, "vocab_size": 3,
            "activation_buffer_dtypes": {
                "left_indices": "i32", "right_indices": "i32"}},
            "entries": [], "quant_summary": {}, "template": graph}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            circuit = folder / "circuit.json"
            circuit.write_text(json.dumps(graph))
            registry = build_ir_v8.load_kernel_registry()
            with contextlib.redirect_stdout(io.StringIO()):
                ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode="prefill")
                lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
                layout = build_ir_v8.generate_memory_layout(
                    lower1, source, registry, mode="prefill", context_len=3)
                lower2 = build_ir_v8.generate_ir_lower_2(
                    lower1, layout, source, registry, mode="prefill")
                call = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
            self.assertFalse(call["errors"])
            self.assertEqual(call["operations"][0]["function"],
                             "indexed_sum_rows_checked_f32")
            (folder / "layout.json").write_text(json.dumps(layout))
            (folder / "call.json").write_text(json.dumps(call))
            generated = folder / "generated.c"
            subprocess.run([
                sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
                "--ir", str(folder / "call.json"), "--layout",
                str(folder / "layout.json"), "--output", str(generated),
            ], check=True, capture_output=True, text=True)
            library_path = folder / "generated.so"
            subprocess.run([
                "cc", "-std=c11", "-O2", "-shared", "-fPIC", str(generated),
                str(ROOT / "src/kernels/indexed_sum_rows_checked.c"),
                "-I", str(ROOT / "include"), "-lm", "-o", str(library_path),
            ], check=True, capture_output=True, text=True)
            loaded = ctypes.CDLL(str(library_path))
            fn = loaded.ck_indexed_sum_rows_synthetic
            fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
            fn.restype = ctypes.c_int
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(raw, (-ctypes.addressof(raw)) & 63)
            acts = {entry["name"]: entry for entry in
                    layout["memory"]["activations"]["buffers"]}

            def arr(name, dtype, shape):
                return np.ndarray(shape, dtype=dtype, buffer=arena,
                                  offset=acts[name]["abs_offset"])

            arr("left", np.float32, (2, 3))[:] = [[1, 2, 3], [4, 5, 6]]
            arr("right", np.float32, (2, 3))[:] = [[10, 20, 30], [40, 50, 60]]
            arr("left_indices", np.int32, (3,))[:] = [0, 1, -1]
            arr("right_indices", np.int32, (3,))[:] = [-1, 1, 0]
            output = arr("output", np.float32, (3, 3))
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, [[1, 2, 3], [44, 55, 66], [10, 20, 30]])
            output[:] = -777
            arr("left_indices", np.int32, (3,))[1] = 2
            self.assertNotEqual(fn(arena, len(arena)), 0)
            self.assertTrue(np.all(output == -777))
            arr("left_indices", np.int32, (3,))[1] = 1
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, [[1, 2, 3], [44, 55, 66], [10, 20, 30]])

    def test_oracle_padding_rejection_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            library_path = Path(temp) / "indexed.so"
            subprocess.run([
                "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-shared", "-fPIC", str(ROOT / "src/kernels/indexed_sum_rows_checked.c"),
                "-I", str(ROOT / "include"), "-lm", "-o", str(library_path),
            ], check=True, capture_output=True, text=True)
            library = ctypes.CDLL(str(library_path))
            fn = library.indexed_sum_rows_checked_f32
            f32 = ctypes.POINTER(ctypes.c_float)
            i32 = ctypes.POINTER(ctypes.c_int32)
            fn.argtypes = [f32, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                           f32, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                           i32, ctypes.c_size_t, i32, ctypes.c_size_t,
                           f32, ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.c_size_t, ctypes.c_size_t]
            fn.restype = ctypes.c_int
            left = np.array([[1, 2, 3, 71], [4, 5, 6, 72]], dtype=np.float32)
            right = np.array([[10, 20, 30, 73], [40, 50, 60, 74]], dtype=np.float32)
            li = np.array([0, 1, -1, 0], dtype=np.int32)
            ri = np.array([-1, 1, 0, 1], dtype=np.int32)
            output = np.full((4, 5), -777, dtype=np.float32)

            def invoke(*, left_elements=None, output_elements=None):
                return fn(left.ctypes.data_as(f32), left.size if left_elements is None else left_elements, 4, 2,
                          right.ctypes.data_as(f32), right.size, 4, 2,
                          li.ctypes.data_as(i32), li.size,
                          ri.ctypes.data_as(i32), ri.size,
                          output.ctypes.data_as(f32), output.size if output_elements is None else output_elements,
                          5, 4, 3)

            self.assertEqual(invoke(), 0)
            expected = np.stack([left[0, :3], left[1, :3] + right[1, :3],
                                 right[0, :3], left[0, :3] + right[1, :3]])
            np.testing.assert_array_equal(output[:, :3], expected)
            self.assertTrue(np.all(output[:, 3:] == -777))
            for bad in (-2, 2):
                output[:] = -777
                li[1] = bad
                self.assertNotEqual(invoke(), 0)
                self.assertTrue(np.all(output == -777))
            li[1] = 1
            output[:] = -777
            self.assertNotEqual(invoke(left_elements=6), 0)
            self.assertTrue(np.all(output == -777))
            self.assertNotEqual(invoke(output_elements=17), 0)
            self.assertTrue(np.all(output == -777))
            right[1, 0] = np.inf
            self.assertNotEqual(invoke(), 0)
            self.assertTrue(np.all(output == -777))
            right[1, 0] = 40
            self.assertEqual(invoke(), 0)
            np.testing.assert_array_equal(output[:, :3], expected)
            left[0, 0] = -0.0
            self.assertEqual(invoke(), 0)
            self.assertTrue(np.signbit(output[0, 0]))


if __name__ == "__main__":
    unittest.main()
