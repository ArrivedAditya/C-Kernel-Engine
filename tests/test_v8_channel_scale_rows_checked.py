"""Independent arithmetic and rejection tests for checked channel scaling."""

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph

ROOT = Path(__file__).resolve().parents[1]
F = ctypes.POINTER(ctypes.c_float)
Z = ctypes.c_size_t


def pointer(value):
    return value.ctypes.data_as(F)


class ChannelScaleRowsCheckedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "libscale.so"
        subprocess.run([
            "cc", "-std=gnu11", "-O2", "-shared", "-fPIC", "-fopenmp",
            str(ROOT / "src/kernels/transformer_rows_checked.c"),
            str(ROOT / "src/kernels/rope_kernels.c"),
            str(ROOT / "src/kernels/attention_kernels.c"),
            str(ROOT / "src/kernels/softmax_kernels.c"),
            str(ROOT / "src/kernels/attention_flash_true.c"),
            str(ROOT / "src/ck_threadpool.c"),
            str(ROOT / "src/ckernel_strict.c"),
            "-I", str(ROOT / "include"), "-lm", "-ldl", "-lpthread",
            "-o", str(library),
        ], check=True, capture_output=True, text=True)
        cls.call = ctypes.CDLL(str(library)).channel_scale_rows_checked_f32
        cls.call.argtypes = [F, Z, Z, F, Z, F, Z, Z, Z, Z]
        cls.call.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, x, scale, y, rows, cols, *, x_capacity=None,
               y_capacity=None):
        return self.call(pointer(x), x.size if x_capacity is None else x_capacity,
                         x.shape[1], pointer(scale), scale.size,
                         pointer(y), y.size if y_capacity is None else y_capacity,
                         y.shape[1], rows, cols)

    def test_independent_oracle_and_padding(self):
        rng = np.random.default_rng(20261024)
        for rows, cols, xpad, ypad in ((1, 1, 0, 0), (3, 5, 2, 3),
                                       (17, 9, 1, 4)):
            with self.subTest(rows=rows, cols=cols):
                x = rng.normal(size=(rows, cols + xpad)).astype(np.float32)
                scale = rng.normal(size=cols).astype(np.float32)
                y = np.full((rows, cols + ypad), -123, dtype=np.float32)
                self.assertEqual(self.invoke(x, scale, y, rows, cols), 0)
                np.testing.assert_allclose(y[:, :cols], x[:, :cols] * scale,
                                           atol=0, rtol=0)
                np.testing.assert_array_equal(y[:, cols:], -123)

    def test_rejection_and_recovery(self):
        x = np.array([[1, 2, 33], [3, 4, 44]], dtype=np.float32)
        scale = np.array([5, 6], dtype=np.float32)
        y = np.full((2, 3), -123, dtype=np.float32)
        self.assertNotEqual(self.invoke(x, scale, y, 2, 2, y_capacity=4), 0)
        np.testing.assert_array_equal(y, -123)
        self.assertNotEqual(self.invoke(x, scale, y, 2, 2, x_capacity=4), 0)
        np.testing.assert_array_equal(y, -123)
        self.assertNotEqual(self.invoke(x, scale[:1], y, 2, 2), 0)
        np.testing.assert_array_equal(y, -123)
        x[1, 0] = np.nan
        self.assertNotEqual(self.invoke(x, scale, y, 2, 2), 0)
        np.testing.assert_array_equal(y, -123)
        x[1, 0] = 3
        scale[1] = np.finfo(np.float32).max
        self.assertNotEqual(self.invoke(x, scale, y, 2, 2), 0)
        np.testing.assert_array_equal(y, -123)
        scale[1] = 6
        self.assertEqual(self.invoke(x, scale, y, 2, 2), 0)
        np.testing.assert_array_equal(y[:, :2], [[5, 12], [15, 24]])

    def test_partial_overlap_rejected(self):
        arena = np.arange(16, dtype=np.float32)
        x = arena[:6].reshape(2, 3)
        y = arena[2:8].reshape(2, 3)
        scale = np.ones(2, dtype=np.float32)
        original = y.copy()
        self.assertNotEqual(self.invoke(x, scale, y, 2, 2), 0)
        np.testing.assert_array_equal(y, original)

    def test_model_neutral_generated_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scale = np.array([.25, -2, 3, .5, 1.25], dtype=np.float32)
            op = {
                "id": "scaled", "op": "channel_scale_rows_checked",
                "kernel": "channel_scale_rows_checked_f32", "returns_status": True,
                "weight_refs": {"scale": "generic.scale"},
                "graph_slots": {"inputs": {"input": "external:input"},
                                "outputs": {"output": "scaled"}},
                "params": {"R": 3, "C": 5, "call_constants": {
                    "input_elements": 15, "input_stride": 5,
                    "scale_elements": 5, "output_elements": 15,
                    "output_stride": 5, "rows": 3, "channels": 5}},
            }
            circuit = {
                "version": 3, "name": "synthetic_channel_scale",
                "family": "bounded_graph", "checked_native_entry": True,
                "contract": {"runtime_invariants": {"inference_only": True}},
                "activation_buffers": {"input": {"shape": [3, 5]},
                                       "scaled": {"shape": [3, 5]}},
                "activation_bindings": {"input": "input", "scaled": "scaled"},
                "native_entry": {
                    "function": "ck_synthetic_channel_scale",
                    "params": [{"c_type": "uint8_t *", "name": "arena"},
                               {"c_type": "size_t", "name": "arena_bytes"}],
                    "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
                "sequence": ["component"],
                "block_types": {"component": {
                    "sequence": ["header", "body", "footer"],
                    "header": [], "body": {"type": "dense", "ops": [op]},
                    "footer": []}},
            }
            path = root / "circuit.json"
            path.write_text(json.dumps(circuit))
            source = {
                "config": {"model": circuit["name"], "arch": circuit["name"],
                           "num_layers": 1, "embed_dim": 5, "num_heads": 1,
                           "num_kv_heads": 1, "head_dim": 5,
                           "intermediate_size": 10, "context_length": 3,
                           "max_seq_len": 3, "vocab_size": 1},
                "entries": [{"name": "generic.scale", "shape": [5],
                             "dtype": "F32", "size": scale.nbytes,
                             "file_offset": 4096,
                             "sha256": hashlib.sha256(scale.tobytes()).hexdigest()}],
                "quant_summary": {}, "template": circuit,
            }
            layout, calls, _, _, fn = compile_native_graph(root, source, path)
            self.assertEqual(calls["errors"], [])
            self.assertEqual(calls["operations"][-1]["function"],
                             "channel_scale_rows_checked_f32")
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            weights = {entry["name"]: entry for entry in
                       layout["memory"]["weights"]["entries"]}
            bound_scale = np.ndarray((5,), np.float32, buffer=arena,
                                     offset=weights["generic.scale"]["abs_offset"])
            bound_scale[:] = scale
            acts = {entry["name"]: entry for entry in
                    layout["memory"]["activations"]["buffers"]}
            x = np.ndarray((3, 5), np.float32, buffer=arena,
                           offset=acts["input"]["abs_offset"])
            y = np.ndarray((3, 5), np.float32, buffer=arena,
                           offset=acts["scaled"]["abs_offset"])
            x[:] = np.arange(15, dtype=np.float32).reshape(3, 5) / 7
            y[:] = -123
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(y, x * scale)
            baseline = y.copy()
            y[:] = -123
            bound_scale[1] = np.nan
            self.assertNotEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(y, -123)
            bound_scale[:] = scale
            self.assertNotEqual(fn(arena, size - 1), 0)
            np.testing.assert_array_equal(y, -123)
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(y, baseline)


if __name__ == "__main__":
    unittest.main()
