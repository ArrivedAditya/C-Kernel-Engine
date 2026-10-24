"""Model-neutral generated SnakeBeta with a checked per-request valid length."""

import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class AudioSnakeBetaGeneratedTest(unittest.TestCase):
    def test_runtime_length_both_weights_failure_and_recovery(self):
        alpha = np.array([.2, -.1, .4], dtype=np.float32)
        beta = np.array([-.3, .5, .1], dtype=np.float32)
        weights = (("synthetic.alpha", alpha), ("synthetic.beta", beta))
        entries = []
        offset = 0
        for name, value in weights:
            payload = value.tobytes()
            entries.append({"name": name, "dtype": "fp32", "shape": [3],
                            "file_offset": offset, "size": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()})
            offset += len(payload)
        ops = [
            {"id": "extent", "op": "runtime_extent_sum",
             "kernel": "runtime_extent_sum_i32", "returns_status": True,
             "produces_runtime_lengths": {"frames": "valid_extent"},
             "params": {"T": 1},
             "graph_slots": {"inputs": {"values": "external:durations"},
                             "outputs": {"valid_extent": "frame_extent"}}},
            {"id": "snake_beta", "op": "audio_snake_beta_strided_checked",
             "kernel": "audio_snake_beta_strided_f32_checked",
             "returns_status": True,
             "consumes_runtime_lengths": ["frames"],
             "runtime_scalar_bindings": {"snake_beta_frames": "frames"},
             "weight_refs": {"alpha": "synthetic.alpha", "beta": "synthetic.beta"},
             "params": {"C": 3, "T": 7,
                        "call_constants": {
                 "snake_beta_input_elements": 27,
                 "snake_beta_input_stride": 9,
                 "snake_beta_alpha_elements": 3,
                 "snake_beta_beta_elements": 3,
                 "snake_beta_output_elements": 30,
                 "snake_beta_output_stride": 10,
                 "snake_beta_channels": 3}},
             "graph_slots": {"inputs": {"input": "external:features"},
                             "outputs": {"output": "output"}}},
        ]
        buffers = {"durations": {"shape": [1]},
                   "frame_extent": {"shape": [1]},
                   "features": {"shape": [3, 9]},
                   "output": {"shape": [3, 10]}}
        circuit = {
            "version": 3, "name": "synthetic_channelwise_snake_beta",
            "family": "bounded_graph", "checked_native_entry": True,
            "contract": {"runtime_invariants": {"inference_only": True}},
            "activation_buffers": buffers,
            "activation_bindings": {name: name for name in buffers},
            "runtime_constants": {"value_elements": 1, "phoneme_count": 1,
                                  "max_duration": 7, "expanded_capacity": 7},
            "runtime_lengths": {"frames": {
                "producer": "extent", "result": "valid_extent",
                "capacity": 7, "allow_zero": False}},
            "native_entry": {
                "function": "ck_synthetic_channelwise_snake_beta",
                "params": [{"c_type": "uint8_t *", "name": "arena"},
                           {"c_type": "size_t", "name": "arena_bytes"},
                           {"c_type": "int32_t *", "name": "out_frames"}],
                "arena": {"pointer": "arena", "bytes": "arena_bytes"},
                "runtime_length_outputs": {"frames": "out_frames"}},
            "sequence": ["component"], "block_types": {"component": {
                "sequence": ["header", "body", "footer"],
                "header": ops[:1], "body": {"type": "dense", "ops": ops[1:]},
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
                "intermediate_size": 3, "context_length": 7,
                "max_seq_len": 7, "vocab_size": 1,
                "activation_buffer_dtypes": {
                    "durations": "i32", "frame_extent": "i32"}},
                "entries": entries, "quant_summary": {}, "template": circuit}
            layout, calls, _library, _loaded, function = compile_native_graph(
                root, source, path)
            self.assertEqual(calls["errors"], [])
            self.assertEqual(calls["operations"][-1]["function"],
                             "audio_snake_beta_strided_f32_checked")
            function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_int32)]
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            weight_entries = {e["name"]: e for e in
                              layout["memory"]["weights"]["entries"]}
            for name, value in weights:
                begin = weight_entries[name]["abs_offset"]
                arena[begin:begin + value.nbytes] = value.tobytes()
            locations = {e["name"]: e for e in
                         layout["memory"]["activations"]["buffers"]}

            def view(name, dtype, shape):
                return np.ndarray(shape, dtype, buffer=arena,
                                  offset=locations[name]["abs_offset"])

            durations = view("durations", np.int32, (1,))
            features = view("features", np.float32, (3, 9))
            output = view("output", np.float32, (3, 10))
            first = None
            for valid in (7, 5, 7):
                durations[0] = valid
                features[:] = np.nan
                features[:, :valid] = np.arange(3 * valid,
                    dtype=np.float32).reshape(3, valid) / 5 - 2
                output[:] = -91
                published = ctypes.c_int32(-999)
                self.assertEqual(function(arena, len(arena),
                                          ctypes.byref(published)), 0)
                self.assertEqual(published.value, valid)
                x = features[:, :valid]
                expected = x + (1 / (np.exp(beta[:, None]) + np.float32(1e-9))) * (
                    np.sin(x * np.exp(alpha[:, None])) ** 2)
                np.testing.assert_allclose(output[:, :valid], expected,
                                           atol=3e-6, rtol=3e-6)
                np.testing.assert_array_equal(output[:, valid:], -91)
                if valid == 7:
                    if first is None:
                        first = output.copy()
                    else:
                        np.testing.assert_array_equal(output, first)
            for failure in ("duration", "beta"):
                durations[0] = 8 if failure == "duration" else 7
                bound_beta = np.ndarray((3,), np.float32, buffer=arena,
                    offset=weight_entries["synthetic.beta"]["abs_offset"])
                if failure == "beta":
                    bound_beta[-1] = np.nan
                output[:] = -91
                published = ctypes.c_int32(-999)
                self.assertNotEqual(function(arena, len(arena),
                                             ctypes.byref(published)), 0)
                self.assertEqual(published.value, -999)
                np.testing.assert_array_equal(output, -91)
                bound_beta[-1] = beta[-1]
            durations[0] = 7
            self.assertEqual(function(arena, len(arena),
                                      ctypes.byref(published)), 0)


if __name__ == "__main__":
    unittest.main()
