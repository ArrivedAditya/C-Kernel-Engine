"""Model-independent normal-v8-codegen exercise of full token-major attention."""
import contextlib
import ctypes
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import build_ir_v8
from tests.test_v8_attention_full_token_major_oracle import numpy_oracle


class GeneratedFullAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        temp = Path(cls.temp.name)
        activation_buffers = {name: {"shape": [3, 10]}
                              for name in ("query", "key", "value", "output")}
        circuit = {
            "version": 3, "name": "full_attention_synthetic", "family": "bounded_graph",
            "contract": {"runtime_invariants": {"inference_only": True,
                                                 "production_kernel_heap_allocation": False}},
            "checked_native_entry": True,
            "activation_buffers": activation_buffers,
            "activation_bindings": {name: name for name in activation_buffers},
            "native_entry": {"function": "ck_full_attention_synthetic",
                             "params": [{"c_type": "uint8_t *", "name": "arena"},
                                        {"c_type": "size_t", "name": "arena_bytes"}],
                             "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
            "runtime_constants": {},
            "sequence": ["attention_block"],
            "block_types": {"attention_block": {
                "sequence": ["header", "body", "footer"], "header": [],
                "body": {"type": "dense", "ops": [{
                    "id": "full_attention", "op": "attention_full_token_major_checked",
                    "kernel": "attention_full_token_major_f32_checked", "returns_status": True,
                    "graph_slots": {"inputs": {name: f"external:{name}" for name in
                                               ("query", "key", "value")},
                                    "outputs": {"output": "output"}},
                    "params": {"T": 3, "H": 2, "D": 5,
                               "call_constants": {
                                   "attention_query_elements": 30,
                                   "attention_key_elements": 30,
                                   "attention_value_elements": 30,
                                   "attention_output_elements": 30,
                                   "attention_tokens": 3, "attention_heads": 2,
                                   "attention_head_dim": 5}}}]}, "footer": []}},
        }
        path = temp / "circuit.json"
        path.write_text(json.dumps(circuit))
        source = {"config": {"model": circuit["name"], "arch": circuit["name"],
                             "num_layers": 1, "embed_dim": 10, "num_heads": 2,
                             "num_kv_heads": 2, "head_dim": 5,
                             "intermediate_size": 20, "context_length": 3,
                             "max_seq_len": 3, "T": 3, "C": 10},
                  "entries": [], "quant_summary": {}, "template": circuit}
        cls.source, cls.circuit_path = source, path
        registry = build_ir_v8.load_kernel_registry()
        with contextlib.redirect_stdout(io.StringIO()):
            ir1 = build_ir_v8.build_ir1_direct(source, path, mode="prefill")
            lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
            layout = build_ir_v8.generate_memory_layout(
                lower1, source, registry, mode="prefill", context_len=3)
            lower2 = build_ir_v8.generate_ir_lower_2(
                lower1, layout, source, registry, mode="prefill")
            call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
        cls.layout, cls.call_ir = layout, call_ir
        (temp / "layout.json").write_text(json.dumps(layout))
        (temp / "calls.json").write_text(json.dumps(call_ir))
        generated = temp / "generated.c"
        subprocess.run([sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
                        "--ir", str(temp / "calls.json"), "--layout", str(temp / "layout.json"),
                        "--output", str(generated)], check=True, capture_output=True, text=True)
        library = temp / "generated.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-pedantic", "-shared", "-fPIC", str(generated),
                        str(ROOT / "src/kernels/attention_full_token_major.c"),
                        "-I", str(ROOT / "include"), "-lm", "-o", str(library)],
                       check=True, capture_output=True, text=True)
        cls.fn = ctypes.CDLL(str(library)).ck_full_attention_synthetic
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        cls.fn.restype = ctypes.c_int
        cls.buffers = {item["name"]: item for item in layout["memory"]["activations"]["buffers"]}
        cls.arena_bytes = layout["memory"]["arena"]["total_size"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_generated_parity_and_failure_propagation(self):
        self.assertEqual(self.call_ir["errors"], [])
        self.assertEqual([op["function"] for op in self.call_ir["operations"]],
                         ["attention_full_token_major_f32_checked"])
        arguments = {arg["name"]: arg for arg in self.call_ir["operations"][0]["args"]}
        scratch = self.buffers[arguments["scratch"]["buffer_ref"]]
        self.assertEqual(int(arguments["scratch_bytes"]["expr"]), 3 * 4)
        scratch_begin = scratch["abs_offset"]
        scratch_end = scratch_begin + 3 * 4
        for name in ("query", "key", "value", "output"):
            planned = self.buffers[name]
            self.assertTrue(scratch_end <= planned["abs_offset"] or
                            planned["abs_offset"] + planned["size"] <= scratch_begin)
        raw = (ctypes.c_uint8 * (self.arena_bytes + 63))()
        offset = (-ctypes.addressof(raw)) & 63
        arena = (ctypes.c_uint8 * self.arena_bytes).from_buffer(raw, offset)
        def tensor(name):
            return np.ndarray((3, 10), dtype=np.float32, buffer=arena,
                              offset=self.buffers[name]["abs_offset"])
        rng = np.random.default_rng(37)
        arrays = [rng.normal(size=(3, 10)).astype(np.float32) for _ in range(3)]
        for name, array in zip(("query", "key", "value"), arrays):
            tensor(name)[:] = array
        tensor("output")[:] = -777.0
        self.assertEqual(self.fn(arena, len(arena)), 0)
        np.testing.assert_allclose(tensor("output"), numpy_oracle(*arrays, 2),
                                   rtol=1e-6, atol=2e-6)
        tensor("output")[:] = -777.0
        self.assertEqual(self.fn(arena, len(arena) - 1), -2)
        self.assertTrue(np.all(tensor("output") == -777.0))
        tensor("query")[0, 0] = np.nan
        self.assertNotEqual(self.fn(arena, len(arena)), 0)
        self.assertTrue(np.all(tensor("output") == -777.0))

    def test_wrong_shape_and_claimed_capacity_rejected_before_codegen(self):
        for field, value in (("attention_query_elements", 31),
                             ("attention_heads", 3)):
            with self.subTest(field=field):
                circuit = json.loads(self.circuit_path.read_text())
                operation = circuit["block_types"]["attention_block"]["body"]["ops"][0]
                operation["params"]["call_constants"][field] = value
                path = Path(self.temp.name) / f"invalid-{field}.json"
                path.write_text(json.dumps(circuit))
                source = dict(self.source, template=circuit)
                registry = build_ir_v8.load_kernel_registry()
                with contextlib.redirect_stdout(io.StringIO()):
                    ir1 = build_ir_v8.build_ir1_direct(source, path, mode="prefill")
                    with self.assertRaisesRegex(RuntimeError, "HARD CALL CONSTANT FAULT"):
                        lower1 = build_ir_v8.generate_ir_lower_1(
                            ir1, registry, source, "prefill")
                        layout = build_ir_v8.generate_memory_layout(
                            lower1, source, registry, mode="prefill", context_len=3)
                        lower2 = build_ir_v8.generate_ir_lower_2(
                            lower1, layout, source, registry, mode="prefill")
                        build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")


if __name__ == "__main__":
    unittest.main()
