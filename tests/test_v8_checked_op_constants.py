"""Model-neutral normal-codegen proof for per-operation checked ABI constants."""
import contextlib
import copy
import hashlib
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
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import build_ir_v8


def write_synthetic_weights(root, tensors):
    """Write only the planner's entry/payload contract for this compiler test."""
    payload = bytearray()
    entries = []
    for name, value in sorted(tensors.items()):
        payload.extend(b"\0" * ((-len(payload)) & 63))
        offset = len(payload)
        raw = np.ascontiguousarray(value, dtype="<f4").tobytes()
        payload.extend(raw)
        entries.append({"name": name, "dtype": "fp32", "shape": list(value.shape),
                        "file_offset": offset, "size": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    (root / "weights.fixture").write_bytes(payload)
    return {"entries": entries}


def linear_op(name, input_name, output_name, k, n, weight, bias):
    return {
        "id": name, "op": "linear_rows_checked", "kernel": "linear_rows_checked_f32",
        "returns_status": True,
        "weight_refs": {"weight": weight, "bias": bias},
        "graph_slots": {"inputs": {"input": input_name},
                        "outputs": {"output": output_name}},
        "params": {"M": 2, "K": k, "N": n, "call_constants": {
            "linear_input_elements": 2 * k, "linear_input_stride": k,
            "linear_weight_elements": n * k, "linear_weight_stride": k,
            "linear_bias_elements": n, "linear_output_elements": 2 * n,
            "linear_output_stride": n, "linear_rows": 2,
            "linear_input_channels": k, "linear_output_channels": n}},
    }


class CheckedOpConstantsTest(unittest.TestCase):
    def compile_graph(self, *, corrupt=None, mutate=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        tensors = {
            "fixture.first.weight": np.arange(12, dtype=np.float32).reshape(4, 3) / 10,
            "fixture.first.bias": np.arange(4, dtype=np.float32) / 10,
            "fixture.second.weight": np.arange(8, dtype=np.float32).reshape(2, 4) / 20,
            "fixture.second.bias": np.arange(2, dtype=np.float32) / 10,
        }
        bundle = write_synthetic_weights(root, tensors)
        first = linear_op("first", "external:input", "middle", 3, 4,
                          "fixture.first.weight", "fixture.first.bias")
        second = linear_op("second", "middle", "output", 4, 2,
                           "fixture.second.weight", "fixture.second.bias")
        first["params"]["call_constants"].update(
            linear_output_elements=10, linear_output_stride=5)
        second["params"]["call_constants"].update(
            linear_input_elements=10, linear_input_stride=5)
        if corrupt is not None:
            second["params"]["call_constants"]["linear_input_stride"] = corrupt
        template = {
            "version": 3, "name": "two_different_linears", "family": "bounded_graph",
            "contract": {"runtime_invariants": {"inference_only": True,
                                                 "production_kernel_heap_allocation": False}},
            "checked_native_entry": True,
            "activation_buffers": {"input": {"shape": [2, 3]},
                                   "middle": {"shape": [2, 5]},
                                   "output": {"shape": [2, 2]}},
            "activation_bindings": {"input": "input", "middle": "middle",
                                    "output": "output"},
            "native_entry": {"function": "ck_two_linears", "params": [
                {"c_type": "uint8_t *", "name": "arena"},
                {"c_type": "size_t", "name": "arena_bytes"}],
                "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
            "sequence": ["component"],
            "block_types": {"component": {"sequence": ["header", "body", "footer"],
                                           "header": [],
                                           "body": {"type": "dense", "ops": [first, second]},
                                           "footer": []}},
        }
        if mutate is not None:
            mutate(template)
        circuit = root / "circuit.json"
        circuit.write_text(json.dumps(template))
        source = {"config": {
            "model": "two_different_linears", "arch": "two_different_linears",
            "num_layers": 1, "embed_dim": 3, "num_heads": 1, "num_kv_heads": 1,
            "head_dim": 3, "intermediate_size": 4, "context_length": 2,
            "max_seq_len": 2, "vocab_size": 1},
            "entries": bundle["entries"], "quant_summary": {}, "template": template}
        registry = build_ir_v8.load_kernel_registry()
        with contextlib.redirect_stdout(io.StringIO()):
            ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode="prefill")
            lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
            layout = build_ir_v8.generate_memory_layout(
                lower1, source, registry, mode="prefill", context_len=2)
            lower2 = build_ir_v8.generate_ir_lower_2(lower1, layout, source, registry,
                                                      mode="prefill")
            call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
        return root, bundle, tensors, layout, call_ir

    def test_two_different_shapes_execute_through_normal_codegen(self):
        root, bundle, tensors, layout, call_ir = self.compile_graph()
        self.assertEqual(call_ir["errors"], [])
        self.assertEqual([op["function"] for op in call_ir["operations"]],
                         ["linear_rows_checked_f32"] * 2)
        self.assertEqual([arg["expr"] for arg in call_ir["operations"][0]["args"]
                          if arg["name"] == "input_channels"], ["3"])
        self.assertEqual([arg["expr"] for arg in call_ir["operations"][1]["args"]
                          if arg["name"] == "input_channels"], ["4"])
        (root / "layout.json").write_text(json.dumps(layout))
        (root / "call.json").write_text(json.dumps(call_ir))
        generated = root / "generated.c"
        subprocess.run([sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
                        "--ir", str(root / "call.json"), "--layout", str(root / "layout.json"),
                        "--output", str(generated)], check=True, capture_output=True, text=True)
        library = root / "generated.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC", str(generated),
                        str(ROOT / "src/kernels/linear_rows_checked.c"),
                        "-I", str(ROOT / "include"), "-lm", "-o", str(library)],
                       check=True, capture_output=True, text=True)
        fn = ctypes.CDLL(str(library)).ck_two_linears
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        fn.restype = ctypes.c_int
        size = layout["memory"]["arena"]["total_size"]
        raw = (ctypes.c_uint8 * (size + 63))()
        arena = (ctypes.c_uint8 * size).from_buffer(raw, (-ctypes.addressof(raw)) & 63)
        bump = (root / "weights.fixture").read_bytes()
        entries = {entry["name"]: entry for entry in bundle["entries"]}
        for planned in layout["memory"]["weights"]["entries"]:
            entry = entries[planned["name"]]
            arena[planned["abs_offset"]:planned["abs_offset"] + entry["size"]] = bump[
                entry["file_offset"]:entry["file_offset"] + entry["size"]]
        activations = {item["name"]: item for item in
                       layout["memory"]["activations"]["buffers"]}
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        np.ndarray((2, 3), dtype=np.float32, buffer=arena,
                   offset=activations["input"]["abs_offset"])[:] = x
        middle = np.ndarray((2, 5), dtype=np.float32, buffer=arena,
                            offset=activations["middle"]["abs_offset"])
        middle[:] = -777
        output = np.ndarray((2, 2), dtype=np.float32, buffer=arena,
                            offset=activations["output"]["abs_offset"])
        output[:] = -777
        self.assertEqual(fn(arena, len(arena)), 0)
        expected = (x @ tensors["fixture.first.weight"].T + tensors["fixture.first.bias"])
        expected = expected @ tensors["fixture.second.weight"].T + tensors["fixture.second.bias"]
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
        self.assertTrue(np.all(middle[:, 4] == -777))
        output[:] = -777
        self.assertEqual(fn(arena, len(arena) - 1), -2)
        self.assertTrue(np.all(output == -777))
        first_weight = next(item for item in layout["memory"]["weights"]["entries"]
                            if item["name"] == "fixture.first.weight")
        np.ndarray((1,), dtype=np.float32, buffer=arena,
                   offset=first_weight["abs_offset"])[0] = np.nan
        self.assertNotEqual(fn(arena, len(arena)), 0)
        self.assertTrue(np.all(output == -777))

    def test_invalid_per_call_constant_rejected_before_codegen(self):
        for value in (True, -1, "4;abort()", 1 << 64):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "HARD CALL CONSTANT FAULT"):
                    self.compile_graph(corrupt=value)

    def test_unknown_nonobject_length_override_and_pointer_rejected(self):
        mutations = [
            lambda t: t["block_types"]["component"]["body"]["ops"][1]["params"].update(
                call_constants={"unknown_capacity": 1}),
            lambda t: t["block_types"]["component"]["body"]["ops"][1]["params"].update(
                call_constants=[1, 2]),
        ]
        for mutate in mutations:
            with self.assertRaisesRegex(RuntimeError, "HARD CALL CONSTANT FAULT"):
                self.compile_graph(mutate=mutate)
        kernel_map = json.loads((ROOT / "version/v8/kernel_maps/linear_rows_checked_f32.json").read_text())
        op = {"params": {"M": 2, "K": 3, "N": 4,
                         "call_constants": {"linear_rows": 2}},
              "runtime_extent_contract": {"runtime_lengths": {"linear_rows": {}}}}
        with self.assertRaisesRegex(RuntimeError, "not an overridable size_t scalar"):
            build_ir_v8._validated_call_constants(op, kernel_map)
        pointer_map = copy.deepcopy(kernel_map)
        pointer_map["call_abi"]["params"][-3] = {
            "name": "state", "source": "runtime:state", "cast": "float*"}
        op["runtime_extent_contract"] = {}
        op["params"]["call_constants"] = {"state": 0}
        with self.assertRaisesRegex(RuntimeError, "not an overridable size_t scalar"):
            build_ir_v8._validated_call_constants(op, pointer_map)

    def test_shape_stride_and_capacity_disagreements_rejected(self):
        def change(key, value):
            def mutate(template):
                op = template["block_types"]["component"]["body"]["ops"][1]
                if key in {"M", "K", "N"}:
                    op["params"][key] = value
                else:
                    op["params"]["call_constants"][key] = value
            return mutate
        cases = [
            ("K", 5), ("linear_input_channels", 5),
            ("linear_input_stride", 6),
            ("linear_input_elements", 11),
            ("linear_weight_elements", 9),
            ("linear_output_elements", 5),
            ("linear_rows", 3),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(RuntimeError, "HARD CALL CONSTANT FAULT"):
                    self.compile_graph(mutate=change(key, value))


if __name__ == "__main__":
    unittest.main()
