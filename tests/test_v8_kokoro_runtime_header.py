"""One generated Kokoro embedding/projection entry with checked input lengths."""

import contextlib
import copy
import ctypes
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tests.test_v8_checked_op_constants import write_synthetic_weights


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
sys.path.insert(0, str(ROOT / "version/v8/tts"))
import build_ir_v8
from build_kokoro_encoder_circuit import build_circuit as full_encoder_circuit


WEIGHTS = {
    "word": "phoneme_encoder.embeddings.word_embeddings.weight",
    "position": "phoneme_encoder.embeddings.position_embeddings.weight",
    "token_type": "phoneme_encoder.embeddings.token_type_embeddings.weight",
    "gamma": "phoneme_encoder.embeddings.LayerNorm.weight",
    "beta": "phoneme_encoder.embeddings.LayerNorm.bias",
    "projection_weight": "phoneme_encoder.encoder.embedding_hidden_mapping_in.weight",
    "projection_bias": "phoneme_encoder.encoder.embedding_hidden_mapping_in.bias",
}


def bounded_header():
    # Derive the bounded fixture from canonical model authoring, never from a
    # previous test circuit. This exercises the first two real model edges.
    graph = full_encoder_circuit(36)
    graph["name"] = "kokoro_runtime_length_header"
    block = graph["block_types"]["phoneme_encoder"]
    block["body"] = {"type": "dense", "ops": []}
    block["footer"] = []
    names = ("word_ids", "type_ids", "embedding_output", "projection_output")
    graph["activation_buffers"] = {
        name: graph["activation_buffers"][name] for name in names}
    graph["activation_bindings"] = {
        name: graph["activation_bindings"][name] for name in names}
    exports = graph["semantic_checkpoints"]["exports"]
    graph["semantic_checkpoints"]["exports"] = {
        name: exports[name] for name in
        ("albert_embeddings", "albert_input_projection")}
    requirements = graph["required_numerical_contracts"]
    graph["required_numerical_contracts"] = {
        name: requirements[name] for name in
        ("embedding_three_table_layer_norm", "linear_rows_checked")}
    graph["activation_buffers"].update(
        request_length={"shape": [1]}, valid_length={"shape": [1]})
    graph["activation_bindings"].update(
        request_length="request_length", valid_length="valid_length")
    graph["native_entry"]["function"] = "ck_kokoro_runtime_header"
    graph["native_entry"]["params"].append(
        {"c_type": "int32_t *", "name": "out_tokens"})
    graph["native_entry"]["runtime_length_outputs"] = {
        "valid_tokens": "out_tokens"}
    graph["runtime_lengths"] = {"valid_tokens": {
        "producer": "check_tokens", "result": "valid_extent",
        "capacity": 36, "allow_zero": False}}
    graph["runtime_constants"].update(
        value_elements=1, phoneme_count=1, max_duration=36,
        expanded_capacity=36)
    block["header"].insert(0, {
        "id": "check_tokens", "op": "runtime_extent_sum",
        "kernel": "runtime_extent_sum_i32", "returns_status": True,
        "produces_runtime_lengths": {"valid_tokens": "valid_extent"},
        "graph_slots": {
            "inputs": {"values": "external:request_length"},
            "outputs": {"valid_extent": "valid_length"}}})
    embedding = block["header"][1]
    embedding["consumes_runtime_lengths"] = ["valid_tokens"]
    embedding["runtime_scalar_bindings"] = {"tokens": "valid_tokens"}
    embedding["params"] = {"T": 36, "C": 128, "call_constants": {
        "id_elements": 36, "output_elements": 36 * 128,
        "channels": 128, "output_stride": 128}}
    projection = block["header"][2]
    projection["consumes_runtime_lengths"] = ["valid_tokens"]
    projection["runtime_scalar_bindings"] = {"linear_rows": "valid_tokens"}
    projection["params"] = {"M": 36, "K": 128, "N": 768,
        "call_constants": {
            "linear_input_elements": 36 * 128, "linear_input_stride": 128,
            "linear_weight_elements": 768 * 128, "linear_weight_stride": 128,
            "linear_bias_elements": 768,
            "linear_output_elements": 36 * 768, "linear_output_stride": 768,
            "linear_input_channels": 128, "linear_output_channels": 768}}
    return graph


class KokoroRuntimeHeaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        embed_path = ROOT / "tests/fixtures/tts/bert_embedding_pinned.npz"
        projection_path = ROOT / "tests/fixtures/tts/kokoro_projection_pinned.npz"
        embed_meta = json.loads(embed_path.with_suffix(".json").read_text())
        projection_meta = json.loads(projection_path.with_suffix(".json").read_text())
        for path, meta in ((embed_path, embed_meta),
                           (projection_path, projection_meta)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta["fixture_sha256"]:
                raise RuntimeError("pinned Kokoro header fixture hash mismatch")
        cls.embed = dict(np.load(embed_path))
        cls.projection = dict(np.load(projection_path))
        tensors = {WEIGHTS[name]: cls.embed[name] for name in
                   ("word", "position", "token_type", "gamma", "beta")}
        tensors[WEIGHTS["projection_weight"]] = cls.projection["weight"]
        tensors[WEIGHTS["projection_bias"]] = cls.projection["bias"]
        bundle = write_synthetic_weights(root, tensors)
        cls.entries = {item["name"]: item for item in bundle["entries"]}
        cls.bump = (root / "weights.fixture").read_bytes()
        template = bounded_header()
        circuit = root / "circuit.json"
        circuit.write_text(json.dumps(template))
        source = {"config": {
            "model": template["name"], "arch": template["name"],
            "num_layers": 1, "embed_dim": 128, "num_heads": 1,
            "num_kv_heads": 1, "head_dim": 128, "intermediate_size": 256,
            "context_length": 36, "max_seq_len": 36, "vocab_size": 178,
            "T": 36, "C": 128, "epsilon": embed_meta["epsilon"],
            "activation_buffer_dtypes": {
                "word_ids": "i32", "type_ids": "i32",
                "request_length": "i32", "valid_length": "i32"}},
            "entries": bundle["entries"], "quant_summary": {},
            "template": template}
        cls.source = source
        cls.circuit = circuit
        registry = build_ir_v8.load_kernel_registry()
        with contextlib.redirect_stdout(io.StringIO()):
            ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode="prefill")
            lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
            layout = build_ir_v8.generate_memory_layout(
                lower1, source, registry, mode="prefill", context_len=36)
            lower2 = build_ir_v8.generate_ir_lower_2(
                lower1, layout, source, registry, mode="prefill")
            call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
        cls.layout, cls.call_ir = layout, call_ir
        (root / "layout.json").write_text(json.dumps(layout))
        (root / "call.json").write_text(json.dumps(call_ir))
        generated = root / "generated.c"
        subprocess.run([sys.executable,
                        str(ROOT / "version/v8/scripts/codegen_v8.py"),
                        "--ir", str(root / "call.json"),
                        "--layout", str(root / "layout.json"),
                        "--output", str(generated)],
                       check=True, capture_output=True, text=True)
        library = root / "generated.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra",
                        "-Werror", "-pedantic", "-shared", "-fPIC",
                        str(generated),
                        str(ROOT / "src/kernels/runtime_extent.c"),
                        str(ROOT / "src/kernels/embedding_three_table_layer_norm.c"),
                        str(ROOT / "src/kernels/linear_rows_checked.c"),
                        "-I", str(ROOT / "include"), "-lm", "-o", str(library)],
                       check=True, capture_output=True, text=True)
        cls.function = ctypes.CDLL(str(library)).ck_kokoro_runtime_header
        cls.function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_int32)]
        cls.function.restype = ctypes.c_int
        cls.activations = {item["name"]: item for item in
                           layout["memory"]["activations"]["buffers"]}
        cls.weights = {item["name"]: item for item in
                       layout["memory"]["weights"]["entries"]}
        cls.arena_size = layout["memory"]["arena"]["total_size"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, shape, dtype=np.float32):
        return np.ndarray(shape, dtype=dtype, buffer=arena,
                          offset=self.activations[name]["abs_offset"])

    def arena(self):
        raw = (ctypes.c_uint8 * (self.arena_size + 63))()
        arena = (ctypes.c_uint8 * self.arena_size).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        for name, planned in self.weights.items():
            entry = self.entries[name]
            payload = self.bump[entry["file_offset"]:
                                entry["file_offset"] + entry["size"]]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
            arena[planned["abs_offset"]:planned["abs_offset"] + len(payload)] = payload
        self.view(arena, "word_ids", (36,), np.int32)[:] = self.embed["ids"]
        self.view(arena, "type_ids", (36,), np.int32)[:] = self.embed["type_ids"]
        return arena

    def test_same_generated_entry_handles_eight_and_thirty_six_tokens(self):
        self.assertEqual(self.call_ir["errors"], [])
        self.assertEqual([op["function"] for op in self.call_ir["operations"]], [
            "ck_runtime_sum_i32_checked", "embedding_three_table_layer_norm_f32",
            "linear_rows_checked_f32"])
        arena = self.arena()
        request = self.view(arena, "request_length", (1,), np.int32)
        embedding = self.view(arena, "embedding_output", (36, 128))
        projection = self.view(arena, "projection_output", (36, 768))
        out_tokens = ctypes.c_int32(-1)
        worst_embedding = 0.0
        worst_projection = 0.0
        for count in (8, 36, 8):
            request[0] = count
            embedding[:] = -777
            projection[:] = -888
            out_tokens.value = -1
            self.assertEqual(self.function(arena, len(arena),
                                           ctypes.byref(out_tokens)), 0)
            self.assertEqual(out_tokens.value, count)
            self.assertTrue(np.isfinite(embedding[:count]).all())
            self.assertTrue(np.isfinite(projection[:count]).all())
            worst_embedding = max(worst_embedding, float(np.max(np.abs(
                embedding[:count] - self.embed["expected"][:count]))))
            worst_projection = max(worst_projection, float(np.max(np.abs(
                projection[:count] - self.projection["expected"][:count]))))
            np.testing.assert_allclose(embedding[:count],
                                       self.embed["expected"][:count],
                                       atol=2e-5, rtol=0)
            np.testing.assert_allclose(projection[:count],
                                       self.projection["expected"][:count],
                                       atol=3e-5, rtol=0)
            self.assertTrue(np.all(embedding[count:] == -777))
            self.assertTrue(np.all(projection[count:] == -888))
        for count in (0, 37, 2147483647):
            request[0] = count
            projection[:] = -888
            out_tokens.value = -1
            self.assertNotEqual(self.function(arena, len(arena),
                                              ctypes.byref(out_tokens)), 0)
            self.assertEqual(out_tokens.value, -1)
            self.assertTrue(np.all(projection == -888))
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.generated-runtime-token-header-v1",
            "name": "One generated Kokoro header entry with changing valid token count",
            "provider": "embedding_three_table_layer_norm_f32+linear_rows_checked_f32",
            "dtype": "fp32", "direction": "inference",
            "oracle": "committed pinned PyTorch 36-token embedding and projection capture",
            "status": "pass", "gate": "composition", "blocking": True,
            "max_diff": max(worst_embedding, worst_projection),
            "tolerance": 3e-5, "metric": "maximum absolute checkpoint error",
            "embedding_max_abs": worst_embedding,
            "projection_max_abs": worst_projection,
            "configuration": "one capacity-36 generated entry; valid 8/36/8 prefix lengths; "
                             "not a distinct 8-token full-model capture",
            "reproduction_command": "python3 -m unittest "
                "tests.test_v8_kokoro_runtime_header",
        }, sort_keys=True))

    def test_bad_token_stops_projection_and_recovery(self):
        arena = self.arena()
        request = self.view(arena, "request_length", (1,), np.int32)
        ids = self.view(arena, "word_ids", (36,), np.int32)
        projection = self.view(arena, "projection_output", (36, 768))
        out_tokens = ctypes.c_int32(-1)
        request[0] = 8
        ids[0] = 178
        projection[:] = -888
        self.assertNotEqual(self.function(arena, len(arena),
                                          ctypes.byref(out_tokens)), 0)
        self.assertEqual(out_tokens.value, -1)
        self.assertTrue(np.all(projection == -888))
        ids[0] = self.embed["ids"][0]
        self.assertEqual(self.function(arena, len(arena),
                                       ctypes.byref(out_tokens)), 0)
        self.assertEqual(out_tokens.value, 8)

    def test_embedding_planned_capacity_and_shape_mismatches_reject(self):
        def lower(mutated):
            registry = build_ir_v8.load_kernel_registry()
            with contextlib.redirect_stdout(io.StringIO()):
                ir1 = build_ir_v8.build_ir1_direct(
                    mutated, self.circuit, mode="prefill")
                low1 = build_ir_v8.generate_ir_lower_1(
                    ir1, registry, mutated, "prefill")
                layout = build_ir_v8.generate_memory_layout(
                    low1, mutated, registry, mode="prefill", context_len=36)
                low2 = build_ir_v8.generate_ir_lower_2(
                    low1, layout, mutated, registry, mode="prefill")
                build_ir_v8.generate_ir_lower_3(low2, mode="prefill")

        for key, bad in (("id_elements", 37), ("output_elements", 4609),
                         ("output_stride", 129), ("channels", 127),
                         ("T", 35)):
            mutated = copy.deepcopy(self.source)
            embedding = mutated["template"]["block_types"]["phoneme_encoder"]["header"][1]
            if key == "T":
                embedding["params"]["T"] = bad
            else:
                embedding["params"]["call_constants"][key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(
                    RuntimeError, "HARD CALL CONSTANT FAULT"):
                lower(mutated)


if __name__ == "__main__":
    unittest.main()
