"""Complete first Kokoro ALBERT layer through ordinary generated native execution."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import build_xray_checkpoint_manifest_v8 as xray_builder
import xray_numerical_parity_v8 as xray

sys.path.insert(0, str(ROOT / "version/v8/tts"))
import export_kokoro_bump as exporter

PREFIX = "phoneme_encoder.encoder.albert_layer_groups.0.albert_layers.0.attention"
BASE_WEIGHTS = {
    "word": "phoneme_encoder.embeddings.word_embeddings.weight",
    "position": "phoneme_encoder.embeddings.position_embeddings.weight",
    "token_type": "phoneme_encoder.embeddings.token_type_embeddings.weight",
    "gamma": "phoneme_encoder.embeddings.LayerNorm.weight",
    "beta": "phoneme_encoder.embeddings.LayerNorm.bias",
    "projection_weight": "phoneme_encoder.encoder.embedding_hidden_mapping_in.weight",
    "projection_bias": "phoneme_encoder.encoder.embedding_hidden_mapping_in.bias",
}


class KokoroGeneratedAlbertLayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = ROOT / "tests/fixtures/tts"
        cls.embed = dict(np.load(fixtures / "bert_embedding_pinned.npz"))
        cls.projection = dict(np.load(fixtures / "kokoro_projection_pinned.npz"))
        fixture_path = fixtures / "kokoro_qkv_pinned.npz"
        cls.qkv = dict(np.load(fixture_path))
        cls.meta = json.loads(fixture_path.with_suffix(".json").read_text())
        context_path = fixtures / "kokoro_attention_context_pinned.npz"
        cls.context = dict(np.load(context_path))
        cls.context_meta = json.loads(context_path.with_suffix(".json").read_text())
        if hashlib.sha256(context_path.read_bytes()).hexdigest() != cls.context_meta["fixture_sha256"]:
            raise RuntimeError("attention-context fixture hash mismatch")
        if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != cls.meta["fixture_sha256"]:
            raise RuntimeError("Q/K/V fixture hash mismatch")
        layer_path = fixtures / "kokoro_albert_layer_pinned.npz"
        cls.layer = dict(np.load(layer_path))
        cls.layer_meta = json.loads(layer_path.with_suffix(".json").read_text())
        if hashlib.sha256(layer_path.read_bytes()).hexdigest() != cls.layer_meta["fixture_sha256"]:
            raise RuntimeError("first-layer oracle fixture hash mismatch")
        evidence = os.environ.get("CKE_KOKORO_LAYER_EVIDENCE_DIR")
        if evidence:
            path = Path(evidence).resolve()
            path.mkdir(parents=True, exist_ok=True)
            cls.temp = SimpleNamespace(name=str(path), cleanup=lambda: None)
        else:
            cls.temp = tempfile.TemporaryDirectory()
        temp = Path(cls.temp.name)
        tensors = {BASE_WEIGHTS[key]: cls.embed[key] for key in
                   ("word", "position", "token_type", "gamma", "beta")}
        tensors[BASE_WEIGHTS["projection_weight"]] = cls.projection["weight"]
        tensors[BASE_WEIGHTS["projection_bias"]] = cls.projection["bias"]
        for name in ("query", "key", "value"):
            for suffix in ("weight", "bias"):
                tensors[f"{PREFIX}.{name}.{suffix}"] = cls.qkv[f"{name}_{suffix}"]
        for name, tensor in cls.layer.items():
            if name.startswith("weight__"):
                canonical = PREFIX.removesuffix(".attention") + "." + name.removeprefix("weight__").replace("__", ".")
                tensors[canonical] = tensor
        origins = {name: {"source_name": name, "transform": "identity"} for name in tensors}
        bundle = exporter.write_bundle(temp, tensors, origins, {
            "n_token": 178, "hidden_dim": 512,
            "plbert": {"intermediate_size": 2048, "max_position_embeddings": 512,
                       "num_attention_heads": 12}},
            {"source": "pinned embedding, projection and PyTorch Q/K/V fixtures"})
        exporter.verify_bundle(temp)
        cls.bump = (temp / "weights.bump").read_bytes()
        cls.entries = {item["name"]: item for item in bundle["entries"]}
        circuit = fixtures / "kokoro_first_albert_layer_generated_circuit.json"
        template = json.loads(circuit.read_text())
        source = {
            "config": {
                "model": template["name"], "arch": template["name"],
                "num_layers": 1, "embed_dim": 128, "num_heads": 1,
                "num_kv_heads": 1, "head_dim": 128, "intermediate_size": 256,
                "context_length": 36, "max_seq_len": 36, "vocab_size": 178,
                "T": 36, "C": 128, "epsilon": 1e-12,
                "activation_buffer_dtypes": {"word_ids": "i32", "type_ids": "i32"},
            },
            "entries": bundle["entries"], "quant_summary": {}, "template": template,
        }
        from tests.v8_checked_graph_test_support import compile_native_graph
        cls.layout, cls.call_ir, cls.library, cls.loaded, cls.fn = compile_native_graph(temp, source, circuit)
        layout = cls.layout
        cls.weights = {item["name"]: item for item in layout["memory"]["weights"]["entries"]}
        cls.activations = {item["name"]: item for item in
                           layout["memory"]["activations"]["buffers"]}
        cls.arena_size = layout["memory"]["arena"]["total_size"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def vector(self, arena, name, dtype, count):
        return np.ndarray((count,), dtype=dtype, buffer=arena,
                          offset=self.activations[name]["abs_offset"])

    def arena(self):
        raw = (ctypes.c_uint8 * (self.arena_size + 63))()
        offset = (-ctypes.addressof(raw)) & 63
        arena = (ctypes.c_uint8 * self.arena_size).from_buffer(raw, offset)
        for name, planned in self.weights.items():
            entry = self.entries[name]
            payload = self.bump[entry["file_offset"]:entry["file_offset"] + entry["size"]]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
            arena[planned["abs_offset"]:planned["abs_offset"] + len(payload)] = payload
        self.vector(arena, "word_ids", np.int32, 36)[:] = self.embed["ids"]
        self.vector(arena, "type_ids", np.int32, 36)[:] = self.embed["type_ids"]
        for name, planned in self.activations.items():
            if name not in ("word_ids", "type_ids"):
                self.vector(arena, name, np.float32, planned["size"] // 4)[:] = -777.
        return arena

    def test_complete_generated_first_layer_and_xray(self):
        self.assertEqual(self.call_ir["errors"], [])
        self.assertEqual(len(self.call_ir["operations"]), 14)
        for op in self.call_ir["operations"]:
            for checkpoint in op.get("semantic_checkpoints", []):
                self.assertNotEqual(checkpoint["resolved_contract_id"], "unresolved")
        self.assertEqual(self.call_ir["operations"][-1]["function"], "layernorm_rows_checked_f32")
        arena = self.arena()
        self.assertEqual(self.fn(arena, len(arena)), 0)
        boundaries = [
            ("embedding_output", self.layer["embedding_expected"],
             "kokoro.phoneme_encoder.embeddings"),
            ("projection_output", self.layer["projection_expected"],
             "kokoro.phoneme_encoder.input_projection"),
        ] + [(f"{name}_output", self.layer[f"{name}_expected"],
              f"kokoro.phoneme_encoder.albert0.{name}") for name in ("query", "key", "value")] + [
            ("attention_context_output", self.layer["attention_projection_input"],
             "kokoro.phoneme_encoder.albert0.attention_context")]
        boundaries += [(name, self.layer[reference], "kokoro.phoneme_encoder.albert0." + ("layer" if name == "albert_layer_output" else name.removesuffix("_output")))
                       for name, reference in (
                           ("attention_projection_output", "attention_projection_expected"),
                           ("attention_residual_output", "attention_norm_input"),
                           ("attention_norm_output", "attention_norm_expected"),
                           ("ffn_linear_output", "ffn_linear_expected"),
                           ("ffn_gelu_output", "ffn_projection_input"),
                           ("ffn_projection_output", "ffn_projection_expected"),
                           ("layer_residual_output", "layer_output_input"),
                           ("albert_layer_output", "layer_output_expected"))]
        report = {"torch": {"tensors": {}}, "comparisons": {}}
        errors = {}
        tolerances = {}
        # Existing prefix retains its 3e-5 bound. New normalized checkpoints use
        # the LayerNorm family's 1e-5 bound. The new, large-valued FFN projection
        # and residual have their own explicit 5e-5 FP32-versus-FP64 comparison
        # contract, additionally checked against independent FP64 arithmetic.
        # No existing primitive fixture or numerical threshold is changed.

        for name, expected, checkpoint in boundaries:
            actual = self.vector(arena, name, np.float32, expected.size).reshape(expected.shape)
            self.assertTrue(np.isfinite(actual).all())
            error = np.abs(actual - expected)
            errors[checkpoint] = float(np.max(error))
            tolerance = (1e-5 if name in ("attention_norm_output", "albert_layer_output")
                         else 5e-5 if name in ("ffn_projection_output", "layer_residual_output")
                         else 3e-5)
            tolerances[checkpoint] = tolerance
            print("CKE_NUMERICAL_CASE " + json.dumps({
                "case_id": checkpoint + ".generated-vs-pytorch28", "name": checkpoint,
                "configuration": str(list(expected.shape)), "shape": list(expected.shape),
                "provider": next(point["kernel_id"] for op in self.call_ir["operations"]
                                 for point in op.get("semantic_checkpoints", []) if point["tensor"] == name),
                "dtype": "fp32", "direction": "inference", "oracle": "pytorch",
                "backend_version": self.layer_meta["dependencies"]["torch"],
                "evidence_kind": "numerical", "max_diff": errors[checkpoint],
                "tolerance": tolerance, "status": "pass" if errors[checkpoint] <= tolerance else "fail",
                "worst_sample": list(map(int,np.unravel_index(error.argmax(),error.shape))),
                "reproduction_command": "python3 -m unittest " + self.id(),
            }, sort_keys=True))
            self.assertLessEqual(errors[checkpoint], tolerance)
            candidate_path = Path(self.temp.name) / f"{name}-candidate.f32"
            oracle_path = Path(self.temp.name) / f"{name}-oracle.f32"
            actual.tofile(candidate_path)
            expected.tofile(oracle_path)
            selector = f"{name}@0" if name not in ("embedding_output", "albert_layer_output") else name
            report["torch"]["tensors"][selector] = {
                "path": str(oracle_path), "shape": list(expected.shape)}
            report["comparisons"][selector] = {
                "ck_path": str(candidate_path), "shape": list(expected.shape),
                "physical_shape": list(expected.shape), "capacity_shape": list(expected.shape),
                "valid_shape": list(expected.shape), "physical_strides": [expected.shape[1], 1]}
        runtime = xray_builder.capture_runtime_library_identity(
            self.loaded, "ck_kokoro_first_albert_layer")
        candidate = xray_builder.build_manifest(
            backend="ck", call_ir=self.call_ir, tensor_report=report,
            model="kokoro_first_albert_layer_generated", source="generated_native",
            phase="prefill", loaded_library=self.library, runtime_library=runtime)
        oracle = xray_builder.build_manifest(
            backend="pytorch", call_ir=self.call_ir, tensor_report=report,
            model="kokoro_first_albert_layer_generated", source="pinned_capture",
            phase="prefill")
        order = [checkpoint for _name, _expected, checkpoint in boundaries]
        profile = {
            "schema": "cke.parity_profile", "schema_version": 1, "name": "kokoro_attention_context",
            "backend": "pytorch", "contract_schema_version": 1,
            "required_match_fields": ["checkpoint_id", "producer", "logical_layout",
                                      "axis_names", "resolved_contract_id", "kernel_id", "function"],
            "observed_storage": {"default": "fp32", "checkpoints": {}},
            "dtype_thresholds": {"fp32": {"cosine_min": 0.99999, "rmse_max": 3e-5,
                                           "relative_rmse_max": 3e-5, "max_abs_max": 3e-5,
                                           "finite_required": True}},
            "checkpoint_order": order, "interval_expansions": {}, "backend_mappings": {},
        }
        (Path(self.temp.name) / "ck-checkpoints.json").write_text(json.dumps(candidate, indent=2))
        (Path(self.temp.name) / "oracle-checkpoints.json").write_text(json.dumps(oracle, indent=2))
        xray_reports = []
        # Existing X-Ray accepts a threshold profile per comparison. Keep each
        # checkpoint's actual bound explicit rather than weakening all FP32 rows.
        for checkpoint in order:
            bounded = json.loads(json.dumps(profile))
            bounded["name"] = "kokoro_first_layer_" + checkpoint.rsplit(".", 1)[-1]
            bounded["checkpoint_order"] = [checkpoint]
            bounded["dtype_thresholds"]["fp32"]["max_abs_max"] = tolerances[checkpoint]
            result = xray.compare_manifests(candidate, oracle, bounded)
            xray_reports.append(result)
            self.assertEqual(result["status"], "pass", result)
        (Path(self.temp.name) / "xray-reports.json").write_text(json.dumps(xray_reports, indent=2))
        print("KOKORO_GENERATED_FIRST_LAYER_EVIDENCE " + json.dumps({
            "status": "PASS", "oracle": self.layer_meta["dependencies"], "max_abs_errors": errors, "checkpoint_absolute_bounds": tolerances,
            "generated_library_sha256": hashlib.sha256(self.library.read_bytes()).hexdigest(),
            "complete_first_albert_layer": "PASS", "complete_phoneme_encoder": "NOT_TESTED", "generated_waveform": "NOT_TESTED"},
            sort_keys=True))

    def test_failed_qkv_producer_preserves_dependent_outputs(self):
        arena = self.arena()
        item = self.weights[f"{PREFIX}.query.weight"]
        np.ndarray((1,), dtype=np.float32, buffer=arena,
                   offset=item["abs_offset"])[0] = np.nan
        self.assertNotEqual(self.fn(arena, len(arena)), 0)
        for name in ("query_output", "key_output", "value_output", "attention_context_output"):
            self.assertTrue(np.all(self.vector(arena, name, np.float32, 27648) == -777.))

    def test_repeated_requests_and_bounds(self):
        arena = self.arena()
        self.assertEqual(self.fn(arena, len(arena)), 0)
        first = self.vector(arena, "albert_layer_output", np.float32, 27648).copy()
        self.vector(arena, "word_ids", np.int32, 36)[:] = 0
        self.assertEqual(self.fn(arena, len(arena)), 0)
        self.assertFalse(np.array_equal(first,
                          self.vector(arena, "albert_layer_output", np.float32, 27648)))
        self.vector(arena, "word_ids", np.int32, 36)[:] = self.embed["ids"]
        self.assertEqual(self.fn(arena, len(arena)), 0)
        np.testing.assert_array_equal(first, self.vector(arena, "albert_layer_output", np.float32, 27648))
        self.vector(arena, "albert_layer_output", np.float32, 27648)[:] = -777.
        self.assertEqual(self.fn(arena, len(arena) - 1), -2)
        self.assertTrue(np.all(self.vector(arena, "albert_layer_output", np.float32, 27648) == -777.))

    def test_exported_bump_first_layer_when_available(self):
        bundle_dir = os.environ.get("CKE_KOKORO_BUMP_DIR")
        if not bundle_dir:
            self.skipTest("CKE_KOKORO_BUMP_DIR unset; exported bundle NOT_TESTED")
        bundle_dir = Path(bundle_dir)
        self.assertEqual(hashlib.sha256(
            (bundle_dir / "weights_manifest.json").read_bytes()).hexdigest(),
            self.layer_meta["bump_bundle_manifest_sha256"])
        entries = {entry["name"]: entry for entry in
                   exporter.verify_bundle(bundle_dir)["entries"]}
        arena = self.arena()
        with (bundle_dir / "weights.bump").open("rb") as stream:
            for name, planned in self.weights.items():
                entry = entries[name]
                self.assertEqual(entry["size"], planned["size"])
                self.assertEqual(entry["sha256"], self.entries[name]["sha256"])
                stream.seek(entry["file_offset"])
                payload = stream.read(entry["size"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                start = planned["abs_offset"]
                arena[start:start + len(payload)] = payload
        self.assertEqual(self.fn(arena, len(arena)), 0)
        for name in ("query", "key", "value"):
            actual = self.vector(arena, f"{name}_output", np.float32, 27648)
            self.assertLessEqual(float(np.max(np.abs(
                actual - self.qkv[f"{name}_expected"].ravel()))), 3e-5)
        actual = self.vector(arena, "attention_context_output", np.float32, 27648)
        self.assertLessEqual(float(np.max(np.abs(
            actual - self.context["context_expected"].ravel()))), 3e-5)
        final = self.vector(arena, "albert_layer_output", np.float32, 27648)
        self.assertTrue(np.isfinite(final).all())
        self.assertLessEqual(float(np.max(np.abs(final - self.layer["layer_output_expected"].ravel()))), 3e-5)

    def test_standalone_native_replay_outside_checkout(self):
        # One native entry executes the whole circuit. Python only prepares the
        # fixture image, builds the host, and compares its returned features.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arena = self.arena()
            (root / "arena.bin").write_bytes(bytes(arena))
            shutil.copyfile(self.library, root / "generated.so")
            output = self.activations["albert_layer_output"]
            source = root / "host.c"
            source.write_text(f"""#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
extern int ck_kokoro_first_albert_layer(uint8_t *, size_t);
int main(int argc, char **argv) {{
    if (argc != 3) return 10;
    const size_t bytes = {self.arena_size}u;
    uint8_t *arena = aligned_alloc(64, (bytes + 63u) & ~(size_t)63u);
    if (!arena) return 11;
    FILE *input = fopen(argv[1], "rb");
    if (!input) {{ free(arena); return 12; }}
    size_t received = fread(arena, 1, bytes, input);
    int extra = fgetc(input);
    fclose(input);
    if (received != bytes || extra != EOF) {{ free(arena); return 13; }}
    int status = ck_kokoro_first_albert_layer(arena, bytes);
    if (status) {{ free(arena); return 14; }}
    FILE *out = fopen(argv[2], "wb");
    if (!out) {{ free(arena); return 15; }}
    size_t written = fwrite(arena + {output['abs_offset']}u, 1, 27648u * sizeof(float), out);
    int close_status = fclose(out);
    free(arena);
    return written == 27648u * sizeof(float) && close_status == 0 ? 0 : 16;
}}
""")
            executable = root / "native-first-layer"
            subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                            str(source), "-L", str(root), "-l:generated.so", "-Wl,-rpath,$ORIGIN",
                            "-o", str(executable)], check=True, capture_output=True, text=True)
            env = dict(os.environ)
            env.pop("LD_LIBRARY_PATH", None)
            subprocess.run([str(executable), "arena.bin", "features.f32"], cwd=root,
                           env=env, check=True, capture_output=True, text=True)
            actual = np.fromfile(root / "features.f32", dtype='f').reshape(36, 768)
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, self.layer["layer_output_expected"], rtol=0, atol=1e-5)
            # Retain a runnable replay alongside the optional generated evidence.
            if os.environ.get("CKE_KOKORO_LAYER_EVIDENCE_DIR"):
                target = Path(self.temp.name) / "standalone"
                target.mkdir(exist_ok=True)
                for name in ("host.c", "native-first-layer", "generated.so", "arena.bin", "features.f32"):
                    shutil.copy2(root / name, target / name)
            (root / "arena.bin").write_bytes(b"invalid")
            rejected = subprocess.run([str(executable), "arena.bin", "rejected.f32"], cwd=root, env=env)
            self.assertEqual(rejected.returncode, 13)
            self.assertFalse((root / "rejected.f32").exists())

    def test_affine_reduction_against_independent_fp64_oracle(self):
        from tests import test_v8_linear_rows_oracle as linear_tests
        linear_tests.LinearRowsOracleTest.setUpClass()
        try:
            linear = linear_tests.LinearRowsOracleTest()
            for stage, weight in (("attention_projection", "attention__dense"),
                                  ("ffn_linear", "ffn"), ("ffn_projection", "ffn_output")):
                with self.subTest(stage=stage):
                    x = self.layer[stage + "_input"]
                    w = self.layer["weight__" + weight + "__weight"]
                    b = self.layer["weight__" + weight + "__bias"]
                    actual = np.empty_like(self.layer[stage + "_expected"])
                    self.assertEqual(linear.call(x, w, b, actual, len(x), x.shape[1], w.shape[0]), 0)
                    expected = (x.astype(np.float64) @ w.astype(np.float64).T + b.astype(np.float64)).astype(np.float32)
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_array_equal(actual, expected)
                    # Also retain the independently captured FP32 reference result.
                    delta = float(np.abs(actual - self.layer[stage + "_expected"]).max())
                    bound = 5e-5 if stage == "ffn_projection" else 3e-5
                    self.assertLessEqual(delta, bound)
                    print("CKE_NUMERICAL_CASE " + json.dumps({
                        "case_id": stage + ".native-vs-numpy-fp64", "name": stage,
                        "configuration": str(list(actual.shape)), "shape": list(actual.shape),
                        "provider": "linear_rows_checked_f32", "dtype": "fp32", "direction": "inference",
                        "oracle": "numpy-fp64", "backend_version": np.__version__,
                        "evidence_kind": "numerical", "status": "pass", "max_diff": 0.0, "tolerance": 0.0,
                        "pytorch_fp32_max_diff": delta, "pytorch_fp32_absolute_bound": bound,
                        "reproduction_command": "python3 -m unittest " + self.id(),
                    }, sort_keys=True))
        finally:
            linear_tests.LinearRowsOracleTest.tearDownClass()

    def test_failed_norm_preserves_its_output_and_consumers(self):
        arena = self.arena()
        name = PREFIX + ".LayerNorm.weight"
        np.ndarray((1,), dtype=np.float32, buffer=arena,
                   offset=self.weights[name]["abs_offset"])[0] = np.nan
        self.assertNotEqual(self.fn(arena, len(arena)), 0)
        for name in ("attention_norm_output", "ffn_linear_output", "ffn_gelu_output",
                     "ffn_projection_output", "layer_residual_output", "albert_layer_output"):
            self.assertTrue(np.all(self.vector(arena, name, np.float32,
                              self.activations[name]["size"] // 4) == -777.))


if __name__ == "__main__":
    unittest.main()
