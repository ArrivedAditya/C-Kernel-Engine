"""Pinned phoneme IDs and BUMP embedding weights through normal generated C."""
import contextlib
import copy
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import build_ir_v8
import codegen_checked_calls_v8 as checked_codegen
import build_xray_checkpoint_manifest_v8 as xray_builder
import xray_numerical_parity_v8 as xray

EXPORTER = ROOT / "version/v8/tts/export_kokoro_bump.py"
SPEC = importlib.util.spec_from_file_location("kokoro_embedding_exporter", EXPORTER)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)

WEIGHT_NAMES = {
    "word": "phoneme_encoder.embeddings.word_embeddings.weight",
    "position": "phoneme_encoder.embeddings.position_embeddings.weight",
    "token_type": "phoneme_encoder.embeddings.token_type_embeddings.weight",
    "gamma": "phoneme_encoder.embeddings.LayerNorm.weight",
    "beta": "phoneme_encoder.embeddings.LayerNorm.bias",
}


class KokoroGeneratedEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_path = ROOT / "tests/fixtures/tts/bert_embedding_pinned.npz"
        cls.fixture_manifest = json.loads(cls.fixture_path.with_suffix(".json").read_text())
        cls.fixture = dict(np.load(cls.fixture_path))
        if hashlib.sha256(cls.fixture_path.read_bytes()).hexdigest() != cls.fixture_manifest["fixture_sha256"]:
            raise RuntimeError("pinned embedding fixture hash mismatch")
        cls.temp = tempfile.TemporaryDirectory()
        temp = Path(cls.temp.name)
        tensors = {name: cls.fixture[key] for key, name in WEIGHT_NAMES.items()}
        origins = {name: {"source_name": name, "transform": "identity"}
                   for name in tensors}
        bundle_config = {
            "n_token": 178, "hidden_dim": 512,
            "plbert": {"intermediate_size": 2048,
                       "max_position_embeddings": 512,
                       "num_attention_heads": 12},
        }
        cls.bundle = exporter.write_bundle(
            temp, tensors, origins, bundle_config,
            {"source": "pinned Kokoro ALBERT embedding PyTorch fixture"},
        )
        exporter.verify_bundle(temp)
        cls.bump = (temp / "weights.bump").read_bytes()
        circuit = ROOT / "tests/fixtures/tts/kokoro_embedding_generated_circuit.json"
        template = json.loads(circuit.read_text())
        source = {
            "config": {
                "model": template["name"], "arch": template["name"],
                "num_layers": 1, "embed_dim": 128, "num_heads": 1,
                "num_kv_heads": 1, "head_dim": 128, "intermediate_size": 256,
                "context_length": 36, "max_seq_len": 36, "vocab_size": 178,
                "T": 36, "C": 128, "epsilon": cls.fixture_manifest["epsilon"],
                "activation_buffer_dtypes": {"word_ids": "i32", "type_ids": "i32"},
            },
            "entries": cls.bundle["entries"], "quant_summary": {},
            "template": template,
        }
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
        (temp / "layout.json").write_text(json.dumps(layout))
        (temp / "call.json").write_text(json.dumps(call_ir))
        generated = temp / "generated.c"
        subprocess.run([
            sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
            "--ir", str(temp / "call.json"), "--layout", str(temp / "layout.json"),
            "--output", str(generated),
        ], check=True, capture_output=True, text=True)
        library = temp / "generated.so"
        subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", str(generated),
            str(ROOT / "src/kernels/embedding_three_table_layer_norm.c"),
            "-I", str(ROOT / "include"), "-lm", "-o", str(library),
        ], check=True)
        cls.library = library
        cls.library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
        cls.loaded = ctypes.CDLL(str(library))
        cls.fn = cls.loaded.ck_kokoro_embedding_boundary
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        cls.fn.restype = ctypes.c_int
        cls.weights = {item["name"]: item for item in
                       layout["memory"]["weights"]["entries"]}
        cls.activations = {item["name"]: item for item in
                           layout["memory"]["activations"]["buffers"]}
        cls.arena_size = layout["memory"]["arena"]["total_size"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def arena(self):
        raw = (ctypes.c_uint8 * (self.arena_size + 63))()
        offset = (-ctypes.addressof(raw)) & 63
        arena = (ctypes.c_uint8 * self.arena_size).from_buffer(raw, offset)
        by_name = {item["name"]: item for item in self.bundle["entries"]}
        for name, planned in self.weights.items():
            entry = by_name[name]
            payload = self.bump[entry["file_offset"]:entry["file_offset"] + entry["size"]]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
            self.assertEqual(len(payload), planned["size"])
            start = planned["abs_offset"]
            arena[start:start + len(payload)] = payload
        self.vector(arena, "word_ids", np.int32, 36)[:] = self.fixture["ids"]
        self.vector(arena, "type_ids", np.int32, 36)[:] = self.fixture["type_ids"]
        self.vector(arena, "embedding_output", np.float32, 4608)[:] = -777.0
        return arena

    def vector(self, arena, name, dtype, count):
        return np.ndarray((count,), dtype=dtype, buffer=arena,
                          offset=self.activations[name]["abs_offset"])

    def test_generated_c_matches_pinned_embedding_boundary(self):
        self.assertEqual(self.call_ir["errors"], [])
        self.assertEqual([op["function"] for op in self.call_ir["operations"]],
                         ["embedding_three_table_layer_norm_f32"])
        args = self.call_ir["operations"][0]["args"]
        self.assertEqual({arg["weight_ref"] for arg in args if "weight_ref" in arg},
                         set(WEIGHT_NAMES.values()))
        arena = self.arena()
        before_weights = {
            name: hashlib.sha256(bytes(arena[item["abs_offset"]:
                                             item["abs_offset"] + item["size"]])).hexdigest()
            for name, item in self.weights.items()
        }
        self.assertEqual(self.fn(arena, len(arena)), 0)
        actual = self.vector(arena, "embedding_output", np.float32, 4608).reshape(36, 128)
        expected = self.fixture["expected"]
        self.assertTrue(np.isfinite(actual).all())
        error = np.abs(actual - expected)
        worst = np.unravel_index(np.argmax(error), error.shape)
        self.assertLessEqual(float(error[worst]), 2e-5)
        for name, item in self.weights.items():
            self.assertEqual(hashlib.sha256(bytes(arena[item["abs_offset"]:
                                                         item["abs_offset"] + item["size"]])).hexdigest(),
                             before_weights[name])
        print("TTS_GENERATED_EMBEDDING_EVIDENCE " + json.dumps({
            "status": "PASS", "graph": "kokoro_embedding_generated_boundary",
            "oracle": self.fixture_manifest["oracle"],
            "provider": self.call_ir["operations"][0]["function"],
            "shape": [36, 128], "max_abs_error": float(error[worst]),
            "worst_token_channel": [int(value) for value in worst],
            "generated_library_sha256": self.library_sha256,
            "reproduce": "python3 -m unittest tests.test_v8_kokoro_generated_embedding",
            "complete_kokoro_encoder": "NOT_TESTED", "generated_waveform": "NOT_TESTED",
        }, sort_keys=True))

    def test_repeated_requests_and_failure_preserves_output(self):
        arena = self.arena()
        self.assertEqual(self.fn(arena, len(arena)), 0)
        first = self.vector(arena, "embedding_output", np.float32, 4608).copy()
        ids = self.vector(arena, "word_ids", np.int32, 36)
        ids[:] = 0
        self.assertEqual(self.fn(arena, len(arena)), 0)
        self.assertFalse(np.array_equal(first, self.vector(arena, "embedding_output", np.float32, 4608)))
        ids[-1] = 178
        self.vector(arena, "embedding_output", np.float32, 4608)[:] = -777.0
        self.assertNotEqual(self.fn(arena, len(arena)), 0)
        self.assertTrue(np.all(self.vector(arena, "embedding_output", np.float32, 4608) == -777.0))

    def test_exported_kokoro_bump_embedding_boundary_when_available(self):
        bundle_dir = os.environ.get("CKE_KOKORO_BUMP_DIR")
        if not bundle_dir:
            self.skipTest("CKE_KOKORO_BUMP_DIR is unset; exported Kokoro weights NOT_TESTED")
        bundle_dir = Path(bundle_dir)
        manifest = exporter.verify_bundle(bundle_dir)
        entries = {item["name"]: item for item in manifest["entries"]}
        arena = self.arena()
        with (bundle_dir / "weights.bump").open("rb") as stream:
            for name, planned in self.weights.items():
                entry = entries[name]
                self.assertEqual(entry["size"], planned["size"])
                stream.seek(entry["file_offset"])
                payload = stream.read(entry["size"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                start = planned["abs_offset"]
                arena[start:start + len(payload)] = payload
                pinned = self.fixture[next(key for key, value in WEIGHT_NAMES.items()
                                          if value == name)]
                self.assertEqual(payload, pinned.tobytes())
        self.assertEqual(self.fn(arena, len(arena)), 0)
        actual = self.vector(arena, "embedding_output", np.float32, 4608)
        self.assertTrue(np.isfinite(actual).all())
        self.assertLessEqual(float(np.max(np.abs(actual - self.fixture["expected"].ravel()))),
                             2e-5)

    def test_xray_compares_generated_embedding_with_pinned_oracle(self):
        arena = self.arena()
        self.assertEqual(self.fn(arena, len(arena)), 0)
        candidate_path = Path(self.temp.name) / "generated-embedding.f32"
        oracle_path = Path(self.temp.name) / "oracle-embedding.f32"
        self.vector(arena, "embedding_output", np.float32, 4608).tofile(candidate_path)
        self.fixture["expected"].tofile(oracle_path)
        report = {
            "torch": {"tensors": {"embedding_output": {
                "path": str(oracle_path), "shape": [36, 128]
            }}},
            "comparisons": {"embedding_output": {
                "ck_path": str(candidate_path), "shape": [36, 128],
                "physical_shape": [36, 128], "capacity_shape": [36, 128],
                "valid_shape": [36, 128], "physical_strides": [128, 1],
            }},
        }
        runtime = xray_builder.capture_runtime_library_identity(
            self.loaded, "ck_kokoro_embedding_boundary")
        candidate = xray_builder.build_manifest(
            backend="ck", call_ir=self.call_ir, tensor_report=report,
            model="kokoro_embedding_generated_boundary", source="generated_native",
            phase="prefill", loaded_library=self.library, runtime_library=runtime)
        oracle = xray_builder.build_manifest(
            backend="pytorch", call_ir=self.call_ir, tensor_report=report,
            model="kokoro_embedding_generated_boundary", source="pinned_capture",
            phase="prefill")
        checkpoint = "kokoro.phoneme_encoder.embeddings"
        profile = {
            "schema": "cke.parity_profile", "schema_version": 1,
            "name": "kokoro_embedding_generated_boundary", "backend": "pytorch",
            "contract_schema_version": 1,
            "required_match_fields": ["checkpoint_id", "producer", "logical_layout",
                                      "axis_names", "resolved_contract_id", "kernel_id",
                                      "function"],
            "observed_storage": {"default": "fp32", "checkpoints": {}},
            "dtype_thresholds": {"fp32": {"cosine_min": 0.99999,
                                           "rmse_max": 2e-5, "relative_rmse_max": 2e-5,
                                           "max_abs_max": 2e-5, "finite_required": True}},
            "checkpoint_order": [checkpoint], "interval_expansions": {},
            "backend_mappings": {},
        }
        comparison = xray.compare_manifests(candidate, oracle, profile,
                                             checkpoint_order=[checkpoint])
        self.assertEqual(comparison["status"], "pass", comparison)
        self.assertEqual(len(comparison["comparisons"]), 1)
        self.assertLessEqual(comparison["comparisons"][0]["metrics"]["max_abs"], 2e-5)
        self.assertEqual(candidate["run"]["runtime_library"]["path"],
                         str(self.library.resolve()))
        self.assertEqual(candidate["run"]["artifact_library"]["sha256"],
                         candidate["run"]["runtime_library"]["sha256"])

    def test_undersized_or_misaligned_arena_rejects(self):
        arena = self.arena()
        self.assertEqual(self.fn(arena, len(arena) - 1), -2)
        self.assertTrue(np.all(self.vector(arena, "embedding_output", np.float32, 4608) == -777.0))
        misaligned = ctypes.cast(ctypes.byref(arena, 1), ctypes.POINTER(ctypes.c_uint8))
        self.assertEqual(self.fn(misaligned, len(arena) - 1), -2)

    def test_invalid_planned_weight_offsets_reject_during_codegen(self):
        malformed = copy.deepcopy(self.call_ir)
        malformed["memory"]["weights"]["entries"][0]["abs_offset"] = self.arena_size
        with self.assertRaisesRegex(checked_codegen.CheckedCallCodegenError,
                                    "exceeds arena"):
            checked_codegen.emit_checked_calls(malformed, ROOT)
        malformed = copy.deepcopy(self.call_ir)
        malformed["memory"]["weights"]["entries"][1]["define"] = (
            malformed["memory"]["weights"]["entries"][0]["define"])
        with self.assertRaisesRegex(checked_codegen.CheckedCallCodegenError,
                                    "invalid or duplicate planned buffer"):
            checked_codegen.emit_checked_calls(malformed, ROOT)


if __name__ == "__main__":
    unittest.main()
