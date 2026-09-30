"""Oracle-fed Kokoro features through two compiler-generated duration expansions."""

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if np is not None:
    from tests.test_v8_duration_logits_runtime_graph import lower_fixture
    import build_xray_checkpoint_manifest_v8 as xray_builder
    import xray_numerical_parity_v8 as xray


class DurationTwoStreamGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if np is None:
            raise unittest.SkipTest("NumPy is required for the committed feature oracle")
        fixture_dir = ROOT / "tests/fixtures/tts"
        cls.layout_source, cls.layout, cls.call_ir = lower_fixture(
            fixture_dir / "duration_two_stream_runtime_extent_circuit.json")
        cls.logit_fixture = json.loads((fixture_dir / "duration_logits_kokoro_torch.json").read_text())
        cls.meta = json.loads((fixture_dir / "duration_two_stream_reference.json").read_text())
        capture = json.loads((ROOT / cls.meta["source_manifest"]).read_text())
        for tensor, digest in cls.meta["source_sha256"].items():
            if capture["tensors"][tensor]["sha256"] != digest:
                raise AssertionError(f"source capture hash mismatch: {tensor}")
        with np.load(fixture_dir / "duration_two_stream_reference.npz", allow_pickle=False) as archive:
            cls.reference = {name: archive[name].copy() for name in archive.files}
        for name, digest in cls.meta["arrays_sha256"].items():
            if hashlib.sha256(cls.reference[name].tobytes()).hexdigest() != digest:
                raise AssertionError(f"committed feature hash mismatch: {name}")
        if cls.meta["durations"] != cls.logit_fixture["durations"]:
            raise AssertionError("captured durations disagree with pinned logit fixture")
        cls.temp = tempfile.TemporaryDirectory()
        temp = Path(cls.temp.name)
        call_file, layout_file = temp / "call.json", temp / "layout.json"
        call_file.write_text(json.dumps(cls.call_ir))
        layout_file.write_text(json.dumps(cls.layout))
        generated = temp / "generated.c"
        subprocess.run([sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
                        "--ir", str(call_file), "--layout", str(layout_file),
                        "--output", str(generated)],
                       check=True, capture_output=True, text=True)
        library = temp / "generated.so"
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                        "-shared", "-fPIC", str(generated),
                        str(ROOT / "src/kernels/audio_duration_logits.c"),
                        str(ROOT / "src/kernels/audio_duration_expand.c"),
                        str(ROOT / "src/kernels/runtime_extent.c"),
                        "-I", str(ROOT / "include"), "-lm", "-o", str(library)], check=True)
        cls.library = library
        cls.loaded = ctypes.CDLL(str(library))
        cls.function = cls.loaded.ck_test_duration_two_stream_graph
        cls.function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_int32)]
        cls.function.restype = ctypes.c_int
        cls.buffers = {item["name"]: item for item in
                       cls.layout["memory"]["activations"]["buffers"]}
        cls.arena_size = cls.layout["memory"]["arena"]["total_size"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _vector(self, arena, name, dtype, count):
        return np.frombuffer(arena, dtype=dtype, count=count,
                             offset=self.buffers[name]["abs_offset"])

    def _matrix(self, arena, name):
        return self._vector(arena, name, np.float32, 640 * 128).reshape(640, 128)

    def _arena(self, logits=None):
        raw = (ctypes.c_uint8 * (self.arena_size + 63))()
        aligned = (-ctypes.addressof(raw)) & 63
        arena = (ctypes.c_uint8 * self.arena_size).from_buffer(raw, aligned)
        logits = logits if logits is not None else np.asarray(self.logit_fixture["logits"],
                                                               dtype=np.float32).ravel()
        self._vector(arena, "audio_duration_logits", np.float32, 1800)[:] = logits
        self._vector(arena, "runtime_values", np.int32, 36)[:] = -777
        self._vector(arena, "runtime_valid_extent", np.int32, 1)[:] = -777
        for name in ("duration_features", "text_features"):
            self._vector(arena, name, np.float32, 640 * 40)[:] = self.reference[name].ravel()
        for name, marker in (("duration_expanded", -99), ("text_expanded", -98),
                             ("runtime_valid_copy", -88)):
            self._matrix(arena, name)[:] = marker
        return arena

    def _assert_unwritten(self, arena):
        for name, marker in (("duration_expanded", -99), ("text_expanded", -98),
                             ("runtime_valid_copy", -88)):
            self.assertTrue(np.all(self._matrix(arena, name) == marker), name)

    def test_generated_graph_matches_both_independent_feature_captures(self):
        self.assertEqual([op["function"] for op in self.call_ir["operations"]], [
            "audio_duration_logits_to_frames_f32",
            "audio_duration_expand_channel_major_f32",
            "audio_duration_expand_channel_major_f32",
            "ck_runtime_copy_valid_f32",
        ])
        self.assertTrue(all(not op["errors"] for op in self.call_ir["operations"]))
        for operation, channels, source, destination in (
            (self.call_ir["operations"][1], "640", "A_DURATION_FEATURES", "A_DURATION_EXPANDED"),
            (self.call_ir["operations"][2], "512", "A_TEXT_FEATURES", "A_TEXT_EXPANDED"),
        ):
            args = {item["name"]: item["expr"] for item in operation["args"]}
            self.assertEqual((args["channels"], args["input_elements"], args["input_stride"],
                              args["output_elements"], args["output_stride"]),
                             (channels, "25600", "40", "81920", "128"))
            self.assertEqual(args["expanded_frames"], "runtime_extents.expanded_frames")
            self.assertIn(source, args["features"])
            self.assertIn(destination, args["output"])
        arena = self._arena()
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, self.meta["frames"])
        np.testing.assert_array_equal(self._vector(arena, "runtime_values", np.int32, 36),
                                      self.meta["durations"])
        for name, expected, channels, marker in (
            ("duration_expanded", "duration_expected", 640, -99),
            ("text_expanded", "text_expected", 512, -98),
        ):
            actual = self._matrix(arena, name)
            np.testing.assert_array_equal(actual[:channels, :frames.value],
                                          self.reference[expected])
            self.assertTrue(np.all(actual[:channels, frames.value:] == marker))
            self.assertTrue(np.all(actual[channels:] == marker))
        for name in ("duration_features", "text_features"):
            input_after = self._vector(arena, name, np.float32, 640 * 40).reshape(640, 40)
            np.testing.assert_array_equal(input_after, self.reference[name])
            self.assertTrue(np.all(input_after[:, 36:] == -77))
        self.assertTrue(np.all(self.reference["text_features"][512:] == -77))
        copy = self._matrix(arena, "runtime_valid_copy")
        np.testing.assert_array_equal(copy[:, :frames.value],
                                      self.reference["duration_expected"])
        self.assertTrue(np.all(copy[:, frames.value:] == -88))

    def test_xray_compares_generated_valid_regions_to_captured_alignment(self):
        arena = self._arena()
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), 0)
        report = {"torch": {"tensors": {}}, "comparisons": {}}
        order = []
        for name, expected, channels in (
            ("duration", "duration_expected", 640),
            ("text", "text_expected", 512),
        ):
            key = f"{name}_expanded@0"
            checkpoint = f"kokoro.duration.{name}.expanded"
            order.append(checkpoint)
            candidate_path = Path(self.temp.name) / f"{name}-generated.f32"
            oracle_path = Path(self.temp.name) / f"{name}-captured.f32"
            self._matrix(arena, f"{name}_expanded").tofile(candidate_path)
            self.reference[expected].tofile(oracle_path)
            shape = [channels, frames.value]
            report["torch"]["tensors"][key] = {"path": str(oracle_path), "shape": shape}
            report["comparisons"][key] = {
                "ck_path": str(candidate_path), "shape": shape,
                "physical_shape": [640, 128], "capacity_shape": [channels, 128],
                "valid_shape": shape, "physical_strides": [128, 1],
            }
        runtime = xray_builder.capture_runtime_library_identity(
            self.loaded, "ck_test_duration_two_stream_graph")
        candidate = xray_builder.build_manifest(
            backend="ck", call_ir=self.call_ir, tensor_report=report,
            model="kokoro_duration_two_stream", source="generated_native",
            phase="prefill", loaded_library=self.library, runtime_library=runtime)
        oracle = xray_builder.build_manifest(
            backend="pytorch", call_ir=self.call_ir, tensor_report=report,
            model="kokoro_duration_two_stream", source="pinned_capture",
            phase="prefill")
        profile = {
            "schema": "cke.parity_profile", "schema_version": 1,
            "name": "kokoro_duration_two_stream", "backend": "pytorch",
            "contract_schema_version": 1,
            "required_match_fields": ["checkpoint_id", "producer", "logical_layout",
                                      "axis_names", "resolved_contract_id", "kernel_id",
                                      "function"],
            "observed_storage": {"default": "fp32", "checkpoints": {}},
            "dtype_thresholds": {"fp32": {"cosine_min": 0.99999,
                                           "rmse_max": 0.0, "relative_rmse_max": 0.0,
                                           "max_abs_max": 0.0, "finite_required": True}},
            "checkpoint_order": order, "interval_expansions": {},
            "backend_mappings": {},
        }
        comparison = xray.compare_manifests(candidate, oracle, profile,
                                             checkpoint_order=order)
        self.assertEqual(comparison["status"], "pass", comparison)
        self.assertEqual(len(comparison["comparisons"]), 2)
        self.assertEqual(comparison["unresolved_contract_checkpoints"], [])
        expected_contract = "audio_duration_expand_checked_strided_copy_fp32"
        for checkpoint, row in zip(order, comparison["comparisons"]):
            self.assertEqual(row["checkpoint_id"], checkpoint)
            for manifest in (candidate, oracle):
                point = next(item for item in manifest["checkpoints"]
                             if item["checkpoint_id"] == checkpoint)
                self.assertEqual(point["resolved_contract_id"], expected_contract)
                self.assertEqual(point["kernel_id"],
                                 "audio_duration_expand_channel_major_f32")
                self.assertEqual(point["function"],
                                 "audio_duration_expand_channel_major_f32")
            self.assertEqual(row["metrics"]["max_abs"], 0.0)
        self.assertEqual(candidate["run"]["runtime_library"]["path"],
                         str(self.library.resolve()))
        self.assertEqual(candidate["run"]["artifact_library"]["sha256"],
                         candidate["run"]["runtime_library"]["sha256"])
        print("TTS_TWO_STREAM_EVIDENCE " + json.dumps({
            "status": "PASS", "oracle": self.meta["reference"],
            "durations": 36, "valid_frames": frames.value,
            "duration_channels": 640, "text_channels": 512,
            "input_stride": 40, "output_stride": 128,
            "max_abs_error": max(row["metrics"]["max_abs"]
                                 for row in comparison["comparisons"]),
            "runtime_symbol": runtime["symbol"],
            "runtime_library_sha256": runtime["sha256"],
            "resolved_contract": expected_contract,
            "reproduce": "python3 -m unittest tests.test_v8_duration_two_stream_runtime_graph",
        }, sort_keys=True))

    def test_rejected_duration_stops_both_expansions_and_extent_publication(self):
        arena = self._arena(np.zeros(1800, dtype=np.float32))
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), -2)
        self.assertEqual(frames.value, -1)
        self.assertTrue(np.all(self._vector(arena, "runtime_values", np.int32, 36) == -777))
        self.assertEqual(self._vector(arena, "runtime_valid_extent", np.int32, 1)[0], -777)
        self._assert_unwritten(arena)

    def test_exact_capacity_on_both_streams(self):
        logits = []
        for token in range(36):
            active = 46 if token == 35 else 6
            logits.extend([0.0] * active + [-100.0] * (50 - active))
        arena = self._arena(np.asarray(logits, dtype=np.float32))
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 128)
        durations = [3] * 35 + [23]
        np.testing.assert_array_equal(self._vector(arena, "runtime_values", np.int32, 36),
                                      durations)
        for name, source, channels, marker in (
            ("duration_expanded", "duration_features", 640, -99),
            ("text_expanded", "text_features", 512, -98),
        ):
            np.testing.assert_array_equal(
                self._matrix(arena, name)[:channels],
                np.repeat(self.reference[source][:channels, :36], durations, axis=1))
            self.assertTrue(np.all(self._matrix(arena, name)[channels:] == marker))

    def test_repeated_request_uses_new_valid_extent_and_preserves_padding(self):
        arena = self._arena()
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        self._vector(arena, "audio_duration_logits", np.float32, 1800)[:] = -100
        frames.value = -1
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 36)
        np.testing.assert_array_equal(self._vector(arena, "runtime_values", np.int32, 36),
                                      [1] * 36)
        for name, source, channels, old in (
            ("duration_expanded", "duration_features", 640, "duration_expected"),
            ("text_expanded", "text_features", 512, "text_expected"),
        ):
            actual = self._matrix(arena, name)
            np.testing.assert_array_equal(actual[:channels, :36],
                                          self.reference[source][:channels, :36])
            np.testing.assert_array_equal(actual[:channels, 36:103],
                                          self.reference[old][:, 36:103])
        np.testing.assert_array_equal(self._matrix(arena, "runtime_valid_copy")[:, :36],
                                      self.reference["duration_features"][:, :36])

    def test_one_frame_above_minimum_uses_shared_extent(self):
        logits = np.full((36, 50), -100, dtype=np.float32)
        logits[:, :2] = 0
        logits[0, :4] = 0
        arena = self._arena(logits.ravel())
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 37)
        durations = [2] + [1] * 35
        np.testing.assert_array_equal(self._vector(arena, "runtime_values", np.int32, 36),
                                      durations)
        for name, source, channels, marker in (
            ("duration_expanded", "duration_features", 640, -99),
            ("text_expanded", "text_features", 512, -98),
        ):
            actual = self._matrix(arena, name)
            np.testing.assert_array_equal(
                actual[:channels, :frames.value],
                np.repeat(self.reference[source][:channels, :36], durations, axis=1))
            self.assertTrue(np.all(actual[:channels, frames.value:] == marker))
            self.assertTrue(np.all(actual[channels:] == marker))

    def test_undersized_arena_rejected_before_any_write(self):
        arena = self._arena()
        before = bytes(arena)
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena) - 1, ctypes.byref(frames)), -2)
        self.assertEqual(frames.value, -1)
        self.assertEqual(bytes(arena), before)


if __name__ == "__main__":
    if np is None:
        print("TEST SKIPPED: NumPy is required for the committed feature oracle")
    else:
        unittest.main()
