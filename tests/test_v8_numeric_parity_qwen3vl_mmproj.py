import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "version" / "v8" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numeric_parity_qwen3vl_mmproj_v8 as npv8  # type: ignore  # noqa: E402
import qwen3vl_encoder_prefix_parity_suite_v8 as prefix_suite  # type: ignore  # noqa: E402


class NumericParityQwen3VLMmprojV8Tests(unittest.TestCase):
    def test_decode_source_rgb8_preserves_ppm_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "sample.ppm"
            image.write_bytes(b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00")
            self.assertEqual(
                npv8._decode_source_rgb8(image),
                (2, 1, b"\xff\x00\x00\x00\xff\x00"),
            )

    def test_independent_preprocess_rejects_inconsistent_input(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid decoded RGB"):
            npv8._run_llamacpp_preprocess(Path("shim"), Path("model"), b"\0", 2, 2, None, None)

    def test_independent_preprocess_rejects_unavailable_api(self) -> None:
        class FakeLibrary:
            def ck_mtmd_clip_init(self, *_args):
                return 1

            def ck_mtmd_clip_preprocess_rgb8(self, *_args):
                return -2

            def ck_mtmd_clip_free(self, _ctx):
                return None

        with mock.patch.object(npv8, "_load_mtmd_shim", return_value=FakeLibrary()):
            with self.assertRaisesRegex(RuntimeError, "lacks the independent preprocessing"):
                npv8._run_llamacpp_preprocess(
                    Path("shim"), Path("model"), b"\x00\x00\x00", 1, 1, None, None,
                )

    def test_preprocess_parity_requires_finite_bounded_error(self) -> None:
        self.assertTrue(npv8._preprocess_parity_pass({"max_abs": 5.0e-8}))
        self.assertFalse(npv8._preprocess_parity_pass({"max_abs": 0.01}))
        self.assertFalse(npv8._preprocess_parity_pass({"max_abs": float("nan")}))
        self.assertFalse(npv8._preprocess_parity_pass({"max_abs": 0.0, "rmse": float("nan")}))

    def test_image_parity_input_uses_production_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "red.ppm"
            image.write_bytes(b"P6\n1 1\n255\n\xff\x00\x00")
            report = npv8._load_image_file(
                image,
                1,
                1,
                {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]},
            )
        self.assertEqual(report["interleaved"], [1.0, -1.0, -1.0])
        self.assertEqual(report["planar"], [1.0, -1.0, -1.0])
        self.assertIn("normalize_mean_std", report["preprocess"])

    def test_bicubic_center_padding_preserves_aspect_and_black_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "wide.ppm"
            image.write_bytes(b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 255, 0]))
            report = npv8._load_image_file(
                image, 4, 4,
                {"image_resize_algorithm": "bicubic", "image_resize_padding": "center_ceil"},
            )
        values = report["interleaved"]
        self.assertEqual(values[:12], [-1.0] * 12)
        self.assertEqual(values[-12:], [-1.0] * 12)
        self.assertNotEqual(values[12:24], [-1.0] * 12)
        self.assertIn("bicubic_center_ceil", report["preprocess"])

    def test_unsupported_resize_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "sample.ppm"
            image.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
            with self.assertRaisesRegex(RuntimeError, "unsupported image resize contract"):
                npv8._load_image_file(
                    image, 1, 1, {"image_resize_algorithm": "bicubic", "image_resize_padding": "none"},
                )

    def test_activation_runtime_base_uses_aligned_arena_boundary(self) -> None:
        layout = {
            "memory": {
                "arena": {"activations_base": 752269632},
                "weights": {"base_offset": 508, "size": 752269120},
            }
        }
        image = {"offset": 80640}

        self.assertEqual(npv8._activation_runtime_base(layout), 752269632)
        self.assertEqual(npv8._activation_runtime_offset(layout, image), 752350272)

    def test_activation_runtime_base_supports_legacy_layouts(self) -> None:
        layout = {"memory": {"weights": {"base_offset": 508, "size": 752269120}}}

        self.assertEqual(npv8._activation_runtime_base(layout), 752269628)

    def test_strict_ggml_loader_uses_active_build_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bin_dir = root / "build" / "bin"
            bin_dir.mkdir(parents=True)
            expected = [
                bin_dir / "libggml-base.so",
                bin_dir / "libggml.so",
                bin_dir / "libggml-cpu.so",
            ]
            for path in expected:
                path.write_bytes(b"library")
            stale = bin_dir / "libggml-cpu.so.0.9.8"
            stale.write_bytes(b"stale")

            loaded: list[Path] = []

            def fake_cdll(path: str, **_kwargs):
                loaded.append(Path(path))
                return object()

            with (
                mock.patch.object(npv8, "LLAMA_CPP_ROOT", root),
                mock.patch.object(npv8.ctypes, "CDLL", side_effect=fake_cdll),
            ):
                cpu_path = npv8._load_ggml_cpu_global()

        self.assertEqual(loaded, expected)
        self.assertEqual(cpu_path, expected[-1].resolve())
        self.assertNotIn(stale, loaded)

    def test_generated_engine_prefers_runtime_local_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = Path(tmpdir)
            model = runtime / "libmodel.so"
            engine = runtime / "libckernel_engine.so"
            model.write_bytes(b"model")
            engine.write_bytes(b"engine")
            self.assertEqual(npv8._resolve_generated_engine(model), engine.resolve())

    def test_generated_engine_honors_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "runtime" / "libmodel.so"
            engine = root / "explicit" / "libckernel_engine.so"
            model.parent.mkdir()
            engine.parent.mkdir()
            model.write_bytes(b"model")
            engine.write_bytes(b"engine")
            with mock.patch.dict("os.environ", {"CK_ENGINE_SO": str(engine)}):
                self.assertEqual(npv8._resolve_generated_engine(model), engine.resolve())

    def test_llama_reference_output_name_uses_public_encode_for_qwen3vl(self) -> None:
        config = {
            "projector_out_dim": 4096,
            "projector_total_out_dim": 16384,
        }
        self.assertIsNone(npv8._llama_reference_output_name(config))

    def test_llama_reference_output_name_uses_public_encode_when_dims_match(self) -> None:
        config = {
            "projector_out_dim": 4096,
            "projector_total_out_dim": 4096,
        }
        self.assertIsNone(npv8._llama_reference_output_name(config))

    def test_resolve_llama_reference_output_name_honors_explicit_named_dump(self) -> None:
        config = {
            "projector_out_dim": 4096,
            "projector_total_out_dim": 16384,
        }
        self.assertEqual(npv8._resolve_llama_reference_output_name(config, "projector_out"), "projector_out")
        self.assertIsNone(npv8._resolve_llama_reference_output_name(config, "clip_encode_float_image"))

    def test_resolve_ck_output_contract_supports_explicit_bridge_alias(self) -> None:
        layout = {
            "config": {
                "projection_dim": 4096,
                "projector_out_dim": 4096,
                "projector_total_out_dim": 16384,
                "vision_merged_tokens": 576,
            }
        }
        offsets = {
            "embedded_input": {"size_bytes": 576 * 4096 * 4},
            "vision_output": {"size_bytes": 576 * 16384 * 4},
        }
        contract = npv8._resolve_ck_output_contract(layout, offsets, "vision_bridge_output")
        self.assertEqual(contract["named_activation"], "vision_bridge_output")
        self.assertEqual(contract["fallback_buffer_name"], "embedded_input")
        self.assertEqual(contract["used_nbytes"], 576 * 4096 * 4)
        self.assertEqual(contract["resolved_output"], "vision_bridge_output")

    def test_load_runtime_metadata_recovers_config_and_weights_bump(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text(json.dumps({"image_size": 768}), encoding="utf-8")
            (root / "weights.bump").write_bytes(b"stub")
            report = npv8._load_runtime_metadata({"metrics": {"max_abs": 1.0}}, root)
        self.assertEqual(report["config"]["image_size"], 768)
        self.assertTrue(str(report["weights_bump"]).endswith("weights.bump"))

    def test_read_named_llama_dump_tensor_flattens_named_record(self) -> None:
        fake_dump = SimpleNamespace(op_name="projector_out", data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        with mock.patch.object(npv8.parity_test, "read_dump_file", return_value=[fake_dump]):
            values = npv8._read_named_llama_dump_tensor(Path("/tmp/fake.bin"), "projector_out")
        self.assertEqual(list(values), [1.0, 2.0, 3.0, 4.0])


    def test_encoder_prefix_suite_loads_limited_summary_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text(
                json.dumps({
                    "results": [
                        {"id": "1 81", "image": "/tmp/1_81.ppm"},
                        {"id": "Fake 1", "image": "/tmp/Fake_1.ppm"},
                        {"id": "missing"},
                    ]
                }),
                encoding="utf-8",
            )
            specs = prefix_suite._load_image_specs(summary_path, [], 2)
        self.assertEqual(specs, [
            {"id": "1 81", "image": "/tmp/1_81.ppm"},
            {"id": "Fake 1", "image": "/tmp/Fake_1.ppm"},
        ])
        self.assertEqual(prefix_suite._sanitize_id("1 81"), "1_81")

    def test_encoder_prefix_suite_thresholds_shape_and_metrics(self) -> None:
        values = 36 * 28 * 16384
        sample = {
            "id": "sample",
            "status": "complete",
            "grid": [36, 28],
            "num_values": values,
            "raw_num_values": {"ck": values, "llama": values},
            "metrics": {"cosine": 0.999, "rmse": 0.01, "mean_abs": 0.001, "max_abs": 1.0},
            "shape_ok": True,
            "expected_values": values,
        }
        shape_ok, expected = prefix_suite._shape_status(sample, 16384)
        self.assertTrue(shape_ok)
        self.assertEqual(expected, values)
        args = SimpleNamespace(min_cosine=0.99, max_rmse=0.03, max_abs=None)
        self.assertEqual(prefix_suite._evaluate_samples([sample], args), [])
        args = SimpleNamespace(min_cosine=0.9999, max_rmse=0.03, max_abs=None)
        self.assertIn("cosine", prefix_suite._evaluate_samples([sample], args)[0])

    def test_encoder_prefix_suite_rejects_incomplete_or_malformed_measurements(self) -> None:
        args = SimpleNamespace(min_cosine=0.99, max_rmse=0.03, max_abs=None)
        base = {
            "id": "sample", "status": "complete", "shape_ok": True,
            "metrics": {"cosine": 1.0, "rmse": 0.0, "mean_abs": 0.0, "max_abs": 0.0},
        }
        self.assertEqual(prefix_suite._evaluate_samples([base], args), [])
        for metrics in ({}, {"cosine": 1.0}, {"cosine": 1.0, "rmse": 0.0, "mean_abs": 0.0},
                        {"cosine": 1.0, "rmse": "0", "mean_abs": 0.0, "max_abs": 0.0},
                        {"cosine": 1.0, "rmse": float("nan"), "mean_abs": 0.0, "max_abs": 0.0}):
            with self.subTest(metrics=metrics):
                failures = prefix_suite._evaluate_samples([{**base, "metrics": metrics}], args)
                self.assertTrue(any("metrics" in failure for failure in failures))
        failures = prefix_suite._evaluate_samples([{**base, "status": "fail"}], args)
        self.assertTrue(any("not a complete measurement" in failure for failure in failures))

    def test_encoder_prefix_suite_requires_positive_exact_geometry(self) -> None:
        sample = {"grid": [1, 2], "num_values": 8, "raw_num_values": {"ck": 8, "llama": 8}}
        self.assertEqual(prefix_suite._shape_status(sample, 4), (True, 8))
        for broken in (
            {**sample, "grid": [0, 2]},
            {**sample, "grid": [1.5, 2]},
            {**sample, "num_values": 8.0},
            {**sample, "raw_num_values": {"ck": 8}},
            {**sample, "raw_num_values": {"ck": 8, "llama": 7}},
        ):
            with self.subTest(broken=broken):
                self.assertFalse(prefix_suite._shape_status(broken, 4)[0])

    def test_encoder_prefix_suite_rejects_stale_invocation_and_changed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "image.ppm"
            image.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
            gguf = root / "mmproj.gguf"
            gguf.write_bytes(b"model")
            runtime = root / "runtime"
            runtime.mkdir()
            paths = {
                "mmproj_gguf": gguf,
                "ck_model_library": runtime / "libqwen3vl_mmproj_v8.so",
                "ck_generated_source": runtime / "qwen3_vl_mmproj_v8.c",
                "ck_weights": runtime / "weights.bump",
                "ck_manifest": runtime / "weights_manifest.map",
                "llama_shim_library": runtime / "libmtmd_clip_shim.so",
                "ck_engine_library": runtime / "libckernel_engine.so",
                "llama_mtmd_library": root / "build" / "bin" / "libmtmd.so",
            }
            for path in paths.values():
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"valid")
            args = SimpleNamespace(
                gguf=gguf, output_dir=root, runtime_dir=runtime,
                threads=1, ck_threads=1, image_min_tokens=None, image_max_tokens=8,
                embed_dim=4, reuse_reports=False, independent_preprocess=True,
            )
            spec = {"id": "image", "image": str(image)}
            report_path = root / "reports" / "01_image.json"

            def write_report(cmd, **_kwargs):
                invocation = cmd[cmd.index("--invocation-id") + 1]
                report_path.write_text(json.dumps({
                    "status": "complete", "invocation_id": invocation,
                    "gguf": str(gguf), "image_path": str(image),
                    "source_image_sha256": prefix_suite.hashlib.sha256(image.read_bytes()).hexdigest(),
                    "input_provenance": "independently_preprocessed_from_shared_decoded_rgb8",
                    "preprocess_evidence": {"verdict": "pass"},
                    "merged_grid": [1, 1], "num_values": 4,
                    "raw_num_values": {"ck": 4, "llama": 4},
                    "metrics": {"cosine": 1.0, "rmse": 0.0, "mean_abs": 0.0, "max_abs": 0.0},
                    "artifact_identity_kind": "selected_file_hashes_after_execution",
                    "artifact_identity": {
                        role: {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                               "sha256": prefix_suite.hashlib.sha256(path.read_bytes()).hexdigest()}
                        for role, path in paths.items()
                    },
                }), encoding="utf-8")
                return SimpleNamespace(returncode=0)

            env = {"CK_LLAMA_CPP_ROOT": str(root)}
            with mock.patch.object(prefix_suite.subprocess, "run", side_effect=write_report):
                valid = prefix_suite._run_one(spec=spec, index=1, args=args, env=env)
            self.assertNotIn("evidence_error", valid)

            def stale_report(cmd, **kwargs):
                result = write_report(cmd, **kwargs)
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                payload["invocation_id"] = "older-run"
                report_path.write_text(json.dumps(payload), encoding="utf-8")
                return result

            with mock.patch.object(prefix_suite.subprocess, "run", side_effect=stale_report):
                stale = prefix_suite._run_one(spec=spec, index=1, args=args, env=env)
            self.assertIn("invocation ID differs", stale["evidence_error"])

            def changed_artifact(cmd, **kwargs):
                result = write_report(cmd, **kwargs)
                paths["ck_model_library"].write_bytes(b"changed")
                return result

            with mock.patch.object(prefix_suite.subprocess, "run", side_effect=changed_artifact):
                changed = prefix_suite._run_one(spec=spec, index=1, args=args, env=env)
            self.assertIn("ck_model_library artifact", changed["evidence_error"])

    def test_encoder_prefix_suite_retains_report_after_child_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "image.ppm"
            image.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
            args = SimpleNamespace(
                gguf=root / "mmproj.gguf", output_dir=root, runtime_dir=root / "runtime",
                threads=1, ck_threads=1, image_min_tokens=None, image_max_tokens=8,
                embed_dim=4, reuse_reports=False, independent_preprocess=True,
                min_cosine=0.99, max_rmse=0.03, max_abs=None,
            )
            spec = {"id": "image", "image": str(image)}

            def run_and_fail(_cmd, **_kwargs):
                self.assertIn("--independent-preprocess", _cmd)
                report = root / "reports" / "01_image.json"
                report.write_text(json.dumps({
                    "gguf": str(args.gguf), "image_path": str(image),
                    "source_image_sha256": prefix_suite.hashlib.sha256(image.read_bytes()).hexdigest(),
                    "input_provenance": "independently_preprocessed_from_shared_decoded_rgb8",
                    "preprocess_evidence": {"verdict": "fail"},
                    "merged_grid": [1, 1], "num_values": 4,
                    "raw_num_values": {"ck": 4, "llama": 4},
                    "metrics": {"cosine": 0.5, "rmse": 1.0, "max_abs": 2.0},
                }), encoding="utf-8")
                return SimpleNamespace(returncode=1)

            with mock.patch.object(prefix_suite.subprocess, "run", side_effect=run_and_fail):
                sample = prefix_suite._run_one(spec=spec, index=1, args=args, env={})
            self.assertEqual(sample["preprocess_evidence"]["verdict"], "fail")
            self.assertEqual(sample["metrics"]["rmse"], 1.0)
            failures = prefix_suite._evaluate_samples([sample], args)
            self.assertTrue(any("numeric parity process exited 1" in item for item in failures))
            self.assertTrue(any("independent preprocessing did not pass" in item for item in failures))

    def test_encoder_prefix_suite_rejects_stale_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = root / "reports" / "01_image.json"
            report.parent.mkdir()
            report.write_text('{"metrics": {"cosine": 1.0}}', encoding="utf-8")
            args = SimpleNamespace(
                gguf=root / "mmproj.gguf", output_dir=root, runtime_dir=root / "runtime",
                threads=1, ck_threads=1, image_min_tokens=None, image_max_tokens=None,
                embed_dim=4, reuse_reports=False, independent_preprocess=False,
            )
            with mock.patch.object(prefix_suite.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                sample = prefix_suite._run_one(
                    spec={"id": "image", "image": str(root / "image.ppm")},
                    index=1, args=args, env={},
                )
            self.assertIn("no fresh report", sample["execution_error"])
            self.assertFalse(sample["shape_ok"])

    def test_encoder_prefix_suite_cannot_suppress_independent_failures(self) -> None:
        for bypass in ("--reuse-reports", "--no-fail"):
            with self.subTest(bypass=bypass), self.assertRaises(SystemExit) as exit_info:
                prefix_suite.main(["--gguf", "missing.gguf", "--independent-preprocess", bypass])
            self.assertEqual(exit_info.exception.code, 2)

    def test_encoder_prefix_suite_summary_keeps_failed_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            specs = [
                {"id": "failed", "image": str(root / "failed.ppm")},
                {"id": "passed", "image": str(root / "passed.ppm")},
            ]
            passed = {
                "id": "passed", "status": "complete", "shape_ok": True,
                "metrics": {"cosine": 1.0, "rmse": 0.0, "mean_abs": 0.0, "max_abs": 0.0},
            }
            failed = {"id": "failed", "shape_ok": False, "execution_error": "child failed"}
            with (
                mock.patch.object(prefix_suite, "_load_image_specs", return_value=specs),
                mock.patch.object(prefix_suite, "_run_one", side_effect=[failed, passed]),
                mock.patch("builtins.print"),
            ):
                code = prefix_suite.main(["--gguf", "fixture.gguf", "--output-dir", str(root)])
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual([item["id"] for item in summary["samples"]], ["failed", "passed"])
            self.assertTrue(any("child failed" in item for item in summary["failures"]))


if __name__ == "__main__":
    unittest.main()
