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
            "grid": [36, 28],
            "num_values": values,
            "raw_num_values": {"ck": values, "llama": values},
            "metrics": {"cosine": 0.999, "rmse": 0.01, "max_abs": 1.0},
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


if __name__ == "__main__":
    unittest.main()
