from __future__ import annotations

from array import array
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/vision_mtmd_prefix_parity_v8.py"
BRIDGE = ROOT / "version/v8/scripts/run_multimodal_bridge_v8.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = load_module("vision_mtmd_prefix_parity_v8_test", SCRIPT)
bridge = load_module("run_multimodal_bridge_v8_prefix_test", BRIDGE)


class MtmdPrefixParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="v8_mtmd_prefix_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.ppm"
        self.image.write_bytes(b"P6\n2 1\n255\n" + bytes((255, 0, 0, 0, 0, 255)))
        self.cke = self.root / "cke.f32"
        self.cke.write_bytes(array("f", [1.0, 2.0, 3.0, 4.0]).tobytes())
        self.oracle = self.root / "oracle.f32"
        self.oracle.write_bytes(array("f", [1.0, 2.0, 3.0, 4.0]).tobytes())
        self.library = self.root / "libencoder.so"
        self.library.write_bytes(b"test library")
        loaded = bridge._load_image_file(self.image, 1, 2)
        self.report = {
            "status": "ok",
            "prefix_source": "encoder",
            "prefix_tokens": 2,
            "prefix_embed_dim": 2,
            "prefix_dump_path": str(self.cke),
            "prefix_dump_sha256": parity.sha256(self.cke),
            "encoder_runtime": {"so_path": str(self.library)},
            "encoder_report": {
                "image_source": "file",
                "image_path": str(self.image),
                "image_sha256": loaded["image_sha256"],
                "decoded_rgb8_sha256": loaded["decoded_rgb8_sha256"],
                "prefix_tokens": 2,
                "embed_dim": 2,
                "model_library_sha256": parity.sha256(self.library),
            },
        }

    def test_file_image_records_encoded_and_decoded_identities(self) -> None:
        result = bridge._load_image_file(self.image, 1, 2)
        self.assertEqual(result["image_sha256"], hashlib.sha256(self.image.read_bytes()).hexdigest())
        self.assertEqual(result["decoded_rgb8_sha256"], hashlib.sha256(bytes((255, 0, 0, 0, 0, 255))).hexdigest())

    def test_valid_report_and_exact_comparison(self) -> None:
        self.assertEqual(parity.validate_cke_report(self.report, self.image), (self.cke, 2, 2))
        result = parity.compare_prefixes(self.cke, self.oracle, 2, 2)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["max_abs"], 0.0)

    def test_rejects_shared_or_synthetic_prefix(self) -> None:
        self.report["prefix_source"] = "synthetic_zero"
        with self.assertRaisesRegex(ValueError, "generated encoder"):
            parity.validate_cke_report(self.report, self.image)

    def test_rejects_swapped_image_and_modified_bytes(self) -> None:
        other = self.root / "other.ppm"
        other.write_bytes(self.image.read_bytes())
        with self.assertRaisesRegex(ValueError, "image path"):
            parity.validate_cke_report(self.report, other)
        self.image.write_bytes(self.image.read_bytes() + b"X")
        with self.assertRaisesRegex(ValueError, "image bytes changed"):
            parity.validate_cke_report(self.report, self.image)

    def test_rejects_missing_hashes_and_stale_prefix(self) -> None:
        del self.report["encoder_report"]["decoded_rgb8_sha256"]
        with self.assertRaisesRegex(ValueError, "decoded RGB8"):
            parity.validate_cke_report(self.report, self.image)
        self.report["encoder_report"]["decoded_rgb8_sha256"] = "0" * 64
        self.cke.write_bytes(array("f", [0.0, 2.0, 3.0, 4.0]).tobytes())
        with self.assertRaisesRegex(ValueError, "hash is missing or stale"):
            parity.validate_cke_report(self.report, self.image)

    def test_rejects_stale_encoder_library(self) -> None:
        self.library.write_bytes(b"different library")
        with self.assertRaisesRegex(ValueError, "library changed"):
            parity.validate_cke_report(self.report, self.image)

    def test_rejects_wrong_extent_and_nonfinite_values(self) -> None:
        self.oracle.write_bytes(array("f", [1.0]).tobytes())
        with self.assertRaisesRegex(ValueError, "extents disagree"):
            parity.compare_prefixes(self.cke, self.oracle, 2, 2)
        self.oracle.write_bytes(array("f", [1.0, float("nan"), 3.0, 4.0]).tobytes())
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            parity.compare_prefixes(self.cke, self.oracle, 2, 2)

    def test_numerical_mismatch_is_not_coverage_failure(self) -> None:
        self.oracle.write_bytes(array("f", [1.0, 2.0, 3.0, 5.0]).tobytes())
        result = parity.compare_prefixes(self.cke, self.oracle, 2, 2)
        self.assertEqual(result["rmse"], 0.5)
        self.assertEqual(result["max_abs"], 1.0)
        self.assertEqual((result["worst_token"], result["worst_channel"]), (1, 1))

    def test_failed_invocation_clears_prior_pass(self) -> None:
        old_result = self.root / "prefix-parity.json"
        old_result.write_text('{"status":"pass"}', encoding="utf-8")
        argv = [
            "vision_mtmd_prefix_parity_v8.py", "--cke-report", str(self.root / "missing.json"),
            "--image", str(self.image), "--model", str(self.root / "missing.gguf"),
            "--mmproj", str(self.root / "missing-mmproj.gguf"),
            "--llama-root", str(self.root), "--llama-build", str(self.root),
            "--output-dir", str(self.root), "--max-rmse", "0.001", "--max-abs", "0.01",
        ]
        with mock.patch("sys.argv", argv), self.assertRaises(FileNotFoundError):
            parity.main()
        self.assertFalse(old_result.exists())


if __name__ == "__main__":
    unittest.main()
