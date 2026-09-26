"""Cheap format, provenance and rejection tests for Kokoro's offline importer."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "version/v8/tts/export_kokoro_bump.py"
SPEC = importlib.util.spec_from_file_location("kokoro_bump_export", EXPORTER)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class KokoroBumpExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.config = {
            "n_token": 178, "hidden_dim": 512,
            "plbert": {
                "intermediate_size": 2048,
                "max_position_embeddings": 512,
                "num_attention_heads": 12,
            },
        }
        self.tensors = {
            "phoneme_encoder.embedding.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
            "voice.fixed.predictor": np.array([0.25, -0.5], dtype=np.float32),
        }
        self.origins = {
            name: {"source_name": name, "transform": "identity"}
            for name in self.tensors
        }

    def write(self):
        return exporter.write_bundle(
            self.output, self.tensors, self.origins, self.config,
            {"fixture": "synthetic; no model execution"},
        )

    def test_canonical_names_and_parametrization_rejection(self):
        self.assertEqual(
            exporter.canonical_name("bert.embeddings.word_embeddings.weight"),
            "phoneme_encoder.embeddings.word_embeddings.weight",
        )
        self.assertEqual(
            exporter.canonical_name("decoder.decode.0.conv1.weight"),
            "waveform_decoder.decode.0.conv1.weight",
        )
        for name in (
            "decoder.decode.0.conv1.weight_g",
            "decoder.decode.0.conv1.parametrizations.weight.original0",
            "unknown.layer.weight",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                exporter.canonical_name(name)

    def test_roundtrip_header_manifest_and_exact_payloads(self):
        manifest = self.write()
        verified = exporter.verify_bundle(self.output)
        self.assertEqual(verified, manifest)
        self.assertEqual([entry["name"] for entry in manifest["entries"]], sorted(self.tensors))
        for entry in manifest["entries"]:
            self.assertEqual(entry["file_offset"] % 64, 0)
            mapped = np.memmap(
                self.output / "weights.bump", mode="r", dtype="<f4",
                offset=entry["file_offset"], shape=tuple(entry["shape"]),
            )
            self.assertTrue(np.array_equal(mapped, self.tensors[entry["name"]]))
        self.assertEqual(
            json.loads((self.output / "config.json").read_text()), self.config,
        )

    def test_corrupted_data_manifest_and_footer_rejected(self):
        self.write()
        path = self.output / "weights.bump"
        manifest_path = self.output / "weights_manifest.json"
        original_data = path.read_bytes()
        original_manifest = manifest_path.read_bytes()
        entry = json.loads(original_manifest)["entries"][0]
        with path.open("r+b") as stream:
            stream.seek(entry["file_offset"])
            stream.write(b"\xff")
        with self.assertRaisesRegex(ValueError, "data hash mismatch"):
            exporter.verify_bundle(self.output)
        path.write_bytes(original_data)
        manifest = json.loads(original_manifest)
        manifest["entries"][0]["size"] += 4
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            exporter.verify_bundle(self.output)
        manifest_path.write_bytes(original_manifest)
        with path.open("r+b") as stream:
            stream.seek(-48, 2)
            stream.write(b"BADMAGIC")
        with self.assertRaisesRegex(ValueError, "metadata footer"):
            exporter.verify_bundle(self.output)

    def test_reject_nonfinite_or_mismatched_origins(self):
        self.tensors["voice.fixed.predictor"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite FP32"):
            self.write()
        self.assertFalse((self.output / "weights.bump").exists())
        self.tensors["voice.fixed.predictor"][0] = 0.25
        self.origins.pop("voice.fixed.predictor")
        with self.assertRaisesRegex(ValueError, "origin sets"):
            self.write()
        self.origins["voice.fixed.predictor"] = {
            "source_name": "voice", "transform": "identity", "file_offset": 0,
        }
        with self.assertRaisesRegex(ValueError, "expected source_name"):
            self.write()

    def test_missing_and_mismatched_pinned_assets_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing pinned"):
            exporter.verify_pinned_assets(self.output)
        (self.output / "config.json").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "does not match pinned"):
            exporter.verify_pinned_assets(self.output)


if __name__ == "__main__":
    unittest.main()
