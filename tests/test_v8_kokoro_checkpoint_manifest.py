"""Validate the pinned predictor checkpoint geometry and capture inventory."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MANIFEST = Path(__file__).resolve().parents[1] / "version/v8/tts/reference/fixture_manifest.json"


class KokoroCheckpointManifestTests(unittest.TestCase):
    def test_preprocessing_rejects_wrong_pinned_config_before_model_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")
            result = subprocess.run([
                sys.executable,
                str(MANIFEST.with_name("capture_kokoro_v1.py")),
                "--model-dir", str(model_dir),
                "--output-dir", str(Path(temporary) / "out"),
                "--preprocess-only",
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA-256", result.stderr)

    def test_predictor_scan_and_adaln_edges_are_captured(self):
        manifest = json.loads(MANIFEST.read_text())
        hooks = manifest["fine_predictor_hooks"]
        self.assertEqual(hooks, [
            f"predictor.text_encoder.lstms.{index}" for index in range(6)
        ])
        tensors = manifest["tensors"]
        self.assertEqual(len(tensors), 63)
        for index in (0, 2, 4):
            stem = f"predictor_text_encoder_lstms_{index}"
            with self.subTest(module=stem):
                self.assertEqual(tensors[f"{stem}_input_0_0"]["shape"], [36, 640])
                self.assertEqual(tensors[f"{stem}_output_0_0"]["shape"], [36, 512])
                for suffix in ("input_0_1", "output_0_1"):
                    self.assertEqual(tensors[f"{stem}_{suffix}"]["shape"], [36])
                    self.assertEqual(tensors[f"{stem}_{suffix}"]["dtype"], "int64")
                for suffix in ("output_1_0", "output_1_1"):
                    self.assertEqual(tensors[f"{stem}_{suffix}"]["shape"], [2, 1, 256])
        for index in (1, 3, 5):
            stem = f"predictor_text_encoder_lstms_{index}"
            with self.subTest(module=stem):
                self.assertEqual(tensors[f"{stem}_input_0"]["shape"], [1, 36, 512])
                self.assertEqual(tensors[f"{stem}_input_1"]["shape"], [1, 128])
                self.assertEqual(tensors[f"{stem}_output"]["shape"], [1, 36, 512])
        self.assertEqual(manifest["waveform"]["frames"], 61800)
        self.assertEqual(manifest["tensors"]["waveform_f32"]["sha256"],
                         "6c651c788bb7cc71969d2f2b9b710d6a08cd0c32e4ee3f3161388e9f78f9c48a")


if __name__ == "__main__":
    unittest.main()
