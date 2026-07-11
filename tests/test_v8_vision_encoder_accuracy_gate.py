#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/vision_encoder_accuracy_gate_v8.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("vision_encoder_accuracy_gate_v8", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisionEncoderAccuracyGateTests(unittest.TestCase):
    def test_relative_rmse_is_scale_aware(self) -> None:
        import numpy as np
        spec = importlib.util.spec_from_file_location("compare_qwen3vl_bf16", ROOT / "version/v8/scripts/compare_qwen3vl_bf16_vision_hidden_v8.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = module._metrics(np.array([1000.0, -1000.0], dtype=np.float32), np.array([1010.0, -1010.0], dtype=np.float32))
        self.assertAlmostEqual(metrics["rmse"], 10.0, places=5)
        self.assertAlmostEqual(metrics["relative_rmse"], 0.01, places=5)

    def test_large_public_fixture_is_deterministic_ppm(self) -> None:
        gate = _load_gate()
        with tempfile.TemporaryDirectory(prefix="vision_encoder_fixture_") as tmp:
            first = Path(tmp) / "first.ppm"
            second = Path(tmp) / "second.ppm"
            gate._write_large_form(first)
            gate._write_large_form(second)
            data = first.read_bytes()
            self.assertTrue(data.startswith(b"P6\n1152 896\n255\n"))
            self.assertEqual(data, second.read_bytes())
            self.assertEqual(len(data), len(b"P6\n1152 896\n255\n") + 1152 * 896 * 3)

    def test_missing_artifact_is_skip_or_required_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vision_encoder_gate_") as tmp:
            out = Path(tmp) / "skip"
            base = [
                sys.executable,
                str(SCRIPT),
                "--mode",
                "q4",
                "--allow-non-avx512",
                "--output-dir",
                str(out),
            ]
            skipped = subprocess.run(base, cwd=ROOT, check=False, capture_output=True, text=True)
            self.assertEqual(skipped.returncode, 0, skipped.stdout + skipped.stderr)
            self.assertEqual(json.loads((out / "summary.json").read_text())["status"], "skip")

            required_out = Path(tmp) / "required"
            required = subprocess.run(
                [*base[:-1], str(required_out), "--require-artifacts"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(required.returncode, 0)
            report = json.loads((required_out / "summary.json").read_text())
            self.assertEqual(report["status"], "fail")
            self.assertTrue(report["failures"])


if __name__ == "__main__":
    unittest.main()
