#!/usr/bin/env python3
"""Unit tests for monotonic X-ray provider promotion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "compare_xray_monotonic_v8.py"
SPEC = importlib.util.spec_from_file_location("compare_xray_monotonic_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, relative_rmse: float, max_abs: float) -> None:
    path.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "checkpoint_id": "vision.layer.0.output",
                        "metrics": {"relative_rmse": relative_rmse, "max_abs": max_abs},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class XrayMonotonicProviderGateTests(unittest.TestCase):
    def test_accepts_bounded_capture_report_and_canonicalizes_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "baseline.json", root / "candidate.json"
            baseline.write_text(
                json.dumps(
                    {
                        "comparisons": {
                            "layer_out@24": {
                                "relative_rmse": 0.02,
                                "max_abs": 0.5,
                            },
                            "vision_output": {
                                "relative_rmse": 0.03,
                                "max_abs": 0.75,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(
                    {
                        "comparisons": {
                            "layer_out@24": {
                                "relative_rmse": 0.019,
                                "max_abs": 0.5,
                            },
                            "vision_output": {
                                "relative_rmse": 0.029,
                                "max_abs": 0.5,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            report = MODULE.compare_reports(
                baseline,
                candidate,
                ("vision.layer.24.output", "vision.prefix.output"),
                1.0e-12,
            )
            self.assertEqual(report["status"], "pass")

    def test_accepts_non_increasing_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "baseline.json", root / "candidate.json"
            _write(baseline, 0.01, 0.25)
            _write(candidate, 0.009, 0.125)
            report = MODULE.compare_reports(
                baseline, candidate, ("vision.layer.0.output",), 1.0e-12
            )
            self.assertEqual(report["status"], "pass")

    def test_rejects_relative_or_worst_case_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "baseline.json", root / "candidate.json"
            _write(baseline, 0.01, 0.25)
            _write(candidate, 0.011, 0.5)
            report = MODULE.compare_reports(
                baseline, candidate, ("vision.layer.0.output",), 1.0e-12
            )
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["failures"], ["vision.layer.0.output"])

    def test_missing_checkpoint_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "baseline.json", root / "candidate.json"
            _write(baseline, 0.01, 0.25)
            _write(candidate, 0.009, 0.125)
            report = MODULE.compare_reports(
                baseline, candidate, ("vision.layer.8.output",), 1.0e-12
            )
            self.assertEqual(report["status"], "fail")
            self.assertIn("missing checkpoint", report["failures"][0])


if __name__ == "__main__":
    unittest.main()
