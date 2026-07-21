#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "xray_attention_sensitivity_v8.py"


def load_module():
    spec = importlib.util.spec_from_file_location("xray_attention_sensitivity_v8", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xray = load_module()


try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class XRayAttentionSensitivityTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(7)
        values = {
            "q": rng.standard_normal((2, 5, 8), dtype=np.float32),
            "k": rng.standard_normal((2, 5, 8), dtype=np.float32),
            "v": rng.standard_normal((2, 5, 8), dtype=np.float32),
        }
        return values, {name: value.copy() for name, value in values.items()}

    def test_identical_inputs_report_no_surviving_delta(self):
        reference, subject = self.fixture()
        report = xray.analyze_attention_sensitivity(
            reference, subject, storage_dtype="bf16", query_start=1, query_count=2
        )
        xray.validate_report(report)
        self.assertEqual(report["status"], "observed")
        for case in report["cases"].values():
            self.assertEqual(case["observation"], "NO_SURVIVING_STORAGE_DELTA")
            self.assertIsNone(case["forward_amplification_l2"])
            self.assertIsNone(case["backward_amplification_l2"])

    def test_q_perturbation_reports_forward_and_vjp_sensitivity(self):
        reference, subject = self.fixture()
        subject["q"][0, 2, 3] += np.float32(0.25)
        report = xray.analyze_attention_sensitivity(
            reference,
            subject,
            storage_dtype="bf16",
            query_start=1,
            query_count=3,
            probe_seed=11,
        )
        q_only = report["cases"]["q_only"]
        self.assertEqual(q_only["observation"], "SENSITIVITY_OBSERVED")
        self.assertGreater(q_only["input_delta_l2"], 0.0)
        self.assertGreater(q_only["output_delta"]["l2"], 0.0)
        self.assertGreater(q_only["vjp_delta_l2"], 0.0)
        self.assertIsNotNone(q_only["forward_amplification_l2"])
        self.assertEqual(report["cases"]["k_only"]["observation"], "NO_SURVIVING_STORAGE_DELTA")

    def test_fixed_probe_is_deterministic(self):
        reference, subject = self.fixture()
        subject["v"][1, 4, 2] += np.float32(0.5)
        first = xray.analyze_attention_sensitivity(
            reference, subject, storage_dtype="bf16", probe_seed=19, threads=1
        )
        second = xray.analyze_attention_sensitivity(
            reference, subject, storage_dtype="bf16", probe_seed=19, threads=1
        )
        self.assertEqual(first["cases"], second["cases"])

    def test_shape_mismatch_fails_before_execution(self):
        reference, subject = self.fixture()
        subject["k"] = subject["k"][:, :-1, :]
        with self.assertRaisesRegex(xray.SensitivityError, "shape mismatch"):
            xray.analyze_attention_sensitivity(reference, subject)

    def test_bounded_causal_query_is_rejected(self):
        reference, subject = self.fixture()
        with self.assertRaisesRegex(xray.SensitivityError, "not position-equivalent"):
            xray.analyze_attention_sensitivity(
                reference, subject, causal=True, query_start=1, query_count=2
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
