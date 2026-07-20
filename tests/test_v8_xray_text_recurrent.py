from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("xray_text_recurrent_v8", SCRIPT_DIR / "xray_text_recurrent_v8.py")
XRAY = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(XRAY)


class TextRecurrentXRayTests(unittest.TestCase):
    def test_named_axis_state_transform_prevents_flat_layout_false_positive(self) -> None:
        ck = np.arange(2 * 128 * 128, dtype=np.float32).reshape(2, 128, 128)
        oracle = ck.transpose(0, 2, 1).copy()
        result = XRAY.compare_arrays("new_state", ck.reshape(-1), oracle.reshape(-1))
        self.assertEqual(result["status"], "exact")
        self.assertIn("head,value,key", result["axis_transform"])

    def test_exact_input_then_projection_difference_is_provider_mismatch(self) -> None:
        schedules = {"ck": "sequential_decode", "oracle_prefix": "batched", "oracle_decode": "sequential"}
        rows = [
            {"logical_token": 3, "layer": 0, "boundary": "attn_norm", "status": "exact"},
            {"logical_token": 3, "layer": 0, "boundary": "linear_attn_qkv_mixed", "status": "different", "max_abs_diff": 0.0047},
        ]
        result = XRAY.classify(rows, schedules)
        self.assertEqual(result["classification"], "PROJECTION_PROVIDER_MISMATCH")
        self.assertEqual(result["previous_exact_boundary"], "attn_norm")

    def test_batched_prompt_rows_are_compared_by_logical_token(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(12, dtype=np.float32).reshape(3, 4)
            values.tofile(root / "attn_norm-0-token-000002-occ-000.bin")
            row = XRAY._load_oracle_row(root, "attn_norm", 0, 1, 3, 4)
            np.testing.assert_array_equal(row, values[1])

    def test_schedule_metadata_is_mandatory_in_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = XRAY.analyze_capture(root / "ck", root / "llama", 3, 3, 0)
            self.assertEqual(report["schedules"]["ck"], "sequential_decode")
            self.assertEqual(report["schedules"]["oracle_prefix"], "batched")
            self.assertEqual(report["schedules"]["oracle_decode"], "sequential")
            self.assertIn("only prioritizes attribution", report["acceptance_policy"])

    def test_state_axis_extent_is_not_hardcoded_to_one_model(self) -> None:
        ck = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        oracle = ck.transpose(0, 2, 1).copy()
        result = XRAY.compare_arrays("new_state", ck.reshape(-1), oracle.reshape(-1), state_size=4)
        self.assertEqual(result["status"], "exact")


if __name__ == "__main__":
    unittest.main()
