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

    def test_missing_checkpoint_does_not_hide_previous_exact_boundary(self) -> None:
        schedules = {"ck": "sequential_decode", "oracle_prefix": "batched", "oracle_decode": "sequential"}
        rows = [
            {"logical_token": 3, "layer": 7, "boundary": "attn_pregate", "status": "exact"},
            {"logical_token": 3, "layer": 7, "boundary": "attn_out", "status": "missing_or_incompatible"},
            {"logical_token": 3, "layer": 7, "boundary": "out_proj", "status": "different", "max_abs_diff": 0.0047},
        ]
        result = XRAY.classify(rows, schedules)
        self.assertEqual(result["boundary"], "out_proj")
        self.assertEqual(result["previous_exact_boundary"], "attn_pregate")

    def test_previous_boundary_is_scoped_to_same_token_and_layer(self) -> None:
        schedules = {"ck": "sequential_decode", "oracle_prefix": "batched", "oracle_decode": "sequential"}
        rows = [
            {"logical_token": 2, "layer": 3, "boundary": "layer_out", "status": "exact"},
            {"logical_token": 3, "layer": 3, "boundary": "attn_norm", "status": "different", "max_abs_diff": 0.0047},
        ]
        result = XRAY.classify(rows, schedules)
        self.assertIsNone(result["previous_exact_boundary"])

    def test_small_normalization_difference_then_gate_drift_is_amplification(self) -> None:
        schedules = {"ck": "batched_then_sequential", "oracle_prefix": "batched", "oracle_decode": "sequential"}
        rows = [
            {"logical_token": 16, "layer": 0, "boundary": "attn_norm", "status": "different", "max_abs_diff": 9.54e-7},
            {"logical_token": 16, "layer": 0, "boundary": "linear_attn_qkv_mixed", "status": "different", "max_abs_diff": 3.81e-6},
            {"logical_token": 16, "layer": 0, "boundary": "alpha", "status": "different", "max_abs_diff": 1.60e-3},
        ]
        result = XRAY.classify(rows, schedules)
        self.assertEqual(result["classification"], "NORMALIZATION_TO_QUANTIZATION_AMPLIFICATION")
        self.assertEqual(result["amplification_source"], "attn_norm")

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

    def test_hybrid_ck_prompt_rows_are_loaded_from_full_prefill_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(12, dtype=np.float32).reshape(3, 4)
            values.tofile(root / "tok_0000_layer_000_z.f32")
            row = XRAY._load_ck_row(root, "z", 0, 2, 3, 4, "hybrid")
            np.testing.assert_array_equal(row, values[2])

    def test_hybrid_ck_attention_rows_are_canonicalized_from_head_major(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
            values.tofile(root / "tok_0000_layer_000_attn_pregate.f32")
            row = XRAY._load_ck_row(
                root, "attn_pregate", 0, 1, 3, 4, "hybrid", attention_heads=2
            )
            np.testing.assert_array_equal(row, values[:, 1, :].reshape(-1))

    def test_hybrid_ck_gated_attention_rows_remain_token_major_after_transpose(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(12, dtype=np.float32).reshape(3, 4)
            values.tofile(root / "tok_0000_layer_000_attn_out.f32")
            row = XRAY._load_ck_row(root, "attn_out", 0, 1, 3, 4, "hybrid", attention_heads=2)
            np.testing.assert_array_equal(row, values[1])

    def test_hybrid_schedule_metadata_matches_batched_oracle_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = XRAY.analyze_capture(
                root / "ck", root / "llama", 3, 3, 0, ck_prefill_mode="hybrid"
            )
            self.assertEqual(report["schedules"]["ck_prefix"], "batched")
            self.assertEqual(report["schedules"]["oracle_prefix"], "batched")
            self.assertEqual(report["schedules"]["ck_decode"], "sequential")

    def test_circuit_checkpoint_maps_to_oracle_graph_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(12, dtype=np.float32).reshape(3, 4)
            values.tofile(root / "attn_residual-0-token-000002-occ-000.bin")
            row = XRAY._load_oracle_row(root, "after_attn", 0, 1, 3, 4)
            np.testing.assert_array_equal(row, values[1])

    def test_layer_composition_edges_are_part_of_xray_contract(self) -> None:
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["after_attn"], "attn_residual")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["out_proj"], "attn_output")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["q_proj"], "Qcur_full")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["attn_gate"], "gate_reshaped")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["attn_pregate"], "attn_pregate")
        self.assertLess(XRAY.BOUNDARIES.index("attn_pregate"), XRAY.BOUNDARIES.index("attn_out"))
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["attn_out"], "attn_gated")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["post_attn_norm"], "attn_post_norm")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["mlp_swiglu"], "ffn_swiglu")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["mlp_down"], "ffn_out")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["layer_out"], "l_out")
        for boundary in XRAY.ORACLE_BOUNDARY_NAMES:
            self.assertIn(boundary, XRAY.BOUNDARIES)

    def test_reused_oracle_graph_label_selects_declared_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = np.arange(4, dtype=np.float32)
            second = first + np.float32(10.0)
            first.tofile(root / "Kcur-3-token-000004-occ-000.bin")
            second.tofile(root / "Kcur-3-token-000004-occ-001.bin")
            row = XRAY._load_oracle_row(root, "rope_k", 3, 4, 4, 4)
            np.testing.assert_array_equal(row, second)

    def test_state_axis_extent_is_not_hardcoded_to_one_model(self) -> None:
        ck = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        oracle = ck.transpose(0, 2, 1).copy()
        result = XRAY.compare_arrays("new_state", ck.reshape(-1), oracle.reshape(-1), state_size=4)
        self.assertEqual(result["status"], "exact")


if __name__ == "__main__":
    unittest.main()
