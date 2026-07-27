#!/usr/bin/env python3
"""Regression tests for xray_qwen3vl_llamacpp_v8 producer canonicalization."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "version" / "v8" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import xray_qwen3vl_llamacpp_v8 as producer


def _profile() -> dict:
    return {
        "schema": "cke.parity_profile",
        "backend": "llamacpp",
        "checkpoint_order": [
            "vision.frontend.patch_bias.output",
            "vision.frontend.position.output",
            "vision.layer.{layer}.attention.output",
        ],
        "observed_storage": {"default": "fp32", "checkpoints": {}},
        "backend_mappings": {
            "vision.frontend.patch_bias.output": {
                "producer": "patch_bias_add",
                "logical_layout": "token_major",
                "axis_names": ["token", "channel"],
                "capture_tensor": "patch_bias",
                "result_tensor": "patch_bias",
                "result_layer": -1,
            },
            "vision.frontend.position.output": {
                "producer": "position_embeddings",
                "logical_layout": "token_major",
                "axis_names": ["token", "channel"],
                "capture_tensor": "inp_pos_emb",
                "result_tensor": "inp_pos_emb",
                "result_layer": -1,
            },
            "vision.layer.{layer}.attention.output": {
                "producer": "attn",
                "logical_layout": "token_major",
                "axis_names": ["token", "channel"],
                "capture_tensor": "kqv_out",
                "result_tensor": "kqv_out",
            },
        },
    }


def _row(layer: int, op: str, status: str, **metrics) -> dict:
    row = {"layer": layer, "op": op, "status": status}
    row.update(metrics)
    return row


class TestNormalizeCaptureReport(unittest.TestCase):
    """Verify the llama.cpp producer emits the canonical X-ray checkpoint contract."""

    def test_all_profile_checkpoints_emitted_in_order(self) -> None:
        capture = {
            "results": [
                _row(-1, "patch_bias", "FAIL", max_abs_diff=1.95, mean_abs_diff=0.024),
                _row(-1, "inp_pos_emb", "FAIL", max_abs_diff=1.95, mean_abs_diff=0.024),
                _row(0, "kqv_out", "FAIL", max_abs_diff=0.4, mean_abs_diff=0.01),
            ]
        }
        report = producer.normalize_capture_report(capture, _profile(), 0)
        ids = [c["checkpoint_id"] for c in report["comparisons"]]
        self.assertEqual(
            ids,
            [
                "vision.frontend.patch_bias.output",
                "vision.frontend.position.output",
                "vision.layer.0.attention.output",
            ],
        )

    def test_mae_never_populates_rmse(self) -> None:
        capture = {
            "results": [
                _row(
                    -1, "patch_bias", "FAIL",
                    max_abs_diff=1.95, mean_abs_diff=0.0238,
                    max_rel_err=4957.5, mean_rel_err=0.737,
                )
            ]
        }
        report = producer.normalize_capture_report(capture, _profile(), 0)
        metrics = report["comparisons"][0]["metrics"]
        self.assertEqual(metrics["max_abs"], 1.95)
        self.assertEqual(metrics["mean_abs"], 0.0238)
        self.assertIsNone(metrics["rmse"])
        self.assertIsNone(metrics["relative_rmse"])
        self.assertEqual(metrics["mean_rel_err"], 0.737)

    def test_rmse_passthrough_when_capture_provides_it(self) -> None:
        capture = {"results": [_row(-1, "patch_bias", "FAIL", max_abs_diff=1.0, rmse=0.5)]}
        report = producer.normalize_capture_report(capture, _profile(), 0)
        self.assertEqual(report["comparisons"][0]["metrics"]["rmse"], 0.5)

    def test_unresolved_execution_fields_stay_null(self) -> None:
        capture = {"results": [_row(0, "kqv_out", "PASS", max_abs_diff=0.0, mean_abs_diff=0.0)]}
        report = producer.normalize_capture_report(capture, _profile(), 0)
        comp = report["comparisons"][2]
        exec_meta = comp["resolved_execution"]
        # Semantic fields from the profile are kept.
        self.assertEqual(exec_meta["producer"], "attn")
        self.assertEqual(exec_meta["phase"], "prefill")
        self.assertEqual(exec_meta["layer"], 0)
        # Nothing is fabricated.
        self.assertIsNone(exec_meta["op_idx"])
        self.assertIsNone(exec_meta["function"])
        self.assertIsNone(exec_meta["kernel_id"])
        self.assertIsNone(exec_meta["resolved_contract_id"])
        self.assertIsNone(exec_meta["storage_dtype"])
        self.assertIsNone(exec_meta["exported_dtype"])

    def test_profile_declared_dtype_is_observed_not_resolved(self) -> None:
        profile = _profile()
        profile["observed_storage"] = {
            "default": "q8_0",
            "checkpoints": {"vision.layer.{layer}.attention.output": "q8_0"},
        }
        capture = {"results": [_row(0, "kqv_out", "PASS", max_abs_diff=0.0)]}
        report = producer.normalize_capture_report(capture, profile, 0)
        exec_meta = report["comparisons"][2]["resolved_execution"]
        self.assertEqual(exec_meta["observed_dtype"], "q8_0")
        self.assertIsNone(exec_meta["storage_dtype"])

    def test_missing_checkpoint_explicit(self) -> None:
        capture = {"results": []}
        report = producer.normalize_capture_report(capture, _profile(), 0)
        self.assertEqual(len(report["comparisons"]), 3)
        for comp in report["comparisons"]:
            self.assertEqual(comp["status"], "fail")
            self.assertEqual(comp["classification"], "MISSING_CHECKPOINT")

    def test_later_failures_are_observed_not_propagated(self) -> None:
        capture = {
            "results": [
                _row(-1, "patch_bias", "FAIL", max_abs_diff=1.95),
                _row(-1, "inp_pos_emb", "FAIL", max_abs_diff=1.95),
                _row(0, "kqv_out", "FAIL", max_abs_diff=0.4),
            ]
        }
        report = producer.normalize_capture_report(capture, _profile(), 0)
        comps = report["comparisons"]
        self.assertEqual(comps[0]["classification"], "KERNEL_IMPLEMENTATION_DIVERGENCE")
        self.assertEqual(comps[1]["classification"], "OBSERVED_DIVERGENCE")
        self.assertEqual(comps[2]["classification"], "OBSERVED_DIVERGENCE")
        self.assertEqual(
            report["first_divergence"]["checkpoint_id"], "vision.frontend.patch_bias.output"
        )

    def test_unmatched_capture_rows_disclosed(self) -> None:
        capture = {
            "results": [
                _row(-1, "ffn_gelu", "MISSING"),
                _row(-1, "patch_bias", "FAIL", max_abs_diff=1.95),
            ]
        }
        report = producer.normalize_capture_report(capture, _profile(), 0)
        unmatched = report["unmatched_capture_rows"]
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched[0]["op"], "ffn_gelu")
        self.assertEqual(unmatched[0]["layer"], -1)

    def test_circuit_scope_declared(self) -> None:
        capture = {"results": []}
        report = producer.normalize_capture_report(capture, _profile(), 0)
        self.assertEqual(report["circuit_scope"], "vision_encoder")

    def test_byte_exact_when_max_abs_zero(self) -> None:
        capture = {"results": [_row(0, "kqv_out", "PASS", max_abs_diff=0.0, mean_abs_diff=0.0)]}
        report = producer.normalize_capture_report(capture, _profile(), 0)
        metrics = report["comparisons"][2]["metrics"]
        self.assertTrue(metrics["byte_exact"])
        self.assertEqual(metrics["exact_ratio"], 1.0)


class TestProvenance(unittest.TestCase):
    """Run provenance: complete runtime fingerprints, fail-closed when missing."""

    def _args(self, gguf: Path) -> argparse.Namespace:
        return argparse.Namespace(gguf=gguf)

    def _capture(self, llama_oracle: dict | None) -> dict:
        binary = {
            "engine": {"sha256": "b" * 64},
            "generated_model": {"sha256": "e" * 64},
            "llama_shim": {"sha256": "c" * 64},
        }
        if llama_oracle is not None:
            binary["llama_oracle"] = llama_oracle
        return {"binary_provenance": binary, "llama_flash_attn": "disabled"}

    def test_phase_is_producer_owned_prefill(self) -> None:
        self.assertEqual(producer.CAPTURE_PHASE, "prefill")
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "model.gguf"
            gguf.write_bytes(b"gguf")
            prov = producer._build_provenance(
                self._args(gguf), self._capture(None), "run-1", producer.CAPTURE_PHASE
            )
        self.assertEqual(prov["phase"], "prefill")

    def test_complete_oracle_fingerprint_passes_through(self) -> None:
        oracle = {
            "commit": "abc123",
            "mode": "flash-disabled",
            "components": [{"name": "libmtmd_clip_shim.so", "path": "/x", "sha256": "c" * 64}],
            "fingerprint_sha256": "f" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "model.gguf"
            gguf.write_bytes(b"gguf")
            prov = producer._build_provenance(
                self._args(gguf), self._capture(oracle), "run-1", "prefill"
            )
        self.assertEqual(prov["oracle"]["fingerprint_sha256"], "f" * 64)
        self.assertEqual(prov["oracle"]["commit"], "abc123")
        self.assertEqual(prov["oracle"]["mode"], "flash-disabled")
        self.assertEqual(len(prov["oracle"]["components"]), 1)
        self.assertEqual(prov["subject"]["runtime_sha256"], "b" * 64)
        self.assertEqual(prov["subject"]["generated_model_sha256"], "e" * 64)

    def test_missing_oracle_identity_stays_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "model.gguf"
            gguf.write_bytes(b"gguf")
            prov = producer._build_provenance(
                self._args(gguf), self._capture(None), "run-1", "prefill"
            )
        self.assertIsNone(prov["oracle"]["fingerprint_sha256"])
        self.assertIsNone(prov["oracle"]["commit"])
        # mode is still derivable from the capture's flash-attn flag
        self.assertEqual(prov["oracle"]["mode"], "flash-disabled")


if __name__ == "__main__":
    unittest.main()
