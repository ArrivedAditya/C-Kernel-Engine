#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xray = load_module("xray_numerical_parity_v8", SCRIPTS / "xray_numerical_parity_v8.py")
builder = load_module("build_xray_checkpoint_manifest_v8", SCRIPTS / "build_xray_checkpoint_manifest_v8.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class XRayNumericalParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="xray_v8_")
        self.root = Path(self.temp.name)
        self.profile = {
            "schema": "cke.parity_profile", "schema_version": 1, "name": "test", "backend": "pytorch",
            "contract_schema_version": 1,
            "required_match_fields": ["checkpoint_id", "producer", "logical_layout", "axis_names", "resolved_contract_id", "kernel_id", "function"],
            "observed_storage": {"default": "fp32", "checkpoints": {}},
            "dtype_thresholds": {
                "fp32": {"cosine_min": 0.99999, "rmse_max": 1e-4, "relative_rmse_max": 1e-4, "max_abs_max": 1e-3, "finite_required": True},
                "bf16": {"cosine_min": 0.999, "rmse_max": 0.02, "relative_rmse_max": 0.02, "max_abs_max": 0.25, "finite_required": True},
            },
            "checkpoint_order": ["vision.layer.0.output", "vision.layer.8.output"],
            "interval_expansions": {}, "backend_mappings": {},
        }

    def tearDown(self):
        self.temp.cleanup()

    def entry(self, checkpoint: str, path: Path, *, storage="fp32", producer="block", physical_axes=None):
        return {
            "checkpoint_id": checkpoint, "producer": producer, "phase": "prefill", "layer": 0,
            "tensor_path": str(path), "storage_dtype": storage, "exported_dtype": "fp32",
            "logical_shape": [2, 3], "physical_shape": [2, 3], "logical_layout": "token_major",
            "axis_names": ["token", "channel"], "physical_axis_names": physical_axes or ["token", "channel"],
            "resolved_contract_id": "contract.test", "kernel_id": "kernel.test", "function": "kernel_test",
            "sha256": digest(path),
        }

    def manifest(self, backend: str, entries):
        return {"schema": "cke.checkpoint_manifest", "schema_version": 1, "backend": backend,
                "run": {"model": "fixture", "phase": "prefill", "source": "unit"}, "checkpoints": entries}

    def test_matching_tensors_pass_and_report_worst_coordinate(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        values.tofile(a); values.tofile(b)
        left = self.manifest("ck", [self.entry("vision.layer.0.output", a)])
        right = self.manifest("pytorch", [self.entry("vision.layer.0.output", b)])
        result = xray.compare_manifests(left, right, self.profile, checkpoint_order=["vision.layer.0.output"])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["comparisons"][0]["classification"], "MATCH")

    def test_named_axes_are_canonicalized_before_comparison(self):
        logical = np.arange(6, dtype=np.float32).reshape(2, 3)
        a = self.root / "a.f32"; b = self.root / "b.f32"
        logical.T.tofile(a); logical.tofile(b)
        left_entry = self.entry("vision.layer.0.output", a, physical_axes=["channel", "token"])
        left_entry["physical_shape"] = [3, 2]
        left = self.manifest("ck", [left_entry])
        right = self.manifest("pytorch", [self.entry("vision.layer.0.output", b)])
        self.assertEqual(xray.compare_manifests(left, right, self.profile, checkpoint_order=["vision.layer.0.output"])["status"], "pass")

    def test_storage_mismatch_is_classified_before_value_comparison(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        np.zeros(6, np.float32).tofile(a); np.zeros(6, np.float32).tofile(b)
        left = self.manifest("ck", [self.entry("vision.layer.0.output", a, storage="fp32")])
        right = self.manifest("pytorch", [self.entry("vision.layer.0.output", b, storage="bf16")])
        result = xray.compare_manifests(left, right, self.profile, checkpoint_order=["vision.layer.0.output"])
        self.assertEqual(result["first_divergence"]["classification"], "STORAGE_CONTRACT_MISMATCH")
        self.assertEqual(result["first_divergence"]["fix_owner"], "circuit_and_kernel_map")
        self.assertIn("Do not add model-name", result["architecture_policy"]["forbidden_fix"])

    def test_producer_mismatch_is_not_mislabeled_as_kernel_math(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        np.zeros(6, np.float32).tofile(a); np.ones(6, np.float32).tofile(b)
        left = self.manifest("ck", [self.entry("vision.layer.0.output", a, producer="wrong")])
        right = self.manifest("pytorch", [self.entry("vision.layer.0.output", b)])
        result = xray.compare_manifests(left, right, self.profile, checkpoint_order=["vision.layer.0.output"])
        self.assertEqual(result["first_divergence"]["classification"], "CIRCUIT_PRODUCER_MISMATCH")
        self.assertNotIn("metrics", result["first_divergence"])

    def test_value_failure_reports_logical_coordinate(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        got = np.zeros((2, 3), np.float32); ref = np.zeros((2, 3), np.float32)
        got[1, 2] = 1.0
        got.tofile(a); ref.tofile(b)
        result = xray.compare_manifests(
            self.manifest("ck", [self.entry("vision.layer.0.output", a)]),
            self.manifest("pytorch", [self.entry("vision.layer.0.output", b)]), self.profile,
            checkpoint_order=["vision.layer.0.output"],
        )
        divergence = result["first_divergence"]
        self.assertEqual(divergence["classification"], "KERNEL_IMPLEMENTATION_DIVERGENCE")
        self.assertEqual(divergence["metrics"]["worst_coordinate"], {"token": 1, "channel": 2})

    def test_ranking_failure_is_reported_after_tensor_passes(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        np.arange(6, dtype=np.float32).tofile(a); np.arange(6, dtype=np.float32).tofile(b)
        ranking = {"schema": "cke.xray_ranking_report", "schema_version": 1,
                   "checks": [{"kind": "teacher_forced", "position": 12, "status": "fail", "ck_top1": 4, "oracle_top1": 5}]}
        result = xray.compare_manifests(
            self.manifest("ck", [self.entry("vision.layer.0.output", a)]),
            self.manifest("pytorch", [self.entry("vision.layer.0.output", b)]), self.profile, ranking,
            checkpoint_order=["vision.layer.0.output"],
        )
        self.assertEqual(result["first_divergence"]["classification"], "RANKING_DIVERGENCE")

    def test_builder_uses_call_ir_as_checkpoint_authority(self):
        tensor = self.root / "tensor.f32"; np.arange(6, dtype=np.float32).tofile(tensor)
        checkpoint = {
            "id": "vision.layer.0.output", "producer": "mlp_residual", "tensor": "layer_out",
            "logical_layout": "token_major", "axis_names": ["token", "channel"], "storage_dtype": "fp32",
            "phase": "prefill", "layer": 0, "kernel_id": "residual", "function": "residual_forward",
            "resolved_contract_id": "contract.residual",
        }
        call_ir = {"operations": [{"semantic_checkpoints": [checkpoint]}]}
        report = {"torch": {"tensors": {"layer_out@0": {"path": str(tensor), "shape": [2, 3]}}},
                  "comparisons": {"layer_out@0": {"ck_path": str(tensor), "shape": [6]}}}
        manifest = builder.build_manifest(
            backend="pytorch", call_ir=call_ir, tensor_report=report, model="fixture", source="unit",
            phase="prefill", storage_dtype_override="bf16",
        )
        self.assertEqual(manifest["checkpoints"][0]["producer"], "mlp_residual")
        self.assertEqual(manifest["checkpoints"][0]["kernel_id"], "residual")
        self.assertEqual(manifest["checkpoints"][0]["storage_dtype"], "bf16")

    def test_requested_missing_checkpoint_is_a_diagnostic_failure(self):
        a = self.root / "a.f32"; np.zeros(6, np.float32).tofile(a)
        result = xray.compare_manifests(
            self.manifest("ck", [self.entry("vision.layer.0.output", a)]),
            self.manifest("pytorch", [self.entry("vision.layer.0.output", a)]), self.profile,
        )
        self.assertEqual(result["first_divergence"]["classification"], "MISSING_CHECKPOINT")
        self.assertIn("exporter", result["first_divergence"]["recommended_action"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
