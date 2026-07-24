#!/usr/bin/env python3
"""Prove that X-ray diagnoses known injected faults in causal order."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NUMERICAL = load_module("xray_numerical_selftest", SCRIPTS / "xray_numerical_parity_v8.py")
EXECUTION = load_module("xray_execution_selftest", SCRIPTS / "xray_execution_state_v8.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class XRayValidatorSelfTest(unittest.TestCase):
    """Use intentionally bad fixtures; production kernels contain no fault switches."""

    CHECKPOINTS = [
        "vision.layer.0.output",
        "vision.layer.1.attention.output",
        "vision.layer.1.output",
        "vision.prefix.output",
        "decoder.layer.0.output",
    ]

    @classmethod
    def setUpClass(cls):
        cls.all_report_rows: list[dict[str, Any]] = []

    @classmethod
    def tearDownClass(cls):
        report_path = os.environ.get("CK_XRAY_SELFTEST_REPORT")
        if not report_path:
            return
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": "cke.xray_validator_selftest",
            "schema_version": 1,
            "status": "pass",
            "scenario_count": len(cls.all_report_rows),
            "scenarios": cls.all_report_rows,
        }, indent=2) + "\n", encoding="utf-8")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="xray_validator_selftest_")
        self.root = Path(self.temp.name)
        self.report_rows: list[dict[str, Any]] = []
        self.profile = {
            "schema": "cke.parity_profile",
            "schema_version": 1,
            "name": "xray-validator-selftest",
            "backend": "pytorch",
            "contract_schema_version": 1,
            "required_match_fields": [
                "checkpoint_id", "producer", "logical_layout", "axis_names",
                "resolved_contract_id", "kernel_id", "function",
            ],
            "observed_storage": {"default": "bf16", "checkpoints": {}},
            "dtype_thresholds": {
                "bf16": {
                    "cosine_min": 0.9999,
                    "rmse_max": 1.0e-4,
                    "relative_rmse_max": 1.0e-4,
                    "max_abs_max": 1.0e-3,
                    "finite_required": True,
                }
            },
            "checkpoint_order": self.CHECKPOINTS,
            "interval_expansions": {},
            "backend_mappings": {},
        }

    def tearDown(self):
        self.__class__.all_report_rows.extend(self.report_rows)
        self.temp.cleanup()

    def tensor(self, name: str, values: np.ndarray) -> Path:
        path = self.root / f"{name}.f32"
        np.asarray(values, dtype=np.float32).tofile(path)
        return path

    @staticmethod
    def metadata(checkpoint: str) -> tuple[str, str, str, int]:
        layer = 1 if ".layer.1." in checkpoint else 0
        if checkpoint == "vision.prefix.output":
            return "vision_projector", "projector.bf16", "projector_bf16", layer
        if checkpoint.startswith("decoder"):
            return "decoder_block", "decoder.block.bf16", "decoder_block_bf16", layer
        if ".attention." in checkpoint:
            return "attention", "attention.bf16", "attention_bf16", layer
        return "vision_block", "vision.block.bf16", "vision_block_bf16", layer

    def entry(self, checkpoint: str, path: Path) -> dict[str, Any]:
        producer, kernel_id, function, layer = self.metadata(checkpoint)
        return {
            "checkpoint_id": checkpoint,
            "producer": producer,
            "phase": "prefill",
            "layer": layer,
            "tensor_path": str(path),
            "storage_dtype": "bf16",
            "exported_dtype": "fp32",
            "logical_shape": [2, 4],
            "physical_shape": [2, 4],
            "logical_layout": "token_major",
            "axis_names": ["token", "channel"],
            "physical_axis_names": ["token", "channel"],
            "resolved_contract_id": f"contract.{kernel_id}",
            "kernel_id": kernel_id,
            "function": function,
            "sha256": sha256(path),
        }

    @staticmethod
    def manifest(backend: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema": "cke.checkpoint_manifest",
            "schema_version": 1,
            "backend": backend,
            "run": {"model": "synthetic-two-layer-vl", "phase": "prefill", "source": "xray-selftest"},
            "checkpoints": entries,
        }

    def numerical_pair(self) -> tuple[dict[str, Any], dict[str, Any]]:
        reference = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(8.0)
        subject_entries = []
        oracle_entries = []
        for index, checkpoint in enumerate(self.CHECKPOINTS):
            subject = self.tensor(f"subject-{index}", reference)
            oracle = self.tensor(f"oracle-{index}", reference)
            subject_entries.append(self.entry(checkpoint, subject))
            oracle_entries.append(self.entry(checkpoint, oracle))
        return self.manifest("ck", subject_entries), self.manifest("pytorch", oracle_entries)

    def record(self, name: str, result: dict[str, Any], expected_edge: str | None, expected_class: str | None):
        first = result.get("first_divergence")
        observed_edge = first.get("checkpoint_id") if first else None
        observed_class = first.get("classification") if first else None
        self.assertEqual(observed_edge, expected_edge, name)
        self.assertEqual(observed_class, expected_class, name)
        self.report_rows.append({
            "name": name,
            "status": "pass",
            "expected_edge": expected_edge,
            "observed_edge": observed_edge,
            "expected_classification": expected_class,
            "observed_classification": observed_class,
            "last_passing_checkpoint": result.get("last_passing_checkpoint"),
        })

    def test_numerical_and_circuit_faults_are_attributed_to_first_edge(self):
        subject, oracle = self.numerical_pair()
        clean = NUMERICAL.compare_manifests(subject, oracle, self.profile)
        self.assertEqual(clean["status"], "pass")
        self.record("clean-control", clean, None, None)

        storage_subject, storage_oracle = self.numerical_pair()
        storage_oracle["checkpoints"][1]["storage_dtype"] = "fp32"
        result = NUMERICAL.compare_manifests(storage_subject, storage_oracle, self.profile)
        self.record(
            "encoder-storage-boundary",
            result,
            "vision.layer.1.attention.output",
            "STORAGE_CONTRACT_MISMATCH",
        )

        circuit_subject, circuit_oracle = self.numerical_pair()
        circuit_subject["checkpoints"][3]["producer"] = "wrong_visual_consumer"
        result = NUMERICAL.compare_manifests(circuit_subject, circuit_oracle, self.profile)
        self.record(
            "bridge-producer-consumer",
            result,
            "vision.prefix.output",
            "CIRCUIT_PRODUCER_MISMATCH",
        )

        math_subject, math_oracle = self.numerical_pair()
        broken = np.fromfile(math_subject["checkpoints"][4]["tensor_path"], dtype=np.float32).reshape(2, 4)
        broken[1, 2] += np.float32(0.25)
        broken.tofile(math_subject["checkpoints"][4]["tensor_path"])
        math_subject["checkpoints"][4]["sha256"] = sha256(Path(math_subject["checkpoints"][4]["tensor_path"]))
        result = NUMERICAL.compare_manifests(math_subject, math_oracle, self.profile)
        self.record(
            "decoder-kernel-arithmetic",
            result,
            "decoder.layer.0.output",
            "KERNEL_IMPLEMENTATION_DIVERGENCE",
        )
        self.assertEqual(result["first_divergence"]["metrics"]["worst_coordinate"], {"token": 1, "channel": 2})

    @staticmethod
    def call(call_id: str, kind: str, start: int, count: int, action: str) -> dict[str, Any]:
        return {
            "call_id": call_id,
            "kind": kind,
            "start": start,
            "count": count,
            "position_start": start,
            "cache_action": action,
            "kernel_batches": [{
                "checkpoint_id": "decoder.layer.0.attention.output",
                "kernel_id": "attention.bf16",
                "numerical_contract_id": "attention.bf16.segmented",
                "effective_contract_id": "attention.bf16.segmented",
                "m": count,
                "n": 8,
                "k": 8,
            }],
        }

    def calls(self) -> list[dict[str, Any]]:
        return [
            self.call("text-before", "text", 0, 2, "reset"),
            self.call("visual", "visual", 2, 4, "append"),
            self.call("text-after", "text", 6, 2, "append"),
        ]

    def artifact(self, name: str, role: str, values: np.ndarray) -> dict[str, Any]:
        path = self.root / f"{name}.f16"
        np.asarray(values, dtype=np.float16).tofile(path)
        return {
            "checkpoint_id": f"decoder.layer.0.{role}",
            "role": role,
            "tensor_path": str(path),
            "dtype": "fp16",
            "shape": list(values.shape),
            "row_axis": 0,
            "sha256": sha256(path),
        }

    def trace(self, backend: str, calls: list[dict[str, Any]], artifacts: list[dict[str, Any]] | None = None):
        return {
            "schema": "cke.xray_execution_trace",
            "schema_version": 1,
            "backend": backend,
            "run": {"model": "synthetic-vl-decoder", "phase": "mixed_prefill", "source": "xray-selftest"},
            "execution": {"policy_id": "segmented_mixed_prefill", "calls": calls},
            "state": {
                "position_policy_id": "mrope",
                "position": [8, 8, 8],
                "cache_token_count": 8,
                "append_index": 7,
                "cache_layout": {
                    "layout_id": "layer_head_token_channel",
                    "base_address": "0x1000",
                    "layer_stride_bytes": 1024,
                    "head_stride_bytes": 256,
                    "token_stride_bytes": 16,
                    "channel_stride_bytes": 2,
                },
            },
            "artifacts": artifacts or [],
        }

    def assert_execution_fault(self, name: str, subject, oracle, expected_class: str, expected_stage: str):
        result = EXECUTION.compare_traces(subject, oracle)
        first = result["first_divergence"]
        self.assertEqual(first["classification"], expected_class, name)
        self.assertEqual(first["stage"], expected_stage, name)
        self.report_rows.append({
            "name": name,
            "status": "pass",
            "expected_stage": expected_stage,
            "observed_stage": first["stage"],
            "expected_classification": expected_class,
            "observed_classification": first["classification"],
        })

    def test_stateful_execution_faults_are_classified_before_arithmetic(self):
        combined = [self.call("combined", "mixed", 0, 8, "reset")]
        subject = self.trace("ck", combined)
        subject["execution"]["policy_id"] = "combined_mixed_prefill"
        self.assert_execution_fault(
            "segmented-prefill-contract",
            subject,
            self.trace("pytorch", self.calls()),
            "EXECUTION_POLICY_MISMATCH",
            "execution_contract",
        )

        subject = self.trace("ck", self.calls())
        subject["state"]["append_index"] = 6
        self.assert_execution_fault(
            "kv-cache-append-index",
            subject,
            self.trace("pytorch", self.calls()),
            "CACHE_STATE_METADATA_MISMATCH",
            "cache_metadata",
        )

        query = np.arange(16, dtype=np.float16).reshape(2, 8)
        expected = np.ones((2, 8), dtype=np.float16)
        actual = expected.copy()
        actual[1, 3] += np.float16(0.125)
        self.assert_execution_fault(
            "attention-identical-input-arithmetic",
            self.trace("ck", self.calls(), [
                self.artifact("subject-query", "query", query),
                self.artifact("subject-attention", "attention_output", actual),
            ]),
            self.trace("pytorch", self.calls(), [
                self.artifact("oracle-query", "query", query),
                self.artifact("oracle-attention", "attention_output", expected),
            ]),
            "ATTENTION_ARITHMETIC_DIVERGENCE",
            "attention_output",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
