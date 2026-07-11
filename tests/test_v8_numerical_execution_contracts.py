#!/usr/bin/env python3
"""Fail-closed tests for generic v8 numerical execution contracts."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "resolve_numerical_execution_contracts_v8.py"
SPEC = importlib.util.spec_from_file_location("resolve_numerical_execution_contracts_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)

PLANNER_SCRIPT = ROOT / "version" / "v8" / "scripts" / "plan_parity_bisection_v8.py"
PLANNER_SPEC = importlib.util.spec_from_file_location("plan_parity_bisection_v8", PLANNER_SCRIPT)
assert PLANNER_SPEC is not None and PLANNER_SPEC.loader is not None
planner = importlib.util.module_from_spec(PLANNER_SPEC)
PLANNER_SPEC.loader.exec_module(planner)


CONTRACT_ID = "bf16_weight_bf16_input_fp32_dot_fp32_output"


def circuit(validation: str = "observed"):
    return {
        "name": "contract_test",
        "required_numerical_contracts": {
            "gemm": {
                "op": "gemm",
                "template_ops": ["mlp_up"],
                "phases": {
                    "prefill": {
                        "contract_id": CONTRACT_ID,
                        "validation": validation,
                        "evidence": "tests/test_v8_numerical_execution_contracts.py",
                    }
                },
                "checkpoint": {
                    "id": "vision.layer.0.mlp.up",
                    "producer": "mlp_up",
                    "logical_layout": "token_major",
                    "axis_names": ["token", "channel"],
                },
            }
        },
    }


class NumericalExecutionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts = resolver.load_json(resolver.DEFAULT_CONTRACTS)
        cls.kernels = resolver.load_kernel_capabilities(contracts=cls.contracts)

    def test_exact_bf16_kernel_resolution_preserves_semantics(self):
        plan = resolver.resolve_contract(
            circuit(), self.contracts, self.kernels, "gemm", "prefill"
        )
        self.assertEqual(plan["kernel"]["id"], "gemm_nt_bf16")
        self.assertEqual(plan["kernel"]["function"], "gemm_nt_bf16")
        self.assertEqual(plan["contract"]["semantics"]["compute"]["input"], "bf16_rne")
        self.assertEqual(plan["contract"]["semantics"]["reduction"]["order"], "ascending_k")
        self.assertFalse(
            plan["contract"]["semantics"]["threading"]["thread_count_changes_arithmetic_order"]
        )
        self.assertEqual(plan["checkpoint"]["axis_names"], ["token", "channel"])

    def test_zero_provider_is_hard_failure(self):
        kernels = copy.deepcopy(self.kernels)
        kernels["kernels"] = {}
        with self.assertRaisesRegex(resolver.ContractError, "resolved to 0 kernels"):
            resolver.resolve_contract(circuit(), self.contracts, kernels, "gemm", "prefill")

    def test_multiple_providers_are_hard_failure(self):
        kernels = copy.deepcopy(self.kernels)
        duplicate = copy.deepcopy(kernels["kernels"]["gemm_nt_bf16"])
        duplicate["id"] = "gemm_nt_bf16_duplicate"
        kernels["kernels"][duplicate["id"]] = duplicate
        with self.assertRaisesRegex(resolver.ContractError, "resolved to 2 kernels"):
            resolver.resolve_contract(circuit(), self.contracts, kernels, "gemm", "prefill")

    def test_decode_cannot_silently_use_prefill_provider(self):
        doc = circuit()
        doc["required_numerical_contracts"]["gemm"]["phases"]["decode"] = copy.deepcopy(
            doc["required_numerical_contracts"]["gemm"]["phases"]["prefill"]
        )
        with self.assertRaisesRegex(resolver.ContractError, "resolved to 0 kernels"):
            resolver.resolve_contract(doc, self.contracts, self.kernels, "gemm", "decode")

    def test_production_rejects_observed_contract(self):
        with self.assertRaisesRegex(resolver.ContractError, "production resolution uses unvalidated"):
            resolver.resolve_contract(
                circuit(), self.contracts, self.kernels, "gemm", "prefill", mode="production"
            )

    def test_arithmetic_capability_mismatch_is_hard_failure(self):
        capability = copy.deepcopy(
            self.kernels["kernels"]["gemm_nt_bf16"]["capabilities"][0]
        )
        capability["arithmetic"]["thread_count_changes_arithmetic_order"] = True
        with self.assertRaisesRegex(resolver.ContractError, "arithmetic metadata disagrees"):
            resolver._validate_capability_against_contract(
                "gemm_nt_bf16", capability, self.contracts["contracts"][CONTRACT_ID]
            )

    def test_multisection_rope_sections_must_mean_axis_selection(self):
        contracts = copy.deepcopy(self.contracts)
        base = copy.deepcopy(contracts["contracts"][CONTRACT_ID])
        base["position_transform"] = {
            "pairing": "multi_section",
            "rotary_width": "mrope_n_dims",
            "head_width": "head_dim",
            "position_rank": 3,
            "axis_order": ["temporal", "height", "width"],
            "section_interpretation": "contiguous_widths",
            "frequency_compute": "fp32",
            "intermediate_compute": "fp32",
            "rounding_points": ["output_store"],
            "threading": "independent_tokens",
        }
        contracts["contracts"]["qwen_mrope_invalid"] = base
        with self.assertRaisesRegex(resolver.ContractError, "redefines rotary width"):
            resolver.validate_contract_registry(contracts)

    def test_mrope_width_must_match_full_rotary_width(self):
        contracts = copy.deepcopy(self.contracts)
        base = copy.deepcopy(contracts["contracts"][CONTRACT_ID])
        base["position_transform"] = {
            "pairing": "multi_section",
            "rotary_width": "mrope_n_dims",
            "head_width": "head_dim",
            "rotary_width_value": 128,
            "head_width_value": 128,
            "mrope_n_dims_value": 64,
            "position_rank": 3,
            "axis_order": ["temporal", "height", "width"],
            "section_interpretation": "axis_selection",
            "frequency_compute": "fp32",
            "intermediate_compute": "fp32",
            "rounding_points": ["output_store"],
            "threading": "independent_tokens",
        }
        contracts["contracts"]["qwen_mrope_bad_width"] = base
        with self.assertRaisesRegex(resolver.ContractError, "inconsistent M-RoPE width"):
            resolver.validate_contract_registry(contracts)

    def test_sparse_failure_produces_bounded_granular_request(self):
        profile = planner.load(
            ROOT / "version" / "v8" / "parity_profiles" / "qwen3vl_pytorch_bf16_v1.json"
        )
        report = {
            "comparisons": [
                {"checkpoint_id": "vision.frontend.position.output", "status": "pass"},
                {"checkpoint_id": "vision.layer.0.output", "status": "pass"},
                {"checkpoint_id": "vision.layer.8.output", "status": "pass"},
                {"checkpoint_id": "vision.layer.16.output", "status": "fail"},
            ]
        }
        result = planner.plan(profile, report)
        self.assertEqual(result["status"], "granular")
        self.assertEqual(result["interval"], "vision.layer.8.output->vision.layer.16.output")
        self.assertEqual(result["next_checkpoints"][0], "vision.layer.9.output")
        self.assertEqual(result["next_checkpoints"][-1], "vision.layer.15.output")

    def test_graph_ir_metadata_retains_contract_and_checkpoint(self):
        scripts = ROOT / "version" / "v8" / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("build_ir_v8_contract_test", scripts / "build_ir_v8.py")
            assert spec is not None and spec.loader is not None
            build_ir = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(build_ir)
        finally:
            sys.path.pop(0)
        plan = resolver.resolve_contract(
            circuit(), self.contracts, self.kernels, "gemm", "prefill"
        )
        metadata = build_ir._graph_ir_contract_metadata(plan)
        self.assertEqual(metadata["required_contract_id"], CONTRACT_ID)
        self.assertEqual(metadata["resolved_contract_id"], CONTRACT_ID)
        self.assertEqual(metadata["kernel_id"], "gemm_nt_bf16")
        self.assertEqual(metadata["function"], "gemm_nt_bf16")
        self.assertEqual(metadata["semantics"]["rounding"]["points"], ["input_load"])
        self.assertEqual(metadata["checkpoint"]["id"], "vision.layer.0.mlp.up")


if __name__ == "__main__":
    unittest.main(verbosity=2)
