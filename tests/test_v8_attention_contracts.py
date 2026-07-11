#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
V8_ROOT = REPO_ROOT / "version" / "v8"
SCRIPT = V8_ROOT / "scripts" / "resolve_attention_contracts_v8.py"
SPEC = importlib.util.spec_from_file_location("resolve_attention_contracts_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resolver
SPEC.loader.exec_module(resolver)


class AttentionContractV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.circuit_path = V8_ROOT / "circuits" / "qwen3vl.json"
        cls.circuit = resolver.load_json(cls.circuit_path)
        cls.vision_circuit_path = V8_ROOT / "circuits" / "qwen3_vl_vision.json"
        cls.vision_circuit = resolver.load_json(cls.vision_circuit_path)
        cls.contracts = resolver.load_json(resolver.DEFAULT_CONTRACTS)
        cls.kernels = resolver.load_kernel_capabilities()

    def resolve(self, phase: str, mode: str = "bringup"):
        return resolver.resolve_contract(
            copy.deepcopy(self.circuit),
            copy.deepcopy(self.contracts),
            copy.deepcopy(self.kernels),
            operation="decoder.attention",
            phase=phase,
            mode=mode,
            source_circuit_path=self.circuit_path,
        )

    def test_registry_defines_complete_semantics(self) -> None:
        resolver.validate_contract_registry(copy.deepcopy(self.contracts))

    def test_kernel_overlay_matches_v8_kernel_maps(self) -> None:
        resolver.validate_kernel_overlay(copy.deepcopy(self.kernels))

    def test_kernel_overlay_rejects_function_drift(self) -> None:
        kernels = copy.deepcopy(self.kernels)
        kernels["kernels"]["attention_forward_decode_head_major_gqa_flash_f16cache"][
            "supported_reductions"
        ]["f16_online_fp32_merge"]["function"] = "wrong_function"
        with self.assertRaisesRegex(resolver.ContractError, "kernel map names"):
            resolver.validate_kernel_overlay(kernels)

    def test_decode_bringup_resolves_requested_contract(self) -> None:
        result = self.resolve("decode")
        self.assertEqual(result["reduction"]["id"], "f16_online_fp32_merge")
        self.assertEqual(
            result["kernel"]["id"],
            "attention_forward_decode_head_major_gqa_flash_f16cache",
        )
        self.assertFalse(result["kernel"]["explicit_selector"])
        self.assertIn("kernel uses legacy implicit selection", result["production_blockers"])

    def test_prefill_bringup_resolves_separately(self) -> None:
        result = resolver.resolve_contract(
            copy.deepcopy(self.vision_circuit),
            copy.deepcopy(self.contracts),
            copy.deepcopy(self.kernels),
            operation="vision_encoder.attention",
            phase="prefill",
            mode="bringup",
            source_circuit_path=self.vision_circuit_path,
        )
        self.assertEqual(result["reduction"]["id"], "f16_kv_fp32_online")
        self.assertEqual(result["phase"], "prefill")

    def test_production_rejects_unvalidated_circuit_request(self) -> None:
        with self.assertRaisesRegex(resolver.ContractError, "Production contract resolution rejected"):
            self.resolve("decode", mode="production")

    def test_plain_dtype_is_not_a_reduction_contract(self) -> None:
        circuit = copy.deepcopy(self.circuit)
        circuit["required_contracts"]["decoder.attention"]["phases"]["decode"]["requires"][
            "numerics.attention_reduction"
        ] = "fp32"
        with self.assertRaisesRegex(resolver.ContractError, "ambiguous reduction"):
            resolver.resolve_contract(
                circuit,
                copy.deepcopy(self.contracts),
                copy.deepcopy(self.kernels),
                operation="decoder.attention",
                phase="decode",
                mode="bringup",
            )

    def test_unsupported_kernel_reduction_fails_without_fallback(self) -> None:
        circuit = copy.deepcopy(self.circuit)
        circuit["required_contracts"]["decoder.attention"]["phases"]["decode"]["requires"][
            "numerics.attention_reduction"
        ] = "fp32_online"
        with self.assertRaisesRegex(resolver.ContractError, "No kernel satisfies circuit requirements"):
            resolver.resolve_contract(
                circuit,
                copy.deepcopy(self.contracts),
                copy.deepcopy(self.kernels),
                operation="decoder.attention",
                phase="decode",
                mode="bringup",
            )

    def test_production_accepts_only_fully_validated_explicit_route(self) -> None:
        circuit = copy.deepcopy(self.circuit)
        contracts = copy.deepcopy(self.contracts)
        kernels = copy.deepcopy(self.kernels)
        circuit["required_contracts"]["decoder.attention"]["phases"]["decode"]["validation"] = "validated"
        contracts["contracts"]["f16_online_fp32_merge"]["status"] = "validated"
        implementation = kernels["kernels"][
            "attention_forward_decode_head_major_gqa_flash_f16cache"
        ]["supported_reductions"]["f16_online_fp32_merge"]
        implementation["status"] = "validated"
        implementation["explicit_selector"] = True
        result = resolver.resolve_contract(
            circuit,
            contracts,
            kernels,
            operation="decoder.attention",
            phase="decode",
            mode="production",
        )
        self.assertEqual(result["production_blockers"], [])

    def test_multiple_matching_kernels_fail_deterministically(self) -> None:
        kernels = copy.deepcopy(self.kernels)
        duplicate = copy.deepcopy(
            kernels["kernels"]["attention_forward_decode_head_major_gqa_flash_f16cache"]
        )
        with tempfile.TemporaryDirectory(prefix="cke_v8_kernel_map_") as tmp:
            map_path = Path(tmp) / "attention_decode_duplicate.json"
            base_map = resolver.load_json(
                V8_ROOT
                / "kernel_maps"
                / "attention_forward_decode_head_major_gqa_flash_f16cache.json"
            )
            base_map["id"] = "attention_decode_duplicate"
            map_path.write_text(json.dumps(base_map), encoding="utf-8")
            duplicate["base_kernel_map"] = str(map_path)
            kernels["kernels"]["attention_decode_duplicate"] = duplicate
            with self.assertRaisesRegex(resolver.ContractError, "Ambiguous kernel selection"):
                resolver.resolve_contract(
                    copy.deepcopy(self.circuit),
                    copy.deepcopy(self.contracts),
                    kernels,
                    operation="decoder.attention",
                    phase="decode",
                    mode="bringup",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
