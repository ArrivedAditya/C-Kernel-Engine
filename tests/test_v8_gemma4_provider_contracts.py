from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v8" / "scripts"))
from build_ir_v8 import build_ir1_direct
from audit_kernel_map_interfaces_v8 import _validate_provider_constraints


PREPARE_CONTRACT = "gemma4_prepare_llama_cpu_bf16_input_rne_v2"
EMBED_CONTRACT = "gemma4_embed_llama_cpu_fp16_table_gelu_v1"


def _manifest(kernel: str, op: str, contract: str, embed_dim: int) -> dict:
    return {
        "config": {
            "model": "toy", "arch": "toy", "num_layers": 0,
            "embed_dim": embed_dim, "vocab_size": 2, "context_length": 1,
        },
        "quant_summary": {},
        "entries": [],
        "template": {
            "version": 2, "name": "toy", "sequence": ["decoder"],
            "block_types": {"decoder": {"sequence": ["header"],
                                      "header": [{"op": op, "kernel": kernel}]}},
            "contract": {"provider_numerical_requirements": {op: contract}},
        },
    }


class Gemma4ProviderContractTests(unittest.TestCase):
    def _lower(self, manifest: dict, mode: str) -> list[dict]:
        with contextlib.redirect_stdout(io.StringIO()):
            return build_ir1_direct(manifest, ROOT / "synthetic-gemma4.json", mode=mode)

    def test_prepare_contract_and_capacity_in_both_phases(self) -> None:
        for kernel in ("gemma4_per_layer_prepare_forward",
                       "gemma4_per_layer_prepare_bf16_forward"):
            for mode in ("prefill", "decode"):
                with self.subTest(kernel=kernel, mode=mode):
                    manifest = _manifest(kernel, "gemma4_per_layer_prepare",
                                         PREPARE_CONTRACT, 16384)
                    ops = self._lower(manifest, mode)
                    self.assertEqual(ops[0]["resolved_provider_numerical_contract"]["id"],
                                     PREPARE_CONTRACT)
                    self.assertEqual(ops[0]["resolved_provider_numerical_contract"]["projection_input"],
                                     "fp32_rounded_to_bf16_rne_before_dot")
                    self.assertEqual(ops[0]["resolved_provider_numerical_contract"]["projection_reduction"],
                                     "ggml_cpu_bf16_avx2_four_accumulators_with_scalar_fallback")
                    manifest["config"]["embed_dim"] = 16385
                    with self.assertRaisesRegex(RuntimeError, "HARD PROVIDER DIMENSION FAULT.*embed_dim=16385"):
                        self._lower(manifest, mode)
                    manifest["config"]["embed_dim"] = 8
                    manifest["template"]["contract"]["provider_numerical_requirements"][
                        "gemma4_per_layer_prepare"] = "ordinary_fp32_input"
                    with self.assertRaisesRegex(RuntimeError, "HARD PROVIDER NUMERICAL FAULT"):
                        self._lower(manifest, mode)

    def test_embed_contract_is_explicit(self) -> None:
        manifest = _manifest("gemma4_per_layer_embed_forward",
                             "gemma4_per_layer_embed", EMBED_CONTRACT, 8)
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                ops = self._lower(manifest, mode)
                contract = ops[0]["resolved_provider_numerical_contract"]
                self.assertEqual(contract["id"], EMBED_CONTRACT)
                self.assertEqual(contract["activation"],
                                 "gelu_fp16_table_input_output_rne_with_signed_10_clamps")

    def test_missing_and_malformed_circuit_requirements_fail_in_lowering(self) -> None:
        manifest = _manifest("gemma4_per_layer_prepare_bf16_forward",
                             "gemma4_per_layer_prepare", PREPARE_CONTRACT, 8)
        contract = manifest["template"]["contract"]
        for malformed in (None, [], {"gemma4_per_layer_prepare": None},
                          {"gemma4_per_layer_prepare": ""}):
            with self.subTest(malformed=malformed):
                contract["provider_numerical_requirements"] = malformed
                with self.assertRaisesRegex(RuntimeError, "HARD PROVIDER NUMERICAL FAULT"):
                    self._lower(manifest, "prefill")
        del contract["provider_numerical_requirements"]
        with self.assertRaisesRegex(RuntimeError, "HARD PROVIDER NUMERICAL FAULT"):
            self._lower(manifest, "decode")

    def test_map_constraint_schema_rejects_malformed_metadata(self) -> None:
        path = ROOT / "version/v8/kernel_maps/gemma4_per_layer_prepare_forward.json"
        provider = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(provider["constraints"]["rounded_input_stack_bytes_per_worker_max"],
                         4 * provider["constraints"]["dimensions"]["embed_dim"]["max"])
        _validate_provider_constraints(provider, path)
        mutations = (
            ("provider_numerical_contract_schema", None),
            ("numerical_contract", None),
            ("numerical_contract", "gemma4_prepare_llama_cpu_bf16_input_rne_v2"),
            ("numerical_contract", {"circuit_binding_required": True}),
            ("numerical_contract", {"id": PREPARE_CONTRACT,
                                    "circuit_binding_required": "true"}),
            ("constraints", {"dimensions": {"embed_dim": {"min": 1, "max": True}}}),
            ("constraints", None),
            ("constraints", {"rounded_input_stack_bytes_per_worker_max": "65536"}),
            ("constraints", {"dimensions": {"embed_dim": {"min": 1, "max": 16384}},
                             "rounded_input_stack_bytes_per_worker_max": 65535}),
        )
        for field, malformed in mutations:
            with self.subTest(field=field, malformed=malformed):
                candidate = copy.deepcopy(provider)
                candidate[field] = malformed
                with self.assertRaises(RuntimeError):
                    _validate_provider_constraints(candidate, path)


if __name__ == "__main__":
    unittest.main()
