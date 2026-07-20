#!/usr/bin/env python3
"""Fail-closed circuit and kernel-map tests for reusable audio transformers."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "version" / "v8"
RESOLVER_PATH = V8 / "scripts" / "resolve_numerical_execution_contracts_v8.py"
BUILD_IR_PATH = V8 / "scripts" / "build_ir_v8.py"
NIGHTLY_PATH = ROOT / "scripts" / "nightly_runner.py"
if str(BUILD_IR_PATH.parent) not in sys.path:
    sys.path.insert(0, str(BUILD_IR_PATH.parent))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


resolver = _load_module("audio_encoder_contract_resolver", RESOLVER_PATH)
build_ir = _load_module("audio_encoder_build_ir", BUILD_IR_PATH)
nightly = _load_module("audio_encoder_nightly", NIGHTLY_PATH)


class AudioEncoderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuit_path = V8 / "circuits" / "audio_transformer_encoder.json"
        cls.circuit = resolver.load_json(cls.circuit_path)
        cls.frontend = resolver.load_json(
            V8 / "circuits" / "whisper_audio_frontend.json"
        )
        cls.contracts = resolver.load_json(resolver.DEFAULT_CONTRACTS)
        cls.kernels = resolver.load_kernel_capabilities(contracts=cls.contracts)

    def test_audio_encoder_contracts_resolve_exact_providers(self):
        expected = {
            "audio.encoder.stem.conv1": "audio_conv1d_channel_major_f32",
            "audio.encoder.stem.conv2": "audio_conv1d_channel_major_f32",
            "audio.encoder.layout": "audio_transpose_channel_to_token_f32",
            "audio.encoder.attention": "attention_forward_query_key_head_major_f32",
        }
        for requirement, kernel_id in expected.items():
            with self.subTest(requirement=requirement):
                plan = resolver.resolve_contract(
                    self.circuit,
                    self.contracts,
                    self.kernels,
                    requirement,
                    "prefill",
                    mode="production",
                )
                self.assertEqual(plan["kernel"]["id"], kernel_id)

    def test_audio_primitive_contracts_resolve_all_exact_providers(self):
        cases = {
            "audio_pcm_s16_mono_fp32": (
                "audio_pcm_decode", "audio_pcm_s16_to_mono_f32"
            ),
            "audio_resample_linear_rational_fp32": (
                "audio_resample", "audio_resample_linear_f32"
            ),
            "audio_stft_centered_hann_precomputed_fp32": (
                "audio_stft", "audio_stft_power_precomputed_f32"
            ),
            "audio_conv1d_channel_major_ascending_fp32": (
                "audio_conv1d", "audio_conv1d_channel_major_f32"
            ),
            "layout_channel_to_token_copy_fp32": (
                "layout_transform", "audio_transpose_channel_to_token_f32"
            ),
            "attention_query_key_scaled_ordered_fp32": (
                "attention", "attention_forward_query_key_head_major_f32"
            ),
        }
        for contract_id, (operator, kernel_id) in cases.items():
            circuit = {
                "required_numerical_contracts": {
                    "test": {
                        "op": operator,
                        "template_ops": ["test_op"],
                        "phases": {
                            "prefill": {
                                "contract_id": contract_id,
                                "validation": "validated",
                                "evidence": "synthetic exact-provider resolution test",
                            }
                        },
                        "checkpoint": {
                            "id": "test.output",
                            "producer": "test_op",
                            "logical_layout": "test_layout",
                            "axis_names": ["element"],
                        },
                    }
                }
            }
            with self.subTest(contract=contract_id):
                plan = resolver.resolve_contract(
                    circuit,
                    self.contracts,
                    self.kernels,
                    "test",
                    "prefill",
                    mode="production",
                )
                self.assertEqual(plan["kernel"]["id"], kernel_id)

    def test_frontend_and_encoder_do_not_name_concrete_kernels(self):
        self.assertNotIn("kernels", self.frontend)
        self.assertNotIn("kernels", self.circuit)

    def test_audio_ops_are_generic_dsl_vocabulary(self):
        expected = {
            "audio_pcm_decode": "audio_pcm_decode",
            "audio_resample": "audio_resample",
            "audio_stft": "audio_stft",
            "audio_log_mel": "audio_log_mel",
            "audio_conv1d_stem_1": "audio_conv1d",
            "audio_conv1d_stem_2": "audio_conv1d",
            "layout_channel_to_token": "layout_transform",
            "cross_attn": "attention",
        }
        for op, family in expected.items():
            self.assertEqual(build_ir.TEMPLATE_TO_KERNEL_OP.get(op), family)
            self.assertIn(op, build_ir.OP_DATAFLOW)

    def test_shared_cross_attention_provider_is_not_audio_named(self):
        kernel = json.loads(
            (V8 / "kernel_maps" / "attention_forward_query_key_head_major_f32.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(kernel["op"], "attention")
        identity = f"{kernel['id']} {kernel['impl']['function']}".lower()
        self.assertNotIn("audio", identity)
        self.assertNotIn("whisper", identity)

    def test_unknown_resampling_semantics_are_a_hard_failure(self):
        circuit = copy.deepcopy(self.frontend)
        request = circuit["required_numerical_contracts"]["audio.frontend.log_mel"]
        request["op"] = "audio_resample"
        request["template_ops"] = ["audio_resample"]
        request["phases"]["prefill"]["contract_id"] = (
            "audio_resample_bandlimited_sinc_fp32"
        )
        with self.assertRaises(resolver.ContractError):
            resolver.resolve_contract(
                circuit,
                self.contracts,
                self.kernels,
                "audio.frontend.log_mel",
                "prefill",
                mode="production",
            )

    def test_audio_primitive_matrix_is_a_visible_nightly_row(self):
        suite = nightly.TEST_SUITES["audio_transformer_primitives"]
        self.assertEqual(suite.name, "Audio Transformer Primitives")
        self.assertEqual(suite.category, "kernels")
        self.assertEqual(suite.test_file.name, "test_audio_encoder.py")
        parsed = nightly.parse_sub_tests(
            "audio_encoder_self_attention_equal "
            "max_diff=2.98e-08 tol=2.0e-06 [PASS]\n"
        )
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].status, "pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
