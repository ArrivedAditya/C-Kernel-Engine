#!/usr/bin/env python3
"""Fail-closed ownership tests for the v8 audio frontend contract."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "resolve_numerical_execution_contracts_v8.py"
SPEC = importlib.util.spec_from_file_location("audio_contract_resolver", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


class AudioFrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuit = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "whisper_audio_frontend.json"
        )
        cls.contracts = resolver.load_json(resolver.DEFAULT_CONTRACTS)
        cls.kernels = resolver.load_kernel_capabilities(contracts=cls.contracts)

    def test_log_mel_resolves_exact_reference_provider(self):
        plan = resolver.resolve_contract(
            self.circuit,
            self.contracts,
            self.kernels,
            "audio.frontend.log_mel",
            "prefill",
            mode="production",
        )
        self.assertEqual(plan["kernel"]["id"], "audio_whisper_log_mel_reference_f32")
        self.assertEqual(
            plan["kernel"]["function"], "audio_whisper_log_mel_reference_f32"
        )
        semantics = plan["contract"]["semantics"]
        self.assertEqual(semantics["operator_family"], "audio_log_mel")
        self.assertEqual(semantics["reduction"]["kind"], "composite")
        self.assertEqual(semantics["reduction"]["order"], "stage_ordered")
        self.assertEqual(
            plan["checkpoint"]["axis_names"], ["mel", "frame"]
        )

    def test_unregistered_audio_arithmetic_is_a_hard_failure(self):
        circuit = copy.deepcopy(self.circuit)
        circuit["required_numerical_contracts"]["audio.frontend.log_mel"]["phases"][
            "prefill"
        ]["contract_id"] = "audio_whisper_log_mel_unregistered_fp16"
        with self.assertRaises(resolver.ContractError):
            resolver.resolve_contract(
                circuit,
                self.contracts,
                self.kernels,
                "audio.frontend.log_mel",
                "prefill",
                mode="production",
            )

    def test_frontend_parameters_are_circuit_owned(self):
        frontend = self.circuit["contract"]["audio_frontend"]
        self.assertEqual(frontend["sample_rate"], 16000)
        self.assertEqual(frontend["n_fft"], 400)
        self.assertEqual(frontend["hop_length"], 160)
        self.assertEqual(frontend["centering"], "reflect")
        self.assertEqual(frontend["window"], "periodic_hann")
        self.assertEqual(frontend["mel_scale"], "slaney")


if __name__ == "__main__":
    unittest.main()
