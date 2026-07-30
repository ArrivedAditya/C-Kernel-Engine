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

    def test_complete_frontend_resolves_exact_providers(self):
        expected = {
            "audio.frontend.wav_decode": "audio_wav_decode_memory_pcm16_mono_window_f32",
            "audio.frontend.resample": "audio_resample_windowed_sinc_f32",
            "audio.frontend.pad": "audio_pad_or_truncate_f32",
            "audio.frontend.stft_tables": "audio_stft_precompute_tables_f32",
            "audio.frontend.stft": "audio_stft_power_fft400_f32",
            "audio.frontend.mel_filters": "audio_whisper_mel_filters_slaney_f32",
            "audio.frontend.log_mel": "audio_whisper_log_mel_from_power_f32",
            "audio.frontend.feature_window": "audio_whisper_log_mel_window_wav_pcm16_f32",
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

    def test_frontend_sequence_is_explicit_and_complete(self):
        sequence = self.circuit["block_types"]["audio_frontend"]["sequence"]
        self.assertEqual(
            [row["op"] for row in sequence],
            [
                "audio_wav_decode",
                "audio_resample",
                "audio_pad_or_truncate",
                "audio_stft_tables",
                "audio_stft",
                "audio_mel_filters",
                "audio_log_mel",
                "audio_feature_window",
            ],
        )

    def test_unregistered_audio_arithmetic_is_a_hard_failure(self):
        circuit = copy.deepcopy(self.circuit)
        circuit["required_numerical_contracts"]["audio.frontend.stft"]["phases"][
            "prefill"
        ]["contract_id"] = "audio_whisper_log_mel_unregistered_fp16"
        with self.assertRaises(resolver.ContractError):
            resolver.resolve_contract(
                circuit,
                self.contracts,
                self.kernels,
                "audio.frontend.stft",
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

    def test_frontend_does_not_override_resolved_kernel_identity(self):
        self.assertNotIn("kernels", self.circuit)


if __name__ == "__main__":
    unittest.main()
