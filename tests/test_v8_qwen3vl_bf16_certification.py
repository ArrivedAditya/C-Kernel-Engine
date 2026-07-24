#!/usr/bin/env python3
"""Regression tests for BF16 Qwen3-VL certification provenance."""

from __future__ import annotations

import importlib.util
import inspect
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/certify_qwen3vl_bf16_corpus_v8.py"
SPEC = importlib.util.spec_from_file_location("certify_qwen3vl_bf16", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CustomPrefixProvenanceTests(unittest.TestCase):
    class _TorchThreads:
        @staticmethod
        def get_num_threads() -> int:
            return 24

        @staticmethod
        def get_num_interop_threads() -> int:
            return 1

    def test_numerical_thread_provenance_is_explicit(self) -> None:
        environment = {
            "CK_NUM_THREADS": "24",
            "OMP_NUM_THREADS": "24",
            "MKL_NUM_THREADS": "24",
        }
        with mock.patch.dict(MODULE.os.environ, environment, clear=False):
            report = MODULE._numerical_thread_provenance(self._TorchThreads(), 24)
        self.assertEqual(report["torch_num_threads"], 24)
        self.assertEqual(report["omp_num_threads"], 24)
        self.assertTrue(report["thread_count_changes_arithmetic_order"])

    def test_numerical_thread_provenance_rejects_cross_thread_oracle(self) -> None:
        environment = {
            "CK_NUM_THREADS": "20",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "20",
        }
        with mock.patch.dict(MODULE.os.environ, environment, clear=False):
            with self.assertRaisesRegex(RuntimeError, "thread provenance mismatch"):
                MODULE._numerical_thread_provenance(self._TorchThreads(), 24)

    def test_teacher_forced_trace_uses_backend_top1_not_forced_tokens(self) -> None:
        reference = [
            {"top_k": [{"token_id": 10, "logit": 2.0}]},
            {"top_k": [{"token_id": 20, "logit": 3.0}]},
        ]
        subject = [
            {"top_k": [{"token_id": 10, "logit": 1.5}]},
            {"top_k": [{"token_id": 21, "logit": 2.5}]},
        ]
        self.assertEqual(MODULE._first_trace_top1_difference(reference, subject), 1)

    def test_teacher_forced_trace_detects_missing_steps(self) -> None:
        row = {"top_k": [{"token_id": 10, "logit": 2.0}]}
        self.assertEqual(MODULE._first_trace_top1_difference([row, row], [row]), 1)

    def test_corpus_trace_normalizes_directly_for_xray(self) -> None:
        normalizer = MODULE._load_ranking_normalizer()
        result = normalizer.normalize({
            "ck_logit_trace": [
                {"step": 0, "top_k": [{"token_id": 10, "logit": 2.0}]},
            ],
            "torch_logit_trace": [
                {"step": 0, "top_k": [{"token_id": 11, "logit": 2.0}]},
            ],
        }, "teacher_forced")
        self.assertEqual(result["checks"][0]["status"], "fail")
        self.assertEqual(result["checks"][0]["ck_top1"], 10)
        self.assertEqual(result["checks"][0]["oracle_top1"], 11)

    def test_failure_histogram_groups_divergence_steps(self) -> None:
        rows = [
            {"exact_pre_eos": True, "first_divergence": None},
            {"exact_pre_eos": False, "first_divergence": 25},
            {"exact_pre_eos": False, "first_divergence": 20},
            {"exact_pre_eos": False, "first_divergence": 25},
            {"exact_pre_eos": False, "first_divergence": None},
        ]
        self.assertEqual(
            MODULE._failure_step_histogram(rows),
            {"20": 1, "25": 2, "unknown": 1},
        )

    def test_bridge_encoder_accepts_processor_planar_override(self) -> None:
        bridge = MODULE._load_bridge_module()
        encoder_parameters = inspect.signature(bridge._run_encoder).parameters
        decoder_parameters = inspect.signature(bridge._run_decoder).parameters
        self.assertIn("planar_override", encoder_parameters)
        self.assertIn("forced_generation_token_ids", decoder_parameters)
        self.assertIn("generation_trace_top_k", decoder_parameters)

    def _fixture(self, root: Path, source_bytes: bytes) -> tuple[Path, Path]:
        source = root / "source.ppm"
        source.write_bytes(source_bytes)
        prefix_dir = root / "capture" / "ck"
        prefix_dir.mkdir(parents=True)
        prefix = prefix_dir / "vision_output.f32"
        prefix.write_bytes(b"\0" * 32)
        report = {
            "image": str(source),
            "torch": {
                "grid_thw": [[1, 2, 3]],
                "tensors": {"vision_output": {"shape": [6, 4]}},
            },
        }
        (prefix_dir.parent / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return source, prefix

    def test_matching_image_shape_and_grid_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, prefix = self._fixture(root, b"same image")
            report = MODULE._validate_custom_prefix_provenance(
                prefix, source, (6, 4), (1, 2, 3)
            )
            self.assertEqual(report, (prefix.parent.parent / "report.json").resolve())

    def test_different_same_shape_image_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, prefix = self._fixture(root, b"prefix image")
            selected = root / "selected.ppm"
            selected.write_bytes(b"different image")
            with self.assertRaisesRegex(RuntimeError, "custom prefix image mismatch"):
                MODULE._validate_custom_prefix_provenance(
                    prefix, selected, (6, 4), (1, 2, 3)
                )

    def test_isolated_decoder_report_preserves_stage_timings(self) -> None:
        report = MODULE._serializable_decoder_report({
            "generated_token_ids": [1, 2],
            "teacher_forced_input_ids": [3],
            "generation_logit_trace": [{"step": 0}],
            "timings": {
                "decoder_forward_mixed_ms": 21.5,
                "decoder_generation_ms": 12.25,
                "decoder_generation_tok_s": 5.0,
            },
            "logits": object(),
        })

        self.assertEqual(report["generated_token_ids"], [1, 2])
        self.assertEqual(report["teacher_forced_input_ids"], [3])
        self.assertEqual(report["generation_logit_trace"], [{"step": 0}])
        self.assertEqual(report["timings"]["decoder_forward_mixed_ms"], 21.5)
        self.assertNotIn("logits", report)


if __name__ == "__main__":
    unittest.main()
