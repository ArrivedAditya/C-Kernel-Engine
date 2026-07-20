#!/usr/bin/env python3
"""Regression tests for BF16 Qwen3-VL certification provenance."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/certify_qwen3vl_bf16_corpus_v8.py"
SPEC = importlib.util.spec_from_file_location("certify_qwen3vl_bf16", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CustomPrefixProvenanceTests(unittest.TestCase):
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
