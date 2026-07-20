#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v8" / "scripts"
sys.path[:0] = [str(SCRIPT_DIR), str(ROOT / "scripts")]
SPEC = importlib.util.spec_from_file_location(
    "certify_text_prompt_parity_v8_tests",
    SCRIPT_DIR / "certify_text_prompt_parity_v8.py",
)
assert SPEC is not None and SPEC.loader is not None
certifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certifier)


class TextPromptCertificationTests(unittest.TestCase):
    def test_qwen35_fixture_has_strict_promotion_matrix(self) -> None:
        fixture = certifier.load_prompt_set(
            ROOT / "version" / "v8" / "test_assets" / "qwen35_text_parity_prompts.json"
        )
        self.assertEqual(fixture["stages"], [64, 128, 256])
        self.assertEqual([row["id"] for row in fixture["prompts"]], [
            "hello", "c-python-sql", "pure-c", "thanks",
        ])
        self.assertEqual(fixture["stop_token_ids"], [248046])
        self.assertEqual(len(fixture["llama_cpp_commit"]), 40)
        self.assertTrue(all(row["tokens"] for row in fixture["prompts"]))

    def test_invalid_stage_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "stages": [128, 64],
                "prompts": [{"id": "one", "text": "one", "tokens": [1]}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "increasing"):
                certifier.load_prompt_set(path)

    def test_eos_report_satisfies_larger_stage(self) -> None:
        report = {"pass": True, "matched_stop_token": 248046, "steps": [{}] * 12}
        self.assertTrue(certifier.report_satisfies_stage(report, 256))

    def test_unfinished_report_does_not_satisfy_stage(self) -> None:
        report = {"pass": True, "matched_stop_token": None, "steps": [{}] * 63}
        self.assertFalse(certifier.report_satisfies_stage(report, 64))

    def test_utf8_corruption_markers_are_rejected(self) -> None:
        self.assertTrue(certifier.decoded_text_is_clean("Hello \u2014 \u4f60\u597d \U0001f60a"))
        for text in ("bad \\uFFFD", "bad \ufffd", "bad \u00c3\u00a9", "bad \ufffd\u0141"):
            self.assertFalse(certifier.decoded_text_is_clean(text))

    def test_eos_report_is_reused_for_larger_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello-64.json"
            source.write_text(json.dumps({
                "pass": True, "matched_stop_token": 248046, "steps": [{}] * 14,
            }), encoding="utf-8")
            self.assertEqual(
                certifier.reusable_report_path(root, "hello", [64, 128, 256], 256),
                source,
            )

    def test_failure_handoff_uses_recurrent_xray(self) -> None:
        command = certifier.xray_handoff(
            Path("/model"), Path("/model.gguf"), Path("/out/fail.json"), Path("/out"), 20
        )
        self.assertIn("xray_text_recurrent_v8.py", command)
        self.assertIn("--parity-report /out/fail.json", command)
        self.assertIn("--ck-prefill-mode hybrid", command)


if __name__ == "__main__":
    unittest.main()
