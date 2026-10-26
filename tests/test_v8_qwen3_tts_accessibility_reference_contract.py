"""Cheap corpus and stop-verdict controls for Qwen technical speech trials."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from version.v8.tts.evaluate_qwen3_tts_accessibility_reference import (
    DEFAULT_CORPUS, classify_termination, load_corpus,
)

EVIDENCE = DEFAULT_CORPUS.parents[3] / "version/v8/reports/qwen3_tts_accessibility_policy_p3_20261026.json"


class Qwen3TTSAccessibilityReferenceContractTest(unittest.TestCase):
    def test_retained_literal_and_explicit_expansions(self):
        cases = load_corpus(DEFAULT_CORPUS)
        literal = cases["literal_path_signed"]
        self.assertEqual(literal["source_text"],
                         "Read /home/antshiv/Workspace/C-Kernel-Engine/"
                         "server_summary.txt. The value is -3.5, not +3.5.")
        self.assertNotIn("spoken_expansion", literal)
        signed = cases["signed_numbers"]
        self.assertIn("-3.5", signed["source_text"])
        self.assertIn("+3.5", signed["source_text"])
        self.assertIn("minus three point five", signed["spoken_expansion"])
        self.assertIn("plus three point five", signed["spoken_expansion"])
        self.assertIn("not", signed["spoken_expansion"])
        self.assertEqual(len(signed["spoken_critical_facts"]),
                         len(signed["critical_facts"]))

    def test_corpus_rejects_missing_fact_and_unexplained_expansion(self):
        source = json.loads(DEFAULT_CORPUS.read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            source["cases"][0]["critical_facts"] = ["unspoken"]
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "critical facts"):
                load_corpus(path)
            source["cases"][0]["critical_facts"] = ["CKE"]
            source["cases"][0]["spoken_expansion"] = "different words"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "expansion"):
                load_corpus(path)

    def test_eos_and_budget_exhaustion_remain_distinct(self):
        eos = 2150
        self.assertEqual(classify_termination([10, 11, eos], eos, 3), "eos")
        self.assertEqual(classify_termination([10, 11, 12], eos, 3), "capacity")
        self.assertEqual(classify_termination([10, 11], eos, 3), "unknown")
        self.assertEqual(classify_termination([], eos, 3), "missing_sequence")

    def test_retained_reference_evidence_keeps_verdicts_separate(self):
        report = json.loads(EVIDENCE.read_text())
        self.assertEqual(report["model"]["revision"],
                         "85e237c12c027371202489a0ec509ded67b5e4b5")
        self.assertEqual(report["corpus_sha256"], hashlib.sha256(
            DEFAULT_CORPUS.read_bytes()).hexdigest())
        greedy = {case["id"]: case for case in report["policies"]["greedy"]["cases"]}
        sampled = {case["id"]: case for case in report["policies"]["sampled"]["cases"]}
        self.assertEqual(greedy["plain_update"]["termination"], "eos")
        self.assertEqual(greedy["literal_path_signed"]["termination"], "capacity")
        self.assertEqual(sampled["literal_path_signed"]["termination"], "eos")
        self.assertEqual(greedy["literal_path_signed"]["wav_sha256"],
                         "7acde7852540df24fdbef22aac4f77b24742bf1d7d0b29133312fbcc2b07a946")
        self.assertEqual(greedy["literal_path_signed"]["source_text"],
                         sampled["literal_path_signed"]["source_text"])
        for policy in report["policies"].values():
            for case in policy["cases"]:
                self.assertEqual(case["cke_execution"], "NOT_TESTED")
                self.assertEqual(case["human_listening"], "NOT_TESTED")
        self.assertTrue(report["policies"]["corpus"]["all_natural_eos"])
        self.assertEqual(len(report["policies"]["corpus"]["cases"]), 6)
        repeated = report["policies"]["repeated"]["repeated_timing"]["plain_update"]
        self.assertEqual(repeated["runs"], 3)
        self.assertTrue(repeated["wav_hashes_identical"])
        supporting_asr = report["supporting_asr"]
        literal_asr = supporting_asr["literal_original_sampled"]
        expanded_asr = supporting_asr["signed_expansion_sampled"]
        self.assertEqual(literal_asr["status"], "FAIL_CRITICAL_FACTS")
        self.assertFalse(literal_asr["critical_fact_checks"]["minus_spoken"])
        self.assertFalse(literal_asr["critical_fact_checks"]["full_path_exact"])
        self.assertEqual(expanded_asr["status"], "PASS_ASR_CRITICAL_FACTS")
        self.assertTrue(expanded_asr["critical_fact_checks"]["minus_spoken"])
        self.assertTrue(expanded_asr["critical_fact_checks"]["plus_spoken"])
        self.assertEqual(literal_asr["human_listening"], "NOT_TESTED")


if __name__ == "__main__":
    unittest.main()
