#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "normalize_xray_ranking_report_v8.py"
SPEC = importlib.util.spec_from_file_location("normalize_xray_ranking_report_v8_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
normalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalizer)


class NormalizeXRayRankingReportTests(unittest.TestCase):
    def test_normalizes_bf16_corpus_teacher_forced_traces(self) -> None:
        source = {
            "ck_logit_trace": [
                {"step": 0, "top_k": [{"token_id": 10, "logit": 4.0}, {"token_id": 11, "logit": 3.0}]},
                {"step": 1, "top_k": [{"token_id": 20, "logit": 2.1}, {"token_id": 21, "logit": 2.0}]},
            ],
            "torch_logit_trace": [
                {"step": 0, "top_k": [{"token_id": 10, "logit": 4.0}, {"token_id": 12, "logit": 3.5}]},
                {"step": 1, "top_k": [{"token_id": 21, "logit": 2.0}, {"token_id": 20, "logit": 1.9}]},
            ],
        }
        report = normalizer.normalize(source, "teacher_forced")
        self.assertEqual([row["status"] for row in report["checks"]], ["pass", "fail"])
        self.assertEqual(report["checks"][1]["ck_top1"], 20)
        self.assertEqual(report["checks"][1]["oracle_top1"], 21)
        self.assertEqual(report["checks"][1]["topk_overlap_count"], 2)
        self.assertAlmostEqual(report["checks"][0]["ck_top1_margin"], 1.0)

    def test_rejects_misaligned_trace_steps(self) -> None:
        source = {
            "ck_logit_trace": [{"step": 2, "top_k": [{"token_id": 1, "logit": 1.0}]}],
            "torch_logit_trace": [{"step": 3, "top_k": [{"token_id": 1, "logit": 1.0}]}],
        }
        with self.assertRaisesRegex(ValueError, "step mismatch"):
            normalizer.normalize(source, "teacher_forced")


if __name__ == "__main__":
    unittest.main(verbosity=2)
