#!/usr/bin/env python3
"""Asset-free controls for the shared full/chunked X-Ray execution gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location(
    "compare_chunked_prefill_equivalence_tests", SCRIPT_DIR / "compare_chunked_prefill_v8.py")
assert spec and spec.loader
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ExecutionEquivalenceMatrixTests(unittest.TestCase):
    def test_matrix_covers_group_tails_appends_and_thread_counts(self):
        cases = compare._matrix_cases("125,126,127,128", "4,8", "1,4", "0,120", 429)
        self.assertEqual(len(cases), 32)
        self.assertIn((125, 4, 1, 120), cases)
        self.assertIn((128, 8, 4, None), cases)
        for args in (("0", "4", "1", "0", 429),
                     ("125", "125", "1", "0", 429),
                     ("125", "4", "1", "125", 429),
                     ("500", "4", "1", "0", 429)):
            with self.assertRaises(ValueError):
                compare._matrix_cases(*args)

    def test_wrong_tail_result_fails_matrix_and_retains_child_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = root / "tokens.json"
            tokens.write_text(json.dumps(list(range(10))))
            args = argparse.Namespace(full=root / "full", chunked=root / "chunked",
                                      tokens=tokens, continuation=None,
                                      json_out=root / "matrix.json", matrix_lengths="5",
                                      matrix_chunk_sizes="4", matrix_threads="1",
                                      matrix_splits="0", matrix_case_timeout_sec=60)

            def wrong_tail(command, **_):
                report = Path(command[command.index("--json-out") + 1])
                report.write_text(json.dumps({"exact": False, "same_engine_build": True,
                                              "first_observed_divergence": {
                                                  "classification": "NUMERICAL_OUTPUT_DIVERGENCE"}}))
                return subprocess.CompletedProcess(command, 1, "", "")

            with (patch.object(compare.subprocess, "run", side_effect=wrong_tail),
                  patch.object(compare, "_source_revision", return_value="fixture")):
                self.assertEqual(compare._run_matrix(args, list(range(10))), 1)
            report = json.loads(args.json_out.read_text())
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["cases"][0]["status"], "fail")
            self.assertTrue(Path(report["cases"][0]["report"]).is_file())
            self.assertIn("--require-exact", report["cases"][0]["command"])

    def test_missing_execution_is_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = root / "tokens.json"
            tokens.write_text("[1,2,3,4,5]\n")
            args = argparse.Namespace(full=root / "full", chunked=root / "chunked",
                                      tokens=tokens, continuation=None,
                                      json_out=root / "matrix.json", matrix_lengths="5",
                                      matrix_chunk_sizes="4", matrix_threads="1",
                                      matrix_splits="0", matrix_case_timeout_sec=60)
            with (patch.object(compare.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 2, "", "missing model bundle")),
                  patch.object(compare, "_source_revision", return_value="fixture")):
                self.assertEqual(compare._run_matrix(args, [1, 2, 3, 4, 5]), 1)
            case = json.loads(args.json_out.read_text())["cases"][0]
            self.assertEqual(case["status"], "not_tested")
            self.assertIn("missing model bundle", case["error"])

    def test_timeout_and_malformed_child_report_fail_without_crashing_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = root / "tokens.json"
            tokens.write_text("[1,2,3,4,5]\n")
            args = argparse.Namespace(full=root / "full", chunked=root / "chunked",
                                      tokens=tokens, continuation=None,
                                      json_out=root / "matrix.json", matrix_lengths="5",
                                      matrix_chunk_sizes="4", matrix_threads="1",
                                      matrix_splits="0", matrix_case_timeout_sec=60)
            with (patch.object(compare.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("fixture", 60)),
                  patch.object(compare, "_source_revision", return_value="fixture")):
                self.assertEqual(compare._run_matrix(args, [1, 2, 3, 4, 5]), 1)
            self.assertEqual(json.loads(args.json_out.read_text())["cases"][0]["status"],
                             "fail")

            def malformed(command, **_):
                report = Path(command[command.index("--json-out") + 1])
                report.write_text("{bad json")
                return subprocess.CompletedProcess(command, 0, "", "")

            with (patch.object(compare.subprocess, "run", side_effect=malformed),
                  patch.object(compare, "_source_revision", return_value="fixture")):
                self.assertEqual(compare._run_matrix(args, [1, 2, 3, 4, 5]), 1)
            self.assertEqual(json.loads(args.json_out.read_text())["cases"][0]["status"],
                             "fail")

    def test_swapped_or_unbound_child_report_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = root / "tokens.json"
            tokens.write_text("[1,2,3,4,5]\n")
            args = argparse.Namespace(full=root / "full", chunked=root / "chunked",
                                      tokens=tokens, continuation=None,
                                      json_out=root / "matrix.json", matrix_lengths="5",
                                      matrix_chunk_sizes="4", matrix_threads="1",
                                      matrix_splits="0", matrix_case_timeout_sec=60)

            def child(command, **_):
                report = Path(command[command.index("--json-out") + 1])
                token_path = Path(command[command.index("--tokens") + 1])
                report.write_text(json.dumps({
                    "schema": "cke.chunked_prefill_comparison.v1", "exact": True,
                    "same_engine_build": True, "prompt_tokens": 5,
                    "append_chunk_size": 4, "split_at": None,
                    "token_sha256": compare._sha256(token_path),
                    "execution": {"omp_num_threads": "1",
                                  "native_threads": {"full": 1, "chunked": 1}}}))
                return subprocess.CompletedProcess(command, 0, "", "")

            with (patch.object(compare.subprocess, "run", side_effect=child),
                  patch.object(compare, "_source_revision", return_value="fixture")):
                self.assertEqual(compare._run_matrix(args, [1, 2, 3, 4, 5]), 0)
            self.assertEqual(json.loads(args.json_out.read_text())["status"], "pass")

            def swapped(command, **kwargs):
                completed = child(command, **kwargs)
                report = Path(command[command.index("--json-out") + 1])
                detail = json.loads(report.read_text())
                detail["prompt_tokens"] = 4
                report.write_text(json.dumps(detail))
                return completed

            with (patch.object(compare.subprocess, "run", side_effect=swapped),
                  patch.object(compare, "_source_revision", return_value="fixture")):
                self.assertEqual(compare._run_matrix(args, [1, 2, 3, 4, 5]), 1)
            self.assertEqual(json.loads(args.json_out.read_text())["cases"][0]["status"], "fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
