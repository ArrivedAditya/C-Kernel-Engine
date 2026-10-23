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
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location(
    "compare_chunked_prefill_equivalence_tests", SCRIPT_DIR / "compare_chunked_prefill_v8.py")
assert spec and spec.loader
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
from prefill_chunk_contract_v8 import plan_prefill_chunks


class ExecutionEquivalenceMatrixTests(unittest.TestCase):
    def test_aligned_planner_preserves_boundaries_and_final_tail(self):
        self.assertEqual(plan_prefill_chunks(129, 7, 4)[:2], ((0, 4), (4, 4)))
        self.assertEqual(plan_prefill_chunks(129, 7, 4)[-2:], ((120, 4), (124, 5)))
        self.assertEqual(plan_prefill_chunks(129, 7, 4, initial_prefix=124),
                         ((0, 124), (124, 5)))
        self.assertEqual(plan_prefill_chunks(129, 1, 4, initial_prefix=128),
                         ((0, 128), (128, 1)))
        self.assertEqual(plan_prefill_chunks(5, 7, 4), ((0, 5),))

    def test_aligned_planner_rejects_unsafe_position_budget_and_packing(self):
        for kwargs in ({"initial_prefix": 127}, {"packing": "combined"},
                       {"row_group": 0}, {"max_chunk_tokens": 3},
                       {"total_tokens": 0}):
            arguments = dict(total_tokens=129, max_chunk_tokens=7, row_group=4)
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_prefill_chunks(**arguments)
        with self.assertRaises(ValueError):
            plan_prefill_chunks(129, 1, 4, initial_prefix=124)

    def test_invalid_aligned_plan_is_rejected_before_model_initialization(self):
        class Symbol:
            def __init__(self, result=0):
                self.result = result
                self.called = 0

            def __call__(self, *_):
                self.called += 1
                return self.result

        symbols = {name: Symbol() for name in (
            "ck_model_init", "ck_model_embed_tokens", "ck_model_forward",
            "ck_model_decode", "ck_model_get_logits", "ck_model_get_vocab_size",
            "ck_model_get_logits_stride", "ck_model_get_active_tokens",
            "ck_model_debug_export_kv", "ck_model_free")}
        symbols["ck_model_prefill_append_row_group"] = Symbol(4)
        fake = SimpleNamespace(**symbols)
        with (patch.object(compare, "_kv_only_state_contract", return_value={}),
              patch.object(compare, "_loaded_identity", return_value={}),
              patch.object(compare.ctypes, "CDLL", return_value=fake)):
            with self.assertRaisesRegex(ValueError, "inside a provider row group"):
                compare._probe(Path("/unused"), list(range(129)), [], Path("/unused"),
                               127, 7, True)
        self.assertEqual(symbols["ck_model_init"].called, 0)
        self.assertEqual(symbols["ck_model_embed_tokens"].called, 0)

    def test_aligned_report_rejects_wrong_group_or_call_boundary(self):
        report = {"prefill_append_row_group": 4,
                  "execution_calls": {"chunked": [
                      {"start": start, "count": count}
                      for start, count in plan_prefill_chunks(9, 5, 4)]}}
        self.assertTrue(compare._aligned_report_plan_matches(report, 9, 5, None))
        report["execution_calls"]["chunked"][0]["count"] = 5
        self.assertFalse(compare._aligned_report_plan_matches(report, 9, 5, None))
        report["prefill_append_row_group"] = 0
        self.assertFalse(compare._aligned_report_plan_matches(report, 9, 5, None))

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
                                      matrix_splits="0", matrix_case_timeout_sec=60,
                                      aligned_chunks=False)

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
                                      matrix_splits="0", matrix_case_timeout_sec=60,
                                      aligned_chunks=False)
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
                                      matrix_splits="0", matrix_case_timeout_sec=60,
                                      aligned_chunks=False)
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
                                      matrix_splits="0", matrix_case_timeout_sec=60,
                                      aligned_chunks=False)

            def child(command, **_):
                report = Path(command[command.index("--json-out") + 1])
                token_path = Path(command[command.index("--tokens") + 1])
                report.write_text(json.dumps({
                    "schema": "cke.chunked_prefill_comparison.v1", "exact": True,
                    "same_engine_build": True, "prompt_tokens": 5,
                    "append_chunk_size": 4, "split_at": None,
                    "aligned_chunks": False,
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
