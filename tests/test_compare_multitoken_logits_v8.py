#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "compare_multitoken_logits_v8_tests",
    SCRIPT_DIR / "compare_multitoken_logits_v8.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class MultitokenParityEOSContractTests(unittest.TestCase):
    def _run(self, ck_logits: np.ndarray, llama_logits: np.ndarray) -> dict:
        with mock.patch.object(runner, "run_llama_logits", return_value={"logits": llama_logits}), \
             mock.patch.object(runner, "load_ck_logits", return_value={"logits": ck_logits}):
            return runner.run_multitoken_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=64,
                ctx_len=128,
                top_k=3,
                threads=1,
                append_on_divergence="stop",
                ck_prefill_mode="batched",
                llama_decode_mode="batched",
                llama_no_repack=False,
                stop_token_ids={2},
            )

    def test_matched_declared_stop_ends_without_post_eos_comparison(self) -> None:
        logits = np.asarray([0.0, 1.0, 4.0], dtype=np.float32)
        report = self._run(logits, logits)
        self.assertTrue(report["pass"])
        self.assertEqual(report["matched_stop_token"], 2)
        self.assertEqual(len(report["steps"]), 1)

    def test_unmatched_stop_candidate_is_still_a_failure(self) -> None:
        ck_logits = np.asarray([0.0, 1.0, 4.0], dtype=np.float32)
        llama_logits = np.asarray([0.0, 5.0, 1.0], dtype=np.float32)
        report = self._run(ck_logits, llama_logits)
        self.assertFalse(report["pass"])
        self.assertIsNone(report["matched_stop_token"])
        self.assertEqual(report["first_divergence"]["ck_next"], 2)

    def test_replay_configures_ck_threads_before_loading_runtime(self) -> None:
        logits = np.asarray([0.0, 1.0, 4.0], dtype=np.float32)
        configured = {
            "CK_NUM_THREADS": "7",
            "CK_THREADPOOL_THREADS": "7",
            "OMP_NUM_THREADS": "7",
        }
        with mock.patch.object(runner, "run_llama_logits", return_value={"logits": logits}), \
             mock.patch.object(runner, "load_ck_logits", return_value={"logits": logits}), \
             mock.patch.object(
                 runner, "_configure_ck_threads", return_value=configured
             ) as configure:
            report = runner.run_multitoken_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=1,
                ctx_len=128,
                top_k=3,
                threads=7,
                append_on_divergence="stop",
                ck_prefill_mode="batched",
                llama_decode_mode="batched",
                llama_no_repack=False,
            )
        configure.assert_called_once_with(7)
        self.assertEqual(report["ck_thread_environment"], configured)


class PersistentTrajectoryParityTests(unittest.TestCase):
    def test_thread_configuration_is_applied_before_runtime_load(self) -> None:
        with mock.patch.dict(runner.os.environ, {}, clear=True):
            configured = runner._configure_ck_threads(12)
            self.assertEqual(
                configured,
                {
                    "CK_NUM_THREADS": "12",
                    "CK_THREADPOOL_THREADS": "12",
                    "OMP_NUM_THREADS": "12",
                },
            )
            self.assertEqual(runner.os.environ["CK_NUM_THREADS"], "12")
            self.assertEqual(runner.os.environ["CK_THREADPOOL_THREADS"], "12")
            self.assertEqual(runner.os.environ["OMP_NUM_THREADS"], "12")

    def test_exact_trajectory_stops_at_shared_eos(self) -> None:
        rows = np.asarray([
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
        ], dtype=np.float32)
        llama = {"logits": rows, "generated_tokens": [1, 2], "meta": {}}
        ck = {"logits": rows, "generated_tokens": [1, 2], "vocab": 3}
        with mock.patch.object(runner, "run_llama_greedy_trajectory", return_value=llama), \
             mock.patch.object(runner, "load_ck_greedy_trajectory_isolated", return_value=ck):
            report = runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=64,
                ctx_len=128,
                top_k=3,
                threads=1,
                llama_no_repack=False,
                stop_token_ids={2},
            )
        self.assertTrue(report["pass"])
        self.assertEqual(report["matched_stop_token"], 2)
        self.assertEqual(report["final_prefix"], [7, 1])
        self.assertEqual(report["execution_mode"], "persistent_greedy_trajectory")

    def test_trajectory_reports_first_top1_divergence(self) -> None:
        ck_rows = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        llama_rows = np.asarray([[0.0, 1.0, 4.0]], dtype=np.float32)
        with mock.patch.object(runner, "run_llama_greedy_trajectory", return_value={
            "logits": llama_rows, "generated_tokens": [2], "meta": {},
        }), mock.patch.object(runner, "load_ck_greedy_trajectory_isolated", return_value={
            "logits": ck_rows, "generated_tokens": [1], "vocab": 3,
        }):
            report = runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"), gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7], max_new_tokens=64, ctx_len=128, top_k=3,
                threads=1, llama_no_repack=False, stop_token_ids={2},
            )
        self.assertFalse(report["pass"])
        self.assertEqual(report["first_divergence"]["step"], 0)
        self.assertEqual(report["first_divergence"]["ck_next"], 1)
        self.assertEqual(report["first_divergence"]["llama_next"], 2)

    def test_trajectory_captures_isolated_ck_before_llama(self) -> None:
        rows = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        calls = []

        def ck_capture(**_kwargs):
            calls.append("ck")
            return {"logits": rows, "generated_tokens": [1], "vocab": 3}

        def llama_capture(*_args, **_kwargs):
            calls.append("llama")
            return {"logits": rows, "generated_tokens": [1], "meta": {}}

        with mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=ck_capture
        ), mock.patch.object(
            runner, "run_llama_greedy_trajectory", side_effect=llama_capture
        ):
            runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=1,
                ctx_len=128,
                top_k=3,
                threads=1,
                llama_no_repack=False,
            )
        self.assertEqual(calls, ["ck", "llama"])

    def test_requested_threads_reach_isolated_ck_capture(self) -> None:
        rows = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        captured: dict = {}

        def ck_capture(**kwargs):
            captured.update(kwargs)
            return {
                "logits": rows,
                "generated_tokens": [1],
                "vocab": 3,
                "thread_environment": {
                    "CK_NUM_THREADS": "7",
                    "CK_THREADPOOL_THREADS": "7",
                    "OMP_NUM_THREADS": "7",
                },
            }

        with mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=ck_capture
        ), mock.patch.object(
            runner,
            "run_llama_greedy_trajectory",
            return_value={"logits": rows, "generated_tokens": [1], "meta": {}},
        ):
            report = runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=1,
                ctx_len=128,
                top_k=3,
                threads=7,
                llama_no_repack=False,
            )

        self.assertEqual(captured["threads"], 7)
        self.assertEqual(report["threads"], 7)
        self.assertEqual(report["ck_thread_environment"]["CK_NUM_THREADS"], "7")


if __name__ == "__main__":
    unittest.main()
