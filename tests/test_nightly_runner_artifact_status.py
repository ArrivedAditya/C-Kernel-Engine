#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nightly_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("nightly_runner_artifact_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NightlyArtifactStatusTests(unittest.TestCase):
    def test_fresh_artifact_status_overrides_zero_exit(self) -> None:
        runner = _load_runner()
        target = {
            "name": "artifact gate",
            "category": "bf16",
            "target": "fake-gate",
            "timeout_sec": 10,
            "status_artifact": "build/fake/summary.json",
        }
        completed = subprocess.CompletedProcess(["make", "fake-gate"], 0, stdout="", stderr="")
        for artifact_status in ("pass", "skip", "fail"):
            with self.subTest(status=artifact_status):
                with mock.patch.object(runner.subprocess, "run", return_value=completed):
                    with mock.patch.object(
                        runner,
                        "_load_json_if_fresh",
                        return_value={"status": artifact_status},
                    ):
                        result = runner.run_make_target(target)
                self.assertEqual(result.status, artifact_status)

    def test_methodical_qwen3vl_stage_lines_are_visible_subtests(self) -> None:
        runner = _load_runner()
        names = [
            "qwen3vl_checkpoint_coverage",
            "qwen3vl_circuit_codegen",
            "qwen3vl_frontend_mrope",
            "qwen3vl_attention_contract",
            "qwen3vl_q8_projection_matrix",
            "qwen3vl_eos_contract",
        ]
        output = "\n".join(
            f"{name} max_diff=0 tol=0 [PASS]" for name in names
        )
        parsed = runner.parse_sub_tests(output)
        self.assertEqual([row.name for row in parsed], names)
        self.assertTrue(all(row.status == "pass" for row in parsed))

    def test_phase_status_prevents_q8_pass_from_masking_bf16_skip(self) -> None:
        runner = _load_runner()
        target = {
            "name": "BF16 artifact gate",
            "category": "bf16",
            "target": "fake-gate",
            "timeout_sec": 10,
            "status_artifact": "build/fake/summary.json",
            "status_phase": "bf16_pytorch",
        }
        artifact = {
            "status": "pass",
            "phases": {
                "q8_mmproj_llamacpp": {"status": "pass"},
                "bf16_pytorch": {
                    "status": "skip",
                    "reason": "missing BF16 checkpoint",
                },
            },
        }
        completed = subprocess.CompletedProcess(
            ["make", "fake-gate"], 0, stdout="", stderr=""
        )
        with mock.patch.object(runner.subprocess, "run", return_value=completed):
            with mock.patch.object(
                runner, "_load_json_if_fresh", return_value=artifact
            ):
                result = runner.run_make_target(target)
        self.assertEqual(result.status, "skip")
        self.assertEqual(result.error_msg, "missing BF16 checkpoint")


if __name__ == "__main__":
    unittest.main()
