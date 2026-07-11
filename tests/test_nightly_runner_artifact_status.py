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


if __name__ == "__main__":
    unittest.main()
