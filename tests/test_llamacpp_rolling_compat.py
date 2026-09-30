from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_llamacpp_rolling_compat.py"
SPEC = importlib.util.spec_from_file_location("rolling_compat", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RollingCompatReportTests(unittest.TestCase):
    def test_immutable_commit_ref_is_accepted(self) -> None:
        commit = "A" * 40
        self.assertEqual(MODULE.remote_commit(commit), commit.lower())

    def test_parse_quick_log(self) -> None:
        content = """
PERFORMANCE SUMMARY
  Average CK/llama.cpp speedup: 0.90x
  Average CK GFLOPS:            7.26
  Average llama.cpp GFLOPS:     9.77
PARITY SMOKETEST SUMMARY
  Passed:  6
  Failed:  0
  Skipped: 1
"""
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "quick.log"
            path.write_text(content, encoding="utf-8")
            summary = MODULE.parse_quick_log(path)
        self.assertEqual(summary["smoketest"], {"passed": 6, "failed": 0, "skipped": 1})
        self.assertEqual(
            summary["performance_summaries"],
            [{"ck_over_llama_speedup": 0.9, "ck_gflops": 7.26, "llama_gflops": 9.77}],
        )

    def test_legacy_patch_drift_does_not_replace_callback_evidence(self) -> None:
        phases = {
            "patch_compatibility": {"status": "warn", "blocking": False},
            "ck_build": {"status": "pass"},
            "quick_parity": {"status": "pass"},
            "production_graph_parity": {"status": "pass"},
            "mtmd_adapter_build": {"status": "pass"},
            "xray_callback": {"status": "pass"},
        }
        self.assertEqual(MODULE.compatibility_status(phases), "pass")

    def test_missing_capture_is_incomplete(self) -> None:
        self.assertEqual(
            MODULE.compatibility_status({"ck_build": {"status": "pass"}, "quick_parity": {"status": "pass"}}),
            "incomplete",
        )

    def test_missing_production_graph_evidence_is_incomplete(self) -> None:
        phases = {name: {"status": "pass"} for name in
                  ("ck_build", "quick_parity", "mtmd_adapter_build", "xray_callback")}
        self.assertEqual(MODULE.compatibility_status(phases), "incomplete")

    def test_required_phase_failure_remains_blocking(self) -> None:
        for phase_name in ("ck_build", "quick_parity", "production_graph_parity", "mtmd_adapter_build", "xray_callback"):
            with self.subTest(phase=phase_name):
                phases = {
                    "patch_compatibility": {"status": "pass", "blocking": False},
                    "ck_build": {"status": "pass"},
                    "quick_parity": {"status": "pass"},
                    "production_graph_parity": {"status": "pass"},
                    "mtmd_adapter_build": {"status": "pass"},
                    "xray_callback": {"status": "pass"},
                }
                phases[phase_name]["status"] = "fail"
                self.assertEqual(MODULE.compatibility_status(phases), "fail")


if __name__ == "__main__":
    unittest.main()
