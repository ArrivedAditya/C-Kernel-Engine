"""Failure isolation and shared kernel-map/nightly selection regressions."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nightly_runner as nightly
import tests.test_v8_audio_adaptive_layer_norm_oracle as adaln


class KernelOracleReportingTest(unittest.TestCase):
    def test_reference_drift_does_not_hide_native_comparison(self):
        fixture = json.loads(adaln.FIXTURE.read_text())
        case = fixture["cases"][1]
        fake_torch = types.SimpleNamespace(__version__="synthetic-independent-reference")
        comparisons = [(case, case["output"], (1e-3, (0, 1., 0.999)))]
        suite = unittest.TestSuite([
            adaln.AdaptiveLayerNormOracleTest("test_fixture_vs_live_oracle"),
            adaln.AdaptiveLayerNormOracleTest("test_live_torch_oracle"),
        ])
        stream = io.StringIO()
        with mock.patch.object(adaln.AdaptiveLayerNormOracleTest, "capture_live_cases",
                               return_value=(fake_torch, fixture, comparisons)), contextlib.redirect_stdout(stream):
            result = unittest.TestResult()
            suite.run(result)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.errors, [])
        cases = nightly.parse_sub_tests(stream.getvalue())
        self.assertEqual([case.status for case in cases], ["fail", "pass"])
        self.assertNotEqual(cases[0].case_id, cases[1].case_id)
        self.assertEqual(cases[1].metadata["provider"], "audio_adaptive_layer_norm_f32")
        self.assertEqual(cases[1].metadata["execution_settings"]["dispatch_identity"], "NOT_VERIFIED")

    def test_map_selects_existing_registration_without_duplicates(self):
        path = "version/v8/kernel_maps/audio_adaptive_layer_norm_f32.json"
        selected = nightly.select_kernel_map_tests([path, path])
        self.assertEqual(selected, ["tts_adaptive_layer_norm_oracle", "tts_adaptive_layer_norm_live"])
        self.assertNotIn("tts_live_torch_oracles", nightly.TEST_SUITES)
        targets = [target for key in selected for target in nightly.TEST_SUITES[key].unittest_targets]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(len(targets), 6)

    def test_unregistered_map_test_fails_selection(self):
        with mock.patch.object(nightly.ROOT.__class__, "read_text", return_value=json.dumps({
                "tests": {"unit": ["tests/no_nightly_registration.py"]}})):
            with self.assertRaisesRegex(ValueError, "missing nightly registration"):
                nightly.select_kernel_map_tests(["version/v8/kernel_maps/audio_adaptive_layer_norm_f32.json"])

    def test_missing_oracle_is_skip_not_pass(self):
        completed = types.SimpleNamespace(returncode=0, stdout="", stderr="Ran 2 tests in 0.01s\n\nOK (skipped=2)\n")
        with mock.patch.object(nightly.subprocess, "run", return_value=completed):
            result = nightly.run_python_test(nightly.TEST_SUITES["tts_adaptive_layer_norm_live"])
        self.assertEqual(result.status, "skip")
        self.assertIn("all selected", result.error_msg)

    def test_historical_settings_require_an_available_mkl_backend(self):
        torch = types.SimpleNamespace(backends=types.SimpleNamespace(
            mkl=types.SimpleNamespace(is_available=lambda: False)))
        with mock.patch.dict(os.environ, {"CKE_EXPECTED_MKL_CBWR": "AVX2", "MKL_CBWR": "AVX2"}), \
                mock.patch.dict(os.environ, {"CKE_EXPECTED_TORCH_VERSION": ""}):
            with self.assertRaisesRegex(unittest.SkipTest, "MKL reference environment is unavailable"):
                adaln.AdaptiveLayerNormOracleTest().assert_pinned_torch(torch)

    def test_environment_merge_preserves_failures_and_provenance(self):
        identity = {"repository_commit": "commit", "github_run_id": "run", "github_run_attempt": "1"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            primary = root / "primary.json"
            extra = root / "oracle" / "nightly-report-native.json"
            extra.parent.mkdir()
            primary.write_text(json.dumps({"run_identity": identity, "summary": {},
                "results": [{"name": "primary failure", "status": "fail", "duration_sec": 1., "sub_tests": []}]}))
            extra.write_text(json.dumps({"run_identity": identity, "oracle_environment": {
                "lane": "native-dispatch", "torch_version_requested": "2.12.1"},
                "runner_hardware": {"cpu": {"model_name": "independent CPU"}},
                "results": [{"name": "native oracle", "status": "pass", "duration_sec": 2.,
                    "execution_id": "tts_adaptive_layer_norm_live", "execution_kind": "python",
                    "execution_args": [], "sub_tests": []}]}))
            self.assertEqual(nightly.merge_environment_reports(primary, extra.parent, 2), 1)
            report = json.loads(primary.read_text())
            self.assertEqual(report["summary"]["failed"], 2)  # primary + missing environment
            self.assertEqual(report["summary"]["passed"], 1)
            self.assertEqual(report["results"][1]["oracle_report_source"]["runner_hardware"]["cpu"]["model_name"], "independent CPU")
            self.assertIn("native-dispatch", report["results"][1]["name"])

    def test_stale_environment_is_not_attached_as_current(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            primary, extra = root / "primary.json", root / "nightly-report-stale.json"
            primary.write_text(json.dumps({"run_identity": {"repository_commit": "new"},
                                           "summary": {}, "results": []}))
            extra.write_text(json.dumps({"run_identity": {"repository_commit": "old"},
                                         "results": [{"status": "pass"}]}))
            self.assertEqual(nightly.merge_environment_reports(primary, root, 1), 1)
            report = json.loads(primary.read_text())
            self.assertEqual(report["summary"]["passed"], 0)
            self.assertIn("mismatched repository_commit", report["results"][0]["error_msg"])

    def test_required_environment_skip_survives_merge_as_missing_evidence(self):
        for status, cases in (("skip", []), ("pass", [{"status": "not_tested",
                              "evidence_kind": "test_execution"}])):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                primary, extra = root / "primary.json", root / "nightly-report-required.json"
                identity = {"repository_commit": "same"}
                primary.write_text(json.dumps({"run_identity": identity, "summary": {}, "results": []}))
                extra.write_text(json.dumps({"run_identity": identity,
                    "selection": {"require_executed": True},
                    "results": [{"name": "oracle", "status": status,
                                 "duration_sec": 0., "sub_tests": cases}]}))
                self.assertEqual(nightly.merge_environment_reports(primary, root, 1), 1)
                report = json.loads(primary.read_text())
                self.assertEqual(report["results"][0]["status"], status)
                self.assertEqual(report["summary"]["failed"], 1)
                self.assertIn("required oracle execution is missing", report["results"][-1]["error_msg"])

    def test_unittest_dependency_status_keeps_case_identity_and_reason(self):
        cases = nightly.parse_unittest_cases("test_fixture (tests.oracle.Case.test_fixture) ... ok\n"
            "test_live (tests.oracle.Case.test_live) ... skipped 'PyTorch unavailable'\n")
        self.assertEqual([case.status for case in cases], ["pass", "not_tested"])
        self.assertEqual(cases[1].case_id, "tests.oracle.Case.test_live")
        self.assertIn("PyTorch unavailable", cases[1].metadata["outcome"])

    def test_class_setup_dependency_skip_is_not_zero_test_pass(self):
        completed = types.SimpleNamespace(returncode=0, stdout="", stderr=
            "setUpClass (tests.oracle.LiveCase) ... skipped 'PyTorch unavailable'\n"
            "Ran 0 tests in 0.01s\n\nOK (skipped=1)\n")
        with mock.patch.object(nightly.subprocess, "run", return_value=completed):
            result = nightly.run_python_test(nightly.TEST_SUITES["tts_duration_logits_live"])
        self.assertEqual(result.status, "skip")
        self.assertIn("PyTorch unavailable", result.error_msg)
        self.assertEqual(result.sub_tests[0].status, "not_tested")

    def test_zero_case_registration_is_not_pass(self):
        completed = types.SimpleNamespace(returncode=0, stdout="", stderr="Ran 0 tests in 0.01s\n\nOK\n")
        with mock.patch.object(nightly.subprocess, "run", return_value=completed):
            result = nightly.run_python_test(nightly.TEST_SUITES["tts_duration_logits_live"])
        self.assertEqual(result.status, "fail")
        self.assertIn("zero cases", result.error_msg)

    def test_timeout_preserves_completed_cases_and_failure_status(self):
        suite = nightly.TEST_SUITES["tts_kokoro_generated_encoder"]
        record = {"case_id": "encoder.partial", "name": "completed-boundary",
                  "status": "pass", "max_diff": 1e-6, "tolerance": 3e-5}
        stdout = "CKE_NUMERICAL_CASE " + json.dumps(record) + "\n"
        stderr = "test_finished (tests.oracle.Case.test_finished) ... ok\n"
        for binary in (False, True):
            with self.subTest(binary=binary):
                expired = nightly.subprocess.TimeoutExpired("native-oracle", suite.timeout_sec,
                    output=stdout.encode() if binary else stdout,
                    stderr=stderr.encode() if binary else stderr)
                with mock.patch.object(nightly.subprocess, "run", side_effect=expired) as run:
                    result = nightly.run_python_test(suite)
                self.assertEqual(run.call_args.args[0][1], "-u")
                self.assertEqual(result.status, "timeout")
                self.assertEqual(result.duration_sec, 300)
                self.assertIn("300s", result.error_msg)
                self.assertIn("encoder.partial", result.stdout)
                self.assertIn("test_finished", result.stderr)
                self.assertEqual(len(result.sub_tests), 2)
                self.assertTrue(all(case.status == "pass" for case in result.sub_tests))

    def test_timeout_without_output_remains_timeout(self):
        suite = nightly.TEST_SUITES["tts_duration_logits_live"]
        with mock.patch.object(nightly.subprocess, "run",
                side_effect=nightly.subprocess.TimeoutExpired("empty", suite.timeout_sec)):
            result = nightly.run_python_test(suite)
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.sub_tests, [])

    def test_timeout_with_invalid_partial_metrics_retains_diagnostic(self):
        suite = nightly.TEST_SUITES["tts_duration_logits_live"]
        output = 'CKE_NUMERICAL_CASE ' + json.dumps({"case_id": "invalid-partial",
            "name": "invalid", "status": "pass", "max_diff": float("nan"), "tolerance": 1e-5})
        with mock.patch.object(nightly.subprocess, "run",
                side_effect=nightly.subprocess.TimeoutExpired("invalid", suite.timeout_sec, output=output)):
            result = nightly.run_python_test(suite)
        self.assertEqual(result.status, "timeout")
        self.assertIn("partial-output parsing failed", result.error_msg)
        self.assertIn("invalid-partial", result.stdout)
        self.assertEqual(result.sub_tests, [])

    def test_nonfinite_metrics_cannot_report_pass(self):
        for maximum in (float("nan"), float("inf"), 2e-5):
            with self.subTest(maximum=maximum):
                record = {"case_id": "adversarial", "name": "native-vs-live",
                          "status": "pass", "max_diff": maximum, "tolerance": 1e-5}
                with self.assertRaises(ValueError):
                    nightly.parse_sub_tests("CKE_NUMERICAL_CASE " + json.dumps(record))


if __name__ == "__main__":
    unittest.main()
