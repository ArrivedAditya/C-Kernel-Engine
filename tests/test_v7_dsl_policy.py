from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "version" / "v8" / "scripts" / "audit_dsl_policy_v8.py"
POLICY_PATH = ROOT / "version" / "v7" / "dsl_policy.json"


def _load_audit():
    spec = importlib.util.spec_from_file_location("audit_training_dsl_policy_v7", AUDIT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V7TrainingDSLPolicyTests(unittest.TestCase):
    def test_training_and_backprop_compiler_has_no_family_dispatch(self) -> None:
        report = _load_audit().audit(POLICY_PATH)
        self.assertEqual(report["status"], "pass", report.get("findings"))

    def test_training_policy_rejects_hidden_family_alias(self) -> None:
        audit = _load_audit()
        source = """
def synthesize_backward(config):
    architecture = config.get('arch', '')
    selected = architecture
    if selected == 'new_family':
        return 'special_backward'
    return 'declared_backward'
"""
        findings = audit.scan_model_dispatch_source(
            source,
            ["model", "model_type", "arch", "family"],
            path="synthetic_v7.py",
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["function"], "synthesize_backward")


if __name__ == "__main__":
    unittest.main()
