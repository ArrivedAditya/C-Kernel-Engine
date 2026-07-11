from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "version" / "v8" / "scripts").resolve()))


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load("audit_dsl_policy_v8", ROOT / "version" / "v8" / "scripts" / "audit_dsl_policy_v8.py")
build_ir = _load("build_ir_v8_dsl_policy", ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py")


class V8DSLPolicyTests(unittest.TestCase):
    def test_cleaned_compiler_functions_have_no_model_literals(self) -> None:
        report = audit.audit()
        self.assertEqual(report["status"], "pass", report["findings"])
        self.assertGreaterEqual(report["checked_functions"], 3)

    def test_ast_policy_rejects_model_specific_branch(self) -> None:
        source = """
def lower(config):
    if config.get('model') == 'qwen_new':
        return 'special_kernel'
    return 'generic_kernel'
"""
        findings = audit.scan_source(source, ["lower"], ["qwen", "gemma"], path="synthetic.py")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["function"], "lower")

    def test_runtime_defaults_are_key_order_deterministic(self) -> None:
        first = {
            "contract": {
                "runtime_defaults": {
                    "prefer_q8_0_contract": True,
                    "activation_preference_by_op": {"mlp_down": "fp32", "out_proj": "q8_0"},
                }
            }
        }
        second = json.loads(json.dumps(first, sort_keys=True))
        self.assertEqual(
            build_ir._apply_circuit_runtime_defaults({}, first, source="first"),
            build_ir._apply_circuit_runtime_defaults({}, second, source="second"),
        )

    def test_runtime_default_override_is_explicit_and_stable(self) -> None:
        circuit = {
            "contract": {
                "runtime_defaults": {
                    "prefer_q8_0_contract": True,
                    "activation_preference_by_op": {"mlp_down": "fp32", "out_proj": "q8_0"},
                }
            }
        }
        actual = build_ir._apply_circuit_runtime_defaults(
            {"prefer_q8_0_contract": False, "activation_preference_by_op": {"mlp_down": "q8_0"}},
            circuit,
            source="override",
        )
        self.assertFalse(actual["prefer_q8_0_contract"])
        self.assertEqual(actual["activation_preference_by_op"]["mlp_down"], "q8_0")
        self.assertEqual(actual["activation_preference_by_op"]["out_proj"], "q8_0")

    def test_policy_rejects_missing_function_instead_of_weakening_scope(self) -> None:
        policy = {
            "schema": "cke.v8_dsl_policy",
            "schema_version": 1,
            "forbidden_model_literals": ["qwen"],
            "compiler_functions": {"version/v8/scripts/build_ir_v8.py": ["function_does_not_exist"]},
        }
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaisesRegex(audit.DSLPolicyError, "policy function not found"):
                audit.audit(path)


if __name__ == "__main__":
    unittest.main()
