from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_nightly_runner():
    path = ROOT / "scripts" / "nightly_runner.py"
    spec = importlib.util.spec_from_file_location("nightly_runner_v8_dsl_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nightly = _load_nightly_runner()


class V8DSLNightlyRegistrationTests(unittest.TestCase):
    EXPECTED_TARGETS = {
        "test-v8-dsl-policy": "v8 DSL Zero-Hardcoding Policy",
        "test-numerical-contracts": "v8 Numerical Kernel Contracts",
        "test-v8-template-circuit-audit": "v8 Template Circuit/Dataflow Audit",
    }

    def test_full_dsl_gate_dependencies_are_explicit_nightly_rows(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        match = re.search(r"^test-v8-dsl:\s+([^\n]+)$", makefile, re.MULTILINE)
        self.assertIsNotNone(match, "Makefile must define the aggregate test-v8-dsl gate")
        dependencies = set(match.group(1).split())
        self.assertEqual(dependencies, set(self.EXPECTED_TARGETS))

        registered = {
            entry["target"]: entry
            for entry in nightly.MAKE_TARGETS.values()
            if entry.get("target") in self.EXPECTED_TARGETS
        }
        self.assertEqual(set(registered), set(self.EXPECTED_TARGETS))
        for target, expected_name in self.EXPECTED_TARGETS.items():
            with self.subTest(target=target):
                self.assertEqual(registered[target]["name"], expected_name)
                self.assertIn(registered[target]["category"], {"inference", "parity"})

    def test_report_documents_the_three_visible_rows(self) -> None:
        source = (ROOT / "docs" / "site" / "_pages" / "test-report.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("v8 DSL and Codegen Contracts", source)
        self.assertIn('id="v8-dsl-contract-dashboard"', source)
        self.assertIn("function renderV8DSLContracts(results)", source)
        self.assertIn("renderV8DSLContracts(data.results || [])", source)
        for target in self.EXPECTED_TARGETS:
            with self.subTest(target=target):
                self.assertIn(f"<code>{target}</code>", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
