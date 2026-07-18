from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_nightly_runner():
    path = ROOT / "scripts" / "nightly_runner.py"
    spec = importlib.util.spec_from_file_location("nightly_runner_q4k_perf_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Q4KLlamaPerformanceGateTests(unittest.TestCase):
    def test_nightly_registers_comparative_benchmark(self) -> None:
        nightly = _load_nightly_runner()
        entry = nightly.MAKE_TARGETS["q4k_q8k_llama_performance"]
        self.assertEqual(entry["category"], "bench")
        self.assertEqual(entry["target"], "test-q4k-q8k-llama-performance")
        self.assertGreaterEqual(entry["timeout_sec"], 120)

    def test_make_gate_uses_production_shape_and_hard_ratio(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        match = re.search(
            r"^test-q4k-q8k-llama-performance:.*?(?=^\.PHONY:|\Z)",
            makefile,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match)
        recipe = match.group(0)
        self.assertIn("CK_Q4K_PERF_M:-1028", recipe)
        self.assertIn("CK_Q4K_PERF_N:-4096", recipe)
        self.assertIn("CK_Q4K_PERF_K:-4096", recipe)
        self.assertIn("CK_Q4K_LLAMA_MAX_RATIO:-2.5", recipe)
        self.assertIn("--perf", recipe)


if __name__ == "__main__":
    unittest.main(verbosity=2)
