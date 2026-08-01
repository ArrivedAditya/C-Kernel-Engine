from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_nightly_runner():
    path = ROOT / "scripts" / "nightly_runner.py"
    spec = importlib.util.spec_from_file_location("nightly_runner_whisper_f16", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WhisperF16GemmPerformanceGateTests(unittest.TestCase):
    def test_nightly_registers_avx2_shape_gate(self) -> None:
        entry = _load_nightly_runner().MAKE_TARGETS["whisper_f16_gemm_performance"]
        self.assertEqual(entry["category"], "bench")
        self.assertEqual(entry["target"], "test-whisper-f16-gemm-performance")
        self.assertGreaterEqual(entry["timeout_sec"], 120)

    def test_make_gate_checks_real_shapes_and_speed(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        match = re.search(
            r"^test-whisper-f16-gemm-performance:.*?(?=^\.PHONY:|\Z)",
            makefile,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match)
        recipe = match.group(0)
        self.assertIn("base_projection", recipe)
        self.assertIn("base_mlp_up", recipe)
        self.assertIn("base_mlp_down", recipe)
        self.assertIn("CK_WHISPER_F16_GEMM_MIN_SPEEDUP:-1.05", recipe)
        self.assertIn("OPENBLAS_NUM_THREADS=1", recipe)

    def test_benchmark_fails_closed_on_output_or_speed(self) -> None:
        source = (ROOT / "benchmarks" / "bench_whisper_f16_gemm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("np.array_equal(baseline_output, optimized_output)", source)
        self.assertIn("speedup >= args.min_speedup", source)
        self.assertIn('report["status"] = "skip"', source)
        self.assertIn("M4xN2 provider is AVX2-specific", source)


if __name__ == "__main__":
    unittest.main()
