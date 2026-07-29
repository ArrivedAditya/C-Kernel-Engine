from __future__ import annotations

import importlib.util
import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "sweep_q4k_llama_performance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sweep_q4k_llama_performance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Q4KLlamaSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sweep = _load_module()

    def test_shape_parser_uses_explicit_m_n_k(self) -> None:
        shape = self.sweep.parse_shape("qwen36=32x34816x5120")
        self.assertEqual(
            (shape.name, shape.M, shape.N, shape.K),
            ("qwen36", 32, 34816, 5120),
        )
        with self.assertRaisesRegex(ValueError, "divisible by 256"):
            self.sweep.parse_shape("bad=1x8x255")

    def test_mreuse_expands_tiles_but_fixed_providers_do_not(self) -> None:
        jobs = self.sweep.build_provider_jobs(
            ["mreuse", "4m", "8m", "4m-vnni-x8", "16m-vnni-x16"],
            [4, 6, 8, 12, 16, 32],
        )
        self.assertEqual(jobs[:6], [("mreuse", value) for value in [4, 6, 8, 12, 16, 32]])
        self.assertEqual(
            jobs[6:],
            [
                ("4m", 0),
                ("8m", 0),
                ("4m-vnni-x8", 0),
                ("16m-vnni-x16", 0),
            ],
        )

    def test_provider_output_parser_requires_exact_reference(self) -> None:
        row = self.sweep.parse_provider_output(
            "provider=mreuse reference_provider=4m exact=true "
            "M=32 N=4096 K=4096 threads=12 tile_m=8 "
            "time_ms=4.125 gflops=260.300 checksum=-1.25"
        )
        self.assertTrue(row["exact"])
        self.assertEqual(row["reference_provider"], "4m")
        self.assertEqual(row["threads"], 12)
        self.assertAlmostEqual(row["time_ms"], 4.125)

    def test_llama_oracle_parser_records_ratio_and_exactness(self) -> None:
        row = self.sweep.parse_oracle_output(
            "performance-gate output bit_exact (10 values) [PASS]\n"
            "Q4_K performance: M=1028 N=4096 K=4096 repeats=5 "
            "ck_ms=38.282 llama_ms=26.583 ck_over_llama=1.440 "
            "max_ratio=report-only [PASS]"
        )
        self.assertTrue(row["bit_exact"])
        self.assertEqual(row["repeats"], 5)
        self.assertAlmostEqual(row["ratio"], 1.44)

    def test_hardware_report_does_not_include_node_identity(self) -> None:
        hardware = self.sweep.collect_hardware()
        self.assertNotIn("hostname", hardware)
        self.assertNotIn("ip", hardware)
        self.assertIn("cpu", hardware)
        self.assertIn("logical_cpus_visible", hardware)

    def test_default_shapes_are_qwen36_hot_projections(self) -> None:
        shapes = [self.sweep.parse_shape(value) for value in self.sweep.QWEN36_HOT_SHAPES]
        self.assertEqual(len(shapes), 4)
        self.assertEqual({shape.M for shape in shapes}, {33, 1034})
        self.assertEqual({(shape.N, shape.K) for shape in shapes}, {
            (34816, 5120),
            (6144, 5120),
        })

    def test_csv_table_contains_provider_and_oracle_rows(self) -> None:
        report = {
            "hardware": {"cpu": "Test CPU"},
            "software_provenance": {
                "isa_label": "test-isa",
                "compiler": "test-compiler",
                "engine_commit": "abc123",
            },
            "results": [
                {
                    "kind": "cke_provider",
                    "shape_name": "qwen36_prompt33_mlp_gate_up",
                    "M": 33,
                    "requested_M": 33,
                    "N": 34816,
                    "K": 5120,
                    "provider": "mreuse",
                    "threads": 24,
                    "tile_m": 8,
                    "time_ms": 1.25,
                    "exact": True,
                    "status": "pass",
                },
                {
                    "kind": "llama_oracle",
                    "shape_name": "qwen36_prompt33_mlp_gate_up",
                    "M": 33,
                    "N": 34816,
                    "K": 5120,
                    "threads": 24,
                    "ck_ms": 2.0,
                    "llama_ms": 1.0,
                    "ratio": 2.0,
                    "bit_exact": True,
                    "status": "pass",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            self.sweep.write_csv_table(report, path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["kind"] for row in rows], ["cke_provider", "llama_oracle"])
        self.assertEqual(rows[0]["provider"], "mreuse")
        self.assertEqual(rows[0]["phase"], "")
        self.assertEqual(rows[1]["llama_ms"], "1.0")


if __name__ == "__main__":
    unittest.main()
