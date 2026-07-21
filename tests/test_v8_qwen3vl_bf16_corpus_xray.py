#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/xray_qwen3vl_bf16_corpus_failures_v8.py"
SPEC = importlib.util.spec_from_file_location("qwen3vl_bf16_corpus_xray", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CorpusXrayTests(unittest.TestCase):
    def test_sparse_profile_stops_at_first_encoder_interval(self) -> None:
        profile = json.loads(
            (ROOT / "version/v8/parity_profiles/qwen3vl_pytorch_bf16_corpus_sparse_v1.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            profile["checkpoint_order"],
            [
                "vision.frontend.position.output",
                "vision.layer.0.output",
                "vision.layer.8.output",
                "vision.layer.16.output",
            ],
        )
        self.assertIn(
            "vision.layer.0.output->vision.layer.8.output",
            profile["interval_expansions"],
        )

    def test_full_profile_can_bisect_early_and_late_intervals(self) -> None:
        profile = json.loads(
            (ROOT / "version/v8/parity_profiles/qwen3vl_pytorch_bf16_v1.json")
            .read_text(encoding="utf-8")
        )
        expansions = profile["interval_expansions"]
        self.assertIn("vision.layer.0.output->vision.layer.8.output", expansions)
        self.assertIn("vision.layer.8.output->vision.layer.16.output", expansions)

    def test_corpus_driver_defaults_to_certified_sdpa_backend(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('default="sdpa"', source)
        self.assertIn('"--attn-implementation", args.attn_implementation', source)
        self.assertIn('shutil.rmtree(capture_dir)', source)

    def test_runtime_map_requires_call_ir_and_distinct_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            (runtime / "call.json").write_text("{}", encoding="utf-8")
            result = MODULE._runtime_map([f"1x56x72={runtime}"])
            self.assertEqual(result[(1, 56, 72)], runtime.resolve())
            with self.assertRaisesRegex(ValueError, "duplicate runtime geometry"):
                MODULE._runtime_map([f"1x56x72={runtime}", f"1x56x72={runtime}"])

    def test_sample_image_is_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": [{"inputs": [{"path": "form.ppm"}]}]}), encoding="utf-8")
            self.assertEqual(MODULE._sample_image(manifest, 0), (root / "form.ppm").resolve())

    def test_groups_attributions_and_errors(self) -> None:
        rows = [
            {"first_divergent_checkpoint": "vision.layer.8.output", "status": "fail"},
            {"first_divergent_checkpoint": "vision.layer.8.output", "status": "fail"},
            {"status": "error"},
        ]
        self.assertEqual(
            MODULE._group_counts(rows),
            {"error": 1, "vision.layer.8.output": 2},
        )

    def test_corpus_summary_preserves_non_exact_and_material_boundaries(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('final.get("first_non_exact_checkpoint")', source)
        self.assertIn('"first_non_exact_checkpoint": non_exact.get("checkpoint_id")', source)
        self.assertIn('"first_material_checkpoint": divergence.get("checkpoint_id")', source)


if __name__ == "__main__":
    unittest.main()
