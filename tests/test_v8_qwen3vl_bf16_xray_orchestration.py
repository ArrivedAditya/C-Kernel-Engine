#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version/v8/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "qwen3vl_bf16_xray_orchestration",
    SCRIPTS / "xray_qwen3vl_bf16_v8.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Qwen3VLBf16XrayOrchestrationTests(unittest.TestCase):
    def test_observed_storage_overrides_survive_manifest_construction(self) -> None:
        manifest = {
            "checkpoints": [
                {"checkpoint_id": "vision.layer.0.output", "storage_dtype": "fp32"},
                {"checkpoint_id": "vision.layer.0.mlp.up", "storage_dtype": "bf16"},
            ]
        }

        MODULE._apply_observed_storage(manifest, {
            "default": "bf16",
            "checkpoints": {"vision.layer.0.mlp.up": "fp32"},
        })

        self.assertEqual(manifest["checkpoints"][0]["storage_dtype"], "bf16")
        self.assertEqual(manifest["checkpoints"][1]["storage_dtype"], "fp32")


if __name__ == "__main__":
    unittest.main()
