#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
PROFILE = ROOT / "version" / "v8" / "parity_profiles" / "qwen3vl_llamacpp_q8_v1.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import sys
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

frontend = load_module("xray_vision_parity_v8_test", SCRIPTS / "xray_vision_parity_v8.py")
llama = load_module("xray_qwen3vl_llamacpp_v8_test", SCRIPTS / "xray_qwen3vl_llamacpp_v8.py")
xray = load_module("xray_numerical_parity_v8_test_interface", SCRIPTS / "xray_numerical_parity_v8.py")


class XRayVisionInterfaceTests(unittest.TestCase):
    def test_frontend_dispatches_llamacpp_without_owning_backend_arguments(self) -> None:
        with mock.patch.object(frontend.llamacpp_adapter, "main", return_value=7) as adapter:
            result = frontend.dispatch(["--backend", "llamacpp", "--gguf", "model.gguf"])
        self.assertEqual(result, 7)
        adapter.assert_called_once_with(["--gguf", "model.gguf"])

    def test_frontend_dispatches_pytorch_without_owning_backend_arguments(self) -> None:
        with mock.patch.object(frontend.pytorch_adapter, "main", return_value=0) as adapter:
            result = frontend.dispatch(["--backend", "pytorch", "--checkpoint", "model"])
        self.assertEqual(result, 0)
        adapter.assert_called_once_with(["--checkpoint", "model"])

    def test_llama_profile_is_schema_valid(self) -> None:
        profile = xray.load_json(PROFILE)
        xray.validate(profile, xray.PROFILE_SCHEMA, "llama profile")
        self.assertEqual(profile["backend"], "llamacpp")

    def test_legacy_results_are_reordered_by_semantic_circuit_position(self) -> None:
        profile = xray.load_json(PROFILE)
        report = {
            "results": [
                {"layer": 0, "op": "Kcur_rope", "status": "FAIL", "max_abs_diff": 0.2},
                {"layer": 0, "op": "q_proj", "status": "FAIL", "max_abs_diff": 0.1},
                {"layer": 0, "op": "ln1", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": -1, "op": "inp_pos_emb", "status": "PASS", "max_abs_diff": 0.0},
            ]
        }
        result = llama.normalize_capture_report(report, profile, layer=0)
        self.assertEqual(result["last_passing_checkpoint"], "vision.layer.0.norm1.output")
        self.assertEqual(result["first_divergence"]["checkpoint_id"], "vision.layer.0.q.pre_rope")
        self.assertEqual(
            result["first_divergence"]["classification"],
            "KERNEL_IMPLEMENTATION_DIVERGENCE",
        )

    def test_capture_adapter_is_documented_as_internal_to_xray(self) -> None:
        source = (SCRIPTS / "activation_parity_qwen3vl_mmproj_v8.py").read_text(encoding="utf-8")
        self.assertIn("must invoke ``xray_vision_parity_v8.py --backend", source)
        self.assertIn("Do not add model-family branches here", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
