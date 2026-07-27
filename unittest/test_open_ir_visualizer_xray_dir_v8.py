#!/usr/bin/env python3
"""Regression tests for --xray-dir exclusivity in open_ir_visualizer_v8."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "version" / "v8" / "tools" / "open_ir_visualizer_v8.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("open_ir_visualizer_v8", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["open_ir_visualizer_v8"] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class XrayDirExclusivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()

    def _make_model(self, root: Path) -> Path:
        model = root / "model"
        # Minimal required IR files so the loader does not flag missing-required.
        _write(model / "ir1_decode.json", {"operations": []})
        _write(model / "layout_decode.json", {"buffers": []})
        _write(model / "lowered_decode_call.json", {"operations": [{"idx": 0}]})
        # A global X-ray artifact that must NOT be picked up under exclusivity.
        _write(
            model / "xray_qwen3vl_bf16_summary.json",
            {"schema": "cke.xray_orchestration_report", "backend": "pytorch"},
        )
        return model

    def _make_xray_dir(self, root: Path, backend: str = "llamacpp") -> Path:
        xray = root / "xray_out"
        _write(
            xray / "xray_summary.json",
            {"schema": "cke.xray_orchestration_report", "backend": backend},
        )
        return xray

    def test_xray_dir_is_exclusive_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._make_model(Path(tmp))
            xray = self._make_xray_dir(Path(tmp))
            data = self.tool.load_model_data(model, xray_dir=xray)
            # The generic summary binds to the llamacpp card.
            self.assertIn("xray_qwen3vl_llamacpp", data["files"])
            # The model-root pytorch artifact must not leak in.
            self.assertNotIn("xray_qwen3vl_pytorch", data["files"])
            self.assertTrue(data["meta"]["xray_dir_exclusive"])
            self.assertNotIn("xray_fallback_used", data["meta"])

    def test_fallback_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._make_model(Path(tmp))
            xray = self._make_xray_dir(Path(tmp))
            data = self.tool.load_model_data(
                model, xray_dir=xray, allow_xray_fallback=True
            )
            self.assertIn("xray_qwen3vl_llamacpp", data["files"])
            self.assertIn("xray_qwen3vl_pytorch", data["files"])
            self.assertFalse(data["meta"]["xray_dir_exclusive"])
            fallbacks = data["meta"].get("xray_fallback_used") or []
            self.assertTrue(any("xray_qwen3vl_pytorch" in f for f in fallbacks))

    def test_generic_summary_binds_by_backend_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._make_model(Path(tmp))
            xray = self._make_xray_dir(Path(tmp), backend="llamacpp")
            data = self.tool.load_model_data(model, xray_dir=xray)
            payload = data["files"]["xray_qwen3vl_llamacpp"]
            self.assertEqual(payload["backend"], "llamacpp")
            src = data["meta"]["loaded_paths"]["xray_qwen3vl_llamacpp"]
            self.assertEqual(Path(src), xray / "xray_summary.json")
            self.assertIn("xray_qwen3vl_llamacpp", data["meta"]["loaded_hashes"])


class XrayGlobalPathLeakTests(unittest.TestCase):
    """Hard-coded global build/ paths must obey --xray-dir exclusivity too."""

    XRAY_KEYS = (
        "xray_whisper_encoder",
        "xray_decoder_pytorch",
        "xray_qwen3vl_pytorch",
        "xray_qwen3vl_llamacpp",
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()

    def _make_fake_project_root(self, root: Path) -> Path:
        fake = root / "fake_project"
        _write(fake / "build" / "whisper-encoder-xray.json",
               {"schema": "cke.whisper_encoder_pytorch_xray", "checkpoints": []})
        _write(fake / "build" / "xray" / "decoder_pytorch" / "xray_summary.json",
               {"schema": "cke.xray.decoder_pytorch"})
        _write(fake / "build" / "xray" / "qwen3vl_bf16" / "xray_summary.json",
               {"schema": "cke.xray_orchestration_report", "backend": "pytorch"})
        _write(fake / "build" / "xray" / "qwen3vl_llamacpp" / "xray_summary.json",
               {"schema": "cke.xray_orchestration_report", "backend": "llamacpp"})
        return fake

    def _load_with_fake_root(self, model: Path, xray: Path, fallback: bool) -> dict:
        original = self.tool.PROJECT_ROOT
        self.tool.PROJECT_ROOT = self._fake_root
        try:
            return self.tool.load_model_data(
                model, xray_dir=xray, allow_xray_fallback=fallback
            )
        finally:
            self.tool.PROJECT_ROOT = original

    def _setup(self, tmp: str, fallback: bool) -> dict:
        base = Path(tmp)
        model = base / "model"
        _write(model / "ir1_decode.json", {"operations": []})
        _write(model / "layout_decode.json", {"buffers": []})
        _write(model / "lowered_decode_call.json", {"operations": [{"idx": 0}]})
        self._fake_root = self._make_fake_project_root(base)
        xray = base / "empty_xray_dir"
        xray.mkdir(parents=True, exist_ok=True)
        return self._load_with_fake_root(model, xray, fallback)

    def test_exclusive_mode_blocks_hardcoded_global_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = self._setup(tmp, fallback=False)
            for key in self.XRAY_KEYS:
                self.assertNotIn(key, data["files"], f"{key} leaked from global build path")
            self.assertTrue(data["meta"]["xray_dir_exclusive"])
            self.assertNotIn("xray_fallback_used", data["meta"])

    def test_fallback_mode_loads_and_records_every_global_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = self._setup(tmp, fallback=True)
            for key in self.XRAY_KEYS:
                self.assertIn(key, data["files"], f"{key} missing with fallback enabled")
            self.assertFalse(data["meta"]["xray_dir_exclusive"])
            recorded = data["meta"].get("xray_fallback_used") or []
            for key in self.XRAY_KEYS:
                self.assertTrue(
                    any(entry.startswith(key + " ") or entry.startswith(key + " ->") or key in entry
                        for entry in recorded),
                    f"{key} fallback not recorded in meta.xray_fallback_used: {recorded}",
                )


if __name__ == "__main__":
    unittest.main()
