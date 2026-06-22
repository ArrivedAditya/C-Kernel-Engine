#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_ir_v8 = _load_module("build_ir_v8_glm4_tests", V8_BUILD_PATH)


class V8GLM4TemplateTests(unittest.TestCase):
    def test_builtin_template_declares_glm4_contract(self) -> None:
        doc = build_ir_v8._load_builtin_template_doc("glm4")
        self.assertEqual(doc["name"], "glm4")
        self.assertEqual(doc["flags"]["tokenizer"], "bpe")
        self.assertEqual(doc["contract"]["attention_contract"]["rope_layout"], "split")
        self.assertTrue(doc["contract"]["attention_contract"]["partial_rotary"])
        self.assertTrue(doc["contract"]["block_contract"]["qkv_bias"])
        self.assertEqual(doc["contract"]["chat_contract"]["force_bos_text_if_tokenizer_add_bos_false"], "[gMASK]")
        self.assertEqual(
            doc["block_types"]["decoder"]["body"]["ops"],
            [
                "rmsnorm",
                "qkv_proj",
                "rope_qk",
                "attn",
                "out_proj",
                "post_attention_norm",
                "residual_add",
                "rmsnorm",
                "mlp_gate_up",
                "silu_mul",
                "mlp_down",
                "post_ffn_norm",
                "residual_add",
            ],
        )


if __name__ == "__main__":
    unittest.main()
