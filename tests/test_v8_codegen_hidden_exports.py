#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEGEN_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py"
sys.path.insert(0, str(CODEGEN_PATH.parent))


def _load_codegen():
    spec = importlib.util.spec_from_file_location("codegen_core_hidden_export_tests", CODEGEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {CODEGEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


codegen = _load_codegen()


def _arg(name: str, expr: str) -> dict[str, str]:
    return {"name": name, "expr": expr}


class HiddenExportExtentTests(unittest.TestCase):
    def test_mlp_projection_exports_cover_all_rows_and_channels(self) -> None:
        up = codegen.emit_op(
            {
                "op": "mlp_up",
                "function": "gemm_nt_bf16",
                "layer": 1,
                "args": [
                    _arg("a", "A"),
                    _arg("b", "B"),
                    _arg("bias", "BIAS"),
                    _arg("c", "UP"),
                    _arg("m", "4032"),
                    _arg("n", "4304"),
                    _arg("k", "1152"),
                ],
            }
        )
        down = codegen.emit_op(
            {
                "op": "mlp_down",
                "function": "gemm_nt_bf16",
                "layer": 1,
                "args": [
                    _arg("a", "UP"),
                    _arg("b", "B"),
                    _arg("bias", "BIAS"),
                    _arg("c", "DOWN"),
                    _arg("m", "4032"),
                    _arg("n", "1152"),
                    _arg("k", "4304"),
                ],
            }
        )

        self.assertIn('"mlp_up", (const float*)UP, (4032) * (4304)', up)
        self.assertIn('"mlp_up_last"', up)
        self.assertIn('(size_t)(4304)', up)
        self.assertIn('"mlp_down", (const float*)DOWN, (4032) * (1152)', down)
        self.assertIn('"mlp_down_last"', down)
        self.assertIn('(size_t)(1152)', down)

    def test_gelu_exports_the_full_post_activation_tensor(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "gelu",
                "function": "gelu_ggml_inplace",
                "layer": 1,
                "args": [_arg("data", "UP"), _arg("n", "17353728")],
            }
        )

        self.assertIn('"ffn_gelu", (const float*)UP, 17353728', emitted)

    def test_attention_norm_exports_the_projection_input_boundary(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "attn_norm",
                "function": "rmsnorm_forward",
                "layer": 0,
                "args": [
                    _arg("input", "X"),
                    _arg("weight", "W"),
                    _arg("output", "Y"),
                    _arg("bias", "NULL"),
                    _arg("num_tokens", "1"),
                    _arg("dim", "1024"),
                    _arg("aligned_dim", "1024"),
                    _arg("eps", "1e-6f"),
                ],
            }
        )

        self.assertIn('"attn_norm", (const float*)Y, EMBED_DIM', emitted)


if __name__ == "__main__":
    unittest.main()
