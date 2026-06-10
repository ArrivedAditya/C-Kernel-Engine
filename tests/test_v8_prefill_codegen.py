#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEGEN_PREFILL_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_prefill_v8.py"


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


codegen_prefill_v8 = _load_module("codegen_prefill_v8_tests", CODEGEN_PREFILL_PATH)


class TestV8PrefillCodegen(unittest.TestCase):
    def test_recurrent_prefill_seq_len_args_use_runtime_num_tokens(self) -> None:
        op = {
            "function": "recurrent_split_qkv_forward",
            "op": "recurrent_split_qkv",
            "layer": 0,
            "section": "body",
            "args": [
                {"name": "packed_qkv", "expr": "(const float*)(model->bump + A_RECURRENT_PACKED)"},
                {"name": "q", "expr": "(float*)(model->bump + A_RECURRENT_Q)"},
                {"name": "k", "expr": "(float*)(model->bump + A_RECURRENT_K)"},
                {"name": "v", "expr": "(float*)(model->bump + A_RECURRENT_V)"},
                {"name": "seq_len", "expr": "1034"},
                {"name": "q_dim", "expr": "2048"},
                {"name": "k_dim", "expr": "2048"},
                {"name": "v_dim", "expr": "2048"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 7, {"embed_dim": 1024})

        self.assertIn("num_tokens", emitted)
        self.assertNotIn("\n        1034,", emitted)
        self.assertIn("\n        2048,", emitted)


if __name__ == "__main__":
    unittest.main()
