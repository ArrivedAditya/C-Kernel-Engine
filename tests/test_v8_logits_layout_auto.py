#!/usr/bin/env python3
from __future__ import annotations

"""
Regression test: --logits-layout auto must plan last-position logits.

Before this fix, auto resolved to "full" for prefill, which made the v8
prefill memory plan reserve context_len x vocab x fp32 of logits scratch
(35 GiB for a 262144-vocab model at 32k context) even though chat and
serving only consume the last-position logits. That broke small-memory
targets (TDA4VM, 2 GB RAM) and inflated arenas everywhere. Explicit
--logits-layout full (used by training) must keep the full reservation.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_IR_SCRIPT = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"

SPEC = importlib.util.spec_from_file_location("build_ir_v8_logits_layout_test", BUILD_IR_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sys.path.insert(0, str(BUILD_IR_SCRIPT.parent))
build_ir = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_ir
SPEC.loader.exec_module(build_ir)


class LogitsLayoutAutoTests(unittest.TestCase):
    def test_auto_resolves_last_for_prefill(self) -> None:
        self.assertEqual(build_ir._resolve_logits_layout({}, "prefill"), "last")
        self.assertEqual(
            build_ir._resolve_logits_layout({"logits_layout": "auto"}, "prefill"),
            "last",
        )

    def test_auto_resolves_last_for_decode(self) -> None:
        self.assertEqual(build_ir._resolve_logits_layout({}, "decode"), "last")

    def test_explicit_full_is_preserved(self) -> None:
        cfg = {"logits_layout": "full"}
        self.assertEqual(build_ir._resolve_logits_layout(cfg, "prefill"), "full")
        self.assertEqual(build_ir._resolve_logits_layout(cfg, "decode"), "full")

    def test_explicit_last_is_preserved(self) -> None:
        cfg = {"logits_layout": "last"}
        self.assertEqual(build_ir._resolve_logits_layout(cfg, "prefill"), "last")

    def test_auto_prefill_plans_single_logits_row(self) -> None:
        layout = build_ir._resolve_logits_layout({}, "prefill")
        seq = build_ir._logits_seq_for_layout(layout, "prefill", 32768, 32768, {})
        self.assertEqual(seq, 1)

    def test_explicit_full_prefill_plans_full_sequence(self) -> None:
        seq = build_ir._logits_seq_for_layout("full", "prefill", 32768, 32768, {})
        self.assertEqual(seq, 32768)

    def test_invalid_layout_falls_back_to_auto(self) -> None:
        self.assertEqual(
            build_ir._resolve_logits_layout({"logits_layout": "bogus"}, "prefill"),
            "last",
        )


if __name__ == "__main__":
    unittest.main()
