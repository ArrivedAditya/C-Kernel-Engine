from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_nightly_runner():
    path = ROOT / "scripts" / "nightly_runner.py"
    spec = importlib.util.spec_from_file_location(
        "nightly_runner_tokenizer_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nightly = _load_nightly_runner()


class TokenizerNightlyRegistrationTests(unittest.TestCase):
    def test_strict_byte_fallback_parity_is_a_visible_nightly_gate(self) -> None:
        entry = nightly.MAKE_TARGETS["tokenizer_byte_fallback_parity"]
        self.assertEqual(entry["name"], "Tokenizer Byte/UTF-8 Parity")
        self.assertEqual(entry["category"], "inference")
        self.assertEqual(entry["target"], "test-tokenizer-special")
        self.assertGreaterEqual(entry["timeout_sec"], 120)

    def test_gate_invokes_the_hardened_oracle_suite(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("test-tokenizer-special: $(LIB_TOKENIZER)", makefile)
        self.assertIn("unittest/test_true_bpe_special_tokens.py", makefile)
        self.assertIn("CK_TOKENIZER_REQUIRE_HF_ORACLE=1", makefile)


if __name__ == "__main__":
    unittest.main(verbosity=2)
