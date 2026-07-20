from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_RUNNERS = (
    "scripts/v4/ck_run_v4.py",
    "scripts/ck_run_v5.py",
    "scripts/download_model_v6.py",
    "scripts/v6/ck_run_v6.py",
    "scripts/v6.5/ck_run_v6_5.py",
    "scripts/v6.6/ck_run_v6_6.py",
)


class LegacyRunnerKernelSourcePolicyTests(unittest.TestCase):
    def test_wildcard_fallback_excludes_parity_only_ggml_oracles(self) -> None:
        for relative_path in LEGACY_RUNNERS:
            with self.subTest(runner=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn('src_dir.glob("*.c")', source)
                self.assertIn(
                    'if not f.name.endswith("_oracle_ggml.c")',
                    source,
                )

    def test_ordinary_makefile_keeps_oracle_behind_explicit_parity_flag(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        strict_block = makefile.split("ifdef CK_LLAMA_PARITY_ENGINE", 1)[1].split(
            "endif", 1
        )[0]
        self.assertIn("src/kernels/attention_oracle_ggml.c", strict_block)
        self.assertIn("-DCK_ENABLE_LLAMA_CPP_PARITY=1", strict_block)
        self.assertIn("/ggml/include", strict_block)


if __name__ == "__main__":
    unittest.main()
