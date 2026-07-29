#!/usr/bin/env python3
"""Contracts for keeping heavyweight pre-commit model gates host-safe."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "pre-commit"


class PrecommitResourceGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = HOOK.read_text(encoding="utf-8")

    def test_host_wide_lock_precedes_snapshot_and_model_execution(self) -> None:
        lock = self.source.index("flock -n 9")
        snapshot = self.source.index("build_staged_snapshot\n")
        v7 = self.source.index('v7-regression-fast REGRESSION_ARGS=')
        v8 = self.source.index('v8-regression-fast REGRESSION_ARGS=')
        self.assertLess(lock, snapshot)
        self.assertLess(lock, v7)
        self.assertLess(lock, v8)

    def test_resource_floor_is_checked_before_each_heavy_stage(self) -> None:
        self.assertIn(
            'PRECOMMIT_MIN_AVAILABLE_GB="${CK_PRECOMMIT_MIN_AVAILABLE_GB:-16}"',
            self.source,
        )
        self.assertIn(
            'PRECOMMIT_MIN_SWAP_FREE_GB="${CK_PRECOMMIT_MIN_SWAP_FREE_GB:-2}"',
            self.source,
        )
        self.assertGreaterEqual(self.source.count("check_resource_headroom\n"), 3)
        self.assertIn('CK_PRECOMMIT_ALLOW_LOW_RESOURCES', self.source)
        self.assertIn('"${swap_total_kb:-0}" -gt 0', self.source)

    def test_regressions_are_bounded_and_deprioritized(self) -> None:
        self.assertIn(
            'PRECOMMIT_THREADS="${CK_PRECOMMIT_THREADS:-8}"',
            self.source,
        )
        self.assertIn(
            'PRECOMMIT_TIMEOUT_SECONDS="${CK_PRECOMMIT_TIMEOUT_SECONDS:-1800}"',
            self.source,
        )
        self.assertIn('CK_NUM_THREADS="$PRECOMMIT_THREADS"', self.source)
        self.assertIn("OMP_NUM_THREADS=1", self.source)
        self.assertIn("timeout --signal=TERM --kill-after=30s", self.source)
        self.assertIn("nice -n 10 ionice -c 2 -n 7", self.source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
