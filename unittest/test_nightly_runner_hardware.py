#!/usr/bin/env python3
"""Contracts for the hardware evidence embedded in nightly JSON reports."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nightly_runner import (  # noqa: E402
    _parse_lscpu_output,
    _parse_meminfo,
    _parse_proc_cpuinfo,
    _redact_proc_cpuinfo,
    capture_runner_hardware,
)


class RunnerHardwareEvidenceTests(unittest.TestCase):
    def test_lscpu_parser_preserves_cpu_identity_and_topology(self):
        fields = _parse_lscpu_output(
            """Architecture:                         x86_64
Vendor ID:                            AuthenticAMD
Model name:                           AMD EPYC 7763 64-Core Processor
CPU(s):                               4
Thread(s) per core:                   2
Core(s) per socket:                   2
Socket(s):                            1
Flags:                                fma avx avx2
"""
        )
        self.assertEqual(fields["Vendor ID"], "AuthenticAMD")
        self.assertEqual(fields["Model name"], "AMD EPYC 7763 64-Core Processor")
        self.assertEqual(fields["CPU(s)"], "4")
        self.assertIn("avx2", fields["Flags"].split())

    def test_current_linux_host_produces_publishable_evidence(self):
        evidence = capture_runner_hardware()
        self.assertTrue(evidence["available"], evidence.get("error"))
        self.assertTrue(evidence["cpu"]["architecture"])
        self.assertTrue(evidence["cpu"]["vendor_id"])
        self.assertTrue(evidence["cpu"]["model_name"])
        self.assertTrue(evidence["raw_lscpu"].startswith("Architecture:"))
        self.assertIn("avx2", evidence["isa"])
        self.assertTrue(evidence["memory"]["total"])
        self.assertIn("MemTotal:", evidence["raw_meminfo"])
        self.assertIn("Mem:", evidence["raw_free_h"])

    def test_procfs_fallback_parses_cpu_and_redacts_serial(self):
        raw = """processor: 0
vendor_id: AuthenticAMD
model name: AMD EPYC 7763 64-Core Processor
flags: fma avx avx2
Serial: 1234567890

processor: 1
vendor_id: AuthenticAMD
"""
        redacted = _redact_proc_cpuinfo(raw)
        fields = _parse_proc_cpuinfo(redacted)
        self.assertEqual(fields["Model name"], "AMD EPYC 7763 64-Core Processor")
        self.assertEqual(fields["CPU(s)"], "2")
        self.assertNotIn("1234567890", redacted)

    def test_meminfo_parser_keeps_capacity_fields(self):
        fields = _parse_meminfo("MemTotal: 16384000 kB\nMemAvailable: 12000000 kB\n")
        self.assertEqual(fields["MemTotal"], "16384000 kB")
        self.assertEqual(fields["MemAvailable"], "12000000 kB")


if __name__ == "__main__":
    unittest.main()
