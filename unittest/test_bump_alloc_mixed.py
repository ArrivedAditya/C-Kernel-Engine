#!/usr/bin/env python3
"""Regression test for file-backed BUMP arenas with a nonzero metadata header."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MixedBumpAllocatorTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "mixed file-backed allocator is Linux/POSIX-only")
    def test_nonzero_weights_base_maps_absolute_file_offsets(self) -> None:
        source = r'''
#include "ckernel_alloc.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) return 10;
    ck_bump_alloc_t alloc;
    if (ck_bump_alloc_init(&alloc, argv[1], 8192, 3, 4096) != 0) return 11;
    if (alloc.mode != CK_BUMP_MODE_MIXED_FILE_BACKED) return 12;
    for (int i = 0; i < 4096; ++i) {
        if (alloc.base[i] != (uint8_t)(i % 251)) return 13;
    }
    alloc.base[4096] = 0xA5;
    if (alloc.base[4096] != 0xA5) return 14;
    ck_bump_alloc_free(&alloc);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="ck_bump_mixed_") as td:
            work = Path(td)
            bump = work / "weights.bump"
            bump.write_bytes(bytes(i % 251 for i in range(4096)))
            src = work / "probe.c"
            exe = work / "probe"
            src.write_text(source, encoding="ascii")
            subprocess.run(
                [
                    os.environ.get("CC", "cc"),
                    "-std=c11",
                    "-I",
                    str(ROOT / "include"),
                    str(src),
                    str(ROOT / "src/ckernel_alloc.c"),
                    "-lpthread",
                    "-o",
                    str(exe),
                ],
                check=True,
            )
            env = dict(os.environ)
            env["CK_BUMP_FORCE_MIXED"] = "1"
            subprocess.run([str(exe), str(bump)], check=True, env=env)


if __name__ == "__main__":
    unittest.main()
