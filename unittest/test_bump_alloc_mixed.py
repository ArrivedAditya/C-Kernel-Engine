#!/usr/bin/env python3
"""Regression test for file-backed BUMP arenas with a nonzero metadata header."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MixedBumpAllocatorTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "mixed file-backed allocator is Linux-only")
    def test_nonzero_weights_base_maps_absolute_file_offsets(self) -> None:
        source = r'''
#include "ckernel_alloc.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 3) return 10;
    const int file_len = atoi(argv[2]);
    ck_bump_alloc_t alloc;
    if (ck_bump_alloc_init(&alloc, argv[1], 2 * CK_TEST_PAGE_SIZE, 3,
                           CK_TEST_PAGE_SIZE + 13) != 0) return 11;
    if (alloc.mode != CK_BUMP_MODE_MIXED_FILE_BACKED) return 12;
    for (int i = 0; i < CK_TEST_PAGE_SIZE + 13; ++i) {
        uint8_t expected = i < file_len ? (uint8_t)(i % 251) : 0;
        if (alloc.base[i] != expected) return 13;
    }
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) return 15;
    int weight_readonly = 0, boundary_writable = 0;
    char line[512], perms[5];
    unsigned long start, end;
    while (fgets(line, sizeof(line), maps)) {
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3) continue;
        if ((unsigned long)alloc.base >= start && (unsigned long)alloc.base < end)
            weight_readonly = perms[0] == 'r' && perms[1] == '-';
        if ((unsigned long)(alloc.base + CK_TEST_PAGE_SIZE) >= start &&
            (unsigned long)(alloc.base + CK_TEST_PAGE_SIZE) < end)
            boundary_writable = perms[0] == 'r' && perms[1] == 'w';
    }
    fclose(maps);
    if (!weight_readonly || !boundary_writable) return 16;
    alloc.base[CK_TEST_PAGE_SIZE + 13] = 0xA5;
    if (alloc.base[CK_TEST_PAGE_SIZE + 13] != 0xA5) return 14;
    ck_bump_alloc_free(&alloc);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="ck_bump_mixed_") as td:
            work = Path(td)
            bump = work / "weights.bump"
            page_size = os.sysconf("SC_PAGE_SIZE")
            original = bytes(i % 251 for i in range(page_size + 13))
            bump.write_bytes(original)
            src = work / "probe.c"
            exe = work / "probe"
            src.write_text(source, encoding="ascii")
            subprocess.run(
                [
                    os.environ.get("CC", "cc"),
                    "-std=c11",
                    f"-DCK_TEST_PAGE_SIZE={page_size}",
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
            subprocess.run([str(exe), str(bump), str(len(original))], check=True, env=env)
            self.assertEqual(bump.read_bytes(), original)
            short_file = original[: page_size + 3]
            bump.write_bytes(short_file)
            subprocess.run([str(exe), str(bump), str(len(short_file))], check=True, env=env)
            self.assertEqual(bump.read_bytes(), short_file)


if __name__ == "__main__":
    unittest.main()
