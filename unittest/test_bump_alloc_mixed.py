#!/usr/bin/env python3
"""Regression tests for read-only file-backed BUMP arenas."""

from __future__ import annotations

import os
import ctypes
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MixedBumpAllocatorTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "mixed file-backed allocator is Linux-only")
    def test_mixed_layouts_and_failure_cleanup(self) -> None:
        source = r'''
#include "ckernel_alloc.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int writable_at(const uint8_t *address) {
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) return -1;
    char line[512], perms[5];
    unsigned long start, end;
    int writable = -1;
    while (fgets(line, sizeof(line), maps)) {
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3) continue;
        if ((unsigned long)address >= start && (unsigned long)address < end) {
            writable = perms[1] == 'w';
            break;
        }
    }
    fclose(maps);
    return writable;
}

int main(int argc, char **argv) {
    if (argc != 7) return 10;
    const size_t total = (size_t)strtoull(argv[2], NULL, 10);
    const size_t weights_base = (size_t)strtoull(argv[3], NULL, 10);
    const size_t activations_base = (size_t)strtoull(argv[4], NULL, 10);
    const size_t file_len = (size_t)strtoull(argv[5], NULL, 10);
    const int should_succeed = atoi(argv[6]);
    ck_bump_alloc_t alloc = {0};
    for (int attempt = 0; attempt < 2; ++attempt) {
        int rc = ck_bump_alloc_init(&alloc, argv[1], total,
                                    weights_base, activations_base);
        if (!should_succeed) {
            if (rc == 0) { ck_bump_alloc_free(&alloc); return 11; }
            if (alloc.base || alloc.mode != CK_BUMP_MODE_UNINITIALIZED ||
                alloc.total_size != 0) return 12;
            continue;
        }
        if (rc != 0 || alloc.mode != CK_BUMP_MODE_MIXED_FILE_BACKED) return 13;
        for (size_t i = 0; i < activations_base; ++i) {
            uint8_t expected = i < file_len ? (uint8_t)(i % 251) : 0;
            if (alloc.base[i] != expected) return 14;
        }
        if (activations_base >= CK_TEST_PAGE_SIZE && writable_at(alloc.base) != 0)
            return 15;
        if (writable_at(alloc.base + activations_base) != 1) return 16;
        alloc.base[activations_base] = 0xA5;
        if (alloc.base[activations_base] != 0xA5) return 17;
        ck_bump_alloc_free(&alloc);
        ck_bump_alloc_free(&alloc);
        if (alloc.base || alloc.mode != CK_BUMP_MODE_UNINITIALIZED) return 18;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="ck_bump_mixed_") as td:
            work = Path(td)
            bump = work / "weights.bump"
            page_size = os.sysconf("SC_PAGE_SIZE")
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
            cases = (
                ("aligned", 2 * page_size, 3, page_size, page_size, 1),
                ("unaligned", 2 * page_size, 3, page_size + 13, page_size + 13, 1),
                ("padded", 2 * page_size, 3, page_size + 13, page_size + 3, 1),
                ("subpage", page_size, 3, 13, 13, 1),
                ("undersized", 3 * page_size, 3, 2 * page_size + 13, page_size + 3, 0),
                ("bad-weights-base", 2 * page_size, page_size + 14, page_size + 13, page_size + 13, 0),
                ("bad-activations-base", page_size, 3, page_size + 13, page_size + 13, 0),
                ("overflow", 2**(8 * ctypes.sizeof(ctypes.c_size_t)) - 1,
                 3, page_size, page_size, 0),
            )
            for name, total, weights_base, activations_base, file_len, expected in cases:
                with self.subTest(name=name):
                    original = bytes(i % 251 for i in range(file_len))
                    bump.write_bytes(original)
                    subprocess.run(
                        [str(exe), str(bump), str(total), str(weights_base),
                         str(activations_base), str(file_len), str(expected)],
                        check=True, env=env,
                    )
                    self.assertEqual(bump.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
