from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_cachegrind_tree.sh"


def _summary(path: Path) -> list[int]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("summary: "):
            return [int(value) for value in line.removeprefix("summary: ").split()]
    raise AssertionError(f"missing Cachegrind summary in {path}")


class CachegrindTreeTests(unittest.TestCase):
    def test_make_target_uses_process_tree_wrapper(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        target = makefile.split("profile-v7-cachegrind:", 1)[1].split("\n\n", 1)[0]
        self.assertIn(
            "PROFILE_V7_CACHEGRIND_RUNNER := scripts/run_cachegrind_tree.sh",
            makefile,
        )
        self.assertIn("$(PROFILE_V7_CACHEGRIND_RUNNER)", target)
        self.assertIn("build/cachegrind_v7.out", target)
        self.assertIn("build/cachegrind_v7_annotated.txt", target)

    @unittest.skipUnless(
        all(shutil.which(tool) for tool in ("valgrind", "cg_merge", "cg_annotate")),
        "Cachegrind tools are not installed",
    )
    def test_parent_and_child_reports_are_merged_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cachegrind.out"
            annotation = Path(tmp) / "cachegrind.txt"
            command = [
                str(SCRIPT),
                str(output),
                str(annotation),
                "--",
                sys.executable,
                "-c",
                "import subprocess; subprocess.run(['/bin/true'], check=True)",
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            raw_files = sorted(Path(f"{output}.d").glob("cachegrind.*.out"))
            self.assertGreaterEqual(len(raw_files), 2)
            raw_summaries = [_summary(path) for path in raw_files]
            merged = _summary(output)
            self.assertEqual(
                merged,
                [sum(values) for values in zip(*raw_summaries, strict=True)],
            )
            self.assertGreater(annotation.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
