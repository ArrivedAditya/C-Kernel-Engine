from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MtmdCaptureLayoutTests(unittest.TestCase):
    def test_compiled_strided_capture_and_rejections(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_mtmd_capture_layout_") as temp:
            executable = Path(temp) / "capture_layout"
            subprocess.run(
                [
                    "c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "version/v8/tools"),
                    str(ROOT / "tests/fixtures/mtmd_capture_layout_v8.cpp"),
                    "-o", str(executable),
                ],
                check=True, capture_output=True, text=True,
            )
            subprocess.run([str(executable)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
