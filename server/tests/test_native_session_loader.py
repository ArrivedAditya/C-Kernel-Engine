"""The built v8 host binaries must find their sibling tokenizer from any cwd."""

import os
import subprocess
import sys
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parents[2] / "build"
SESSION_LIB_PATH = BUILD_DIR / "libck_session_v8.so"


def test_session_and_cli_load_outside_checkout(tmp_path):
    assert SESSION_LIB_PATH.is_file(), "build with `make ck-session-v8` first"
    cli = BUILD_DIR / "ck-cli-v8"
    assert cli.is_file(), "build with `make ck-cli-v8` first"
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", "import ctypes, sys; ctypes.CDLL(sys.argv[1])", str(SESSION_LIB_PATH)],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [str(cli), "--help"], cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    # The CLI currently returns 1 for --help; the marker proves it reached main.
    assert "Native inference CLI" in result.stderr, result.stderr
