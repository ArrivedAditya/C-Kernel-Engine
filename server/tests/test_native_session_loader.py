"""Relocated v8 host binaries must load their sibling tokenizer library."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_BUILD_DIR = Path(__file__).resolve().parents[2] / "build"


def test_session_and_cli_load_relocated_sibling_tokenizer(tmp_path):
    build_dir = Path(os.environ.get("CK_TEST_BUILD_DIR", DEFAULT_BUILD_DIR)).resolve()
    names = ("libck_session_v8.so", "ck-cli-v8", "libckernel_tokenizer.so")
    relocated = tmp_path / "relocated bundle"
    relocated.mkdir()
    for name in names:
        source = build_dir / name
        assert source.is_file(), f"missing {source}; build with `make ck-session-v8 ck-cli-v8`"
        shutil.copy2(source, relocated / name)

    tokenizer = relocated / "libckernel_tokenizer.so"
    env = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "LD_DEBUG"):
        env.pop(key, None)
    child = """
import ctypes, hashlib, json, sys
from pathlib import Path
ctypes.CDLL(sys.argv[1])
maps = [line.split(maxsplit=5) for line in Path('/proc/self/maps').read_text().splitlines()]
paths = sorted({Path(parts[5]).resolve() for parts in maps
                if len(parts) > 5 and parts[5].endswith('/libckernel_tokenizer.so')})
print(json.dumps([{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in paths]))
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(relocated / "libck_session_v8.so")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [{
        "path": str(tokenizer.resolve()),
        "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
    }]

    cli_env = env.copy()
    cli_env["LD_DEBUG"] = "libs"
    result = subprocess.run(
        [str(relocated / "ck-cli-v8"), "--help"],
        cwd=tmp_path, env=cli_env, capture_output=True, text=True, timeout=10,
    )
    # The CLI currently returns 1 for --help; reaching main proves startup.
    assert result.returncode == 1, result.stderr
    assert "Native inference CLI" in result.stderr, result.stderr
    assert f"calling init: {tokenizer}" in result.stderr, result.stderr
