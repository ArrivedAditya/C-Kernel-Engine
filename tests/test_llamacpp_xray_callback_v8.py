from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "test_llamacpp_xray_callback_v8.py"
SPEC = importlib.util.spec_from_file_location("llamacpp_xray_callback", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CallbackEvidenceTests(unittest.TestCase):
    def test_each_invocation_gets_fresh_capture_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first_run, first_capture = MODULE.new_capture_run(output)
            (first_capture / "index.json").write_text("stale")
            second_run, second_capture = MODULE.new_capture_run(output)
            self.assertNotEqual(first_run, second_run)
            self.assertTrue(first_capture.joinpath("index.json").is_file())
            self.assertEqual(list(second_capture.iterdir()), [])

    def test_wrong_oracle_commit_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oracle = root / "oracle"
            oracle.mkdir()
            (root / "fixture.gguf").write_bytes(b"fixture")
            subprocess.run(["git", "init", "-q", str(oracle)], check=True)
            subprocess.run(["git", "-C", str(oracle), "-c", "user.name=CKE Test",
                            "-c", "user.email=cke@example.invalid", "commit", "--allow-empty", "-qm", "fixture"], check=True)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--model", str(root / "fixture.gguf"),
                 "--llama-root", str(oracle), "--output-dir", str(root / "out"),
                 "--expected-commit", "0" * 40],
                env=os.environ.copy(), capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_capture_requires_complete_extent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "attn_norm-0-token-000000-occ-000"
            record = {"name": name, "base_name": "attn_norm-0", "dtype": 0, "shape": [2],
                      "elem_count": 2, "nbytes": 8}
            (root / "index.json").write_text(json.dumps(record) + "\n")
            (root / f"{name}.bin").write_bytes(b"1234")
            with self.assertRaisesRegex(RuntimeError, "capture size"):
                MODULE.validate_capture(root, "attn_norm-0")
            (root / f"{name}.bin").write_bytes(b"12345678")
            self.assertEqual(MODULE.validate_capture(root, "attn_norm-0")["elements"], 2)

    def test_wrong_dtype_or_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"name": "one", "base_name": "attn_norm-0", "dtype": 1,
                      "shape": [2], "elem_count": 2, "nbytes": 8}
            (root / "index.json").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(RuntimeError, "invalid FP32"):
                MODULE.validate_capture(root, "attn_norm-0")
            record.update(dtype=0, shape=[3])
            (root / "index.json").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(RuntimeError, "invalid FP32"):
                MODULE.validate_capture(root, "attn_norm-0")

    def test_duplicate_capture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"name": "one", "base_name": "attn_norm-0", "elem_count": 2, "nbytes": 8}
            (root / "index.json").write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
            with self.assertRaisesRegex(RuntimeError, "expected one"):
                MODULE.validate_capture(root, "attn_norm-0")

    def test_loaded_library_must_be_from_selected_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "selected"
            other = root / "other"
            selected.mkdir()
            other.mkdir()
            library = other / "libllama.so.0"
            library.touch()
            ldd = mock.Mock(stdout=f"libllama.so.0 => {library} (0x0000)\n")
            with mock.patch.object(MODULE.subprocess, "run", return_value=ldd):
                with self.assertRaisesRegex(RuntimeError, "outside the selected oracle"):
                    MODULE.loaded_oracle_libraries(root / "helper", selected)


if __name__ == "__main__":
    unittest.main()
