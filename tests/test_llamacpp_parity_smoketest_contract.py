from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_parity_smoketest.sh"


class ParitySmoketestContractTests(unittest.TestCase):
    def test_stale_oracle_rejected_before_build_or_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=CKE Test",
                            "-c", "user.email=cke@example.invalid", "commit", "--allow-empty", "-qm", "fixture"], check=True)
            env = os.environ.copy()
            env.update({"LLAMA_CPP_DIR": str(repo), "LLAMA_CPP_COMMIT": "0" * 40})
            result = subprocess.run([str(SCRIPT), "--skip-build", "--kernels", "--require-kernel-parity"],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match required", result.stdout)
            self.assertNotIn("Building CK parity library", result.stdout)

    def test_missing_llama_helper_cannot_fall_back_to_pytorch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "oracle"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=CKE Test",
                            "-c", "user.email=cke@example.invalid", "commit", "--allow-empty", "-qm", "fixture"], check=True)
            revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            tools = root / "bin"
            tools.mkdir()
            fake_make = tools / "make"
            fake_make.write_text("#!/bin/sh\nexit 0\n")
            fake_make.chmod(0o755)
            env = os.environ.copy()
            env.update({"LLAMA_CPP_DIR": str(repo), "LLAMA_CPP_COMMIT": revision,
                        "PATH": f"{tools}:{env['PATH']}"})
            result = subprocess.run([str(SCRIPT), "--skip-build", "--kernels", "--require-kernel-parity"],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Required llama.cpp kernel parity inputs are missing", result.stdout)
            self.assertNotIn("Falling back to PyTorch", result.stdout)


if __name__ == "__main__":
    unittest.main()
