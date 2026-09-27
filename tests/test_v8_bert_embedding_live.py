"""Optional live pinned Kokoro/PyTorch embedding oracle for nightly hosts."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests import test_v8_bert_embedding_oracle as oracle

ROOT = oracle.ROOT


class BertEmbeddingLiveTest(unittest.TestCase):
    def test_live_pinned_model_boundary(self):
        model_dir = Path(os.environ.get("CKE_KOKORO_MODEL_DIR", ""))
        if not model_dir.is_dir() or not (model_dir / "kokoro-v1_0.pth").is_file():
            self.skipTest("pinned Kokoro assets unavailable; set CKE_KOKORO_MODEL_DIR")
        oracle_python = os.environ.get("CKE_KOKORO_ORACLE_PYTHON")
        if not oracle_python:
            self.skipTest("pinned PyTorch/Kokoro oracle unavailable; set CKE_KOKORO_ORACLE_PYTHON")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "embedding.npz"
            subprocess.run([
                oracle_python,
                str(ROOT / "version/v8/tts/reference/capture_bert_embedding.py"),
                "--model-dir", str(model_dir), "--output", str(output),
            ], check=True)
            live = dict(np.load(output))
            metadata = json.loads(output.with_suffix(".json").read_text())
        oracle.BertEmbeddingOracleTest.setUpClass()
        try:
            candidate = oracle.BertEmbeddingOracleTest("test_pinned_kokoro_embedding_boundary")
            candidate.data = live
            candidate.manifest = metadata
            status, actual = candidate.invoke()
            self.assertEqual(status, 0)
            self.assertTrue(np.isfinite(actual).all())
            error = np.abs(actual.reshape(live["expected"].shape) - live["expected"])
            self.assertLessEqual(float(error.max()), 2e-5)
        finally:
            oracle.BertEmbeddingOracleTest.tearDownClass()


if __name__ == "__main__":
    unittest.main()
