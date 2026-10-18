import os
import subprocess
import unittest
from pathlib import Path

import numpy as np

from version.v8.scripts.replay_ggml_attention_packet_v8 import (
    fp64_reference, pinned_tree_identity, run_ggml, validate_oracle_contract,
)
from version.v8.scripts.xray_operation_bundle_v8 import BundleError


ROOT = Path(__file__).resolve().parents[1]


class GgmlAttentionPacketTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"context": {"oracle_mode": "disabled", "valid_cache_positions": [0, 3]}}
        self.call = {"sliding_window": 8, "head_dim": 4, "aligned_head_dim": 4,
                     "selector": "CK_ATTN_REDUCTION_F16_REGULAR_NOSCALE"}

    def test_rejects_incompatible_contracts(self):
        validate_oracle_contract(self.manifest, self.call, [1, 2])
        cases = (
            ({"oracle_mode": "enabled"}, {}, "non-flash"),
            ({"valid_cache_positions": [0, 2]}, {}, "position range"),
            ({}, {"sliding_window": 2}, "wrapped"),
            ({}, {"aligned_head_dim": 8}, "padded"),
            ({}, {"selector": "CK_ATTN_REDUCTION_SCALED"}, "scaling"),
        )
        for context_patch, call_patch, message in cases:
            with self.subTest(message=message):
                manifest = {"context": {**self.manifest["context"], **context_patch}}
                call = {**self.call, **call_patch}
                with self.assertRaisesRegex(BundleError, message):
                    validate_oracle_contract(manifest, call, [1, 2])

    def test_pinned_ggml_graph_bounded_against_small_fp64_case(self):
        tree = Path(os.environ.get("CK_PINNED_LLAMA_TREE", str(ROOT / "llama.cpp")))
        try:
            pinned_tree_identity(tree)
        except (BundleError, OSError, subprocess.CalledProcessError):
            self.skipTest("pinned GGML libraries unavailable; asset-backed replay runs locally")
        q = np.array([[[0.25, -0.5, 0.75, 1.0], [0.4, -0.2, 0.8, -0.1]],
                      [[1.0, 0.5, -0.75, 0.25], [-0.3, 0.2, 0.7, 0.8]],
                      [[0.5, 0.4, -0.6, 0.2], [0.6, -0.7, 0.1, 0.3]]], dtype=np.float32)
        k = np.array([[[0.5, 0.25, -0.25, 0.75]],
                      [[-0.5, 0.1, 0.8, -0.3]],
                      [[0.7, -0.4, 0.2, 0.6]]], dtype=np.float16)
        v = np.array([[[0.1, 0.2, 0.3, 0.4]],
                      [[-0.2, 0.4, 0.8, 0.1]],
                      [[0.9, -0.1, 0.2, 0.7]]], dtype=np.float16)
        actual, identity = run_ggml(tree, q, k, v)
        expected = fp64_reference(q, k, v)
        self.assertEqual(identity["source_revision"], subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip())
        self.assertLess(float(np.max(np.abs(actual - expected))), 2e-4)
        self.assertGreater(float(np.max(np.abs(actual - expected))), 1e-5)
        self.assertGreater(float(np.max(np.abs(actual[0] - actual[-1]))), 0.1)


if __name__ == "__main__":
    unittest.main()
