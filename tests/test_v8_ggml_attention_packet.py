import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from version.v8.scripts.replay_ggml_attention_packet_v8 import (
    FLASH_SOURCE, _compile_ggml_source, fp64_full_flash_reference_rows,
    fp64_reference, pinned_tree_identity, run_ggml, run_ggml_flash_rows,
    validate_oracle_contract,
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

    def test_full_flash_geometry_and_operand_contract(self):
        q = np.zeros((3, 2, 4), dtype=np.float32)
        k = np.zeros((3, 1, 4), dtype=np.float16)
        v = np.zeros_like(k)
        tree = Path("unused-for-rejected-inputs")
        with self.assertRaisesRegex(BundleError, "full-query geometry"):
            run_ggml_flash_rows(tree, q[:1], k, v, scale=1.0)
        with self.assertRaisesRegex(BundleError, "finite operands"):
            run_ggml_flash_rows(tree, q, k, v, scale=float("nan"))
        with self.assertRaisesRegex(BundleError, "matching K/V"):
            run_ggml_flash_rows(tree, q, k, v[:2], scale=1.0)
        with self.assertRaisesRegex(BundleError, "observed FP32 Q"):
            run_ggml_flash_rows(tree, q, k.astype(np.float32), v, scale=1.0)
        with self.assertRaisesRegex(BundleError, "finite operands"):
            bad_q = q.copy()
            bad_q[0, 0, 0] = np.inf
            run_ggml_flash_rows(tree, bad_q, k, v, scale=1.0)

    def test_pinned_full_flash_against_noncausal_fp64(self):
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
        actual, identity = run_ggml_flash_rows(tree, q, k, v, scale=1.0)
        expected = np.empty_like(q)
        for token in range(q.shape[0]):
            for head in range(q.shape[1]):
                scores = k[:, 0].astype(np.float64) @ q[token, head].astype(np.float64)
                weights = np.exp(scores - scores.max())
                expected[token, head] = (weights / weights.sum()) @ v[:, 0].astype(np.float64)
        self.assertEqual(identity["contract"]["query_extent"], "complete")
        self.assertEqual(identity["contract"]["mode"], "full_noncausal_flash")
        self.assertLess(float(np.max(np.abs(actual - expected))), 2e-4)
        np.testing.assert_allclose(
            fp64_full_flash_reference_rows(q, k, v, [0, 2], scale=1.0),
            expected[[0, 2]], rtol=0, atol=1e-7,
        )
        with self.assertRaisesRegex(BundleError, "distinct in-range"):
            fp64_full_flash_reference_rows(q, k, v, [0, 0], scale=1.0)
        changed = v.copy()
        changed[-1, 0, 0] += np.float16(1.0)
        altered, _ = run_ggml_flash_rows(tree, q, k, changed, scale=1.0)
        self.assertGreater(float(abs(altered[0, 0, 0] - actual[0, 0, 0])), 0.05)
        subset, evidence = run_ggml_flash_rows(
            tree, q[:1], k, v, scale=1.0, allow_subset_queries=True,
        )
        self.assertEqual(subset.shape, (1, 2, 4))
        self.assertEqual(evidence["contract"]["query_extent"], "subset_not_connected_fidelity")

    def test_flash_helper_rejects_truncated_operand(self):
        tree = Path(os.environ.get("CK_PINNED_LLAMA_TREE", str(ROOT / "llama.cpp")))
        try:
            pinned_tree_identity(tree)
        except (BundleError, OSError, subprocess.CalledProcessError):
            self.skipTest("pinned GGML libraries unavailable; asset-backed replay runs locally")
        with tempfile.TemporaryDirectory(prefix="cke-flash-truncated-") as temp:
            directory = Path(temp)
            executable = directory / "replay"
            _compile_ggml_source(tree, FLASH_SOURCE, executable)
            q, k, v, output = (directory / name for name in ("q.f32", "k.f16", "v.f16", "out.f32"))
            q.write_bytes(np.zeros((3, 2, 4), dtype="<f4").tobytes())
            k.write_bytes(np.zeros((1,), dtype="<f2").tobytes())
            v.write_bytes(np.zeros((3, 1, 4), dtype="<f2").tobytes())
            env = dict(os.environ, LD_LIBRARY_PATH=str(tree / "build" / "bin"))
            result = subprocess.run(
                [str(executable), str(q), str(k), str(v), str(output),
                 "3", "3", "2", "1", "4", "1", "1"],
                env=env, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("incorrectly sized", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
