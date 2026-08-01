import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "audit_layout_dataflow_v8.py"
SPEC = importlib.util.spec_from_file_location("audit_layout_dataflow_v8", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LayoutDataflowAuditTests(unittest.TestCase):
    @staticmethod
    def _op(idx, name, *, heads=32, kv_heads=8, head_dim=128, dtype="fp32"):
        return {
            "idx": idx,
            "layer": 0,
            "section": "body",
            "op": name,
            "function": "transpose_inplace",
            "params": {
                "num_heads": heads,
                "num_kv_heads": kv_heads,
                "head_dim": head_dim,
            },
            "outputs": {"buf": {"dtype": dtype}},
        }

    def test_dense_gqa_counts_payload_and_two_copy_passes(self):
        payload = {
            "config": {"model": "fixture", "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
            "operations": [
                self._op(1, "transpose_qkv_to_head_major"),
                self._op(2, "transpose_kv_to_head_major"),
                self._op(3, "transpose_kv_to_head_major"),
                self._op(4, "transpose_attn_out_to_token_major"),
            ],
        }
        report = MODULE.audit(payload, tokens=100)
        one_q = 100 * 32 * 128 * 4
        one_kv = 100 * 8 * 128 * 4
        expected_payload = one_q * 2 + one_kv * 2
        self.assertEqual(report["summary"]["layout_conversion_count"], 4)
        self.assertEqual(report["summary"]["payload_bytes"], expected_payload)
        self.assertEqual(report["summary"]["copied_bytes"], expected_payload * 2)
        self.assertEqual(report["summary"]["logical_read_write_bytes"], expected_payload * 4)
        self.assertEqual(report["summary"]["unmapped_conversion_count"], 4)
        self.assertEqual(report["summary"]["unresolved_parallel_ownership_count"], 0)
        self.assertTrue(
            all(
                row["false_sharing_status"] == "not_applicable_serial"
                for row in report["findings"]
            )
        )

    def test_recurrent_path_without_transposes_is_clean(self):
        report = MODULE.audit(
            {
                "config": {"model": "recurrent"},
                "operations": [{"idx": 1, "op": "split_recurrent_qkv"}],
            },
            tokens=1024,
        )
        self.assertEqual(report["summary"]["layout_conversion_count"], 0)
        self.assertEqual(report["summary"]["logical_read_write_bytes"], 0)

    def test_fp16_and_cross_kv_use_declared_width_and_kv_tokens(self):
        report = MODULE.audit(
            {
                "config": {"model": "audio"},
                "operations": [
                    self._op(1, "transpose_cross_kv_to_head_major", dtype="fp16"),
                ],
            },
            tokens=12,
            kv_tokens=1500,
        )
        expected_payload = 1500 * 8 * 128 * 2
        self.assertEqual(report["findings"][0]["tokens"], 1500)
        self.assertEqual(report["summary"]["payload_bytes"], expected_payload)

    def test_malformed_dimension_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "num_heads"):
            MODULE.audit(
                {
                    "config": {},
                    "operations": [
                        {
                            "idx": 1,
                            "op": "transpose_qkv_to_head_major",
                            "params": {"head_dim": 128},
                        }
                    ],
                },
                tokens=10,
            )

    def test_parallel_conversion_without_ownership_fails_audit_status(self):
        op = self._op(1, "transpose_qkv_to_head_major")
        op["parallel"] = {"enabled": True}
        report = MODULE.audit({"config": {"model": "fixture"}, "operations": [op]}, tokens=10)
        self.assertEqual(report["summary"]["unresolved_parallel_ownership_count"], 1)
        self.assertEqual(
            report["findings"][0]["false_sharing_status"],
            "unresolved_parallel_ownership",
        )


if __name__ == "__main__":
    unittest.main()
