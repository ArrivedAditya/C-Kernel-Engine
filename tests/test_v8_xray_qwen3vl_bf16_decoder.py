#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import struct
import tempfile
import unittest
from array import array
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "xray_qwen3vl_bf16_decoder_v8.py"
SPEC = importlib.util.spec_from_file_location("xray_qwen3vl_bf16_decoder_v8_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
decoder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(decoder)


class Qwen3VLBf16DecoderXRayTests(unittest.TestCase):
    def test_prefill_probe_starts_at_layer_input(self) -> None:
        self.assertEqual(decoder.PREFILL_PROBE_NAMES[0], "layer_input")
        self.assertEqual(len(decoder.PREFILL_PROBE_NAMES), len(set(decoder.PREFILL_PROBE_NAMES)))
        self.assertEqual(decoder.PREFILL_PROBE_NAMES[-1], "layer_out")
        self.assertTrue(
            {
                "attn_pregate", "out_proj", "after_attn", "ffn_norm",
                "mlp_gate", "mlp_up", "mlp_swiglu", "mlp_down",
            }.issubset(decoder.PREFILL_PROBE_NAMES)
        )
        self.assertEqual(
            decoder.PREFILL_HEAD_MAJOR_NAMES,
            {"qk_norm_q", "qk_norm_k", "rope_q", "rope_k", "attn_pregate"},
        )

    def test_codegen_exports_the_importable_layer_input_checkpoint(self) -> None:
        source = (ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('ck_debug_import_checkpoint(model, {int(layer)}, "{checkpoint}"', source)
        self.assertIn('ck_debug_export_hidden(model, {int(layer)}, "{checkpoint}"', source)

    def test_requires_a_post_prefill_teacher_forced_failure(self) -> None:
        self.assertEqual(decoder._first_failed_position({"first_divergence": 20}), 20)
        with self.assertRaisesRegex(ValueError, "after position zero"):
            decoder._first_failed_position({"first_divergence": 0})

    def test_operation_metadata_uses_exact_occurrence_and_provider(self) -> None:
        def operation(op: str, function: str, kernel: str, contract: str) -> dict:
            return {
                "layer": 3,
                "op": op,
                "function": function,
                "call_abi": {"kernel_id": kernel},
                "resolved_contract": {"resolved_contract_id": contract},
            }

        call_ir = {"operations": [
            operation("residual_add", "residual_first", "residual.1", "contract.1"),
            operation("residual_add", "residual_second", "residual.2", "contract.2"),
        ]}
        metadata = decoder._operation_metadata(call_ir, 3, "layer_out")
        self.assertEqual(metadata["function"], "residual_second")
        self.assertEqual(metadata["kernel_id"], "residual.2")
        self.assertEqual(metadata["resolved_contract_id"], "contract.2")

    def test_manifest_retains_canonical_edge_and_provider_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer.f32"
            np.arange(8, dtype=np.float32).tofile(path)
            tensor = {
                0: {"layer_out": {
                    "path": str(path), "shape": [1, 8],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }}
            }
            call_ir = {"operations": [
                {
                    "layer": 0, "op": "residual_add", "function": "first",
                    "call_abi": {"kernel_id": "first"},
                    "resolved_contract": {"resolved_contract_id": "contract.first"},
                },
                {
                    "layer": 0, "op": "residual_add", "function": "layer_output",
                    "call_abi": {"kernel_id": "residual.bf16"},
                    "resolved_contract": {"resolved_contract_id": "contract.residual.bf16"},
                },
            ]}
            manifest = decoder._manifest(
                "ck", tensor, call_ir, [(0, "layer_out")], "teacher_forced", "fixture"
            )
        checkpoint = manifest["checkpoints"][0]
        self.assertEqual(checkpoint["checkpoint_id"], "decoder.layer.0.layer_out")
        self.assertEqual(checkpoint["function"], "layer_output")
        self.assertEqual(checkpoint["resolved_contract_id"], "contract.residual.bf16")

    def test_largest_layer_delta_uses_adjacent_relative_rmse_growth(self) -> None:
        rows = [
            {"checkpoint_id": "decoder.layer.0.layer_out", "metrics": {"relative_rmse": 0.001}},
            {"checkpoint_id": "decoder.layer.1.layer_out", "metrics": {"relative_rmse": 0.002}},
            {"checkpoint_id": "decoder.layer.2.layer_out", "metrics": {"relative_rmse": 0.010}},
            {"checkpoint_id": "decoder.layer.3.layer_out", "metrics": {"relative_rmse": 0.011}},
        ]
        self.assertEqual(decoder._largest_layer_delta(rows), 2)

    def test_certified_prefix_loader_enforces_exact_extent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prefix.f32"
            np.arange(8, dtype=np.float32).tofile(path)
            request = {"prefix_f32": str(path), "prefix_tokens": 2, "prefix_embed_dim": 4}
            self.assertEqual(len(decoder._load_certified_prefix(request)), 8)
            request["prefix_embed_dim"] = 3
            with self.assertRaisesRegex(ValueError, "more than 6"):
                decoder._load_certified_prefix(request)

    def test_prefix_comparison_rejects_stale_same_shape_artifacts(self) -> None:
        reference = array("f", [1.0, 2.0, 3.0, 4.0])
        exact = decoder._compare_prefix_arrays(array("f", reference), reference)
        self.assertTrue(exact["byte_exact"])
        self.assertEqual(exact["max_abs"], 0.0)

        stale = decoder._compare_prefix_arrays(array("f", [1.0, 2.0, 3.0, 4.5]), reference)
        self.assertFalse(stale["byte_exact"])
        self.assertTrue(stale["shape_match"])
        self.assertEqual(stale["exact_elements"], 3)
        self.assertEqual(stale["max_abs"], 0.5)

    def test_capture_directory_loaders_restore_canonical_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.arange(8, dtype=np.float32).tofile(root / "layer_002_qk_norm_q.f32")
            np.arange(12, dtype=np.float32).tofile(root / "tok_001_layer_002_mlp_gate_up.f32")
            torch_tensors = decoder._load_pytorch_capture_dir(root)
            ck_tensors, intermediate = decoder._load_ck_capture_dir(root)
            self.assertEqual(torch_tensors[2]["qk_norm_q"]["shape"], [1, 8])
            self.assertEqual(intermediate, 6)
            self.assertEqual(ck_tensors[2]["mlp_gate"]["shape"], [1, 6])
            self.assertEqual(ck_tensors[2]["mlp_up"]["shape"], [1, 6])

    def test_prefill_attention_context_is_canonicalized_from_head_major(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ck_dir = root / "ck"
            torch_dir = root / "torch"
            rows_dir = root / "rows"
            ck_dir.mkdir()
            torch_dir.mkdir()
            # CK prefill attention context is [head, token, channel].
            values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
            values.tofile(ck_dir / "tok_000_layer_000_attn_pregate.f32")
            reference = values[:, 1, :].reshape(-1)
            reference_path = torch_dir / "attn_pregate.f32"
            reference.tofile(reference_path)
            report = decoder._compare_prefill_probe(
                ck_dir,
                {
                    "token": 1,
                    "layer": 0,
                    "tensors": {"attn_pregate": {"path": str(reference_path)}},
                },
                token_count=3,
                head_dim=2,
                output_dir=rows_dir,
            )
        row = next(item for item in report["comparisons"] if item["name"] == "attn_pregate")
        self.assertEqual(row["status"], "exact")
        self.assertEqual(row["exact_elements"], 4)

    def test_kv_cache_comparison_reports_first_logical_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = np.arange(12, dtype=np.uint16).reshape(1, 2, 2, 3)
            value = (key + 100).astype(np.uint16)
            key.tofile(root / "key.bf16")
            value.tofile(root / "value.bf16")
            ck_value = value.copy()
            ck_value[0, 1, 0, 2] += 1
            with (root / "ck.bin").open("wb") as handle:
                handle.write(struct.pack("<8I", 0x564B5843, 1, 2, 2, 2, 32, 3, 0))
                key[0].tofile(handle)
                ck_value[0].tofile(handle)
            report = decoder._compare_kv_cache(root / "ck.bin", {
                "key": {"path": str(root / "key.bf16"), "shape": list(key.shape)},
                "value": {"path": str(root / "value.bf16"), "shape": list(value.shape)},
            })
            self.assertFalse(report["byte_exact"])
            self.assertEqual(report["first_difference"]["kind"], "value")
            self.assertEqual(report["first_difference"]["head"], 1)
            self.assertEqual(report["first_difference"]["token"], 0)
            self.assertEqual(report["first_difference"]["channel"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
