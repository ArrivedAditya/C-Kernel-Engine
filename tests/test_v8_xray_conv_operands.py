"""Convolution X-Ray evidence distinguishes operands from local arithmetic."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.xray_conv_operands_v8 import (
    compare_packets, import_oracle_capture, load_oracle_capture,
)
from version.v8.scripts.xray_operation_bundle_v8 import BundleError, sha256, write_bundle


class ConvOperandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / "libtest.so"
        self.runtime.write_bytes(b"synthetic runtime")

    def packet(self, name, *, changed=None, axes=None, omit=None):
        image = np.arange(30, dtype=np.float32).reshape(1, 2, 3, 5)
        patches = np.arange(48, dtype=np.float16).reshape(1, 2, 3, 8)
        weights = np.arange(24, dtype=np.float16).reshape(3, 8)
        result = np.arange(18, dtype=np.float32).reshape(6, 3)
        tensors = {
            "conv_image": (image, "fp32", ("batch", "channel", "height", "width")),
            "conv_im2col": (patches, "fp16", ("batch", "out_height", "out_width", "reduction")),
            "conv_weight": (weights, "fp16", ("out_channel", "reduction")),
            "conv_result": (result, "fp32", ("row", "out_channel")),
        }
        if changed:
            role, index, value = changed
            values, dtype, labels = tensors[role]
            values = values.copy()
            values[index] = value
            tensors[role] = values, dtype, labels
        if axes:
            role, labels = axes
            values, dtype, _ = tensors[role]
            tensors[role] = values, dtype, labels
        if omit:
            del tensors[omit]
        return write_bundle(self.root / name, operation="conv_patch_projection",
                            provider=name, contract="synthetic_conv", runtime=self.runtime,
                            context={"capture_kind": "observed_graph_edges"}, tensors=tensors)

    def test_identical_operands_isolate_local_output_failure(self):
        cke = self.packet("cke")
        oracle = self.packet("oracle", changed=("conv_result", (4, 2), 999.0))
        report = compare_packets(cke, oracle, 0.0)
        self.assertEqual(report["operand_construction"], "PASS")
        self.assertTrue(report["identical_input_replay_ready"])
        self.assertEqual(report["first_proven_local_failure"], "conv_result")
        self.assertEqual(report["gates"]["conv_result"]["worst_index"], [4, 2])

    def test_input_or_weight_change_blocks_local_attribution(self):
        for changed in (("conv_im2col", (0, 0, 0, 1), 123.0),
                        ("conv_weight", (1, 3), 123.0),
                        ("conv_image", (0, 1, 1, 2), 123.0)):
            with self.subTest(changed=changed[0]):
                cke = self.packet("cke_" + changed[0])
                oracle = self.packet("oracle_" + changed[0], changed=changed)
                report = compare_packets(cke, oracle, 0.0)
                self.assertEqual(report["first_observed_failure"], changed[0])
                self.assertIsNone(report["first_proven_local_failure"])
                self.assertEqual(report["local_arithmetic"], "NOT_ISOLATED")

    def test_rounding_near_miss_is_not_identical_input(self):
        cke = self.packet("cke")
        oracle = self.packet("oracle", changed=("conv_im2col", (0, 0, 0, 1), 1.0009765625))
        report = compare_packets(cke, oracle, 0.01)
        self.assertEqual(report["operand_construction"], "PASS")
        self.assertFalse(report["identical_input_replay_ready"])
        self.assertEqual(report["local_arithmetic"], "NOT_ISOLATED")

    def test_missing_wrong_layout_and_stale_bytes_fail_closed(self):
        cke = self.packet("cke")
        missing = self.packet("missing", omit="conv_im2col")
        missing_report = compare_packets(cke, missing, 0)
        self.assertEqual(missing_report["gates"]["conv_im2col"]["status"], "UNAVAILABLE")
        self.assertEqual(missing_report["status"], "INCOMPLETE")
        wrong_axes = self.packet("axes", axes=("conv_im2col", ("batch", "out_width", "out_height", "reduction")))
        axes_report = compare_packets(cke, wrong_axes, 0)
        self.assertEqual(axes_report["gates"]["conv_im2col"]["status"], "ALIGNMENT_UNRESOLVED")
        self.assertEqual(axes_report["status"], "INCOMPLETE")
        tampered = self.packet("tampered")
        with (tampered.parent / "conv_weight.bin").open("ab") as output:
            output.write(b"x")
        with self.assertRaisesRegex(BundleError, "tensor bytes changed"):
            compare_packets(cke, tampered, 0)

    def test_synthetic_ggml_graph_edge_capture(self):
        build = Path(os.environ.get("CK_TEST_LLAMA_BUILD", str(ROOT / "llama.cpp" / "build")))
        lib_dir = build / "bin"
        if not (lib_dir / "libmtmd.so").is_file():
            self.skipTest("pinned llama.cpp mtmd build unavailable")
        binary = self.root / "capture"
        source = ROOT / "tests" / "fixtures" / "mtmd_conv_operand_capture.cpp"
        cmd = ["g++", "-std=c++17", "-O1",
               "-DCK_MTMD_CLIP_OBJECT_API=1", "-DCK_MTMD_CLIP_VALUE_API=1",
               "-DCK_MTMD_IMAGE_PREPROC_API=1",
               f"-I{ROOT / 'llama.cpp/tools/mtmd'}",
               f"-I{ROOT / 'llama.cpp/ggml/include'}",
               f"-I{ROOT / 'llama.cpp/include'}", str(source),
               f"-L{lib_dir}", "-lmtmd", "-lggml-cpu", "-lggml-base",
               f"-Wl,-rpath,{lib_dir}", "-o", str(binary)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        capture = self.root / "capture_files"
        capture.mkdir()
        execution = subprocess.run([str(binary), str(capture)], capture_output=True, text=True)
        self.assertEqual(execution.returncode, 0, execution.stderr)
        tensors, origins = load_oracle_capture(capture, 1)
        self.assertEqual(tensors["conv_image"][0].shape, (1, 2, 3, 5))
        self.assertEqual(tensors["conv_im2col"][0].shape, (1, 2, 3, 8))
        self.assertEqual(tensors["conv_result"][0].shape, (6, 3))
        self.assertEqual(origins["conv_im2col"]["producer_op"], "IM2COL")
        self.assertEqual(origins["conv_result"]["canonical_transform"],
                         "squeeze_outer_singletons_then_transpose_to_row_out_channel")
        self.assertEqual(tensors["conv_image"][0][0, 0, 0, 0], np.float32(100.0 / 7.0))
        np.testing.assert_array_equal(
            tensors["conv_im2col"][0][0, 0, 0],
            np.array([0, 100 / 7, 0, 105 / 7, 0, 115 / 7, 0, 120 / 7], dtype=np.float16),
        )
        oracle_packet = import_oracle_capture(capture, 1, self.runtime,
                                              self.root / "oracle_packet", "synthetic_conv")
        cke_packet = write_bundle(
            self.root / "cke_packet", operation="conv_patch_projection", provider="synthetic_cke",
            contract="synthetic_conv", runtime=self.runtime,
            context={"capture_kind": "observed_graph_edges"}, tensors=tensors,
        )
        self.assertEqual(compare_packets(cke_packet, oracle_packet, 0)["status"], "PASS")
        report_path = self.root / "report.json"
        command = [sys.executable, str(ROOT / "version/v8/scripts/xray_conv_operands_v8.py"),
                   "--cke-bundle", str(cke_packet), "--oracle-capture", str(capture),
                   "--oracle-runtime", str(self.runtime), "--capture-id", "1",
                   "--oracle-bundle-dir", str(self.root / "oracle_cli_packet"),
                   "--report", str(report_path), "--max-abs", "0", "--contract", "synthetic_conv"]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(report_path.read_text())["evidence_scope"], "packet_numerics_only")
        image_raw = capture / "conv_1_conv_image.raw"
        original_image = image_raw.read_bytes()
        modified_image = bytearray(original_image)
        modified_image[:4] = struct.pack("<f", -123.0)
        image_raw.write_bytes(modified_image)
        try:
            with self.assertRaisesRegex(BundleError, "captured image and im2col disagree"):
                load_oracle_capture(capture, 1)
        finally:
            image_raw.write_bytes(original_image)
        identity = {
            "selected_capture_id": 1, "processed_width": 5, "processed_height": 3,
            "loaded_mtmd_path": str(self.runtime), "loaded_mtmd_sha256": sha256(self.runtime),
        }
        for role in ("image", "projector", "shim"):
            identity[f"{role}_path"] = str(self.runtime)
            identity[f"{role}_sha256"] = sha256(self.runtime)
        (capture / "capture-identity.json").write_text(json.dumps(identity))
        import_oracle_capture(capture, 1, self.runtime, self.root / "bound_packet", "synthetic_conv")
        identity["image_sha256"] = "0" * 64
        (capture / "capture-identity.json").write_text(json.dumps(identity))
        with self.assertRaisesRegex(BundleError, "image artifact changed"):
            import_oracle_capture(capture, 1, self.runtime, self.root / "stale_packet", "synthetic_conv")
        failed_command = command.copy()
        failed_command[failed_command.index("--oracle-bundle-dir") + 1] = str(self.root / "stale_cli_packet")
        failed_command[failed_command.index("--report") + 1] = str(self.root / "failed_report.json")
        failed = subprocess.run(failed_command, capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1)
        self.assertEqual(json.loads((self.root / "failed_report.json").read_text())["status"],
                         "INCOMPLETE")
        rows = [json.loads(line) for line in (capture / "meta.jsonl").read_text().splitlines()]
        self.assertEqual({row["capture_id"] for row in rows}, {1})
        self.assertEqual([row["role"] for row in rows if row["role"].startswith("conv_layout_")],
                         ["conv_layout_0"])
        self.assertEqual(rows[-1]["op"], "CONT")
        rows[1]["status"] = "unavailable"
        (capture / "meta.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(BundleError, "unavailable"):
            load_oracle_capture(capture, 1)
        rows[1]["status"] = "observed"
        rows[1]["convolution"]["stride"] = [0, 1]
        (capture / "meta.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(BundleError, "invalid convolution stride"):
            load_oracle_capture(capture, 1)
        rows[1]["convolution"]["stride"] = [2, 1]
        (capture / "meta.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        raw = capture / rows[1]["raw_file"]
        raw.write_bytes(raw.read_bytes()[:-2])
        with self.assertRaisesRegex(BundleError, "truncated"):
            load_oracle_capture(capture, 1)


if __name__ == "__main__":
    unittest.main()
