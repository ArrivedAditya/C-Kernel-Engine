#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v8" / "scripts"))
import parity_test_v8 as parity
import xray_numerical_parity_v8 as xray


class AttentionParityCaptureTests(unittest.TestCase):
    def test_strided_head_major_dump_is_complete_and_token_major(self) -> None:
        source = r'''
#include "ck_parity_dump.h"

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    float values[2 * 3 * 4];
    for (int h = 0; h < 2; ++h) {
        for (int t = 0; t < 3; ++t) {
            for (int d = 0; d < 4; ++d) {
                values[(h * 3 + t) * 4 + d] = d < 2 ? (float)(h * 100 + t * 10 + d) : -99.0f;
            }
        }
    }
    ck_dump_init(argv[1]);
    ck_dump_tensor_head_major_token_major_strided(values, 0, "kqv_out", 2, 3, 2, 4);
    ck_dump_close();
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="v8_attn_dump_") as tmp:
            directory = Path(tmp)
            c_path = directory / "capture.c"
            executable = directory / "capture"
            c_path.write_text(source, encoding="ascii")
            subprocess.run(
                ["cc", "-std=c11", "-DCK_PARITY_DUMP", "-I", str(ROOT / "version" / "v8" / "src"),
                 str(c_path), "-o", str(executable)],
                check=True, capture_output=True, text=True,
            )
            env = {key: value for key, value in os.environ.items() if not key.startswith("CK_PARITY_")}
            subprocess.run([str(executable), str(directory)], check=True, env=env, capture_output=True)
            dumps = parity.read_dump_file(directory / "dump.bin")
            self.assertEqual(len(dumps), 1)
            self.assertEqual(dumps[0].op_name, "kqv_out")
            mapping = xray.load_json(
                ROOT / "version" / "v8" / "parity_profiles" / "qwen3vl_llamacpp_q8_v1.json"
            )["backend_mappings"]["vision.layer.{layer}.attention.output"]
            self.assertEqual(mapping["capture_tensor"], dumps[0].op_name)
            self.assertEqual(mapping["logical_layout"], "token_major")

            tensor_path = directory / "attention.f32"
            dumps[0].data.tofile(tensor_path)
            entry = {
                "checkpoint_id": "vision.layer.0.attention.output",
                "tensor_path": str(tensor_path),
                "exported_dtype": "fp32",
                "physical_shape": [3, 2, 2],
                "physical_axis_names": mapping["axis_names"],
                "axis_names": ["token", "head", "channel"],
                "logical_shape": [3, 2, 2],
            }
            expected = np.asarray([
                [[0, 1], [100, 101]],
                [[10, 11], [110, 111]],
                [[20, 21], [120, 121]],
            ], dtype=np.float32)
            np.testing.assert_array_equal(xray._load_tensor(entry), expected)

            wrong_layout = {
                **entry,
                "physical_shape": [2, 3, 2],
                "physical_axis_names": ["head", "token", "channel"],
            }
            self.assertFalse(np.array_equal(xray._load_tensor(wrong_layout), expected))

    def test_unequal_extents_cannot_pass_on_matching_prefix(self) -> None:
        reference = parity.ParityDump(0, "kqv_out", np.asarray([1, 2, 3], dtype=np.float32), 0, "fp32")
        subject = parity.ParityDump(0, "kqv_out", np.asarray([1, 2], dtype=np.float32), 0, "fp32")
        result = parity.compare_dumps(reference, subject)
        self.assertEqual(result["status"], "ERROR")
        self.assertTrue(result["size_mismatch"])
        self.assertEqual(result["ref_shape"], [3])
        self.assertEqual(result["test_shape"], [2])


if __name__ == "__main__":
    unittest.main()
