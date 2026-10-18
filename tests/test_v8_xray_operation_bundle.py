"""Operation replay packets reject stale or semantically inconsistent evidence."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from version.v8.scripts.xray_operation_bundle_v8 import BundleError, load_bundle, write_bundle


class OperationBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = self.root / "libtest.so"
        self.runtime.write_bytes(b"fixed test runtime")

    def tearDown(self):
        self.temp.cleanup()

    def bundle(self):
        return write_bundle(
            self.root / "bundle", operation="projection", provider="test_provider",
            contract="bf16_storage_fp32_compute", runtime=self.runtime,
            context={"layer": 0, "valid_rows": 2},
            tensors={
                "input": (np.array([0x3f80, 0x4000], dtype=np.uint16),
                          "bf16_bits", ("reduction",)),
                "output": (np.array([[1.0], [2.0]], dtype=np.float32),
                           "fp32", ("row", "channel")),
            },
        )

    def test_round_trip_and_byte_tampering(self):
        bundle = self.bundle()
        manifest, tensors = load_bundle(bundle)
        self.assertEqual(manifest["provider"], "test_provider")
        np.testing.assert_array_equal(tensors["input"], [0x3f80, 0x4000])
        path = bundle.parent / "input.bin"
        changed = bytearray(path.read_bytes())
        changed[0] ^= 1
        path.write_bytes(changed)
        with self.assertRaisesRegex(BundleError, "tensor bytes changed"):
            load_bundle(bundle)

    def test_stale_runtime_and_invalid_geometry(self):
        bundle = self.bundle()
        self.runtime.write_bytes(b"different runtime")
        with self.assertRaisesRegex(BundleError, "runtime library identity changed"):
            load_bundle(bundle)
        self.runtime.write_bytes(b"fixed test runtime")
        manifest = json.loads(bundle.read_text())
        manifest["tensors"]["output"]["shape"] = [3, 1]
        bundle.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(BundleError, "inconsistent byte extent"):
            load_bundle(bundle)

    def test_no_implicit_storage_cast_or_overwrite(self):
        with self.assertRaisesRegex(BundleError, "storage dtype"):
            write_bundle(
                self.root / "bad", operation="projection", provider="test_provider",
                contract="fp16", runtime=self.runtime, context={},
                tensors={"input": (np.array([1.0], dtype=np.float32),
                                   "fp16", ("channel",))},
            )
        bundle = self.bundle()
        with self.assertRaisesRegex(BundleError, "must be empty"):
            self.bundle()
        self.assertTrue(bundle.exists())


if __name__ == "__main__":
    unittest.main()
