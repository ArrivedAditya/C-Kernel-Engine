"""Independent PyTorch evidence and bounds for duration-logit reduction."""

import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tts/duration_logits_kokoro_torch.json"
FLOAT = ctypes.c_float
INT = ctypes.c_int32
SIZE = ctypes.c_size_t


def compile_kernel(directory):
    library = Path(directory) / "libduration_logits.so"
    subprocess.run([
        "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
        "-shared", "-fPIC", "-I", str(ROOT / "include"),
        str(ROOT / "src/kernels/audio_duration_logits.c"), "-lm", "-o", str(library),
    ], check=True)
    fn = ctypes.CDLL(str(library)).audio_duration_logits_to_frames_f32
    fn.argtypes = [ctypes.POINTER(FLOAT), SIZE, SIZE, SIZE, SIZE, FLOAT,
                   ctypes.POINTER(INT), SIZE, SIZE, SIZE, ctypes.POINTER(INT)]
    fn.restype = ctypes.c_int
    return fn


def invoke(fn, values, tokens, bins, stride=None, speed=1.0, input_elements=None,
           duration_capacity=None, max_duration_per_token=50,
           max_expanded_frames=4096):
    stride = stride if stride is not None else bins
    source = (FLOAT * len(values))(*values)
    output = (INT * max(tokens, 1))(*([-777] * max(tokens, 1)))
    valid = INT(987654)
    status = fn(source, len(values) if input_elements is None else input_elements,
                tokens, bins, stride, speed, output,
                tokens if duration_capacity is None else duration_capacity,
                max_duration_per_token, max_expanded_frames, ctypes.byref(valid))
    return status, list(output), valid.value


class AudioDurationLogitsOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.fn = compile_kernel(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pinned_production_shape_against_committed_torch_fixture(self):
        fixture = json.loads(FIXTURE.read_text())
        capture = json.loads((ROOT / fixture["source_manifest"]).read_text())
        self.assertEqual(capture["tensors"][fixture["source_tensor"]]["sha256"],
                         fixture["source_npy_sha256"])
        tokens, bins = fixture["shape"]
        values = [item for row in fixture["logits"] for item in row]
        self.assertEqual(len(values), tokens * bins)
        self.assertEqual(hashlib.sha256(b"".join(struct.pack("<f", value)
                                                  for value in values)).hexdigest(),
                         fixture["source_payload_sha256"])
        for _ in range(2):
            status, output, valid = invoke(self.fn, values, tokens, bins,
                                           speed=fixture["speed"])
            self.assertEqual(status, 0)
            self.assertEqual(output, fixture["durations"])
            self.assertEqual(valid, fixture["expected_frames"])
        self.assertEqual(valid, 103)
        self.assertEqual(invoke(self.fn, values, tokens, bins,
                                max_expanded_frames=103)[0], 0)
        status, output, extent = invoke(self.fn, values, tokens, bins,
                                        max_expanded_frames=102)
        self.assertEqual(status, -2)
        self.assertEqual(output, [-777] * tokens)
        self.assertEqual(extent, 987654)

    def test_layout_ties_and_clamping(self):
        # Two zero logits yield a sum of one. At speed two, 0.5 rounds to even
        # zero and then clamps to one. At speed 0.4, 2.5 rounds to even two.
        values = [0.0, 0.0, math.nan, 0.0, 0.0]
        self.assertEqual(invoke(self.fn, values, 2, 2, stride=3,
                                speed=2.0)[:], (0, [1, 1], 2))
        self.assertEqual(invoke(self.fn, values, 2, 2, stride=3,
                                speed=0.4)[:], (0, [2, 2], 4))
        # Four zero logits / 0.8 = 2.5, still even two.
        self.assertEqual(invoke(self.fn, [0.0] * 4, 1, 4,
                                speed=0.8)[:], (0, [2], 2))

    def test_rejections_leave_outputs_unchanged(self):
        valid = [0.0] * 6
        cases = [
            dict(tokens=0, bins=2),
            dict(tokens=2, bins=0),
            dict(tokens=2, bins=2, stride=1),
            dict(tokens=2, bins=2, input_elements=3),
            dict(tokens=2, bins=2, duration_capacity=1),
            dict(tokens=2, bins=2, speed=0),
            dict(tokens=2, bins=2, speed=math.inf),
            dict(tokens=2, bins=2, max_duration_per_token=0),
            dict(tokens=2, bins=2, max_duration_per_token=2**31),
            dict(tokens=2, bins=2, max_expanded_frames=1),
        ]
        for case in cases:
            with self.subTest(case=case):
                status, output, extent = invoke(self.fn, valid, **case)
                self.assertNotEqual(status, 0)
                self.assertEqual(output, [-777] * max(case["tokens"], 1))
                self.assertEqual(extent, 987654)
        for bad in (math.nan, math.inf, -math.inf):
            status, output, extent = invoke(self.fn, [0.0, 0.0, bad, 0.0], 2, 2)
            self.assertEqual(status, -1)
            self.assertEqual(output, [-777, -777])
            self.assertEqual(extent, 987654)
        status, output, extent = invoke(self.fn, [10.0] * 4, 2, 2,
                                        max_duration_per_token=1)
        self.assertEqual(status, -2)
        self.assertEqual(output, [-777, -777])
        self.assertEqual(extent, 987654)

    def test_hostile_geometry_rejected_before_buffer_access(self):
        source = (FLOAT * 1)(0.0)
        output = (INT * 1)(-777)
        extent = INT(987654)
        status = self.fn(source, 1, 2, 2, SIZE(-1).value, 1.0,
                         output, 1, 50, 100, ctypes.byref(extent))
        self.assertEqual(status, -3)
        self.assertEqual(output[0], -777)
        self.assertEqual(extent.value, 987654)

if __name__ == "__main__":
    unittest.main()
