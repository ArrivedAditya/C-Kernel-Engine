"""Live PyTorch comparison for the checked duration-logit producer."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ImportError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
    if __name__ == "__main__":
        print(f"TEST SKIPPED: live PyTorch oracle dependency unavailable: {exc}")
        raise SystemExit(0)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_v8_audio_duration_logits_oracle import FIXTURE, compile_kernel, invoke


class AudioDurationLogitsLiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest(
                f"live PyTorch oracle dependency unavailable: {TORCH_IMPORT_ERROR}")
        cls.temp = tempfile.TemporaryDirectory()
        cls.fn = compile_kernel(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pinned_and_varied_shapes_against_torch(self):
        fixture = json.loads(FIXTURE.read_text())
        logits = torch.tensor(fixture["logits"], dtype=torch.float32)
        expected = torch.round(torch.sigmoid(logits).sum(dim=-1) /
                               fixture["speed"]).clamp(min=1).to(torch.int32)
        self.assertEqual(expected.tolist(), fixture["durations"])
        values = [value for row in fixture["logits"] for value in row]
        status, output, valid = invoke(self.fn, values, *fixture["shape"],
                                       speed=fixture["speed"])
        self.assertEqual((status, output, valid),
                         (0, expected.tolist(), fixture["expected_frames"]))

        for tokens, bins, speed in ((1, 1, 1.0), (3, 7, 0.75), (8, 51, 1.25)):
            logits = torch.linspace(-7.0, 5.0, steps=tokens * bins,
                                    dtype=torch.float32).reshape(tokens, bins)
            expected = torch.round(torch.sigmoid(logits).sum(dim=-1) /
                                   speed).clamp(min=1).to(torch.int32).tolist()
            stride = bins + 3
            values = [value for row in logits.tolist()
                      for value in row + [math.nan] * 3]
            status, output, valid = invoke(self.fn, values, tokens, bins,
                                           stride=stride, speed=speed,
                                           max_duration_per_token=100)
            self.assertEqual((status, output, valid), (0, expected, sum(expected)))


if __name__ == "__main__":
    unittest.main()
