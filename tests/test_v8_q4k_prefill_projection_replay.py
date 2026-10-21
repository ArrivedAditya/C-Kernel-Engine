import tempfile
import unittest
from pathlib import Path

import numpy as np

from version.v8.scripts.replay_q4k_prefill_projection_v8 import _metrics, _offset, _rows


class Q4KPrefillProjectionReplayTests(unittest.TestCase):
    def test_explicit_row_slice_preserves_asymmetric_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.f32"
            values = np.arange(35, dtype=np.float32).reshape(7, 5)
            values.tofile(path)
            np.testing.assert_array_equal(_rows(path, 3, 5, 2), values[2:5])
            with self.assertRaisesRegex(ValueError, "cannot provide"):
                _rows(path, 3, 5, 5)
            with self.assertRaisesRegex(ValueError, "cannot provide"):
                _rows(path, 3, 5, -1)
            with self.assertRaisesRegex(ValueError, "cannot provide"):
                _rows(path, 3, 6, 0)

    def test_nonfinite_and_truncated_inputs_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.f32"
            np.array([1, np.nan, 3, 4], dtype=np.float32).tofile(path)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                _rows(path, 2, 2)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "cannot provide"):
                _rows(path, 2, 2)

    def test_weight_symbol_must_be_unique_and_declared(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "generated.c"
            source.write_text("#define W_LAYER_0_WO 123\n", encoding="ascii")
            self.assertEqual(_offset(source, "W_LAYER_0_WO"), 123)
            with self.assertRaisesRegex(ValueError, "generated layer weight"):
                _offset(source, "W_LAYER_0_WO; exit")
            with self.assertRaisesRegex(ValueError, "expected one"):
                _offset(source, "W_LAYER_1_WO")
            source.write_text(source.read_text() * 2, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "expected one"):
                _offset(source, "W_LAYER_0_WO")

    def test_metrics_reject_mismatched_shape(self):
        actual = np.array([[1, 2], [3, 4]], dtype=np.float32)
        expected = np.array([[1, 2], [3, 5]], dtype=np.float32)
        result = _metrics(actual, expected)
        self.assertEqual((result["worst_row"], result["worst_col"]), (1, 1))
        self.assertEqual(result["max_abs"], 1)
        with self.assertRaisesRegex(ValueError, "equal finite"):
            _metrics(actual, expected.ravel())


if __name__ == "__main__":
    unittest.main()
