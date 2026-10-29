"""Checked repetition penalty through a kernel and normal generated execution."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_prompt_first_block import compile_entry, view

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/kernels/repetition_penalty_checked.c"
SELECT_SOURCE = "src/kernels/select_next_id_checked.c"


def synthetic_graph() -> dict:
    shapes = {"scores": [2, 5], "history": [3], "adjusted": [5],
              "selected": [1]}
    return {
        "version": 3, "name": "repetition_penalty_synthetic",
        "family": "bounded_graph", "checked_native_entry": True,
        "contract": {"runtime_invariants": {"inference_only": True}},
        "activation_buffers": {name: {"shape": shape}
                               for name, shape in shapes.items()},
        "activation_bindings": {name: name for name in shapes},
        "native_entry": {"function": "ck_repetition_penalty_synthetic",
                         "params": [{"c_type": "uint8_t *", "name": "arena"},
                                    {"c_type": "size_t", "name": "arena_bytes"}],
                         "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
        "sequence": ["adjust"],
        "block_types": {"adjust": {
            "sequence": ["header", "body", "footer"], "header": [],
            "body": {"type": "dense", "ops": [{
                "id": "penalize", "op": "repetition_penalty_last_row_checked",
                "kernel": "repetition_penalty_last_row_checked_f32",
                "returns_status": True,
                "graph_slots": {"inputs": {"values": "external:scores",
                                           "history": "external:history"},
                                "outputs": {"output": "adjusted"}},
                "params": {"R": 2, "C": 5, "H": 3, "call_constants": {
                    "values_elements": 10, "rows": 2, "channels": 5,
                    "row_stride": 5, "history_elements": 3,
                    "history_count": 3, "penalty_numerator": 105,
                    "penalty_denominator": 100, "output_elements": 5}},
            }, {
                "id": "select_adjusted", "op": "select_next_id_checked",
                "kernel": "select_next_id_checked_f32", "returns_status": True,
                "graph_slots": {"inputs": {"values": "adjusted"},
                                "outputs": {"index": "selected"}},
                "params": {"R": 1, "C": 5, "call_constants": {
                    "values_elements": 5, "rows": 1, "channels": 5,
                    "row_stride": 5, "emitted_frames": 1,
                    "min_frames": 2, "max_frames": 3, "stop_id": 2,
                    "suppressed_begin": 4, "suppressed_end": 5,
                    "selected_elements": 1}},
            }]}, "footer": []}},
        "runtime_constants": {}, "runtime_lengths": {},
    }


class RepetitionPenaltyCheckedTest(unittest.TestCase):
    def test_direct_sign_duplicate_and_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "penalty.so"
            subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                            "-I", str(ROOT / "include"),
                            str(ROOT / SOURCE), "-o", str(library)],
                           check=True, capture_output=True)
            fn = ctypes.CDLL(str(library)).ck_repetition_penalty_last_row_checked_f32
            pfloat = ctypes.POINTER(ctypes.c_float)
            pint = ctypes.POINTER(ctypes.c_int32)
            fn.argtypes = [pfloat, *([ctypes.c_size_t] * 4), pint,
                           *([ctypes.c_size_t] * 4), pfloat, ctypes.c_size_t]
            fn.restype = ctypes.c_int
            scores = np.array([[9, 8, 7, 6, 5],
                               [1.05, -1.05, 0, 2, -2]], dtype=np.float32)
            history = np.array([0, 1, 0], dtype=np.int32)
            output = np.full(5, -719, dtype=np.float32)

            def run(rows=2, channels=5, stride=5, n=105, d=100,
                    count=3, capacity=10):
                return fn(scores.ctypes.data_as(pfloat), capacity, rows,
                          channels, stride, history.ctypes.data_as(pint), 3,
                          count, n, d, output.ctypes.data_as(pfloat), 5)

            self.assertEqual(run(), 0)
            np.testing.assert_array_equal(
                output, np.array([1.0, np.float32(-1.05) * np.float32(1.05),
                                  0, 2, -2], dtype=np.float32))
            for change in ({"rows": 0}, {"stride": 4}, {"capacity": 9},
                           {"count": 4}, {"n": 0}, {"d": 0},
                           {"n": 2**32}):
                with self.subTest(change=change):
                    output.fill(-719)
                    self.assertLess(run(**change), 0)
                    np.testing.assert_array_equal(output, [-719] * 5)
            history[2] = 5
            self.assertLess(run(), 0)
            np.testing.assert_array_equal(output, [-719] * 5)
            history[2] = 0
            scores[1, 2] = np.nan
            self.assertLess(run(), 0)
            np.testing.assert_array_equal(output, [-719] * 5)
            scores[1, 2] = 0
            self.assertLess(fn(scores.ctypes.data_as(pfloat), 10, 2, 5, 5,
                               history.ctypes.data_as(pint), 3, 3, 105, 100,
                               scores.ctypes.data_as(pfloat), 5), 0)
            np.testing.assert_array_equal(scores[1],
                                          np.array([1.05, -1.05, 0, 2, -2],
                                                   dtype=np.float32))
            scores[1, 1] = -3.4e38
            self.assertLess(run(n=2, d=1), 0)
            np.testing.assert_array_equal(output, [-719] * 5)
            scores[1, 1] = -1.05
            self.assertEqual(run(), 0)  # recovery on the same buffers

    def test_generated_policy_and_failure_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            fn, acts, _, layout, calls, _ = compile_entry(
                Path(temp), synthetic_graph(), {},
                extra_sources=(SOURCE, SELECT_SOURCE))
            self.assertEqual([op["function"] for op in calls["operations"]],
                             ["ck_repetition_penalty_last_row_checked_f32",
                              "ck_select_next_id_checked_f32"])
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            scores = view(arena, acts, "scores", np.float32, (2, 5))
            history = view(arena, acts, "history", np.int32, (3,))
            adjusted = view(arena, acts, "adjusted", np.float32, (5,))
            selected = view(arena, acts, "selected", np.int32, (1,))
            scores[:] = [[5, 4, 3, 2, 1], [1.05, -1.05, 0, 1.04, -2]]
            history[:] = [0, 1, 0]
            adjusted.fill(-719)
            selected[0] = -719
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(
                adjusted, [1.0, np.float32(-1.05) * np.float32(1.05),
                           0, np.float32(1.04), -2])
            self.assertEqual(int(selected[0]), 3)  # raw maximum was seen ID 0
            history[2] = -1
            adjusted.fill(-719)
            selected[0] = -719
            self.assertNotEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(adjusted, [-719] * 5)
            self.assertEqual(int(selected[0]), -719)


if __name__ == "__main__":
    unittest.main()
