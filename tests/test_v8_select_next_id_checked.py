"""Checked greedy selection with minimum-frame EOS and budget statuses."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_prompt_first_block import compile_entry, view

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("src/kernels/select_next_id_checked.c",
           "src/kernels/concat_i32_checked.c")


def synthetic_graph(emitted_frames: int) -> dict:
    shapes = {"scores": [1, 4], "selected": [1], "suffix": [1],
              "combined": [2]}
    return {
        "version": 3, "name": "select_next_id_synthetic", "family": "bounded_graph",
        "checked_native_entry": True,
        "contract": {"runtime_invariants": {"inference_only": True}},
        "activation_buffers": {name: {"shape": shape} for name, shape in shapes.items()},
        "activation_bindings": {name: name for name in shapes},
        "native_entry": {"function": "ck_select_next_id_synthetic", "params": [
            {"c_type": "uint8_t *", "name": "arena"},
            {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
        "sequence": ["select_then_copy"],
        "block_types": {"select_then_copy": {
            "sequence": ["header", "body", "footer"], "header": [],
            "body": {"type": "dense", "ops": [
                {"id": "select", "op": "select_next_id_checked",
                 "kernel": "select_next_id_checked_f32", "returns_status": True,
                 "graph_slots": {"inputs": {"values": "external:scores"},
                                 "outputs": {"index": "selected"}},
                 "params": {"R": 1, "C": 4, "call_constants": {
                     "values_elements": 4, "rows": 1, "channels": 4,
                     "row_stride": 4, "emitted_frames": emitted_frames,
                     "min_frames": 2, "max_frames": 3, "stop_id": 2,
                     "suppressed_begin": 3, "suppressed_end": 4,
                     "selected_elements": 1}}},
                {"id": "dependent_copy", "op": "concat_i32_checked",
                 "kernel": "concat_i32_checked", "returns_status": True,
                 "graph_slots": {"inputs": {"left": "selected",
                                            "right": "external:suffix"},
                                 "outputs": {"output": "combined"}},
                 "params": {"L": 1, "R": 1, "O": 2, "call_constants": {
                     "left_elements": 1, "left_count": 1,
                     "right_elements": 1, "right_count": 1,
                     "output_elements": 2}}},
            ]}, "footer": []}},
        "runtime_constants": {}, "runtime_lengths": {},
    }


class SelectNextIdCheckedTest(unittest.TestCase):
    def test_direct_policy_and_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "select.so"
            subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                            "-I", str(ROOT / "include"),
                            str(ROOT / SOURCES[0]), "-o", str(library)],
                           check=True, capture_output=True)
            fn = ctypes.CDLL(str(library)).ck_select_next_id_checked_f32
            pfloat = ctypes.POINTER(ctypes.c_float)
            pint = ctypes.POINTER(ctypes.c_int32)
            fn.argtypes = [pfloat, *([ctypes.c_size_t] * 10), pint,
                           ctypes.c_size_t]
            fn.restype = ctypes.c_int
            scores = np.array([[0.1, 0.9, 2.0, -4.0]], dtype=np.float32)
            selected = np.array([-719], dtype=np.int32)

            def run(emitted=1, minimum=2, maximum=3, stop=2,
                    suppressed_begin=3, suppressed_end=4,
                    rows=1, channels=4, stride=4, capacity=4):
                return fn(scores.ctypes.data_as(pfloat), capacity, rows,
                          channels, stride, emitted, minimum, maximum,
                          stop, suppressed_begin, suppressed_end,
                          selected.ctypes.data_as(pint), 1)

            self.assertEqual(run(), 0)
            self.assertEqual(int(selected[0]), 1)
            selected[0] = -719
            self.assertEqual(run(emitted=2), 1)
            self.assertEqual(int(selected[0]), 2)
            selected[0] = -719
            self.assertEqual(run(emitted=3), 2)
            self.assertEqual(int(selected[0]), -719)
            for change in ({"minimum": 4}, {"stop": 4},
                           {"suppressed_begin": 4, "suppressed_end": 3},
                           {"suppressed_end": 5},
                           {"capacity": 3}, {"rows": 0},
                           {"stride": 3}):
                with self.subTest(change=change):
                    selected[0] = -719
                    self.assertLess(run(**change), 0)
                    self.assertEqual(int(selected[0]), -719)
            scores[0, 0] = np.nan
            self.assertLess(run(), 0)
            self.assertEqual(int(selected[0]), -719)
            self.assertLess(fn(None, 4, 1, 4, 4, 1, 2, 3, 2, 3, 4,
                               selected.ctypes.data_as(pint), 1), 0)
            self.assertEqual(int(selected[0]), -719)
            self.assertLess(fn(scores.ctypes.data_as(pfloat), 4, 1, 4, 4,
                               1, 2, 3, 2, 3, 4,
                               scores.ctypes.data_as(pint), 1), 0)
            self.assertTrue(np.isnan(scores[0, 0]))
            scores[0, 0] = 0.9
            scores[0, 1] = 0.9
            scores[0, 2] = -1.0
            scores[0, 3] = 10.0  # suppressed even though it is the maximum
            self.assertEqual(run(), 0)
            self.assertEqual(int(selected[0]), 0)  # first equal maximum
            self.assertEqual(run(emitted=2, stop=3), 1)
            self.assertEqual(int(selected[0]), 3)  # stop is exempt after min

    def test_generated_control_status_skips_dependents(self):
        for emitted, expected_status, expected_id in ((1, 0, 1),
                                                       (2, 1, 2),
                                                       (3, 2, -719)):
            with self.subTest(emitted=emitted), tempfile.TemporaryDirectory() as temp:
                fn, acts, _, layout, calls, _ = compile_entry(
                    Path(temp), synthetic_graph(emitted), {},
                    extra_sources=SOURCES)
                self.assertEqual([op["function"] for op in calls["operations"]],
                                 ["ck_select_next_id_checked_f32",
                                  "ck_concat_i32_checked"])
                size = layout["memory"]["arena"]["total_size"]
                raw = (ctypes.c_uint8 * (size + 63))()
                arena = (ctypes.c_uint8 * size).from_buffer(
                    raw, (-ctypes.addressof(raw)) & 63)
                view(arena, acts, "scores", np.float32, (1, 4))[:] = [[0.1, 0.9, 2.0, -4.0]]
                view(arena, acts, "suffix", np.int32, (1,))[0] = 17
                selected = view(arena, acts, "selected", np.int32, (1,))
                combined = view(arena, acts, "combined", np.int32, (2,))
                selected[0], combined[:] = -719, -907
                self.assertEqual(fn(arena, size), expected_status)
                self.assertEqual(int(selected[0]), expected_id)
                np.testing.assert_array_equal(
                    combined, [expected_id, 17] if expected_status == 0 else
                    [-907, -907])


if __name__ == "__main__":
    unittest.main()
