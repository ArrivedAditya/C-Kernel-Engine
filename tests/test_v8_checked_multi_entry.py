"""One planned arena shared by two normally lowered checked entry functions."""

from __future__ import annotations

import copy
import ctypes
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_prompt_first_block import compile_entry, view

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("src/kernels/select_next_id_checked.c",
           "src/kernels/concat_i32_checked.c")


def circuit() -> dict:
    shapes = {"scores": [1, 4], "selected": [1], "valid": [1],
              "ready_scalar": [1],
              "suffix": [1], "combined": [2]}
    return {
        "version": 3, "name": "checked_multi_entry_synthetic",
        "family": "bounded_graph", "checked_native_entry": True,
        "contract": {"runtime_invariants": {"inference_only": True}},
        "activation_buffers": {name: {"shape": shape}
                               for name, shape in shapes.items()},
        "activation_bindings": {name: name for name in shapes},
        "native_entry": {
            "function": "ck_synthetic_prefill",
            "params": [{"c_type": "uint8_t *", "name": "arena"},
                       {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"},
            "segments": [
                {"function": "ck_synthetic_prefill", "last_op": "publish_valid"},
                {"function": "ck_synthetic_step", "last_op": "join_state"},
            ],
        },
        "sequence": ["prefill", "step"],
        "block_types": {
            "prefill": {"sequence": ["header", "body", "footer"],
                        "header": [{
                            "id": "clear_valid", "op": "runtime_extent_affine",
                            "kernel": "runtime_extent_affine_i32",
                            "returns_status": True,
                            "params": {"call_constants": {
                                "extent_affine_source": 0,
                                "extent_affine_factor": 1,
                                "extent_affine_offset": 0,
                                "extent_affine_capacity": 1}},
                            "graph_slots": {"inputs": {},
                                            "outputs": {"valid_extent": "valid"}},
                        }],
                        "body": {"type": "dense", "ops": [{
                            "id": "select", "op": "select_next_id_checked",
                            "kernel": "select_next_id_checked_f32",
                            "returns_status": True,
                            "graph_slots": {"inputs": {"values": "external:scores"},
                                            "outputs": {"index": "selected"}},
                            "params": {"R": 1, "C": 4, "call_constants": {
                                "values_elements": 4, "rows": 1,
                                "channels": 4, "row_stride": 4,
                                "emitted_frames": 1, "min_frames": 2,
                                "max_frames": 3, "stop_id": 2,
                                "suppressed_begin": 3, "suppressed_end": 4,
                                "selected_elements": 1}},
                        }],
                        },
                        "footer": [{
                            "id": "publish_valid", "op": "runtime_extent_affine",
                            "kernel": "runtime_extent_affine_i32",
                            "returns_status": True,
                            "params": {"call_constants": {
                                "extent_affine_source": 0,
                                "extent_affine_factor": 1,
                                "extent_affine_offset": 1,
                                "extent_affine_capacity": 1}},
                            "graph_slots": {"inputs": {},
                                            "outputs": {"valid_extent": "valid"}},
                        }]},
            "step": {"sequence": ["header", "body", "footer"],
                     "header": [{
                         "id": "check_valid", "op": "runtime_extent_sum",
                         "kernel": "runtime_extent_sum_i32", "returns_status": True,
                         "produces_runtime_lengths": {"ready": "valid_extent"},
                         "graph_slots": {"inputs": {"values": "valid"},
                                         "outputs": {"valid_extent": "ready_scalar"}},
                         "params": {"T": 1},
                     }],
                     "body": {"type": "dense", "ops": [{
                         "id": "join_state", "op": "concat_i32_checked",
                         "kernel": "concat_i32_checked", "returns_status": True,
                         "consumes_runtime_lengths": ["ready"],
                         "graph_slots": {"inputs": {
                             "left": "selected", "right": "external:suffix"},
                             "outputs": {"output": "combined"}},
                         "params": {"L": 1, "R": 1, "O": 2,
                                    "call_constants": {
                                        "left_elements": 1, "left_count": 1,
                                        "right_elements": 1, "right_count": 1,
                                        "output_elements": 2}},
                     }]}, "footer": []},
        },
        "runtime_constants": {"value_elements": 1, "phoneme_count": 1,
                              "max_duration": 1, "expanded_capacity": 1},
        "runtime_lengths": {"ready": {
            "producer": "check_valid", "result": "valid_extent",
            "capacity": 1, "allow_zero": False}},
    }


class CheckedMultiEntryTest(unittest.TestCase):
    def test_shared_state_rejection_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            graph = circuit()
            prefill, acts, _, layout, calls, library = compile_entry(
                Path(temp), graph, {}, extra_sources=SOURCES)
            self.assertEqual(calls["entry"]["segments"],
                             graph["native_entry"]["segments"])
            native = ctypes.CDLL(str(library))
            step = native.ck_synthetic_step
            step.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
            step.restype = ctypes.c_int
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            scores = view(arena, acts, "scores", np.float32, (1, 4))
            selected = view(arena, acts, "selected", np.int32, (1,))
            valid = view(arena, acts, "valid", np.int32, (1,))
            suffix = view(arena, acts, "suffix", np.int32, (1,))
            combined = view(arena, acts, "combined", np.int32, (2,))
            suffix[0] = 17
            combined[:] = -719
            self.assertNotEqual(step(arena, size), 0)
            np.testing.assert_array_equal(combined, [-719, -719])
            for winning_id in (1, 0, 1):
                scores[:] = [[0.1, 0.2, -1.0, 99.0]]
                scores[0, winning_id] = 0.9
                self.assertEqual(prefill(arena, size), 0)
                self.assertEqual(int(valid[0]), 1)
                self.assertEqual(int(selected[0]), winning_id)
                self.assertEqual(step(arena, size), 0)
                np.testing.assert_array_equal(combined, [winning_id, 17])
            scores[0, 0] = np.nan
            selected[0] = 73
            combined[:] = -719
            self.assertNotEqual(prefill(arena, size), 0)
            self.assertEqual(int(valid[0]), 0)
            self.assertEqual(int(selected[0]), 73)
            self.assertNotEqual(step(arena, size), 0)
            np.testing.assert_array_equal(combined, [-719, -719])
            scores[0, 0] = 0.1
            self.assertEqual(prefill(arena, size), 0)
            self.assertEqual(step(arena, size), 0)
            np.testing.assert_array_equal(combined, [1, 17])
            combined[:] = -719
            self.assertNotEqual(step(arena, size - 1), 0)
            np.testing.assert_array_equal(combined, [-719, -719])

    def test_reject_invalid_segments(self):
        from version.v8.scripts import codegen_checked_calls_v8 as generator
        # The IR validation runs after normal lowering; mutate only the entry
        # declaration to test malformed boundaries independently of arithmetic.
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, _, calls, _ = compile_entry(
                Path(temp), circuit(), {}, extra_sources=SOURCES)
            for segments in (
                [{"function": "ck_synthetic_prefill", "last_op": "missing"},
                 {"function": "ck_synthetic_step", "last_op": "join_state"}],
                [{"function": "ck_synthetic_prefill", "last_op": "publish_valid"},
                 {"function": "ck_synthetic_prefill", "last_op": "join_state"}],
                [{"function": "ck_synthetic_prefill", "last_op": "join_state"},
                 {"function": "ck_synthetic_step", "last_op": "publish_valid"}],
                [{"function": "ck_synthetic_prefill", "last_op": "check_valid"},
                 {"function": "ck_synthetic_step", "last_op": "join_state"}],
                [{"function": "ck_synthetic_prefill", "last_op": "publish_valid"},
                 {"function": "ck_concat_i32_checked", "last_op": "join_state"}],
            ):
                with self.subTest(segments=segments):
                    broken = copy.deepcopy(calls)
                    broken["entry"]["segments"] = segments
                    with self.assertRaises(generator.CheckedCallCodegenError):
                        generator.emit_checked_calls(broken, ROOT)
            broken = copy.deepcopy(calls)
            broken["entry"]["function"] = "ck_concat_i32_checked"
            with self.assertRaises(generator.CheckedCallCodegenError):
                generator.emit_checked_calls(broken, ROOT)

    def test_positive_stop_keeps_step_unready(self):
        graph = circuit()
        graph["block_types"]["prefill"]["body"]["ops"][0]["params"] \
             ["call_constants"]["min_frames"] = 1
        with tempfile.TemporaryDirectory() as temp:
            prefill, acts, _, layout, _, library = compile_entry(
                Path(temp), graph, {}, extra_sources=SOURCES)
            step = ctypes.CDLL(str(library)).ck_synthetic_step
            step.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
            step.restype = ctypes.c_int
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            view(arena, acts, "scores", np.float32, (1, 4))[:] = [
                [0.1, 0.2, 0.9, -4.0]]
            valid = view(arena, acts, "valid", np.int32, (1,))
            combined = view(arena, acts, "combined", np.int32, (2,))
            valid[0] = 1  # stale success from a previous request
            combined[:] = -719
            self.assertEqual(prefill(arena, size), 1)
            self.assertEqual(int(valid[0]), 0)
            self.assertNotEqual(step(arena, size), 0)
            np.testing.assert_array_equal(combined, [-719, -719])


if __name__ == "__main__":
    unittest.main()
