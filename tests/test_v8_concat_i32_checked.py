"""Exact i32 span concatenation through its checked provider and generated C."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_prompt_first_block import compile_entry, view

ROOT = Path(__file__).resolve().parents[1]


def synthetic_graph():
    names = ("first", "second", "combined")
    return {
        "version": 3, "name": "i32_concat_synthetic", "family": "bounded_graph",
        "checked_native_entry": True,
        "contract": {"runtime_invariants": {"inference_only": True}},
        "activation_buffers": {name: {"shape": [n]} for name, n in
                               zip(names, (2, 3, 6))},
        "activation_bindings": {name: name for name in names},
        "native_entry": {"function": "ck_i32_concat_synthetic", "params": [
            {"c_type": "uint8_t *", "name": "arena"},
            {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
        "sequence": ["combine"],
        "block_types": {"combine": {
            "sequence": ["header", "body", "footer"], "header": [],
            "body": {"type": "dense", "ops": [{
                "id": "concat", "op": "concat_i32_checked",
                "kernel": "concat_i32_checked", "returns_status": True,
                "graph_slots": {"inputs": {"left": "external:first",
                                           "right": "external:second"},
                                "outputs": {"output": "combined"}},
                "params": {"L": 2, "R": 3, "O": 6, "call_constants": {
                    "left_elements": 2, "left_count": 2,
                    "right_elements": 3, "right_count": 3,
                    "output_elements": 6}},
            }]}, "footer": []}},
        "runtime_constants": {}, "runtime_lengths": {},
    }


class ConcatI32CheckedTest(unittest.TestCase):
    def test_direct_capacity_overlap_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "concat.so"
            subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                            "-I", str(ROOT / "include"),
                            str(ROOT / "src/kernels/concat_i32_checked.c"),
                            "-o", str(library)], check=True, capture_output=True)
            func = ctypes.CDLL(str(library)).ck_concat_i32_checked
            pointer = ctypes.POINTER(ctypes.c_int32)
            func.argtypes = [pointer, ctypes.c_size_t, ctypes.c_size_t,
                             pointer, ctypes.c_size_t, ctypes.c_size_t,
                             pointer, ctypes.c_size_t]
            func.restype = ctypes.c_int
            left = np.array([2, 7], dtype=np.int32)
            right = np.array([11, 13, 17], dtype=np.int32)
            out = np.full(6, -907, dtype=np.int32)

            def run(left_cap=2, right_cap=3, out_cap=6,
                    output=None):
                target = out if output is None else output
                return func(left.ctypes.data_as(pointer), left_cap, 2,
                            right.ctypes.data_as(pointer), right_cap, 3,
                            target.ctypes.data_as(pointer), out_cap)

            self.assertEqual(run(), 0)
            np.testing.assert_array_equal(out, [2, 7, 11, 13, 17, -907])
            for changes in ({"left_cap": 1}, {"right_cap": 2},
                            {"out_cap": 4}, {"output": left}):
                with self.subTest(changes=changes):
                    out[:] = -907
                    before_left = left.copy()
                    self.assertNotEqual(run(**changes), 0)
                    np.testing.assert_array_equal(out, np.full(6, -907))
                    np.testing.assert_array_equal(left, before_left)
            self.assertEqual(run(), 0)
            np.testing.assert_array_equal(out[:5], np.r_[left, right])

    def test_generated_composition_and_contract_rejection(self):
        graph = synthetic_graph()
        with tempfile.TemporaryDirectory() as temp:
            fn, acts, _, layout, calls, _ = compile_entry(
                Path(temp), graph, {},
                extra_sources=("src/kernels/concat_i32_checked.c",))
            self.assertEqual(calls["operations"][0]["function"],
                             "ck_concat_i32_checked")
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            view(arena, acts, "first", np.int32, (2,))[:] = [2, 7]
            view(arena, acts, "second", np.int32, (3,))[:] = [11, 13, 17]
            result = view(arena, acts, "combined", np.int32, (6,))
            result[:] = -907
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(result, [2, 7, 11, 13, 17, -907])
            result[:] = -907
            self.assertNotEqual(fn(arena, size - 1), 0)
            np.testing.assert_array_equal(result, np.full(6, -907))
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(result[:5], [2, 7, 11, 13, 17])

        for field, value in (("left_elements", 3), ("right_count", 2),
                             ("output_elements", 7)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                bad = synthetic_graph()
                bad["block_types"]["combine"]["body"]["ops"][0]["params"][
                    "call_constants"][field] = value
                with self.assertRaisesRegex(RuntimeError, "HARD CALL CONSTANT FAULT"):
                    compile_entry(Path(temp), bad, {},
                                  extra_sources=("src/kernels/concat_i32_checked.c",))


if __name__ == "__main__":
    unittest.main()
