"""Checked integer state movement through a provider and generated entries."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_prompt_first_block import compile_entry, view

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/kernels/copy_i32_span_checked.c"


def _op(name: str, source: str, destination: str, src_size: int,
        dst_size: int, src_offset: int, dst_offset: int, count: int) -> dict:
    graph = {
        "id": name, "op": "copy_i32_span_checked",
        "kernel": "copy_i32_span_checked", "returns_status": True,
        "graph_slots": {"inputs": {"source": source},
                        "outputs": {"destination": destination}},
        "params": {"S": src_size, "D": dst_size, "call_constants": {
            "source_elements": src_size, "source_offset": src_offset,
            "destination_elements": dst_size,
            "destination_offset": dst_offset, "count": count}},
    }
    return graph


def _circuit() -> dict:
    shapes = {"initial": [4], "state": [4], "replacement": [4],
              "observed": [1]}
    graph = {
        "version": 3, "name": "i32_state_copy_synthetic",
        "family": "bounded_graph", "checked_native_entry": True,
        "contract": {"runtime_invariants": {"inference_only": True}},
        "activation_buffers": {name: {"shape": shape}
                               for name, shape in shapes.items()},
        "activation_bindings": {name: name for name in shapes},
        "native_entry": {
            "function": "ck_state_initialize",
            "params": [{"c_type": "uint8_t *", "name": "arena"},
                       {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"},
            "segments": [
                {"function": "ck_state_initialize", "last_op": "initialize"},
                {"function": "ck_state_step", "last_op": "update"}],
        },
        "sequence": ["initialize_state", "step"],
        "block_types": {
            "initialize_state": {"sequence": ["header", "body", "footer"],
                                 "header": [], "body": {"type": "dense", "ops": [
                _op("initialize", "external:initial", "external:state",
                    4, 4, 0, 0, 4)]}, "footer": []},
            "step": {"sequence": ["header", "body", "footer"],
                     "header": [], "body": {"type": "dense", "ops": [
                _op("read", "external:state", "observed", 4, 1, 3, 0, 1),
                _op("update", "external:replacement", "external:state",
                    4, 4, 0, 0, 4)]}, "footer": []},
        },
        "runtime_constants": {}, "runtime_lengths": {},
    }
    graph["activation_buffers"]["state"]["lifetime"] = "sequence"
    return graph


class CopyI32SpanCheckedTest(unittest.TestCase):
    def test_direct_bounds_overlap_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "copy.so"
            subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                            "-I", str(ROOT / "include"),
                            str(ROOT / SOURCE), "-o", str(library)],
                           check=True, capture_output=True)
            fn = ctypes.CDLL(str(library)).ck_copy_i32_span_checked
            ptr = ctypes.POINTER(ctypes.c_int32)
            fn.argtypes = [ptr, ctypes.c_size_t, ctypes.c_size_t,
                           ptr, ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.c_size_t]
            fn.restype = ctypes.c_int
            src = np.array([4, 7, 11, 13], dtype=np.int32)
            dst = np.full(6, -901, dtype=np.int32)
            self.assertEqual(fn(src.ctypes.data_as(ptr), 4, 1,
                                dst.ctypes.data_as(ptr), 6, 2, 2), 0)
            np.testing.assert_array_equal(dst, [-901, -901, 7, 11, -901, -901])
            for source_size, source_offset, dest_size, dest_offset, count in (
                (4, 3, 6, 2, 2), (4, 0, 6, 5, 2), (4, 0, 6, 0, 0),
            ):
                before = dst.copy()
                self.assertNotEqual(fn(src.ctypes.data_as(ptr), source_size,
                                       source_offset, dst.ctypes.data_as(ptr),
                                       dest_size, dest_offset, count), 0)
                np.testing.assert_array_equal(dst, before)
            before = src.copy()
            self.assertNotEqual(fn(src.ctypes.data_as(ptr), 4, 0,
                                   src.ctypes.data_as(ptr), 4, 1, 2), 0)
            np.testing.assert_array_equal(src, before)
            before = dst.copy()
            self.assertNotEqual(fn(None, 4, 0, dst.ctypes.data_as(ptr),
                                   6, 0, 1), 0)
            self.assertNotEqual(fn(src.ctypes.data_as(ptr), 4, 0,
                                   None, 6, 0, 1), 0)
            np.testing.assert_array_equal(dst, before)
            self.assertEqual(fn(src.ctypes.data_as(ptr), 4, 2,
                                dst.ctypes.data_as(ptr), 6, 4, 2), 0)
            np.testing.assert_array_equal(dst, [-901, -901, 7, 11, 11, 13])

    def test_generated_repeated_state(self):
        with tempfile.TemporaryDirectory() as temp:
            graph = _circuit()
            initialize, acts, _, layout, _, library = compile_entry(
                Path(temp), graph, {}, extra_sources=(SOURCE,))
            self.assertEqual(acts["state"]["lifetime"], "sequence")
            step = ctypes.CDLL(str(library)).ck_state_step
            step.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
            step.restype = ctypes.c_int
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            initial = view(arena, acts, "initial", np.int32, (4,))
            state = view(arena, acts, "state", np.int32, (4,))
            replacement = view(arena, acts, "replacement", np.int32, (4,))
            observed = view(arena, acts, "observed", np.int32, (1,))
            initial[:] = [2, 3, 5, 7]
            state[:] = -1
            replacement[:] = [11, 13, 17, 19]
            self.assertEqual(initialize(arena, size), 0)
            self.assertEqual(step(arena, size), 0)
            self.assertEqual(int(observed[0]), 7)
            np.testing.assert_array_equal(state, replacement)
            replacement[:] = [23, 29, 31, 37]
            self.assertEqual(step(arena, size), 0)
            self.assertEqual(int(observed[0]), 19)
            observed[0] = -901
            self.assertNotEqual(step(arena, size - 1), 0)
            self.assertEqual(int(observed[0]), -901)
            np.testing.assert_array_equal(state, [23, 29, 31, 37])
            self.assertEqual(step(arena, size), 0)
            self.assertEqual(int(observed[0]), 37)
            np.testing.assert_array_equal(state, [23, 29, 31, 37])

    def test_reject_invalid_declared_state_lifetime(self):
        for bad_lifetime in ("unknown", ["sequence"]):
            with self.subTest(lifetime=bad_lifetime):
                graph = _circuit()
                graph["activation_buffers"]["state"]["lifetime"] = bad_lifetime
                with tempfile.TemporaryDirectory() as temp:
                    with self.assertRaisesRegex(RuntimeError, "CIRCUIT BUFFER FAULT"):
                        compile_entry(Path(temp), graph, {}, extra_sources=(SOURCE,))


if __name__ == "__main__":
    unittest.main()
