"""Caller-owned bounded frame-loop shell over checked generated entries."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_prompt_first_block import compile_entry, view
from tests.test_v8_qwen3_tts_talker_prefill import weight_shapes
from tests.test_v8_qwen3_tts_second_generated_frame import prepare_inputs
from tests.test_v8_qwen3_tts_two_phase_speech import SOURCES, segment_function
from version.v8.tts.build_qwen3_tts_second_generated_frame_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
DRIVER = "src/runtime/checked_frame_loop.c"


class Loop(ctypes.Structure):
    _fields_ = [("prefill", ctypes.c_void_p), ("frame", ctypes.c_void_p),
                ("arena", ctypes.POINTER(ctypes.c_uint8)),
                ("arena_bytes", ctypes.c_size_t),
                ("step_index_offset", ctypes.c_size_t),
                ("prefill_valid_code_count_offset", ctypes.c_size_t),
                ("frame_valid_code_count_offset", ctypes.c_size_t),
                ("prefill_code_offsets", ctypes.POINTER(ctypes.c_size_t)),
                ("frame_code_offsets", ctypes.POINTER(ctypes.c_size_t)),
                ("codebooks", ctypes.c_size_t), ("max_frames", ctypes.c_size_t),
                ("staging_codes", ctypes.POINTER(ctypes.c_int32)),
                ("staging_elements", ctypes.c_size_t),
                ("output_codes", ctypes.POINTER(ctypes.c_int32)),
                ("output_elements", ctypes.c_size_t),
                ("valid_frames", ctypes.POINTER(ctypes.c_size_t))]


def run_function(library: Path):
    fn = ctypes.CDLL(str(library)).ck_run_checked_frame_loop
    fn.argtypes = [ctypes.POINTER(Loop)]
    fn.restype = ctypes.c_int
    return fn


class CheckedFrameLoopTest(unittest.TestCase):
    def test_eos_budget_failure_and_recovery(self):
        fake = r'''
#include <stddef.h>
#include <stdint.h>
static int mode = 0;
static int calls = 0;
void set_mode(int value) { mode = value; calls = 0; }
int get_calls(void) { return calls; }
int fake_prefill(uint8_t *arena, size_t bytes) {
    if (bytes < 32) return -1;
    calls++;
    ((int32_t *)arena)[1] = mode == 4 ? 0 : 1;
    ((int32_t *)arena)[3] = 10;
    ((int32_t *)arena)[4] = 20;
    return 0;
}
int fake_frame(uint8_t *arena, size_t bytes) {
    if (bytes < 32) return -1;
    calls++;
    int32_t *values = (int32_t *)arena;
    int32_t step = values[0];
    if (mode == 1 && step == 1) return -3;
    if (mode == 0 && step == 2) return 1;
    if (mode == 2 && step == 1) return 2;
    values[2] = 1;
    values[5] = 11 + step;
    values[6] = 21 + step;
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "fake.c"
            source.write_text(fake)
            library = Path(temp) / "loop.so"
            subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                            "-I", str(ROOT / "include"),
                            str(ROOT / DRIVER), str(source), "-o", str(library)],
                           check=True, capture_output=True)
            native = ctypes.CDLL(str(library))
            native.set_mode.argtypes = [ctypes.c_int]
            native.get_calls.restype = ctypes.c_int
            arena = (ctypes.c_uint8 * 32)()
            first = (ctypes.c_size_t * 2)(12, 16)
            frame = (ctypes.c_size_t * 2)(20, 24)
            staging = np.full(10, -1, dtype=np.int32)
            output = np.full(10, -901, dtype=np.int32)
            valid = ctypes.c_size_t(99)
            request = Loop(ctypes.cast(native.fake_prefill, ctypes.c_void_p),
                           ctypes.cast(native.fake_frame, ctypes.c_void_p),
                           arena, 32, 0, 4, 8, first, frame, 2, 5,
                           staging.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                           10, output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                           10, ctypes.pointer(valid))
            run = run_function(library)
            native.set_mode(0)
            self.assertEqual(run(ctypes.byref(request)), 0)
            self.assertEqual(valid.value, 3)
            np.testing.assert_array_equal(output[:6], [10, 20, 11, 21, 12, 22])
            np.testing.assert_array_equal(output[6:], [-901] * 4)
            for mode, expected in ((1, -2), (2, 2), (3, 2), (4, -3)):
                native.set_mode(mode)
                output[:] = -901
                self.assertEqual(run(ctypes.byref(request)), expected)
                self.assertEqual(valid.value, 0)
                np.testing.assert_array_equal(output, [-901] * 10)
            native.set_mode(0)
            request.frame_code_offsets = (ctypes.c_size_t * 2)(20, 32)
            self.assertEqual(run(ctypes.byref(request)), -1)
            self.assertEqual(native.get_calls(), 0)
            self.assertEqual(valid.value, 0)
            request.frame_code_offsets = frame
            request.output_elements = 9
            self.assertEqual(run(ctypes.byref(request)), -1)
            self.assertEqual(native.get_calls(), 0)
            np.testing.assert_array_equal(output, [-901] * 10)
            request.output_elements = 10
            request.staging_codes = request.output_codes
            self.assertEqual(run(ctypes.byref(request)), -1)
            self.assertEqual(native.get_calls(), 0)
            np.testing.assert_array_equal(output, [-901] * 10)
            request.staging_codes = staging.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int32))
            self.assertEqual(run(ctypes.byref(request)), 0)
            self.assertEqual(valid.value, 3)

    def _run_generated_qwen(self, *, force_eos: bool):
        graph = build_circuit(talker_layers=1, continuation_steps=2,
                              max_generated_frames=5)
        if force_eos:
            selector = graph["block_types"]["second_talker_predictor_bridge"]
            selector["body"]["ops"][1]["params"]["call_constants"]["stop_id"] = 0
        with tempfile.TemporaryDirectory() as temp:
            _, acts, _, layout, _, library = compile_entry(
                Path(temp), graph, weight_shapes(graph),
                extra_sources=(*SOURCES, DRIVER))
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            prepare_inputs(arena, acts)
            cache_capacity = graph["activation_buffers"]["talker_layer0_cache_k"]["shape"][1]
            cache = view(arena, acts, "talker_layer0_cache_k",
                         np.float32, (8, cache_capacity, 128))
            cache[:] = -901.0
            prefill_names = ["talker_first_code_id",
                             "pred_predictor_codebook0_selected_id",
                             "pred_step1_selected_id", "pred_step2_selected_id"]
            frame_names = ["second_talker_first_code_id",
                           "second_predictor_codebook0_selected_id",
                           "second_step1_selected_id", "second_step2_selected_id"]
            prefill_offsets = (ctypes.c_size_t * 4)(
                *(acts[name]["abs_offset"] for name in prefill_names))
            frame_offsets = (ctypes.c_size_t * 4)(
                *(acts[name]["abs_offset"] for name in frame_names))
            staging = np.full(20, -1, dtype=np.int32)
            output = np.full(20, -901, dtype=np.int32)
            valid = ctypes.c_size_t(99)
            request = Loop(
                ctypes.cast(segment_function(library, graph, 0), ctypes.c_void_p),
                ctypes.cast(segment_function(library, graph, 1), ctypes.c_void_p),
                arena, size, acts["talker_step_index_input"]["abs_offset"],
                acts["pred_predictor_valid_code_count"]["abs_offset"],
                acts["second_predictor_valid_code_count"]["abs_offset"],
                prefill_offsets, frame_offsets, 4, 5,
                staging.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 20,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 20,
                ctypes.pointer(valid))
            result = run_function(library)(ctypes.byref(request))
            if force_eos:
                self.assertEqual(result, 0)
                self.assertEqual(valid.value, 2)
                self.assertEqual(int(output[0]), 0)
                self.assertEqual(int(output[4]), 1)
                np.testing.assert_array_equal(output[8:], [-901] * 12)
                self.assertTrue(np.all(cache[:, 20:, :] == -901.0))
                self.assertEqual(int(view(arena, acts, "generation_state_valid",
                                          np.int32, (1,))[0]), 0)
                first_output = output.copy()
                output[:] = -901
                valid.value = 99
                self.assertEqual(run_function(library)(ctypes.byref(request)), 0)
                self.assertEqual(valid.value, 2)
                np.testing.assert_array_equal(output, first_output)
            else:
                self.assertEqual(result, 2)
                self.assertEqual(valid.value, 0)
                np.testing.assert_array_equal(output, [-901] * 20)
                self.assertEqual(int(view(arena, acts, "expected_step_index",
                                          np.int32, (1,))[0]), 4)

    def test_generated_qwen_entries_reach_budget_without_publishing(self):
        self._run_generated_qwen(force_eos=False)

    def test_generated_qwen_entries_stop_on_eos(self):
        self._run_generated_qwen(force_eos=True)


if __name__ == "__main__":
    unittest.main()
