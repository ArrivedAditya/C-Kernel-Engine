"""Normal v8 compilation of logits -> checked duration extent -> features."""

import contextlib
import ctypes
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import build_ir_v8


def lower_fixture(path=None):
    path = path or ROOT / "tests/fixtures/tts/duration_logits_runtime_extent_circuit.json"
    template = json.loads(path.read_text())
    source = {
        "config": {
            "model": template["name"], "arch": template["name"],
            "num_layers": 1, "embed_dim": 4, "num_heads": 1,
            "num_kv_heads": 1, "head_dim": 4, "intermediate_size": 8,
            "context_length": 36, "max_seq_len": 36, "vocab_size": 8,
            "T": 36, "B": 50, "speed": 1.0,
            "activation_buffer_dtypes": {
                "runtime_values": "i32", "runtime_valid_extent": "i32",
            },
        },
        "entries": [], "quant_summary": {}, "template": template,
    }
    registry = build_ir_v8.load_kernel_registry()
    with contextlib.redirect_stdout(io.StringIO()):
        ir1 = build_ir_v8.build_ir1_direct(source, path, mode="prefill")
        lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
        layout = build_ir_v8.generate_memory_layout(
            lower1, source, registry, mode="prefill", context_len=36)
        lower2 = build_ir_v8.generate_ir_lower_2(
            lower1, layout, source, registry, mode="prefill")
        call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
    return source, layout, call_ir


class DurationLogitsRuntimeGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.layout, cls.call_ir = lower_fixture()
        cls.fixture = json.loads((ROOT / "tests/fixtures/tts/duration_logits_kokoro_torch.json").read_text())
        cls.temp = tempfile.TemporaryDirectory()
        temp = Path(cls.temp.name)
        call_path, layout_path = temp / "call.json", temp / "layout.json"
        call_path.write_text(json.dumps(cls.call_ir))
        layout_path.write_text(json.dumps(cls.layout))
        generated = temp / "generated.c"
        subprocess.run([
            sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
            "--ir", str(call_path), "--layout", str(layout_path),
            "--output", str(generated),
        ], check=True, capture_output=True, text=True)
        library = temp / "generated.so"
        subprocess.run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", str(generated),
            str(ROOT / "src/kernels/audio_duration_logits.c"),
            str(ROOT / "src/kernels/audio_duration_expand.c"),
            str(ROOT / "src/kernels/runtime_extent.c"),
            "-I", str(ROOT / "include"), "-lm", "-o", str(library),
        ], check=True)
        cls.function = ctypes.CDLL(str(library)).ck_test_duration_logits_graph
        cls.function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_int32)]
        cls.function.restype = ctypes.c_int
        cls.buffers = {item["name"]: item for item in
                       cls.layout["memory"]["activations"]["buffers"]}
        cls.arena_size = cls.layout["memory"]["arena"]["total_size"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _array(self, arena, name, element_type, length):
        return (element_type * length).from_buffer(arena, self.buffers[name]["abs_offset"])

    def _arena(self, logits=None):
        raw = (ctypes.c_uint8 * (self.arena_size + 63))()
        offset = (-ctypes.addressof(raw)) & 63
        arena = (ctypes.c_uint8 * self.arena_size).from_buffer(raw, offset)
        logits = logits if logits is not None else [value for row in self.fixture["logits"]
                                                 for value in row]
        self._array(arena, "audio_duration_logits", ctypes.c_float, 1800)[:] = logits
        self._array(arena, "runtime_values", ctypes.c_int32, 36)[:] = [-777] * 36
        self._array(arena, "runtime_valid_extent", ctypes.c_int32, 1)[:] = [-777]
        self._array(arena, "audio_features", ctypes.c_float, 72)[:] = (
            list(range(36)) + list(range(100, 136)))
        self._array(arena, "audio_expanded", ctypes.c_float, 256)[:] = [-99] * 256
        self._array(arena, "runtime_valid_copy", ctypes.c_float, 256)[:] = [-88] * 256
        return arena

    def test_normal_lowering_generated_c_matches_pinned_duration(self):
        self.assertEqual([op["function"] for op in self.call_ir["operations"]], [
            "audio_duration_logits_to_frames_f32",
            "audio_duration_expand_channel_major_f32",
            "ck_runtime_copy_valid_f32",
        ])
        self.assertTrue(all(not op["errors"] for op in self.call_ir["operations"]))
        producer_args = {arg["name"]: arg["expr"] for arg in
                         self.call_ir["operations"][0]["args"]}
        expansion_args = {arg["name"]: arg["expr"] for arg in
                          self.call_ir["operations"][1]["args"]}
        self.assertEqual((producer_args["input_elements"], producer_args["input_stride"]),
                         ("1800", "50"))
        self.assertEqual((expansion_args["input_elements"], expansion_args["input_stride"]),
                         ("72", "36"))
        arena = self._arena()
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        durations = list(self._array(arena, "runtime_values", ctypes.c_int32, 36))
        self.assertEqual(durations, self.fixture["durations"])
        for channel in range(2):
            values = range(36) if channel == 0 else range(100, 136)
            expected = [float(value) for value, count in zip(values, durations)
                        for _ in range(count)]
            expanded = list(self._array(arena, "audio_expanded", ctypes.c_float, 256))
            copied = list(self._array(arena, "runtime_valid_copy", ctypes.c_float, 256))
            start = channel * 128
            self.assertEqual(expanded[start:start + 103], expected)
            self.assertEqual(copied[start:start + 103], expected)
            self.assertEqual(expanded[start + 103:start + 128], [-99] * 25)
            self.assertEqual(copied[start + 103:start + 128], [-88] * 25)

    def test_excess_extent_stops_downstream_before_writes(self):
        arena = self._arena([0.0] * 1800)
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), -2)
        self.assertEqual(frames.value, -1)
        self.assertEqual(list(self._array(arena, "runtime_values", ctypes.c_int32, 36)),
                         [-777] * 36)
        self.assertEqual(list(self._array(arena, "audio_expanded", ctypes.c_float, 256)),
                         [-99] * 256)
        self.assertEqual(list(self._array(arena, "runtime_valid_copy", ctypes.c_float, 256)),
                         [-88] * 256)

    def test_exact_capacity_and_nonfinite_rejection(self):
        logits = []
        for token in range(36):
            active = 46 if token == 35 else 6
            logits.extend([0.0] * active + [-100.0] * (50 - active))
        arena = self._arena(logits)
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 128)
        self.assertEqual(list(self._array(arena, "runtime_values", ctypes.c_int32, 36)),
                         [3] * 35 + [23])

        arena = self._arena()
        self._array(arena, "audio_duration_logits", ctypes.c_float, 1800)[1799] = float("nan")
        frames.value = -1
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), -1)
        self.assertEqual(frames.value, -1)
        self.assertEqual(list(self._array(arena, "runtime_values", ctypes.c_int32, 36)),
                         [-777] * 36)
        self.assertEqual(list(self._array(arena, "audio_expanded", ctypes.c_float, 256)),
                         [-99] * 256)

    def test_repeated_valid_extents_on_one_arena(self):
        arena = self._arena()
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        self._array(arena, "audio_duration_logits", ctypes.c_float, 1800)[:] = (
            [-100.0] * 1800)
        frames.value = -1
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 36)
        self.assertEqual(list(self._array(arena, "runtime_values", ctypes.c_int32, 36)),
                         [1] * 36)
        copied = list(self._array(arena, "runtime_valid_copy", ctypes.c_float, 256))
        self.assertEqual(copied[:36], list(range(36)))
        self.assertEqual(copied[128:164], list(range(100, 136)))

    def test_undersized_arena_stops_before_producer(self):
        arena = self._arena()
        before = bytes(arena)
        frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena) - 1,
                                       ctypes.byref(frames)), -2)
        self.assertEqual(frames.value, -1)
        self.assertEqual(bytes(arena), before)


if __name__ == "__main__":
    unittest.main()
