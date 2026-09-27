"""Normal v8 lowering of a fixed-size, status-checked generated C entry."""
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


class CheckedStaticEntryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "tests/fixtures/tts/checked_static_entry_circuit.json"
        template = json.loads(path.read_text())
        source = {
            "config": {
                "model": template["name"], "arch": template["name"],
                "num_layers": 1, "embed_dim": 4, "num_heads": 1,
                "num_kv_heads": 1, "head_dim": 4, "intermediate_size": 8,
                "context_length": 7, "max_seq_len": 7, "vocab_size": 8,
            },
            "entries": [], "quant_summary": {}, "template": template,
        }
        registry = build_ir_v8.load_kernel_registry()
        with contextlib.redirect_stdout(io.StringIO()):
            ir1 = build_ir_v8.build_ir1_direct(source, path, mode="prefill")
            lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
            layout = build_ir_v8.generate_memory_layout(
                lower1, source, registry, mode="prefill", context_len=7)
            lower2 = build_ir_v8.generate_ir_lower_2(
                lower1, layout, source, registry, mode="prefill")
            call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
        cls.call_ir, cls.layout = call_ir, layout
        cls.temp = tempfile.TemporaryDirectory()
        temp = Path(cls.temp.name)
        (temp / "call.json").write_text(json.dumps(call_ir))
        (temp / "layout.json").write_text(json.dumps(layout))
        generated = temp / "generated.c"
        subprocess.run([
            sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
            "--ir", str(temp / "call.json"), "--layout", str(temp / "layout.json"),
            "--output", str(generated),
        ], check=True, capture_output=True, text=True)
        library = temp / "generated.so"
        subprocess.run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", str(generated),
            str(ROOT / "src/kernels/audio_istft_mag_phase.c"),
            "-I", str(ROOT / "include"), "-lm", "-o", str(library),
        ], check=True)
        cls.function = ctypes.CDLL(str(library)).ck_test_checked_static_entry
        cls.function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        cls.function.restype = ctypes.c_int
        cls.buffers = {item["name"]: item for item in
                       layout["memory"]["activations"]["buffers"]}
        cls.arena_size = layout["memory"]["arena"]["total_size"]
        cls.fixture = json.loads((ROOT / "tests/fixtures/tts/istft_torch20_hop5.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def array(self, arena, name, count):
        return (ctypes.c_float * count).from_buffer(
            arena, self.buffers[name]["abs_offset"])

    def arena(self):
        raw = (ctypes.c_uint8 * (self.arena_size + 63))()
        offset = (-ctypes.addressof(raw)) & 63
        arena = (ctypes.c_uint8 * self.arena_size).from_buffer(raw, offset)
        self.array(arena, "audio_magnitude", 77)[:] = [
            value for row in self.fixture["magnitude"] for value in row]
        self.array(arena, "audio_phase", 77)[:] = [
            value for row in self.fixture["phase"] for value in row]
        self.array(arena, "audio_waveform", 30)[:] = [-777.] * 30
        return arena

    def test_normal_codegen_static_graph_matches_oracle(self):
        self.assertEqual(self.call_ir["runtime_extent_contract"]["runtime_lengths"], {})
        self.assertEqual([op["function"] for op in self.call_ir["operations"]],
                         ["audio_istft_mag_phase_f32"])
        arena = self.arena()
        self.assertEqual(self.function(arena, len(arena)), 0)
        actual = self.array(arena, "audio_waveform", 30)
        error = max(abs(x - y) for x, y in zip(actual, self.fixture["waveform"]))
        self.assertLessEqual(error, 2e-5)
        print("TTS_CHECKED_STATIC_EVIDENCE " + json.dumps({
            "status": "PASS", "graph": "checked_static_entry_synthetic",
            "oracle": self.fixture["oracle"], "samples": 30,
            "max_abs_error": error,
            "reproduce": "python3 -m unittest tests.test_v8_checked_static_entry",
            "kokoro_complete_waveform": "NOT_TESTED",
        }, sort_keys=True))

    def test_provider_failure_preserves_output(self):
        arena = self.arena()
        self.array(arena, "audio_magnitude", 77)[0] = -1.0
        self.assertNotEqual(self.function(arena, len(arena)), 0)
        self.assertEqual(list(self.array(arena, "audio_waveform", 30)), [-777.] * 30)

    def test_arena_capacity_and_alignment_rejected(self):
        arena = self.arena()
        self.assertEqual(self.function(arena, len(arena) - 1), -2)
        self.assertEqual(list(self.array(arena, "audio_waveform", 30)), [-777.] * 30)
        misaligned = ctypes.cast(ctypes.byref(arena, 1), ctypes.POINTER(ctypes.c_uint8))
        self.assertEqual(self.function(misaligned, len(arena) - 1), -2)

    def test_checked_entry_requires_native_entry_declaration(self):
        template = json.loads((ROOT / "tests/fixtures/tts/checked_static_entry_circuit.json").read_text())
        del template["native_entry"]
        source = {
            "config": {
                "model": template["name"], "arch": template["name"],
                "num_layers": 1, "embed_dim": 4, "num_heads": 1,
                "num_kv_heads": 1, "head_dim": 4, "intermediate_size": 8,
                "context_length": 7, "max_seq_len": 7, "vocab_size": 8,
            },
            "entries": [], "quant_summary": {}, "template": template,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "requires circuit-declared native_entry"):
                build_ir_v8.build_ir1_direct(source, Path("missing-entry.json"), mode="prefill")
        template["native_entry"] = self.call_ir["entry"]
        template["checked_native_entry"] = "true"
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "checked_native_entry must be boolean"):
                build_ir_v8.build_ir1_direct(source, Path("invalid-entry.json"), mode="prefill")


if __name__ == "__main__":
    unittest.main()
