"""Generated CustomVoice prompt assembly and first talker block."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "version/v8/scripts"), str(ROOT / "version/v8/tts")]
import build_ir_v8
from build_qwen3_tts_prompt_first_block_circuit import (
    OUTPUT, build_circuit, prompt_plan,
)
from tests.test_v8_qwen3_tts_first_block import weight_shapes as block_weight_shapes

FIXTURE = ROOT / "tests/fixtures/tts/qwen3_tts_0b6_customvoice"


def expected_weight_shapes(graph):
    shapes = block_weight_shapes(graph)
    shapes.update({
        "talker.model.text_embedding.weight": (151936, 2048),
        "talker.model.codec_embedding.weight": (3072, 1024),
        "talker.text_projection.linear_fc1.weight": (2048, 2048),
        "talker.text_projection.linear_fc1.bias": (2048,),
        "talker.text_projection.linear_fc2.weight": (1024, 2048),
        "talker.text_projection.linear_fc2.bias": (1024,),
    })
    return shapes


def compile_entry(folder: Path, graph: dict, shapes: dict,
                  *, extra_sources: tuple[str, ...] = ()):
    folder.mkdir(parents=True, exist_ok=True)
    circuit = folder / "circuit.json"
    circuit.write_text(json.dumps(graph), encoding="utf-8")
    entries = []
    offset = 4096
    for name, shape in shapes.items():
        offset = (offset + 63) & ~63
        size = int(np.prod(shape)) * 4
        entries.append({"name": name, "shape": list(shape), "dtype": "F32",
                        "size": size, "file_offset": offset})
        offset += size
    request_rows = graph["contract"]["runtime_invariants"].get("prompt_rows", 1)
    source = {"config": {
        "model": graph["name"], "arch": graph["name"],
        "num_layers": 1, "embed_dim": 1024, "num_heads": 16,
        "num_kv_heads": 8, "head_dim": 128, "intermediate_size": 3072,
        "context_length": request_rows, "max_seq_len": request_rows,
        "vocab_size": 151936,
        "activation_buffer_dtypes": {
            name: "i32" for name in (
                "text_ids", "codec_ids", "left_indices", "right_indices",
                "step_index_input", "step_index_checked", "position_checked")},
    }, "entries": entries, "quant_summary": {}, "template": graph}
    registry = build_ir_v8.load_kernel_registry()
    with contextlib.redirect_stdout(io.StringIO()):
        ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode="prefill")
        lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
        layout = build_ir_v8.generate_memory_layout(
            lower1, source, registry, mode="prefill", context_len=request_rows)
        lower2 = build_ir_v8.generate_ir_lower_2(
            lower1, layout, source, registry, mode="prefill")
        call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
    if call_ir["errors"]:
        raise AssertionError(call_ir["errors"])
    (folder / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    (folder / "call.json").write_text(json.dumps(call_ir), encoding="utf-8")
    generated = folder / "generated.c"
    subprocess.run([
        sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
        "--ir", str(folder / "call.json"), "--layout", str(folder / "layout.json"),
        "--output", str(generated),
    ], check=True, capture_output=True, text=True)
    library = folder / "generated.so"
    subprocess.run([
        "cc", "-std=gnu11", "-O2", "-w", "-shared", "-fPIC",
        "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
        "-Wl,--no-undefined", "-fopenmp", str(generated),
        *(str(ROOT / path) for path in (
            "src/kernels/audio_text_embedding.c",
            "src/kernels/strided_unary_checked.c",
            "src/kernels/indexed_sum_rows_checked.c",
            "src/kernels/transformer_rows_checked.c",
            "src/kernels/linear_rows_checked.c",
            "src/kernels/rope_kernels.c", "src/kernels/attention_kernels.c",
            "src/kernels/softmax_kernels.c",
            "src/kernels/attention_flash_true.c", "src/ck_threadpool.c",
            "src/ckernel_strict.c",
            "src/kernels/attention_decode_append_checked.c",
            "src/kernels/runtime_extent.c", *extra_sources)),
        "-I", str(ROOT / "include"), "-lm", "-ldl", "-lpthread",
        "-o", str(library),
    ], check=True, capture_output=True, text=True)
    loaded = ctypes.CDLL(str(library))
    fn = getattr(loaded, graph["native_entry"]["function"])
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    activations = {entry["name"]: entry for entry in
                   layout["memory"]["activations"]["buffers"]}
    weights = {entry["name"]: entry for entry in
               layout["memory"]["weights"]["entries"]}
    return fn, activations, weights, layout, call_ir, library


def view(arena, activations, name, dtype, shape):
    return np.ndarray(shape, dtype=dtype, buffer=arena,
                      offset=activations[name]["abs_offset"])


def pinned_prompt_ids(model_dir: Path, source_capture):
    input_ids = source_capture["text_token_ids"].reshape(-1)
    context_ids = source_capture["context_token_ids"].reshape(-1)
    config = json.loads((model_dir / "config.json").read_text())
    talker = config["talker_config"]
    text_ids = np.r_[input_ids[:3], [config["tts_pad_token_id"],
                    config["tts_bos_token_id"], config["tts_eos_token_id"]],
                    context_ids].astype(np.int32)
    codec_ids = np.array([
        talker["codec_think_id"], talker["codec_think_bos_id"],
        talker["codec_language_id"]["english"],
        talker["codec_think_eos_id"], talker["spk_id"]["ryan"],
        talker["codec_pad_id"], talker["codec_bos_id"],
    ], dtype=np.int32)
    return text_ids, codec_ids


class Qwen3TTSPromptFirstBlockTest(unittest.TestCase):
    def test_declaration_and_prompt_connection(self):
        graph = build_circuit()
        self.assertEqual(OUTPUT.read_text(), json.dumps(graph, indent=2) + "\n")
        self.assertEqual(graph["sequence"], ["text_projection", "first_block"])
        self.assertEqual(len(prompt_plan(7)["left_indices"]), 18)
        self.assertEqual(graph["block_types"]["text_projection"]["body"]["ops"][-1]["id"],
                         "prompt_assembly")
        self.assertEqual(graph["block_types"]["first_block"]["body"]["ops"][-1]["id"],
                         "block_output")
        for context_tokens in (1, 7, 37, 117):
            with self.subTest(context_tokens=context_tokens):
                alternate = build_circuit(context_tokens)
                plan = alternate["contract"]["runtime_invariants"]["prompt_index_plan"]
                self.assertEqual(len(plan["left_indices"]), context_tokens + 11)
                self.assertEqual(len(plan["right_indices"]), context_tokens + 11)
                self.assertLess(max(plan["left_indices"]), context_tokens + 6)
                self.assertLess(max(plan["right_indices"]), 7)
        for invalid in (0, 118, True, 2.5):
            with self.assertRaises(ValueError):
                build_circuit(invalid)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned Qwen3-TTS weights unavailable; generated prompt NOT_TESTED")
    def test_pinned_generated_prompt_and_block(self):
        from safetensors import safe_open
        from capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        graph = build_circuit()
        shapes = expected_weight_shapes(graph)
        manifest = json.loads((FIXTURE / "qwen3_tts_first_block.json").read_text())
        capture_path = FIXTURE / "qwen3_tts_first_block.npz"
        self.assertEqual(hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        with np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as source_capture, \
             np.load(capture_path) as block_capture, \
             tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            fn, acts, weights, layout, calls, library = compile_entry(folder, graph, shapes)
            self.assertEqual(len(calls["operations"]), 28)
            self.assertEqual(calls["operations"][7]["function"],
                             "indexed_sum_rows_checked_f32")
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(raw, (-ctypes.addressof(raw)) & 63)
            with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
                for name, item in weights.items():
                    value = source.get_tensor(name).float().numpy()
                    self.assertEqual(value.shape, shapes[name])
                    self.assertEqual(value.dtype, np.float32)
                    self.assertTrue(value.flags.c_contiguous)
                    arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()

            context_ids = source_capture["context_token_ids"].reshape(-1)
            self.assertEqual(len(context_ids), 7)
            text_ids, codec_ids = pinned_prompt_ids(model_dir, source_capture)
            view(arena, acts, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(arena, acts, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            for name, values in prompt_plan(7).items():
                view(arena, acts, name, np.int32, (18,))[:] = values
            self.assertEqual(fn(arena, len(arena)), 0)
            expected_prompt = block_capture["call0_input"][0]
            prompt = view(arena, acts, "block_input", np.float32, (18, 1024))
            output = view(arena, acts, "block_output", np.float32, (18, 1024))
            prompt_error = float(np.max(np.abs(prompt - expected_prompt)))
            output_error = float(np.max(np.abs(output - block_capture["call0_output"][0])))
            print("QWEN3_TTS_GENERATED_PROMPT_BLOCK " + json.dumps({
                "scope": "text and codec IDs through first talker block; no codes generated",
                "prompt_max_abs": prompt_error, "block_max_abs": output_error,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "model_revision": model_dir.name,
                "full_talker": "NOT_TESTED", "waveform_from_text": "NOT_TESTED"},
                sort_keys=True))
            self.assertLessEqual(prompt_error, 5e-4)
            self.assertLessEqual(output_error, 5e-4)
            pristine = output.copy()
            # Invalid request IDs must stop before prompt assembly and block execution.
            view(arena, acts, "codec_ids", np.int32, (7,))[0] = 3072
            output[:] = -777
            self.assertNotEqual(fn(arena, len(arena)), 0)
            self.assertTrue(np.all(output == -777))
            view(arena, acts, "codec_ids", np.int32, (7,))[0] = codec_ids[0]
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, pristine)
            # A valid but incorrect row edge must be exposed by prompt parity.
            left_map = view(arena, acts, "left_indices", np.int32, (18,))
            left_map[10] = 3
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertGreater(float(np.max(np.abs(prompt - expected_prompt))), 1e-3)
            left_map[10] = prompt_plan(7)["left_indices"][10]
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, pristine)
            output[:] = -777
            left_map[10] = 13
            self.assertNotEqual(fn(arena, len(arena)), 0)
            self.assertTrue(np.all(output == -777))
            left_map[10] = prompt_plan(7)["left_indices"][10]
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, pristine)
            output[:] = -777
            self.assertNotEqual(fn(arena, len(arena) - 1), 0)
            self.assertTrue(np.all(output == -777))


if __name__ == "__main__":
    unittest.main()
