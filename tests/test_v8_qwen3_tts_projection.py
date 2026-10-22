"""Generated text-ID → talker-projection execution and rejection controls."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
sys.path.insert(0, str(ROOT / "version/v8/tts"))
import build_ir_v8
from build_qwen3_tts_text_projection_circuit import build_circuit

WEIGHTS = (
    "talker.model.text_embedding.weight",
    "talker.text_projection.linear_fc1.weight",
    "talker.text_projection.linear_fc1.bias",
    "talker.text_projection.linear_fc2.weight",
    "talker.text_projection.linear_fc2.bias",
)


def generated_entry(folder: Path, arrays: dict[str, np.ndarray], tokens: int):
    table = arrays[WEIGHTS[0]]
    hidden = arrays[WEIGHTS[3]].shape[0]
    graph = build_circuit(tokens=tokens, vocab=table.shape[0],
                          text_width=table.shape[1], hidden_width=hidden)
    circuit = folder / "circuit.json"
    circuit.write_text(json.dumps(graph))
    config = {
        "model": graph["name"], "arch": graph["name"],
        "num_layers": 1, "embed_dim": table.shape[1], "num_heads": 1,
        "num_kv_heads": 1, "head_dim": table.shape[1],
        "intermediate_size": table.shape[1] * 2,
        "context_length": tokens, "max_seq_len": tokens,
        "vocab_size": table.shape[0],
        "activation_buffer_dtypes": {"text_ids": "i32"},
    }
    entries = []
    file_offset = 4096
    for name in WEIGHTS:
        file_offset = (file_offset + 63) & ~63
        entries.append({"name": name, "shape": list(arrays[name].shape),
                        "dtype": "F32", "size": arrays[name].nbytes,
                        "file_offset": file_offset,
                        "sha256": hashlib.sha256(arrays[name].tobytes()).hexdigest()})
        file_offset += arrays[name].nbytes
    source = {"config": config, "entries": entries,
              "quant_summary": {}, "template": graph}
    registry = build_ir_v8.load_kernel_registry()
    with contextlib.redirect_stdout(io.StringIO()):
        ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode="prefill")
        lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
        layout = build_ir_v8.generate_memory_layout(
            lower1, source, registry, mode="prefill", context_len=tokens)
        lower2 = build_ir_v8.generate_ir_lower_2(
            lower1, layout, source, registry, mode="prefill")
        call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
    if call_ir["errors"]:
        raise AssertionError(call_ir["errors"])
    (folder / "layout.json").write_text(json.dumps(layout))
    (folder / "call.json").write_text(json.dumps(call_ir))
    source_path = folder / "generated.c"
    subprocess.run([
        sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
        "--ir", str(folder / "call.json"),
        "--layout", str(folder / "layout.json"), "--output", str(source_path),
    ], check=True, capture_output=True, text=True)
    library_path = folder / "generated.so"
    subprocess.run([
        "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-shared", "-fPIC", str(source_path),
        str(ROOT / "src/kernels/audio_text_embedding.c"),
        str(ROOT / "src/kernels/strided_unary_checked.c"),
        str(ROOT / "src/kernels/linear_rows_checked.c"),
        "-I", str(ROOT / "include"), "-lm", "-o", str(library_path),
    ], check=True, capture_output=True, text=True)
    loaded = ctypes.CDLL(str(library_path))
    function = loaded.ck_qwen3_tts_text_projection_bounded
    function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    function.restype = ctypes.c_int
    weights = {entry["name"]: entry for entry in
               layout["memory"]["weights"]["entries"]}
    activations = {entry["name"]: entry for entry in
                   layout["memory"]["activations"]["buffers"]}
    size = layout["memory"]["arena"]["total_size"]
    raw = (ctypes.c_uint8 * (size + 63))()
    offset = (-ctypes.addressof(raw)) & 63
    arena = (ctypes.c_uint8 * size).from_buffer(raw, offset)
    for name, entry in weights.items():
        value = arrays[name]
        start = entry["abs_offset"]
        arena[start:start + value.nbytes] = value.tobytes()
    return function, arena, activations, call_ir, library_path


def vector(arena, activations, name: str, dtype, shape):
    return np.ndarray(shape, dtype=dtype, buffer=arena,
                      offset=activations[name]["abs_offset"])


class Qwen3TTSTextProjectionTest(unittest.TestCase):
    def test_pinned_first_block_capture_has_connected_residuals(self):
        folder = ROOT / "tests/fixtures/tts/qwen3_tts_0b6_customvoice"
        manifest = json.loads((folder / "qwen3_tts_first_block.json").read_text())
        fixture_path = folder / "qwen3_tts_first_block.npz"
        self.assertEqual(hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        with np.load(fixture_path) as fixture:
            for call, tokens in ((0, 18), (1, 1)):
                prefix = f"call{call}_"
                value = lambda name: fixture[prefix + name]
                self.assertEqual(value("input").shape, (1, tokens, 1024))
                self.assertEqual(value("q_norm").shape, (1, tokens, 16, 128))
                self.assertEqual(value("k_norm").shape, (1, tokens, 8, 128))
                self.assertTrue(np.array_equal(value("cos")[0], value("cos")[1]))
                self.assertTrue(np.array_equal(value("cos")[0], value("cos")[2]))
                self.assertTrue(np.array_equal(value("sin")[0], value("sin")[1]))
                self.assertTrue(np.array_equal(value("sin")[0], value("sin")[2]))
                for source_name, rotated_name in (("q_norm", "q_after_rope"),
                                                  ("k_norm", "k_after_rope")):
                    source = value(source_name).transpose(0, 2, 1, 3)
                    cos = value("cos")[0, :, :, :][..., None, :, :]
                    sin = value("sin")[0, :, :, :][..., None, :, :]
                    half = source.shape[-1] // 2
                    rotated_half = np.concatenate(
                        (-source[..., half:], source[..., :half]), axis=-1)
                    expected_rope = source * cos + rotated_half * sin
                    np.testing.assert_allclose(value(rotated_name), expected_rope,
                                               atol=2e-6, rtol=2e-6)
                expected = (value("input") + value("attention") + value("down_proj"))
                np.testing.assert_allclose(value("output"), expected,
                                           atol=2e-6, rtol=2e-6)
            np.testing.assert_array_equal(fixture["call0_position_ids"].ravel(),
                                          np.arange(18))
            np.testing.assert_array_equal(fixture["call1_position_ids"].ravel(),
                                          [18])

    def test_generated_synthetic_projection_and_rejection(self):
        rng = np.random.default_rng(20261022)
        arrays = {
            WEIGHTS[0]: rng.normal(size=(11, 8)).astype(np.float32),
            WEIGHTS[1]: rng.normal(size=(8, 8)).astype(np.float32) * .1,
            WEIGHTS[2]: rng.normal(size=(8,)).astype(np.float32) * .1,
            WEIGHTS[3]: rng.normal(size=(4, 8)).astype(np.float32) * .1,
            WEIGHTS[4]: rng.normal(size=(4,)).astype(np.float32) * .1,
        }
        with tempfile.TemporaryDirectory() as temp:
            fn, arena, acts, calls, _ = generated_entry(Path(temp), arrays, 3)
            self.assertEqual([op["function"] for op in calls["operations"]], [
                "audio_text_embedding_channel_major_f32",
                "transpose_strided_f32_checked", "linear_rows_checked_f32",
                "silu_strided_f32_checked", "linear_rows_checked_f32"])
            ids = vector(arena, acts, "text_ids", np.int32, (3,))
            ids[:] = [1, 7, 3]
            self.assertEqual(fn(arena, len(arena)), 0)
            selected = arrays[WEIGHTS[0]][ids]
            fc1 = (selected.astype(np.float64) @ arrays[WEIGHTS[1]].T.astype(np.float64)
                   + arrays[WEIGHTS[2]]).astype(np.float32)
            active = (fc1 / (1 + np.exp(-fc1))).astype(np.float32)
            expected = (active.astype(np.float64) @ arrays[WEIGHTS[3]].T.astype(np.float64)
                        + arrays[WEIGHTS[4]]).astype(np.float32)
            result = vector(arena, acts, "projection_output", np.float32, (3, 4))
            np.testing.assert_allclose(result, expected, atol=3e-7, rtol=3e-7)
            for index, bad_id in ((0, -1), (2, 11)):
                ids[:] = [1, 7, 3]
                ids[index] = bad_id
                result[:] = -777.0
                self.assertNotEqual(fn(arena, len(arena)), 0)
                self.assertTrue(np.all(result == -777.0))
            ids[:] = [1, 7, 3]
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_allclose(result, expected, atol=3e-7, rtol=3e-7)
            result[:] = -777.0
            self.assertNotEqual(fn(arena, len(arena) - 1), 0)
            self.assertTrue(np.all(result == -777.0))
            fc1_weight = next(item for item in calls["memory"]["weights"]["entries"]
                              if item["name"] == WEIGHTS[1])
            weight = np.ndarray((8, 8), dtype=np.float32, buffer=arena,
                                offset=fc1_weight["abs_offset"])
            original = float(weight[0, 0])
            weight[0, 0] = original + 1.0
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertGreater(float(np.max(np.abs(result - expected))), 1e-3)
            weight[0, 0] = original
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_allclose(result, expected, atol=3e-7, rtol=3e-7)

    def test_single_and_five_token_generated_entries(self):
        rng = np.random.default_rng(77)
        arrays = {
            WEIGHTS[0]: rng.normal(size=(9, 4)).astype(np.float32),
            WEIGHTS[1]: rng.normal(size=(4, 4)).astype(np.float32) * .1,
            WEIGHTS[2]: rng.normal(size=4).astype(np.float32) * .1,
            WEIGHTS[3]: rng.normal(size=(3, 4)).astype(np.float32) * .1,
            WEIGHTS[4]: rng.normal(size=3).astype(np.float32) * .1,
        }
        for count in (1, 5):
            with self.subTest(tokens=count), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, _ = generated_entry(Path(temp), arrays, count)
                ids = vector(arena, acts, "text_ids", np.int32, (count,))
                ids[:] = np.arange(count)
                self.assertEqual(fn(arena, len(arena)), 0)
                result = vector(arena, acts, "projection_output", np.float32,
                                (count, 3))
                selected = arrays[WEIGHTS[0]][ids]
                fc1 = (selected.astype(np.float64) @ arrays[WEIGHTS[1]].T.astype(np.float64)
                       + arrays[WEIGHTS[2]]).astype(np.float32)
                active = (fc1 / (1 + np.exp(-fc1))).astype(np.float32)
                expected = (active.astype(np.float64) @ arrays[WEIGHTS[3]].T.astype(np.float64)
                            + arrays[WEIGHTS[4]]).astype(np.float32)
                np.testing.assert_allclose(result, expected, atol=3e-7, rtol=3e-7)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned Qwen3-TTS weights unavailable; real artifact NOT_TESTED")
    def test_pinned_customvoice_projection(self):
        from safetensors import safe_open
        from capture_qwen3_tts_customvoice_oracle import verify_model_dir
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        fixture = np.load(ROOT / "tests/fixtures/tts/qwen3_tts_0b6_customvoice"
                          / "qwen3_tts_customvoice_reference.npz")
        arrays = {}
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
            for name in WEIGHTS:
                arrays[name] = source.get_tensor(name).float().numpy().copy()
        full_tokens = int(fixture["context_token_ids"].shape[-1])
        for tokens in (3, full_tokens):
            with self.subTest(tokens=tokens), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(Path(temp), arrays, tokens)
                ids = vector(arena, acts, "text_ids", np.int32, (tokens,))
                ids[:] = fixture["context_token_ids"].reshape(-1)[:tokens]
                self.assertEqual(fn(arena, len(arena)), 0)
                embedding = vector(arena, acts, "embedding_token", np.float32,
                                   (tokens, arrays[WEIGHTS[0]].shape[1]))
                np.testing.assert_array_equal(
                    embedding, fixture["context_projection_input"].reshape(
                        full_tokens, embedding.shape[1])[:tokens])
                result = vector(arena, acts, "projection_output", np.float32,
                                (tokens, arrays[WEIGHTS[3]].shape[0]))
                reference = fixture["context_projection_output"].reshape(
                    full_tokens, result.shape[1])[:tokens]
                error = float(np.max(np.abs(result - reference)))
                self.assertLessEqual(error, 3e-4)
                print("QWEN3_TTS_GENERATED_PROJECTION " + json.dumps({
                    "scope": "text IDs through two-layer projection only",
                    "tokens": tokens, "max_abs_error": error,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "model_revision": model_dir.name,
                    "complete_talker_block": "NOT_TESTED",
                    "waveform": "NOT_TESTED"}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
