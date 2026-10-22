"""First generated Qwen3-TTS talker block: fixed graph and pinned oracle parity."""

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
from build_qwen3_tts_first_block_circuit import build_circuit, OUTPUT

FIXTURE = ROOT / "tests/fixtures/tts/qwen3_tts_0b6_customvoice"


def weight_shapes(graph):
    shapes = {}
    for op in graph["block_types"]["first_block"]["body"]["ops"]:
        for name in op.get("weight_refs", {}).values():
            if name.endswith(("input_layernorm.weight",
                              "post_attention_layernorm.weight")):
                shape = (1024,)
            elif name.endswith(("q_norm.weight", "k_norm.weight")):
                shape = (128,)
            else:
                shape = (op["params"]["N"], op["params"]["K"])
            shapes[name] = shape
    return shapes


def generated_entry(folder: Path, arrays: dict[str, np.ndarray], tokens: int):
    graph = build_circuit(tokens)
    circuit = folder / "circuit.json"
    circuit.write_text(json.dumps(graph))
    expected = weight_shapes(graph)
    if set(arrays) != set(expected):
        raise ValueError("first-block weights do not match the circuit")
    entries = []
    file_offset = 4096
    for name, shape in expected.items():
        value = arrays[name]
        if value.dtype != np.float32 or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"invalid pinned weight: {name}")
        file_offset = (file_offset + 63) & ~63
        entries.append({"name": name, "shape": list(shape), "dtype": "F32",
                        "size": value.nbytes, "file_offset": file_offset,
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest()})
        file_offset += value.nbytes
    config = {
        "model": graph["name"], "arch": graph["name"], "num_layers": 1,
        "embed_dim": 1024, "num_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "intermediate_size": 3072,
        "context_length": tokens, "max_seq_len": tokens, "vocab_size": 3072,
    }
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
    generated = folder / "generated.c"
    subprocess.run([
        sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
        "--ir", str(folder / "call.json"),
        "--layout", str(folder / "layout.json"),
        "--output", str(generated),
    ], check=True, capture_output=True, text=True)
    library = folder / "generated.so"
    subprocess.run([
        "cc", "-std=gnu11", "-O2", "-w", "-shared", "-fPIC",
        "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
        "-Wl,--no-undefined", "-fopenmp", str(generated),
        str(ROOT / "src/kernels/transformer_rows_checked.c"),
        str(ROOT / "src/kernels/linear_rows_checked.c"),
        str(ROOT / "src/kernels/rope_kernels.c"),
        str(ROOT / "src/kernels/attention_kernels.c"),
        str(ROOT / "src/kernels/softmax_kernels.c"),
        str(ROOT / "src/kernels/attention_flash_true.c"),
        str(ROOT / "src/ck_threadpool.c"),
        str(ROOT / "src/ckernel_strict.c"),
        "-I", str(ROOT / "include"), "-lm", "-ldl", "-lpthread",
        "-o", str(library),
    ], check=True, capture_output=True, text=True)
    loaded = ctypes.CDLL(str(library))
    function = loaded.ck_qwen3_tts_first_block_bounded
    function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    function.restype = ctypes.c_int
    size = layout["memory"]["arena"]["total_size"]
    raw = (ctypes.c_uint8 * (size + 63))()
    offset = (-ctypes.addressof(raw)) & 63
    arena = (ctypes.c_uint8 * size).from_buffer(raw, offset)
    planned_weights = {item["name"]: item for item in
                       layout["memory"]["weights"]["entries"]}
    for name, planned in planned_weights.items():
        value = arrays[name]
        start = planned["abs_offset"]
        arena[start:start + value.nbytes] = value.tobytes()
    activations = {item["name"]: item for item in
                   layout["memory"]["activations"]["buffers"]}
    return function, arena, activations, call_ir, library, layout


def vector(arena, activations, name, shape):
    return np.ndarray(shape, dtype=np.float32, buffer=arena,
                      offset=activations[name]["abs_offset"])


class Qwen3TTSFirstBlockTest(unittest.TestCase):
    def test_circuit_freshness_and_fail_closed_dimensions(self):
        self.assertEqual(OUTPUT.read_text(), json.dumps(build_circuit(), indent=2) + "\n")
        with self.assertRaises(ValueError):
            build_circuit(0)
        with self.assertRaises(ValueError):
            build_circuit(129)
        with self.assertRaises(ValueError):
            build_circuit(True)

    def test_generated_zero_projection_preserves_residual_and_rejects_capacity(self):
        graph = build_circuit(1)
        arrays = {}
        for name, shape in weight_shapes(graph).items():
            arrays[name] = np.zeros(shape, dtype=np.float32)
            if name.endswith(("input_layernorm.weight",
                              "post_attention_layernorm.weight",
                              "q_norm.weight", "k_norm.weight")):
                arrays[name][:] = 1.0
        with tempfile.TemporaryDirectory() as temp:
            fn, arena, acts, calls, _, _ = generated_entry(Path(temp), arrays, 1)
            self.assertEqual(len(calls["operations"]), 20)
            source = vector(arena, acts, "block_input", (1, 1024))
            source[:] = np.linspace(-0.5, 0.5, 1024, dtype=np.float32)
            result = vector(arena, acts, "block_output", (1, 1024))
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(result, source)
            result[:] = -777.0
            self.assertNotEqual(fn(arena, len(arena) - 1), 0)
            self.assertTrue(np.all(result == -777.0))

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned Qwen3-TTS weights unavailable; native block NOT_TESTED")
    def test_pinned_first_block_prefill(self):
        from safetensors import safe_open
        from capture_qwen3_tts_customvoice_oracle import verify_model_dir
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        with np.load(FIXTURE / "qwen3_tts_first_block.npz") as fixture:
            reference = {name: fixture[f"call0_{name}"].copy() for name in (
                "input", "input_norm", "q_proj", "k_proj", "v_proj",
                "q_norm", "k_norm", "q_after_rope", "k_after_rope",
                "attention", "post_attention_norm", "gate_proj", "up_proj",
                "down_proj", "output")}
        arrays = {}
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
            for name in weight_shapes(build_circuit()):
                arrays[name] = source.get_tensor(name).float().numpy().copy()
        stages = {
            "input_norm": "input_norm", "q_proj": "q_proj",
            "k_proj": "k_proj", "v_proj": "v_proj",
            "q_norm": "q_norm", "k_norm": "k_norm",
            "q_head": "q_after_rope", "k_head": "k_after_rope",
            "attention_proj": "attention",
            "post_attention_norm": "post_attention_norm",
            "gate_proj": "gate_proj", "up_proj": "up_proj",
            "down_proj": "down_proj", "block_output": "output",
        }
        for tokens in (8, 18):
            with self.subTest(tokens=tokens), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, library, layout = generated_entry(Path(temp), arrays, tokens)
                self.assertEqual(len(calls["operations"]), 20)
                vector(arena, acts, "block_input", (tokens, 1024))[:] = (
                    reference["input"][0, :tokens])
                self.assertEqual(fn(arena, len(arena)), 0)
                errors = {}
                for name, oracle in stages.items():
                    value = reference[oracle]
                    if name in ("q_head", "k_head"):
                        target = value[0, :, :tokens, :]
                    elif name in ("q_norm", "k_norm"):
                        target = value[0, :tokens].reshape(tokens, -1)
                    else:
                        target = value[0, :tokens]
                    actual = vector(arena, acts, name, target.shape)
                    self.assertTrue(np.isfinite(actual).all(), name)
                    errors[name] = float(np.max(np.abs(actual - target)))
                print("QWEN3_TTS_FIRST_BLOCK " + json.dumps({
                    "scope": "oracle-fed first talker block; no native utterance",
                    "tokens": tokens, "errors": errors,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "model_revision": model_dir.name,
                    "full_talker": "NOT_TESTED", "waveform": "NOT_TESTED"},
                    sort_keys=True))
                self.assertLessEqual(max(errors.values()), 5e-5, errors)
                # A wrong Q projection must fail the same first-stage parity
                # gate rather than merely changing the final generated token.
                weight = next(item for item in layout["memory"]["weights"]["entries"]
                              if item["name"].endswith("self_attn.q_proj.weight"))
                q_weight = np.ndarray((2048, 1024), dtype=np.float32,
                                      buffer=arena, offset=weight["abs_offset"])
                q_weight[0] += 1.0
                self.assertEqual(fn(arena, len(arena)), 0)
                changed_q = vector(arena, acts, "q_proj", (tokens, 2048))
                self.assertGreater(float(np.max(np.abs(
                    changed_q - reference["q_proj"][0, :tokens]))), 5e-5)
                output = vector(arena, acts, "block_output", (tokens, 1024))
                output[:] = -777.0
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                self.assertTrue(np.all(output == -777.0))


if __name__ == "__main__":
    unittest.main()
