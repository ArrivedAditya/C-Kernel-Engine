"""Generated causal depthwise stage of Qwen3-TTS's first codec ConvNeXt block."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_codec_semantic import MODEL, generated_entry, view
from tests.test_v8_qwen3_tts_codec_preconv import fill_ids
from tests.test_v8_qwen3_tts_codec_attention import synthetic_weights
from tests.test_v8_qwen3_tts_codec_pretransformer import (
    add_synthetic_weights, pinned_pretransformer_weights,
)
from tests.test_v8_qwen3_tts_codec_first_upsample import WEIGHT as UPSAMPLE_WEIGHT, BIAS as UPSAMPLE_BIAS
from version.v8.tts.build_qwen3_tts_codec_depthwise_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_depthwise.npz")
WEIGHT = "codec.decoder.upsample.0.1.dwconv.conv.weight"
BIAS = "codec.decoder.upsample.0.1.dwconv.conv.bias"


class Qwen3TTSCodecDepthwiseTest(unittest.TestCase):
    def test_checked_provider_strides_capacity_and_staged_commit(self):
        source = ROOT / "src/kernels/audio_conv1d_depthwise_checked.c"
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "depthwise.so"
            import subprocess
            subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                            "-Isrc/kernels", str(source), "-lm", "-o", str(library)],
                           check=True, cwd=ROOT, capture_output=True)
            function = ctypes.CDLL(str(library)).audio_conv1d_depthwise_channel_major_f32_checked
            pointer = ctypes.POINTER(ctypes.c_float)
            function.argtypes = [pointer, ctypes.c_size_t, ctypes.c_size_t,
                                 pointer, ctypes.c_size_t, pointer, ctypes.c_size_t,
                                 pointer, ctypes.c_size_t, ctypes.c_size_t,
                                 pointer, ctypes.c_size_t, ctypes.c_size_t,
                                 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                 ctypes.c_size_t]
            function.restype = ctypes.c_int
            inp = np.array([[1, 2, 3, 4, 99], [5, 6, 7, 8, 99]], dtype=np.float32)
            weight = np.array([[1, 2], [-1, 3]], dtype=np.float32)
            bias = np.array([.5, -.5], dtype=np.float32)
            out = np.full((2, 5), -123, dtype=np.float32)
            scratch = np.zeros((2, 3), dtype=np.float32)

            def ptr(value):
                return value.ctypes.data_as(pointer)

            def invoke(output_elements=10, weight_value=weight, output_value=out):
                return function(ptr(inp), inp.size, 5, ptr(weight_value),
                                weight_value.size, ptr(bias), bias.size,
                                ptr(output_value), output_elements, 5,
                                ptr(scratch), scratch.size, 2, 4, 2, 1, 3)

            self.assertNotEqual(invoke(output_elements=7), 0)
            np.testing.assert_array_equal(out, -123)
            broken = weight.copy()
            broken[0, 0] = np.inf
            self.assertNotEqual(invoke(weight_value=broken), 0)
            np.testing.assert_array_equal(out, -123)
            finite_overflow = np.full_like(weight, 1e30)
            large_input = inp.copy()
            large_input[:, :4] = 1e30
            original_input = inp.copy()
            inp[:] = large_input
            self.assertNotEqual(invoke(weight_value=finite_overflow), 0)
            np.testing.assert_array_equal(out, -123)
            inp[:] = original_input
            self.assertNotEqual(invoke(output_value=inp), 0)
            np.testing.assert_array_equal(inp[:, -1], 99)
            self.assertEqual(invoke(), 0)
            np.testing.assert_array_equal(out[:, :3],
                                          np.array([[5.5, 8.5, 11.5],
                                                    [12.5, 14.5, 16.5]], dtype=np.float32))
            np.testing.assert_array_equal(out[:, 3:], -123)

    def test_authored_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_depthwise_bounded.json"
        self.assertEqual(json.loads(path.read_text()), build_circuit())
        graph = build_circuit()
        ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
        self.assertEqual([op["id"] for op in ops[-2:]],
                         ["depthwise_left_pad", "depthwise_output"])
        self.assertEqual(ops[-1]["weight_refs"], {"weight": WEIGHT, "bias": BIAS})
        self.assertEqual(ops[-2]["params"]["call_constants"]["zero_pad_left_padding"], 6)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        prior = MODEL.with_name("qwen3_tts_codec_first_upsample.npz")
        prior_manifest = json.loads(prior.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         manifest["source_first_upsample_sha256"])
        for identity in ("model_revision", "reference_revision", "codec_source_sha256"):
            self.assertEqual(manifest[identity], prior_manifest[identity], identity)
        self.assertLessEqual(manifest["source_first_upsample_input_max_abs"], 1e-7)
        self.assertEqual((manifest["frames"], manifest["latent_width"]), (92, 1024))
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files), {"first_upsample", "depthwise_output"})
            self.assertEqual(oracle["depthwise_output"].shape, (1024, 92))
            self.assertTrue(np.isfinite(oracle["depthwise_output"]).all())

    def test_generated_depthwise_padding_rejection_and_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = synthetic_weights(rng, vocab=12)
        add_synthetic_weights(arrays, rng, 2)
        arrays[UPSAMPLE_WEIGHT] = (rng.normal(size=(3, 3, 2)) * .1).astype(np.float32)
        arrays[UPSAMPLE_BIAS] = (rng.normal(size=3) * .1).astype(np.float32)
        arrays[WEIGHT] = (rng.normal(size=(3, 1, 7)) * .1).astype(np.float32)
        arrays[BIAS] = (rng.normal(size=3) * .1).astype(np.float32)
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    first_decoder_depthwise=True, transformer_layers=2,
                    qkv_heads=2, qkv_head_dim=2, sliding_window=3)
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                length = frames * 2
                upsample = view(arena, acts, "first_upsample", np.float32, (3, length))
                padded = view(arena, acts, "depthwise_padded", np.float32,
                              (3, length + 6))
                output = view(arena, acts, "depthwise_output", np.float32,
                              (3, length))
                np.testing.assert_array_equal(padded[:, :6], 0)
                np.testing.assert_array_equal(padded[:, 6:], upsample)
                expected = np.empty_like(output)
                for channel in range(3):
                    for position in range(length):
                        expected[channel, position] = (
                            arrays[BIAS][channel] +
                            np.dot(padded[channel, position:position + 7],
                                   arrays[WEIGHT][channel, 0]))
                np.testing.assert_allclose(output, expected, atol=2e-6, rtol=2e-6)
                baseline = output.copy()
                entry = next(item for item in calls["memory"]["weights"]["entries"]
                             if item["name"] == WEIGHT)
                bound = np.ndarray(arrays[WEIGHT].shape, dtype=np.float32,
                                   buffer=arena, offset=entry["abs_offset"])
                saved = bound.copy()
                bound[0, 0, 0] = np.nan
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, -123)
                bound[:] = saved
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                np.testing.assert_array_equal(output, -123)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; direct model depthwise parity NOT_TESTED")
    def test_pinned_connected_depthwise(self):
        from safetensors import safe_open
        import torch

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_pretransformer_weights(model_dir)
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for name in (UPSAMPLE_WEIGHT, UPSAMPLE_BIAS, WEIGHT, BIAS):
                arrays[name] = np.ascontiguousarray(
                    source.get_tensor(name.removeprefix("codec.")).float().numpy())
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(
                    Path(temp), arrays, len(codes), acoustic_codebooks=15,
                    first_decoder_depthwise=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                errors = {}
                for name in ("first_upsample", "depthwise_output"):
                    expected = oracle[name]
                    actual = view(arena, acts, name, np.float32, expected.shape)
                    diff = np.abs(actual - expected)
                    errors[name] = float(np.max(diff))
                    self.assertTrue(np.isfinite(actual).all(), name)
                actual_input = view(arena, acts, "first_upsample", np.float32,
                                    oracle["first_upsample"].shape)
                actual_output = view(arena, acts, "depthwise_output", np.float32,
                                     oracle["depthwise_output"].shape)
                with torch.inference_mode():
                    same_input_oracle = torch.nn.functional.conv1d(
                        torch.nn.functional.pad(torch.from_numpy(actual_input.copy()[None]),
                                                (6, 0)),
                        torch.from_numpy(arrays[WEIGHT]),
                        torch.from_numpy(arrays[BIAS]), groups=1024)[0].numpy()
                    reference_input_oracle = torch.nn.functional.conv1d(
                        torch.nn.functional.pad(torch.from_numpy(
                            oracle["first_upsample"].copy()[None]), (6, 0)),
                        torch.from_numpy(arrays[WEIGHT]),
                        torch.from_numpy(arrays[BIAS]), groups=1024)[0].numpy()
                local_error = float(np.max(np.abs(actual_output - same_input_oracle)))
                propagated_error = float(np.max(np.abs(
                    same_input_oracle - reference_input_oracle)))
                reference_replay_error = float(np.max(np.abs(
                    reference_input_oracle - oracle["depthwise_output"])))
                print("QWEN3_TTS_CODEC_DEPTHWISE " + json.dumps({
                    "max_abs_errors": errors,
                    "same_input_local_max_abs": local_error,
                    "upstream_propagated_max_abs": propagated_error,
                    "reference_replay_max_abs": reference_replay_error,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codec codes through first ConvNeXt depthwise; full block and waveform NOT_TESTED",
                }, sort_keys=True))
                self.assertLessEqual(errors["first_upsample"], 6e-6)
                self.assertLessEqual(local_error, 1e-6)
                self.assertLessEqual(reference_replay_error, 1e-6)
                self.assertLessEqual(propagated_error, 2.5e-5)
                self.assertLessEqual(errors["depthwise_output"],
                                     local_error + propagated_error +
                                     reference_replay_error + 1e-7)


if __name__ == "__main__":
    unittest.main()
