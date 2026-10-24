"""Generated first complete ConvNeXt block in the Qwen3-TTS codec decoder."""

import ctypes
import hashlib
import json
import math
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
from tests.test_v8_qwen3_tts_codec_depthwise import WEIGHT as DEPTHWISE_WEIGHT, BIAS as DEPTHWISE_BIAS
from version.v8.tts.build_qwen3_tts_codec_convnext_complete_circuit import build_circuit, PREFIX

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_convnext_complete.npz")
WEIGHT_NAMES = (
    f"{PREFIX}.norm.weight", f"{PREFIX}.norm.bias",
    f"{PREFIX}.pwconv1.weight", f"{PREFIX}.pwconv1.bias",
    f"{PREFIX}.pwconv2.weight", f"{PREFIX}.pwconv2.bias",
    f"{PREFIX}.gamma",
)


def add_synthetic_convnext_weights(arrays, rng):
    arrays[UPSAMPLE_WEIGHT] = (rng.normal(size=(3, 3, 2)) * .1).astype(np.float32)
    arrays[UPSAMPLE_BIAS] = (rng.normal(size=3) * .1).astype(np.float32)
    arrays[DEPTHWISE_WEIGHT] = (rng.normal(size=(3, 1, 7)) * .1).astype(np.float32)
    arrays[DEPTHWISE_BIAS] = (rng.normal(size=3) * .1).astype(np.float32)
    arrays[f"{PREFIX}.norm.weight"] = np.array([.8, 1.1, 1.2], dtype=np.float32)
    arrays[f"{PREFIX}.norm.bias"] = np.array([.1, -.2, .3], dtype=np.float32)
    arrays[f"{PREFIX}.pwconv1.weight"] = (rng.normal(size=(6, 3)) * .1).astype(np.float32)
    arrays[f"{PREFIX}.pwconv1.bias"] = (rng.normal(size=6) * .1).astype(np.float32)
    arrays[f"{PREFIX}.pwconv2.weight"] = (rng.normal(size=(3, 6)) * .1).astype(np.float32)
    arrays[f"{PREFIX}.pwconv2.bias"] = (rng.normal(size=3) * .1).astype(np.float32)
    arrays[f"{PREFIX}.gamma"] = np.array([.2, -.1, .3], dtype=np.float32)


class Qwen3TTSCodecConvNextCompleteTest(unittest.TestCase):
    def test_authored_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_convnext_complete_bounded.json"
        self.assertEqual(json.loads(path.read_text()), build_circuit())
        graph = build_circuit()
        ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
        self.assertEqual([op["id"] for op in ops[-8:]], [
            "convnext_depthwise_token", "convnext_norm", "convnext_pw1",
            "convnext_gelu", "convnext_pw2", "convnext_scaled",
            "convnext_scaled_channel", "convnext_output"])
        self.assertEqual(ops[-5]["kernel"], "gelu_rows_erf_checked_f32")
        self.assertEqual(ops[-7]["params"]["epsilon"], 1e-6)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        prior = MODEL.with_name("qwen3_tts_codec_depthwise.npz")
        prior_manifest = json.loads(prior.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         manifest["source_depthwise_sha256"])
        for identity in ("model_revision", "reference_revision", "codec_source_sha256"):
            self.assertEqual(manifest[identity], prior_manifest[identity], identity)
        self.assertLessEqual(max(manifest["source_depthwise_max_abs"].values()), 1e-7)
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files), {
                "first_upsample", "depthwise_output", "convnext_norm",
                "convnext_pw1", "convnext_gelu", "convnext_pw2",
                "convnext_output"})
            self.assertEqual(oracle["convnext_output"].shape, (1024, 92))
            self.assertTrue(np.isfinite(oracle["convnext_output"]).all())

    def test_generated_small_block_and_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = synthetic_weights(rng, vocab=12)
        add_synthetic_weights(arrays, rng, 2)
        add_synthetic_convnext_weights(arrays, rng)
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, library = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    first_convnext_complete=True, convnext_expansion=2,
                    transformer_layers=2, qkv_heads=2, qkv_head_dim=2,
                    sliding_window=3)
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                length = frames * 2
                depthwise = view(arena, acts, "depthwise_output", np.float32,
                                 (3, length))
                depthwise_token = view(arena, acts, "convnext_depthwise_token",
                                       np.float32, (length, 3))
                np.testing.assert_array_equal(depthwise_token, depthwise.T)
                mean = depthwise_token.astype(np.float64).mean(axis=1, keepdims=True)
                variance = np.mean((depthwise_token.astype(np.float64) - mean) ** 2,
                                   axis=1, keepdims=True)
                expected_norm = (depthwise_token - mean) / np.sqrt(variance + 1e-6)
                expected_norm = (expected_norm * arrays[f"{PREFIX}.norm.weight"] +
                                 arrays[f"{PREFIX}.norm.bias"])
                norm = view(arena, acts, "convnext_norm", np.float32, (length, 3))
                np.testing.assert_allclose(norm, expected_norm, atol=2e-5, rtol=2e-5)
                pw1 = view(arena, acts, "convnext_pw1", np.float32, (length, 6))
                np.testing.assert_allclose(pw1,
                    norm @ arrays[f"{PREFIX}.pwconv1.weight"].T +
                    arrays[f"{PREFIX}.pwconv1.bias"], atol=2e-6, rtol=2e-6)
                gelu = view(arena, acts, "convnext_gelu", np.float32, (length, 6))
                expected_gelu = .5 * pw1 * (1 + np.vectorize(math.erf)(
                    pw1.astype(np.float64) / math.sqrt(2)))
                np.testing.assert_allclose(gelu, expected_gelu, atol=2e-7, rtol=2e-7)
                pw2 = view(arena, acts, "convnext_pw2", np.float32, (length, 3))
                np.testing.assert_allclose(pw2,
                    gelu @ arrays[f"{PREFIX}.pwconv2.weight"].T +
                    arrays[f"{PREFIX}.pwconv2.bias"], atol=2e-6, rtol=2e-6)
                scaled = view(arena, acts, "convnext_scaled", np.float32,
                              (length, 3))
                np.testing.assert_allclose(scaled,
                    pw2 * arrays[f"{PREFIX}.gamma"], atol=2e-7, rtol=2e-7)
                upsample = view(arena, acts, "first_upsample", np.float32,
                                (3, length))
                output = view(arena, acts, "convnext_output", np.float32,
                              (3, length))
                np.testing.assert_allclose(output, upsample + scaled.T,
                                           atol=2e-7, rtol=2e-7)
                baseline = output.copy()
                entry = next(item for item in calls["memory"]["weights"]["entries"]
                             if item["name"] == f"{PREFIX}.pwconv2.weight")
                bound = np.ndarray(arrays[f"{PREFIX}.pwconv2.weight"].shape,
                                   dtype=np.float32, buffer=arena,
                                   offset=entry["abs_offset"])
                saved = bound.copy()
                bound[0, 0] = np.nan
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, -123)
                bound[:] = saved
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                np.testing.assert_array_equal(output, -123)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)
                gamma_entry = next(item for item in calls["memory"]["weights"]["entries"]
                                   if item["name"] == f"{PREFIX}.gamma")
                gamma = np.ndarray((3,), dtype=np.float32, buffer=arena,
                                   offset=gamma_entry["abs_offset"])
                saved_gamma = gamma.copy()
                gamma[0] = 0
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output[1:], baseline[1:])
                self.assertGreater(float(np.max(np.abs(output[0] - baseline[0]))), 1e-8)
                gamma[:] = saved_gamma
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)
                if frames == 1:
                    provider = ctypes.CDLL(str(library)).gelu_rows_erf_checked_f32
                    pointer = ctypes.POINTER(ctypes.c_float)
                    provider.argtypes = [pointer, ctypes.c_size_t, ctypes.c_size_t,
                                         pointer, ctypes.c_size_t, ctypes.c_size_t,
                                         pointer, ctypes.c_size_t, ctypes.c_size_t,
                                         ctypes.c_size_t]
                    provider.restype = ctypes.c_int
                    inp = np.array([[-2, -.5, 0, 99], [.5, 2, 3, 99]],
                                   dtype=np.float32)
                    out = np.full((2, 4), -123, dtype=np.float32)
                    scratch = np.zeros(6, dtype=np.float32)

                    def ptr(value):
                        return value.ctypes.data_as(pointer)

                    def invoke(input_value=inp, output_value=out,
                               output_elements=8):
                        return provider(ptr(input_value), input_value.size, 4,
                                        ptr(output_value), output_elements, 4,
                                        ptr(scratch), scratch.size, 2, 3)

                    self.assertNotEqual(invoke(output_elements=6), 0)
                    np.testing.assert_array_equal(out, -123)
                    broken = inp.copy()
                    broken[0, 0] = np.nan
                    self.assertNotEqual(invoke(input_value=broken), 0)
                    np.testing.assert_array_equal(out, -123)
                    self.assertNotEqual(invoke(output_value=inp), 0)
                    self.assertEqual(invoke(), 0)
                    expected = .5 * inp[:, :3] * (1 + np.vectorize(math.erf)(
                        inp[:, :3].astype(np.float64) / math.sqrt(2)))
                    np.testing.assert_allclose(out[:, :3], expected,
                                               atol=1e-7, rtol=1e-7)
                    np.testing.assert_array_equal(out[:, 3:], -123)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; direct-model ConvNeXt parity NOT_TESTED")
    def test_pinned_connected_complete_block(self):
        from safetensors import safe_open
        import torch

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_pretransformer_weights(model_dir)
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for name in (UPSAMPLE_WEIGHT, UPSAMPLE_BIAS,
                         DEPTHWISE_WEIGHT, DEPTHWISE_BIAS, *WEIGHT_NAMES):
                arrays[name] = np.ascontiguousarray(
                    source.get_tensor(name.removeprefix("codec.")).float().numpy())
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(
                    Path(temp), arrays, len(codes), acoustic_codebooks=15,
                    first_convnext_complete=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                errors = {}
                rmses = {}
                worst = {}
                for name in oracle.files:
                    actual = view(arena, acts, name, np.float32, oracle[name].shape)
                    expected = oracle[name]
                    diff = np.abs(actual - expected)
                    errors[name] = float(np.max(diff))
                    rmses[name] = float(np.sqrt(np.mean(np.square(
                        actual.astype(np.float64) - expected.astype(np.float64)))))
                    index = np.unravel_index(int(np.argmax(diff)), diff.shape)
                    worst[name] = {"index": list(map(int, index)),
                                   "actual": float(actual[index]),
                                   "reference": float(expected[index])}
                    self.assertTrue(np.isfinite(actual).all(), name)
                actual_depthwise = view(arena, acts, "convnext_depthwise_token",
                                        np.float32, oracle["convnext_norm"].shape)
                actual_input = view(arena, acts, "convnext_norm", np.float32,
                                    oracle["convnext_norm"].shape)
                actual_pw1 = view(arena, acts, "convnext_pw1", np.float32,
                                  oracle["convnext_pw1"].shape)
                actual_gelu = view(arena, acts, "convnext_gelu", np.float32,
                                   oracle["convnext_gelu"].shape)
                actual_pw2 = view(arena, acts, "convnext_pw2", np.float32,
                                  oracle["convnext_pw2"].shape)
                with torch.inference_mode():
                    same_input_norm = torch.nn.functional.layer_norm(
                        torch.from_numpy(actual_depthwise.copy()), (1024,),
                        torch.from_numpy(arrays[f"{PREFIX}.norm.weight"]),
                        torch.from_numpy(arrays[f"{PREFIX}.norm.bias"]),
                        eps=1e-6).numpy()
                    same_input_pw1 = torch.nn.functional.linear(
                        torch.from_numpy(actual_input.copy()),
                        torch.from_numpy(arrays[f"{PREFIX}.pwconv1.weight"]),
                        torch.from_numpy(arrays[f"{PREFIX}.pwconv1.bias"])).numpy()
                    same_input_gelu = torch.nn.functional.gelu(
                        torch.from_numpy(actual_pw1.copy()), approximate="none").numpy()
                    same_input_pw2 = torch.nn.functional.linear(
                        torch.from_numpy(actual_gelu.copy()),
                        torch.from_numpy(arrays[f"{PREFIX}.pwconv2.weight"]),
                        torch.from_numpy(arrays[f"{PREFIX}.pwconv2.bias"])).numpy()
                same_input_errors = {
                    "convnext_norm": float(np.max(np.abs(
                        actual_input - same_input_norm))),
                    "convnext_pw1": float(np.max(np.abs(
                        actual_pw1 - same_input_pw1))),
                    "convnext_gelu": float(np.max(np.abs(
                        actual_gelu - same_input_gelu))),
                    "convnext_pw2": float(np.max(np.abs(
                        actual_pw2 - same_input_pw2))),
                }
                actual_scaled = view(arena, acts, "convnext_scaled", np.float32,
                                     oracle["convnext_norm"].shape)
                actual_upsample = view(arena, acts, "first_upsample", np.float32,
                                       oracle["first_upsample"].shape)
                actual_output = view(arena, acts, "convnext_output", np.float32,
                                     oracle["convnext_output"].shape)
                residual_error = float(np.max(np.abs(
                    actual_output - (actual_upsample + actual_scaled.T))))
                print("QWEN3_TTS_CODEC_CONVNEXT_COMPLETE " + json.dumps({
                    "max_abs_errors": errors, "rmse": rmses, "worst": worst,
                    "same_input_errors": same_input_errors,
                    "residual_same_input_max_abs": residual_error,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codec codes through first complete ConvNeXt; remaining vocoder and waveform NOT_TESTED",
                }, sort_keys=True))
                for name, limit in LIMITS.items():
                    self.assertLessEqual(errors[name], limit, name)
                for name, limit in LOCAL_LIMITS.items():
                    self.assertLessEqual(same_input_errors[name], limit, name)
                self.assertLessEqual(residual_error, 1e-6)


LIMITS = {
    "first_upsample": 6e-6,
    "depthwise_output": 2.5e-5,
    "convnext_norm": 1e-5,
    "convnext_pw1": 4e-5,
    "convnext_gelu": 4e-5,
    "convnext_pw2": 8e-5,
    "convnext_output": 6e-6,
}
LOCAL_LIMITS = {
    "convnext_norm": 1e-5,
    "convnext_pw1": 1e-5,
    "convnext_gelu": 1e-6,
    "convnext_pw2": 2e-5,
}


if __name__ == "__main__":
    unittest.main()
