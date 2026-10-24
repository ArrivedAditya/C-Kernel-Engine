"""Generated Qwen3-TTS codec through the second complete ConvNeXt block."""

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
from tests.test_v8_qwen3_tts_codec_convnext_complete import (
    WEIGHT_NAMES as FIRST_WEIGHT_NAMES, UPSAMPLE_WEIGHT, UPSAMPLE_BIAS,
    DEPTHWISE_WEIGHT, DEPTHWISE_BIAS, add_synthetic_convnext_weights,
)
from tests.test_v8_qwen3_tts_codec_second_upsample import (
    WEIGHT as SECOND_UPSAMPLE_WEIGHT, BIAS as SECOND_UPSAMPLE_BIAS,
)
from version.v8.tts.build_qwen3_tts_codec_second_convnext_circuit import (
    build_circuit, FIRST_OPS,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_second_convnext.npz")
PREFIX = "codec.decoder.upsample.1.1"
FIRST_PREFIX = "codec.decoder.upsample.0.1"
SECOND_WEIGHT_NAMES = tuple(
    name.replace(FIRST_PREFIX, PREFIX)
    for name in (DEPTHWISE_WEIGHT, DEPTHWISE_BIAS, *FIRST_WEIGHT_NAMES))


def add_synthetic_second_weights(arrays):
    for first, second in zip(
            (DEPTHWISE_WEIGHT, DEPTHWISE_BIAS, *FIRST_WEIGHT_NAMES),
            SECOND_WEIGHT_NAMES):
        arrays[second] = np.ascontiguousarray(
            arrays[first] * np.float32(1.17))


class Qwen3TTSCodecSecondConvNeXtTest(unittest.TestCase):
    def test_authored_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_second_convnext_bounded.json"
        graph = build_circuit()
        self.assertEqual(json.loads(path.read_text()), graph)
        ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
        self.assertEqual(tuple(op["id"] for op in ops[-10:]),
                         tuple("second_" + name for name in FIRST_OPS))
        self.assertEqual(ops[-10]["graph_slots"]["inputs"]["input"],
                         "second_upsample")
        self.assertEqual(ops[-1]["graph_slots"]["inputs"]["left"],
                         "second_upsample")
        self.assertEqual(ops[-9]["weight_refs"]["weight"],
                         f"{PREFIX}.dwconv.conv.weight")
        self.assertEqual(ops[-1]["graph_slots"]["outputs"]["output"],
                         "second_convnext_output")
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        prior = MODEL.with_name("qwen3_tts_codec_second_upsample.npz")
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         manifest["source_second_upsample_sha256"])
        prior_manifest = json.loads(prior.with_suffix(".json").read_text())
        for identity in ("model_revision", "reference_revision", "codec_source_sha256"):
            self.assertEqual(manifest[identity], prior_manifest[identity], identity)
        self.assertLessEqual(manifest["source_second_upsample_max_abs"], 1e-7)
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files), {
                "second_upsample", "second_depthwise_output",
                "second_convnext_norm", "second_convnext_pw1",
                "second_convnext_gelu", "second_convnext_pw2",
                "second_convnext_output"})
            self.assertEqual(oracle["second_convnext_output"].shape, (1024, 184))
            self.assertTrue(np.isfinite(oracle["second_convnext_output"]).all())

    def test_generated_small_second_block_and_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = synthetic_weights(rng, vocab=12)
        add_synthetic_weights(arrays, rng, 2)
        add_synthetic_convnext_weights(arrays, rng)
        arrays[SECOND_UPSAMPLE_WEIGHT] = (
            rng.normal(size=(3, 3, 2)) * .1).astype(np.float32)
        arrays[SECOND_UPSAMPLE_BIAS] = (
            rng.normal(size=3) * .1).astype(np.float32)
        add_synthetic_second_weights(arrays)
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    second_convnext_complete=True, convnext_expansion=2,
                    transformer_layers=2, qkv_heads=2, qkv_head_dim=2,
                    sliding_window=3)
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                length = frames * 4
                depthwise = view(arena, acts, "second_depthwise_output",
                                 np.float32, (3, length))
                token = view(arena, acts, "second_convnext_depthwise_token",
                             np.float32, (length, 3))
                np.testing.assert_array_equal(token, depthwise.T)
                mean = token.astype(np.float64).mean(axis=1, keepdims=True)
                variance = np.mean(
                    (token.astype(np.float64) - mean) ** 2, axis=1,
                    keepdims=True)
                expected_norm = (token - mean) / np.sqrt(variance + 1e-6)
                expected_norm = (expected_norm *
                    arrays[f"{PREFIX}.norm.weight"] +
                    arrays[f"{PREFIX}.norm.bias"])
                norm = view(arena, acts, "second_convnext_norm", np.float32,
                            (length, 3))
                np.testing.assert_allclose(norm, expected_norm,
                                           atol=2e-5, rtol=2e-5)
                pw1 = view(arena, acts, "second_convnext_pw1", np.float32,
                           (length, 6))
                np.testing.assert_allclose(
                    pw1, norm @ arrays[f"{PREFIX}.pwconv1.weight"].T +
                    arrays[f"{PREFIX}.pwconv1.bias"], atol=2e-6, rtol=2e-6)
                gelu = view(arena, acts, "second_convnext_gelu", np.float32,
                            (length, 6))
                expected_gelu = .5 * pw1 * (1 + np.vectorize(math.erf)(
                    pw1.astype(np.float64) / math.sqrt(2)))
                np.testing.assert_allclose(gelu, expected_gelu,
                                           atol=2e-7, rtol=2e-7)
                pw2 = view(arena, acts, "second_convnext_pw2", np.float32,
                           (length, 3))
                np.testing.assert_allclose(
                    pw2, gelu @ arrays[f"{PREFIX}.pwconv2.weight"].T +
                    arrays[f"{PREFIX}.pwconv2.bias"], atol=2e-6, rtol=2e-6)
                scaled = view(arena, acts, "second_convnext_scaled", np.float32,
                              (length, 3))
                np.testing.assert_allclose(
                    scaled, pw2 * arrays[f"{PREFIX}.gamma"],
                    atol=2e-7, rtol=2e-7)
                upsample = view(arena, acts, "second_upsample", np.float32,
                                (3, length))
                output = view(arena, acts, "second_convnext_output", np.float32,
                              (3, length))
                np.testing.assert_allclose(output, upsample + scaled.T,
                                           atol=2e-7, rtol=2e-7)
                baseline = output.copy()
                first = view(arena, acts, "convnext_output", np.float32,
                             (3, frames * 2)).copy()
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
                np.testing.assert_array_equal(
                    view(arena, acts, "convnext_output", np.float32,
                         (3, frames * 2)), first)
                np.testing.assert_array_equal(output[1:], baseline[1:])
                self.assertGreater(float(np.max(np.abs(
                    output[0] - baseline[0]))), 1e-8)
                gamma[:] = saved_gamma
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; full-model second ConvNeXt NOT_TESTED")
    def test_pinned_connected_second_block(self):
        from safetensors import safe_open
        import torch

        torch.set_num_threads(4)
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_pretransformer_weights(model_dir)
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for name in (UPSAMPLE_WEIGHT, UPSAMPLE_BIAS,
                         DEPTHWISE_WEIGHT, DEPTHWISE_BIAS,
                         *FIRST_WEIGHT_NAMES,
                         SECOND_UPSAMPLE_WEIGHT, SECOND_UPSAMPLE_BIAS,
                         *SECOND_WEIGHT_NAMES):
                arrays[name] = np.ascontiguousarray(
                    source.get_tensor(name.removeprefix("codec.")).float().numpy())
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(
                    Path(temp), arrays, len(codes), acoustic_codebooks=15,
                    second_convnext_complete=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                errors, rmses, worst = {}, {}, {}
                for name in oracle.files:
                    expected = oracle[name]
                    actual = view(arena, acts, name, np.float32, expected.shape)
                    diff = np.abs(actual - expected)
                    errors[name] = float(np.max(diff))
                    rmses[name] = float(np.sqrt(np.mean(np.square(
                        actual.astype(np.float64) - expected.astype(np.float64)))))
                    index = np.unravel_index(int(np.argmax(diff)), diff.shape)
                    worst[name] = {"index": list(map(int, index)),
                                   "actual": float(actual[index]),
                                   "reference": float(expected[index])}
                    self.assertTrue(np.isfinite(actual).all(), name)
                source_value = view(arena, acts, "second_upsample",
                                    np.float32, oracle["second_upsample"].shape)
                depthwise = view(arena, acts, "second_depthwise_output",
                                 np.float32, oracle["second_depthwise_output"].shape)
                norm_input = view(arena, acts, "second_convnext_depthwise_token",
                                  np.float32, oracle["second_convnext_norm"].shape)
                norm = view(arena, acts, "second_convnext_norm",
                            np.float32, oracle["second_convnext_norm"].shape)
                pw1 = view(arena, acts, "second_convnext_pw1",
                           np.float32, oracle["second_convnext_pw1"].shape)
                gelu = view(arena, acts, "second_convnext_gelu",
                            np.float32, oracle["second_convnext_gelu"].shape)
                pw2 = view(arena, acts, "second_convnext_pw2",
                           np.float32, oracle["second_convnext_pw2"].shape)
                with torch.inference_mode():
                    same_depthwise = torch.nn.functional.conv1d(
                        torch.nn.functional.pad(
                            torch.from_numpy(source_value.copy()[None]), (6, 0)),
                        torch.from_numpy(arrays[f"{PREFIX}.dwconv.conv.weight"]),
                        torch.from_numpy(arrays[f"{PREFIX}.dwconv.conv.bias"]),
                        groups=1024)[0].numpy()
                    same_norm = torch.nn.functional.layer_norm(
                        torch.from_numpy(norm_input.copy()), (1024,),
                        torch.from_numpy(arrays[f"{PREFIX}.norm.weight"]),
                        torch.from_numpy(arrays[f"{PREFIX}.norm.bias"]),
                        eps=1e-6).numpy()
                    same_pw1 = torch.nn.functional.linear(
                        torch.from_numpy(norm.copy()),
                        torch.from_numpy(arrays[f"{PREFIX}.pwconv1.weight"]),
                        torch.from_numpy(arrays[f"{PREFIX}.pwconv1.bias"])).numpy()
                    same_gelu = torch.nn.functional.gelu(
                        torch.from_numpy(pw1.copy()), approximate="none").numpy()
                    same_pw2 = torch.nn.functional.linear(
                        torch.from_numpy(gelu.copy()),
                        torch.from_numpy(arrays[f"{PREFIX}.pwconv2.weight"]),
                        torch.from_numpy(arrays[f"{PREFIX}.pwconv2.bias"])).numpy()
                local = {
                    "second_depthwise_output": float(np.max(np.abs(
                        depthwise - same_depthwise))),
                    "second_convnext_norm": float(np.max(np.abs(norm - same_norm))),
                    "second_convnext_pw1": float(np.max(np.abs(pw1 - same_pw1))),
                    "second_convnext_gelu": float(np.max(np.abs(gelu - same_gelu))),
                    "second_convnext_pw2": float(np.max(np.abs(pw2 - same_pw2))),
                }
                scaled = view(arena, acts, "second_convnext_scaled",
                              np.float32, oracle["second_convnext_norm"].shape)
                output = view(arena, acts, "second_convnext_output",
                              np.float32, oracle["second_convnext_output"].shape)
                join_error = float(np.max(np.abs(
                    output - (source_value + scaled.T))))
                print("QWEN3_TTS_CODEC_SECOND_CONVNEXT " + json.dumps({
                    "max_abs_errors": errors, "rmse": rmses, "worst": worst,
                    "same_input_local_max_abs": local,
                    "residual_same_input_max_abs": join_error,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codec codes through second complete ConvNeXt; remaining decoder and waveform NOT_TESTED",
                }, sort_keys=True))
                for name, limit in LIMITS.items():
                    self.assertLessEqual(errors[name], limit, name)
                for name, limit in LOCAL_LIMITS.items():
                    self.assertLessEqual(local[name], limit, name)
                self.assertLessEqual(join_error, 1e-6)


LIMITS = {
    "second_upsample": 8e-5,
    # The depthwise operation is exact on identical operands in this capture;
    # its larger connected error is propagated from the second upsample.
    "second_depthwise_output": 3e-4,
    "second_convnext_norm": 2e-5,
    "second_convnext_pw1": 1.2e-4,
    "second_convnext_gelu": 1e-4,
    "second_convnext_pw2": 1.5e-4,
    "second_convnext_output": 6e-5,
}
LOCAL_LIMITS = {
    "second_depthwise_output": 1e-5,
    "second_convnext_norm": 1e-5,
    "second_convnext_pw1": 2e-5,
    "second_convnext_gelu": 1e-6,
    "second_convnext_pw2": 3.5e-5,
}


if __name__ == "__main__":
    unittest.main()
