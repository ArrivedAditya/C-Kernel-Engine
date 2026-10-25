"""Generated Qwen3-TTS codec through all three first-stage residual units."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_codec_semantic import MODEL, generated_entry, view
from tests.test_v8_qwen3_tts_codec_preconv import fill_ids
from tests.test_v8_qwen3_tts_codec_pretransformer import pinned_pretransformer_weights
from tests.test_v8_qwen3_tts_codec_convnext_complete import (
    WEIGHT_NAMES as FIRST_WEIGHT_NAMES, UPSAMPLE_WEIGHT, UPSAMPLE_BIAS,
    DEPTHWISE_WEIGHT, DEPTHWISE_BIAS,
)
from tests.test_v8_qwen3_tts_codec_second_upsample import (
    WEIGHT as SECOND_UPSAMPLE_WEIGHT, BIAS as SECOND_UPSAMPLE_BIAS,
)
from tests.test_v8_qwen3_tts_codec_second_convnext import SECOND_WEIGHT_NAMES
from tests.test_v8_qwen3_tts_codec_decoder_ingress import CONV_WEIGHT, CONV_BIAS
from tests.test_v8_qwen3_tts_codec_snake_beta import ALPHA, BETA
from tests import test_v8_qwen3_tts_codec_decoder_upsample as decoder_upsample_test
from version.v8.tts.build_qwen3_tts_codec_first_residual_pool_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_first_residual_pool.npz")
DILATIONS = (1, 3, 9)


def weight_names(unit, decoder_block=1):
    base = f"codec.decoder.decoder.{decoder_block}.block.{unit + 2}"
    return {
        "a1": f"{base}.act1.alpha", "b1": f"{base}.act1.beta",
        "w1": f"{base}.conv1.conv.weight", "bias1": f"{base}.conv1.conv.bias",
        "a2": f"{base}.act2.alpha", "b2": f"{base}.act2.beta",
        "w2": f"{base}.conv2.conv.weight", "bias2": f"{base}.conv2.conv.bias",
    }


def snake_numpy(x, alpha, beta):
    return x + np.sin(x * np.exp(alpha[:, None])) ** 2 / (
        np.exp(beta[:, None]) + np.float32(1e-9))


def unit_numpy(x, arrays, unit, dilation, decoder_block=1):
    names = weight_names(unit, decoder_block)
    y = snake_numpy(x, arrays[names["a1"]], arrays[names["b1"]])
    padded = np.pad(y, ((0, 0), (6 * dilation, 0)))
    first = np.tile(arrays[names["bias1"]][:, None], (1, x.shape[1])).astype(np.float64)
    for oc in range(x.shape[0]):
        for frame in range(x.shape[1]):
            for ic in range(x.shape[0]):
                for tap in range(7):
                    first[oc, frame] += (float(padded[ic, frame + tap * dilation]) *
                                         float(arrays[names["w1"]][oc, ic, tap]))
    y = snake_numpy(first.astype(np.float32), arrays[names["a2"]],
                    arrays[names["b2"]])
    second = np.tile(arrays[names["bias2"]][:, None], (1, x.shape[1])).astype(np.float64)
    for oc in range(x.shape[0]):
        for frame in range(x.shape[1]):
            for ic in range(x.shape[0]):
                second[oc, frame] += (float(y[ic, frame]) *
                                      float(arrays[names["w2"]][oc, ic, 0]))
    return (second.astype(np.float32) + x).astype(np.float32)


class Qwen3TTSCodecFirstResidualPoolTest(unittest.TestCase):
    def test_circuit_and_fixture_integrity(self):
        graph = build_circuit()
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_first_residual_pool_bounded.json"
        self.assertEqual(json.loads(path.read_text()), graph)
        ops = graph["block_types"]["codec_semantic"]["body"]["ops"][-18:]
        self.assertEqual(len(ops), 18)
        for unit, dilation in enumerate(DILATIONS):
            selected = ops[unit * 6:(unit + 1) * 6]
            self.assertEqual([op["op"] for op in selected], [
                "audio_snake_beta_strided_checked", "audio_zero_pad1d_left_checked",
                "audio_conv1d_dilated_checked", "audio_snake_beta_strided_checked",
                "audio_conv1d_dilated_checked", "audio_scaled_sum_strided_checked"])
            self.assertEqual(selected[1]["params"]["call_constants"]["zero_pad_left_padding"],
                             6 * dilation)
            self.assertEqual(selected[2]["params"]["call_constants"]["conv_dilation"],
                             dilation)
            previous = ("decoder_upsample_output" if unit == 0 else
                        f"decoder_residual_{unit - 1}_output")
            self.assertEqual(selected[-1]["graph_slots"]["inputs"]["right"], previous)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        prior = MODEL.with_name("qwen3_tts_codec_decoder_upsample.npz")
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         manifest["source_decoder_upsample_sha256"])
        prior_manifest = json.loads(prior.with_suffix(".json").read_text())
        for identity in ("model_revision", "reference_revision", "codec_source_sha256"):
            self.assertEqual(manifest[identity], prior_manifest[identity], identity)
        self.assertLessEqual(manifest["source_decoder_upsample_max_abs"], 1e-7)
        self.assertEqual(manifest["dilations"], list(DILATIONS))
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files), {"decoder_upsample_output", *(
                f"decoder_residual_{unit}_output" for unit in range(3))})
            for name in oracle.files:
                self.assertEqual(oracle[name].shape, (768, 1472))
                self.assertTrue(np.isfinite(oracle[name]).all())

    def test_small_generated_three_units_and_failure_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = decoder_upsample_test.Qwen3TTSCodecDecoderUpsampleTest.synthetic_arrays(rng)
        for unit in range(3):
            names = weight_names(unit)
            for part in ("a1", "b1", "a2", "b2"):
                arrays[names[part]] = rng.uniform(-.5, .5, 2).astype(np.float32)
            for part, kernel in (("w1", 7), ("w2", 1)):
                arrays[names[part]] = (rng.normal(size=(2, 2, kernel)) * .1).astype(np.float32)
            for part in ("bias1", "bias2"):
                arrays[names[part]] = (rng.normal(size=2) * .1).astype(np.float32)
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    first_residual_pool=True, decoder_upsample_rate=2,
                    convnext_expansion=2, transformer_layers=2,
                    qkv_heads=2, qkv_head_dim=2, sliding_window=3)
                self.assertEqual(calls["operations"][-1]["function"],
                                 "audio_scaled_sum_strided_f32_checked")
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                length = frames * 8
                expected = view(arena, acts, "decoder_upsample_output", np.float32,
                                (2, length)).copy()
                for unit, dilation in enumerate(DILATIONS):
                    expected = unit_numpy(expected, arrays, unit, dilation)
                    actual = view(arena, acts, f"decoder_residual_{unit}_output",
                                  np.float32, (2, length))
                    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
                output = view(arena, acts, "decoder_residual_2_output",
                              np.float32, (2, length))
                baseline = output.copy()
                fill_ids(arena, acts, (codes + 1) % 12)
                self.assertEqual(fn(arena, len(arena)), 0)
                self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-6)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)
                entries = {e["name"]: e for e in calls["memory"]["weights"]["entries"]}
                final_weight = weight_names(2)["w2"]
                bound = np.ndarray(arrays[final_weight].shape, np.float32,
                                   buffer=arena, offset=entries[final_weight]["abs_offset"])
                saved = bound.copy()
                bound[0, 0, 0] += 2
                self.assertEqual(fn(arena, len(arena)), 0)
                self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-5)
                bound[0, 0, 0] = np.nan
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, -123)
                bound[:] = saved
                first_alpha = weight_names(0)["a1"]
                bound_alpha = np.ndarray((2,), np.float32, buffer=arena,
                                         offset=entries[first_alpha]["abs_offset"])
                original_alpha = bound_alpha[0].copy()
                bound_alpha[0] = np.nan
                self.assertNotEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, -123)
                bound_alpha[0] = original_alpha
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                np.testing.assert_array_equal(output, -123)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; connected residual pool NOT_TESTED")
    def test_pinned_connected_residual_pool(self):
        from safetensors import safe_open
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(4)
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_pretransformer_weights(model_dir)
        names = (UPSAMPLE_WEIGHT, UPSAMPLE_BIAS, DEPTHWISE_WEIGHT,
                 DEPTHWISE_BIAS, *FIRST_WEIGHT_NAMES,
                 SECOND_UPSAMPLE_WEIGHT, SECOND_UPSAMPLE_BIAS,
                 *SECOND_WEIGHT_NAMES, CONV_WEIGHT, CONV_BIAS,
                 ALPHA, BETA, decoder_upsample_test.WEIGHT,
                 decoder_upsample_test.BIAS,
                 *(name for unit in range(3) for name in weight_names(unit).values()))
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for name in names:
                arrays[name] = np.ascontiguousarray(
                    source.get_tensor(name.removeprefix("codec.")).float().numpy())

        def replay(x, unit, dilation):
            keys = weight_names(unit)
            def snake(y, alpha, beta):
                a = torch.from_numpy(arrays[keys[alpha]])[None, :, None]
                b = torch.from_numpy(arrays[keys[beta]])[None, :, None]
                return y + torch.sin(y * torch.exp(a)).pow(2) / (torch.exp(b) + 1e-9)
            residual = x
            y = snake(x, "a1", "b1")
            y = F.conv1d(F.pad(y, (6 * dilation, 0)),
                         torch.from_numpy(arrays[keys["w1"]]),
                         torch.from_numpy(arrays[keys["bias1"]]), dilation=dilation)
            y = snake(y, "a2", "b2")
            y = F.conv1d(y, torch.from_numpy(arrays[keys["w2"]]),
                         torch.from_numpy(arrays[keys["bias2"]]))
            return y + residual

        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(
                    Path(temp), arrays, len(codes), acoustic_codebooks=15,
                    first_residual_pool=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                results = []
                connected_limits = (1e-5, 1.5e-5, 2e-5)
                local_limits = (5e-6, 6e-6, 1e-5)
                previous = "decoder_upsample_output"
                for unit, dilation in enumerate(DILATIONS):
                    name = f"decoder_residual_{unit}_output"
                    input_value = view(arena, acts, previous, np.float32,
                                       oracle[previous].shape)
                    actual = view(arena, acts, name, np.float32, oracle[name].shape)
                    reference = oracle[name]
                    error = np.abs(actual - reference)
                    worst = np.unravel_index(int(np.argmax(error)), error.shape)
                    with torch.inference_mode():
                        same_input = replay(torch.from_numpy(
                            input_value.copy()[None]), unit, dilation)[0].numpy()
                    metrics = {
                        "unit": unit, "dilation": dilation,
                        "input_max_abs": float(np.max(np.abs(
                            input_value - oracle[previous]))),
                        "connected_max_abs": float(np.max(error)),
                        "connected_rmse": float(np.sqrt(np.mean(np.square(
                            actual.astype(np.float64) - reference.astype(np.float64))))),
                        "same_input_local_max_abs": float(np.max(np.abs(
                            actual - same_input))),
                        "connected_limit": connected_limits[unit],
                        "local_limit": local_limits[unit],
                        "worst": {"index": list(map(int, worst)),
                                  "actual": float(actual[worst]),
                                  "reference": float(reference[worst])},
                    }
                    results.append(metrics)
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertLessEqual(metrics["connected_max_abs"],
                                         connected_limits[unit])
                    self.assertLessEqual(metrics["same_input_local_max_abs"],
                                         local_limits[unit])
                    previous = name
                print("QWEN3_TTS_CODEC_FIRST_RESIDUAL_POOL " + json.dumps({
                    "units": results,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codec codes through first decoder residual pool; waveform NOT_TESTED",
                }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
