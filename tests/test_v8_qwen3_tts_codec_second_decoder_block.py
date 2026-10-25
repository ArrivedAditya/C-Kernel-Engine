"""Generated Qwen3-TTS codec through its second complete decoder block."""

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
    WEIGHT_NAMES as FIRST_WEIGHT_NAMES,
    UPSAMPLE_WEIGHT as FIRST_CODEC_UPSAMPLE_WEIGHT,
    UPSAMPLE_BIAS as FIRST_CODEC_UPSAMPLE_BIAS,
    DEPTHWISE_WEIGHT, DEPTHWISE_BIAS,
)
from tests.test_v8_qwen3_tts_codec_second_upsample import (
    WEIGHT as SECOND_UPSAMPLE_WEIGHT, BIAS as SECOND_UPSAMPLE_BIAS,
)
from tests.test_v8_qwen3_tts_codec_second_convnext import SECOND_WEIGHT_NAMES
from tests.test_v8_qwen3_tts_codec_decoder_ingress import CONV_WEIGHT, CONV_BIAS
from tests.test_v8_qwen3_tts_codec_snake_beta import ALPHA, BETA
from tests import test_v8_qwen3_tts_codec_decoder_upsample as decoder_upsample_test
from tests.test_v8_qwen3_tts_codec_first_residual_pool import (
    weight_names as first_weight_names, snake_numpy, unit_numpy,
)
from version.v8.tts.build_qwen3_tts_codec_second_decoder_block_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_second_decoder_block.npz")
DILATIONS = (1, 3, 9)
UPSAMPLE_WEIGHT = "codec.decoder.decoder.2.block.1.conv.weight"
UPSAMPLE_BIAS = "codec.decoder.decoder.2.block.1.conv.bias"
SNAKE_ALPHA = "codec.decoder.decoder.2.block.0.alpha"
SNAKE_BETA = "codec.decoder.decoder.2.block.0.beta"


def weight_names(unit):
    return first_weight_names(unit, decoder_block=2)


class Qwen3TTSCodecSecondDecoderBlockTest(unittest.TestCase):
    def test_circuit_fixture_and_edge_contracts(self):
        graph = build_circuit()
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_second_decoder_block_bounded.json"
        self.assertEqual(json.loads(path.read_text()), graph)
        ops = graph["block_types"]["codec_semantic"]["body"]["ops"][-20:]
        self.assertEqual([op["op"] for op in ops[:2]], [
            "audio_snake_beta_strided_checked",
            "audio_conv_transpose1d_dense_crop_right_checked"])
        self.assertEqual(ops[0]["graph_slots"]["inputs"]["input"],
                         "decoder_residual_2_output")
        self.assertEqual(ops[1]["graph_slots"]["inputs"]["input"],
                         "decoder_second_snake_beta_output")
        self.assertEqual(ops[1]["weight_refs"],
                         {"weight": UPSAMPLE_WEIGHT, "bias": UPSAMPLE_BIAS})
        self.assertEqual(ops[1]["params"]["call_constants"]["cropped_deconv_right_crop"], 5)
        for unit, dilation in enumerate(DILATIONS):
            selected = ops[2 + unit * 6:8 + unit * 6]
            self.assertEqual([op["op"] for op in selected], [
                "audio_snake_beta_strided_checked", "audio_zero_pad1d_left_checked",
                "audio_conv1d_dilated_checked", "audio_snake_beta_strided_checked",
                "audio_conv1d_dilated_checked", "audio_scaled_sum_strided_checked"])
            self.assertEqual(selected[1]["params"]["call_constants"]["zero_pad_left_padding"],
                             6 * dilation)
            self.assertEqual(selected[-1]["graph_slots"]["inputs"]["right"],
                             "decoder_second_upsample_output" if unit == 0 else
                             f"decoder_second_residual_{unit - 1}_output")
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                build_circuit(second_decoder_upsample_rate=invalid)
        with self.assertRaises(ValueError):
            build_circuit(decoder_width=6)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        prior = MODEL.with_name("qwen3_tts_codec_first_residual_pool.npz")
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         manifest["source_first_residual_pool_sha256"])
        prior_manifest = json.loads(prior.with_suffix(".json").read_text())
        for identity in ("model_revision", "reference_revision", "codec_source_sha256"):
            self.assertEqual(manifest[identity], prior_manifest[identity])
        self.assertLessEqual(manifest["source_first_residual_pool_max_abs"], 1e-7)
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files), {
                "decoder_second_snake_beta_output",
                "decoder_second_upsample_output", *(
                    f"decoder_second_residual_{unit}_output" for unit in range(3))})
            for name in oracle.files:
                self.assertEqual(oracle[name].shape,
                                 (768, 1472) if name == "decoder_second_snake_beta_output"
                                 else (384, 7360))
                self.assertTrue(np.isfinite(oracle[name]).all())

    @staticmethod
    def synthetic_arrays(rng):
        arrays = decoder_upsample_test.Qwen3TTSCodecDecoderUpsampleTest.synthetic_arrays(rng)
        for block in (1, 2):
            channels = 2 if block == 1 else 1
            for unit in range(3):
                names = first_weight_names(unit) if block == 1 else weight_names(unit)
                for part in ("a1", "b1", "a2", "b2"):
                    arrays[names[part]] = rng.uniform(-.5, .5, channels).astype(np.float32)
                for part, kernel in (("w1", 7), ("w2", 1)):
                    arrays[names[part]] = (rng.normal(size=(channels, channels, kernel)) * .1).astype(np.float32)
                for part in ("bias1", "bias2"):
                    arrays[names[part]] = (rng.normal(size=channels) * .1).astype(np.float32)
        arrays[SNAKE_ALPHA] = np.array([-.3, .25], dtype=np.float32)
        arrays[SNAKE_BETA] = np.array([.35, -.2], dtype=np.float32)
        arrays[UPSAMPLE_WEIGHT] = (rng.normal(size=(2, 1, 4)) * .1).astype(np.float32)
        arrays[UPSAMPLE_BIAS] = np.array([.05], dtype=np.float32)
        return arrays

    def test_small_generated_lengths_and_failure_recovery(self):
        rng = np.random.default_rng(20261025)
        arrays = self.synthetic_arrays(rng)
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    second_decoder_block=True, decoder_upsample_rate=2,
                    second_decoder_upsample_rate=2,
                    convnext_expansion=2, transformer_layers=2,
                    qkv_heads=2, qkv_head_dim=2, sliding_window=3)
                self.assertEqual(calls["operations"][-1]["function"],
                                 "audio_scaled_sum_strided_f32_checked")
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                source = view(arena, acts, "decoder_residual_2_output", np.float32,
                              (2, frames * 8)).copy()
                expected_snake = snake_numpy(source, arrays[SNAKE_ALPHA], arrays[SNAKE_BETA])
                np.testing.assert_allclose(
                    view(arena, acts, "decoder_second_snake_beta_output", np.float32,
                         expected_snake.shape), expected_snake, atol=1e-6, rtol=1e-6)
                length = frames * 16
                expected = np.full((1, length), arrays[UPSAMPLE_BIAS][0], dtype=np.float64)
                for ic in range(2):
                    for frame in range(frames * 8):
                        for tap in range(4):
                            pos = frame * 2 + tap
                            if pos < length:
                                expected[0, pos] += (float(expected_snake[ic, frame]) *
                                                     float(arrays[UPSAMPLE_WEIGHT][ic, 0, tap]))
                expected = expected.astype(np.float32)
                np.testing.assert_allclose(
                    view(arena, acts, "decoder_second_upsample_output", np.float32,
                         (1, length)), expected, atol=2e-6, rtol=2e-6)
                for unit, dilation in enumerate(DILATIONS):
                    expected = unit_numpy(expected, arrays, unit, dilation,
                                          decoder_block=2)
                    np.testing.assert_allclose(
                        view(arena, acts, f"decoder_second_residual_{unit}_output",
                             np.float32, (1, length)), expected,
                        atol=3e-6, rtol=3e-6)
                output = view(arena, acts, "decoder_second_residual_2_output",
                              np.float32, (1, length))
                baseline = output.copy()
                fill_ids(arena, acts, (codes + 1) % 12)
                self.assertEqual(fn(arena, len(arena)), 0)
                self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-7)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)
                entries = {entry["name"]: entry for entry in calls["memory"]["weights"]["entries"]}
                upsample_weight = np.ndarray(arrays[UPSAMPLE_WEIGHT].shape,
                                             np.float32, buffer=arena,
                                             offset=entries[UPSAMPLE_WEIGHT]["abs_offset"])
                original_weight = upsample_weight[0, 0, 0].copy()
                upsample_weight[0, 0, 0] += 1
                self.assertEqual(fn(arena, len(arena)), 0)
                self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-6)
                upsample_weight[0, 0, 0] = original_weight
                alpha = np.ndarray((2,), np.float32, buffer=arena,
                                   offset=entries[SNAKE_ALPHA]["abs_offset"])
                saved = alpha.copy()
                alpha[0] = np.nan
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, -123)
                alpha[:] = saved
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                np.testing.assert_array_equal(output, -123)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; connected second decoder NOT_TESTED")
    def test_pinned_connected_second_decoder_block(self):
        from safetensors import safe_open
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(4)
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_pretransformer_weights(model_dir)
        names = (FIRST_CODEC_UPSAMPLE_WEIGHT, FIRST_CODEC_UPSAMPLE_BIAS,
                 DEPTHWISE_WEIGHT,
                 DEPTHWISE_BIAS, *FIRST_WEIGHT_NAMES,
                 SECOND_UPSAMPLE_WEIGHT, SECOND_UPSAMPLE_BIAS,
                 *SECOND_WEIGHT_NAMES, CONV_WEIGHT, CONV_BIAS,
                 ALPHA, BETA, decoder_upsample_test.WEIGHT,
                 decoder_upsample_test.BIAS,
                 *(name for unit in range(3) for name in first_weight_names(unit).values()),
                 SNAKE_ALPHA, SNAKE_BETA, UPSAMPLE_WEIGHT, UPSAMPLE_BIAS,
                 *(name for unit in range(3) for name in weight_names(unit).values()))
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for name in names:
                arrays[name] = np.ascontiguousarray(
                    source.get_tensor(name.removeprefix("codec.")).float().numpy())
        prior = MODEL.with_name("qwen3_tts_codec_first_residual_pool.npz")
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle, np.load(prior) as prior_oracle:
            references = {name: oracle[name] for name in oracle.files}
            references["decoder_residual_2_output"] = prior_oracle["decoder_residual_2_output"]
            codes = fixture["codec_tokens"].astype(np.int32)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(
                    Path(temp), arrays, len(codes), acoustic_codebooks=15,
                    second_decoder_block=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                metrics = []
                previous = "decoder_residual_2_output"
                checkpoints = ("decoder_second_snake_beta_output",
                               "decoder_second_upsample_output",
                               *(f"decoder_second_residual_{unit}_output"
                                 for unit in range(3)))
                connected_limits = (3e-5, 1e-5, 3e-5, 6e-5, 9e-5)
                local_limits = (2e-6, 8e-6, 5e-6, 7e-6, 1e-5)
                for index, name in enumerate(checkpoints):
                    input_value = view(arena, acts, previous, np.float32,
                                       references[previous].shape)
                    actual = view(arena, acts, name, np.float32, references[name].shape)
                    reference = references[name]
                    error = np.abs(actual - reference)
                    worst = np.unravel_index(int(np.argmax(error)), error.shape)
                    with torch.inference_mode():
                        tensor = torch.from_numpy(input_value.copy()[None])
                        if name == "decoder_second_snake_beta_output":
                            alpha = torch.from_numpy(arrays[SNAKE_ALPHA])[None, :, None]
                            beta = torch.from_numpy(arrays[SNAKE_BETA])[None, :, None]
                            replay = tensor + torch.sin(tensor * torch.exp(alpha)).pow(2) / (
                                torch.exp(beta) + 1e-9)
                        elif name == "decoder_second_upsample_output":
                            replay = F.conv_transpose1d(
                                tensor, torch.from_numpy(arrays[UPSAMPLE_WEIGHT]),
                                torch.from_numpy(arrays[UPSAMPLE_BIAS]), stride=5)[..., :-5]
                        else:
                            unit = int(name.removeprefix("decoder_second_residual_").removesuffix("_output"))
                            dilation = DILATIONS[unit]
                            keys = weight_names(unit)
                            def snake(value, a, b):
                                return value + torch.sin(value * torch.exp(
                                    torch.from_numpy(arrays[keys[a]])[None, :, None])).pow(2) / (
                                    torch.exp(torch.from_numpy(arrays[keys[b]])[None, :, None]) + 1e-9)
                            value = snake(tensor, "a1", "b1")
                            value = F.conv1d(F.pad(value, (6 * dilation, 0)),
                                             torch.from_numpy(arrays[keys["w1"]]),
                                             torch.from_numpy(arrays[keys["bias1"]]), dilation=dilation)
                            value = snake(value, "a2", "b2")
                            value = F.conv1d(value, torch.from_numpy(arrays[keys["w2"]]),
                                             torch.from_numpy(arrays[keys["bias2"]]))
                            replay = value + tensor
                    replay_value = replay[0].numpy()
                    result = {"checkpoint": name,
                              "input_max_abs": float(np.max(np.abs(
                                  input_value - references[previous]))),
                              "connected_max_abs": float(np.max(error)),
                      "same_input_local_max_abs": float(np.max(np.abs(
                          actual - replay_value))),
                      "connected_limit": connected_limits[index],
                      "local_limit": local_limits[index],
                              "connected_rmse": float(np.sqrt(np.mean(np.square(
                                  actual.astype(np.float64) - reference.astype(np.float64))))),
                              "worst": {"index": list(map(int, worst)),
                                        "actual": float(actual[worst]),
                                        "reference": float(reference[worst])}}
                    metrics.append(result)
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertLessEqual(result["connected_max_abs"],
                                         connected_limits[index])
                    self.assertLessEqual(result["same_input_local_max_abs"],
                                         local_limits[index])
                    previous = name
                print("QWEN3_TTS_CODEC_SECOND_DECODER_BLOCK " + json.dumps({
                    "checkpoints": metrics,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codec codes through second decoder block; waveform NOT_TESTED",
                }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
