import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/xray_decoder_pytorch_v8.py"
SPEC = importlib.util.spec_from_file_location("xray_decoder_pytorch_v8", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DecoderPyTorchXRayTests(unittest.TestCase):
    def test_parse_token_ids_accepts_explicit_csv(self) -> None:
        self.assertEqual(MODULE.parse_token_ids("1, 2,3"), [1, 2, 3])

    def test_teacher_forced_split_requires_prompt_and_suffix(self) -> None:
        self.assertEqual(
            MODULE.split_teacher_forced_tokens([10, 11, 12, 13], 2),
            ([10, 11], [12, 13]),
        )
        with self.assertRaisesRegex(ValueError, "leave at least one"):
            MODULE.split_teacher_forced_tokens([10, 11], 2)

    def test_persistent_capture_rejects_a_trajectory_that_left_the_prefix(self) -> None:
        trajectory = {
            "generated_tokens": [41, 99],
            "logits": [object(), object()],
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                MODULE,
                "load_ck_greedy_trajectory",
                return_value=trajectory,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "left the teacher-forced prefix"
                ):
                    MODULE.capture_ck_persistent(
                        Path(directory),
                        [1, 2],
                        [41, 42],
                        Path(directory) / "capture",
                    )

    def test_run_reports_persistent_vs_replay_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            call_ir = tmp_path / "call.json"
            call_ir.write_text('{"operations": []}\n', encoding="utf-8")

            def tensors(prefix: str, layer_one: list[float]):
                return {
                    0: {
                        "layer_out": MODULE._write_tensor(
                            tmp_path / f"{prefix}_layer0.f32",
                            np.asarray([1.0, 2.0], dtype=np.float32),
                        )
                    },
                    1: {
                        "layer_out": MODULE._write_tensor(
                            tmp_path / f"{prefix}_layer1.f32",
                            np.asarray(layer_one, dtype=np.float32),
                        )
                    },
                }

            replay = tensors("replay", [3.0, 4.0])
            oracle = tensors("oracle", [3.0, 4.0])
            persistent = tensors("persistent", [6.0, 8.0])
            replay_logits = np.asarray([0.0, 2.0, 1.0], dtype=np.float32)
            persistent_logits = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)
            args = SimpleNamespace(
                token_ids="10,11,12",
                prompt_token_count=2,
                output_dir=tmp_path / "report",
                call_ir=call_ir,
                checkpoint=tmp_path,
                runtime=tmp_path,
                threads=1,
                model_name="fixture",
                top_k=2,
            )
            with (
                mock.patch.object(
                    MODULE,
                    "capture_pytorch",
                    return_value=(oracle, replay_logits),
                ),
                mock.patch.object(
                    MODULE,
                    "capture_ck",
                    return_value=(replay, replay_logits),
                ),
                mock.patch.object(
                    MODULE,
                    "capture_ck_persistent",
                    return_value=(persistent, persistent_logits),
                ),
            ):
                result = MODULE.run(args)

            state = result["persistent_vs_replay"]
            self.assertEqual(result["attribution_scope"], "persistent_vs_replay")
            self.assertEqual(result["persistent_state_status"], "diverged")
            self.assertFalse(state["ranking"]["top1_match"])
            self.assertEqual(
                state["first_material"]["checkpoint_id"],
                "decoder.layer.1.layer_out",
            )

    def test_decoder_layers_discovers_nested_language_model_without_model_name(self) -> None:
        layers = [object(), object(), object()]
        model = SimpleNamespace(
            language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
        )
        self.assertEqual(MODULE.decoder_layers(model), layers)

    def test_sparse_manifest_keeps_resolved_provider_identity(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            tensor = tmp_path / "layer.f32"
            tensor.write_bytes(b"\0" * 16)
            record = {
                "path": str(tensor),
                "shape": [4],
                "sha256": MODULE._sha256(tensor),
            }
            call_ir = {
                "operations": [
                    {
                        "op": "residual_add",
                        "layer": 0,
                        "function": "residual_add_f32",
                        "kernel_id": "residual_add_f32",
                        "resolved_contract_id": "residual.fp32",
                    }
                ]
            }
            result = MODULE.manifest(
                "ck",
                "fixture",
                tmp_path,
                {0: {"layer_out": record}},
                call_ir,
                [(0, "layer_out")],
            )
            checkpoint = result["checkpoints"][0]
            self.assertEqual(checkpoint["checkpoint_id"], "decoder.layer.0.layer_out")
            self.assertEqual(checkpoint["kernel_id"], "residual_add_f32")
            self.assertEqual(checkpoint["function"], "residual_add_f32")
            self.assertEqual(checkpoint["resolved_contract_id"], "residual.fp32")


if __name__ == "__main__":
    unittest.main()
