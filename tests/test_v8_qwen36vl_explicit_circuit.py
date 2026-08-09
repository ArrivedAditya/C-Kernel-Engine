from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location("build_ir_v8_qwen36vl_test", SCRIPTS / "build_ir_v8.py")
assert SPEC and SPEC.loader
build_ir_v8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_ir_v8)

BRIDGE_SPEC = importlib.util.spec_from_file_location(
    "run_multimodal_bridge_v8_qwen36vl_test",
    SCRIPTS / "run_multimodal_bridge_v8.py",
)
assert BRIDGE_SPEC and BRIDGE_SPEC.loader
bridge_v8 = importlib.util.module_from_spec(BRIDGE_SPEC)
BRIDGE_SPEC.loader.exec_module(bridge_v8)

CODEGEN_SPEC = importlib.util.spec_from_file_location(
    "codegen_prefill_v8_qwen36vl_test",
    SCRIPTS / "codegen_prefill_v8.py",
)
assert CODEGEN_SPEC and CODEGEN_SPEC.loader
codegen_prefill_v8 = importlib.util.module_from_spec(CODEGEN_SPEC)
CODEGEN_SPEC.loader.exec_module(codegen_prefill_v8)


def _composition_manifest(*, deepstack_layers: int = 0) -> dict:
    return {
        "config": {
            "model": "qwen36vl",
            "component_configs": {
                "vision_encoder": {
                    "model": "qwen3_vl_vision",
                    "num_deepstack_layers": deepstack_layers,
                    "deepstack_layer_indices": [] if deepstack_layers == 0 else [5],
                    "projector_out_dim": 5120,
                },
                "decoder": {
                    "model": "qwen35",
                    "embed_dim": 5120,
                },
            },
        },
        "template": {"name": "qwen36vl"},
        "model": {"weights": []},
    }


class Qwen36VLExplicitCircuitTests(unittest.TestCase):
    def test_composition_references_reusable_circuits_without_copying_them(self) -> None:
        raw = json.loads((ROOT / "version/v8/circuits/qwen36vl.json").read_text())
        self.assertNotIn("block_types", raw)
        self.assertEqual(raw["sequence"], ["vision_encoder", "decoder"])
        self.assertEqual(raw["components"]["vision_encoder"]["circuit"], "qwen3_vl_vision")
        self.assertEqual(raw["components"]["decoder"]["circuit"], "qwen35")
        self.assertEqual(raw["components"]["vision_encoder"]["runtime_role"], "encoder")
        self.assertEqual(raw["components"]["decoder"]["runtime_role"], "decoder")

        hydrated = build_ir_v8._load_builtin_template_doc("qwen36vl")
        self.assertEqual(set(hydrated["block_types"]), {"vision_encoder", "decoder"})
        self.assertEqual(
            hydrated["resolved_components"]["vision_encoder"]["circuit"],
            "qwen3_vl_vision",
        )
        self.assertEqual(hydrated["resolved_components"]["decoder"]["circuit"], "qwen35")

    def test_stitch_is_explicit_and_disables_deepstack(self) -> None:
        plan = build_ir_v8.build_stitch_plan(_composition_manifest())
        self.assertEqual(plan["sequence"], ["vision_encoder", "decoder"])
        self.assertEqual(len(plan["edges"]), 1)
        edge = plan["edges"][0]
        self.assertEqual(edge["op"], "multimodal_prefix_stitch")
        self.assertEqual(edge["from_output"], "vision_embeddings")
        self.assertEqual(edge["to_input"], "visual_prefix")
        self.assertEqual(
            plan["components"]["vision_encoder"]["exports"][edge["from_output"]],
            "vision_output",
        )
        self.assertEqual(
            plan["components"]["decoder"]["imports"][edge["to_input"]],
            "embedded_input",
        )
        self.assertEqual(edge["required_contract"]["deepstack_injections"], 0)
        providers = edge["required_contract"]["providers"]
        self.assertEqual(
            providers["prefix_insert"]["resolved_function"],
            "ck_multimodal_prefix_insert_f32",
        )
        self.assertEqual(
            providers["position_builder"]["resolved_function"],
            "ck_multimodal_mrope_positions_2d",
        )
        self.assertEqual(
            providers["position_transform"]["resolved_function"],
            "mrope_qk_imrope_positions",
        )
        self.assertEqual(
            plan["components"]["vision_encoder"]["config_requires"]["num_deepstack_layers"],
            0,
        )

    def test_each_block_retains_its_source_circuit_contract(self) -> None:
        blocks = build_ir_v8.build_block_manifests(_composition_manifest())
        self.assertEqual([item["block_name"] for item in blocks], ["vision_encoder", "decoder"])
        self.assertEqual(blocks[0]["template"]["name"], "qwen3_vl_vision")
        self.assertEqual(blocks[0]["template"]["sequence"], ["vision_encoder"])
        self.assertEqual(blocks[0]["config"]["projector_out_dim"], 5120)
        self.assertEqual(blocks[1]["template"]["name"], "qwen35")
        self.assertEqual(blocks[1]["template"]["sequence"], ["decoder"])
        self.assertEqual(blocks[1]["config"]["embed_dim"], 5120)

    def test_nonzero_deepstack_is_rejected_instead_of_inferred(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires config num_deepstack_layers=0"):
            build_ir_v8.build_block_manifests(_composition_manifest(deepstack_layers=1))

    def test_bridge_policy_is_loaded_from_explicit_stitch(self) -> None:
        circuit = bridge_v8._load_explicit_composition_circuit("qwen36vl")
        bridge = bridge_v8._composition_bridge_contract(circuit)
        chat = bridge_v8._composition_exported_contract(circuit, "chat_contract")
        self.assertEqual(bridge["position_policy"], "mrope_2d")
        self.assertEqual(bridge["generation_policy"], "incremental_decode_after_prefill")
        self.assertEqual(bridge["deepstack_injections"], 0)
        self.assertEqual(
            bridge["providers"]["position_builder"]["resolved_function"],
            "ck_multimodal_mrope_positions_2d",
        )
        self.assertEqual(chat["name"], "qwen35")

    def test_runtime_must_satisfy_declared_component_and_dimension_contract(self) -> None:
        circuit = bridge_v8._load_explicit_composition_circuit("qwen36vl")
        evidence = bridge_v8._validate_composition_runtime(
            circuit,
            encoder_config={
                "num_deepstack_layers": 0,
                "deepstack_layer_indices": [],
                "projector_out_dim": 5120,
            },
            decoder_config={"embed_dim": 5120},
        )
        self.assertEqual(evidence["status"], "validated")

        with self.assertRaisesRegex(RuntimeError, "dimension mismatch"):
            bridge_v8._validate_composition_runtime(
                circuit,
                encoder_config={
                    "num_deepstack_layers": 0,
                    "deepstack_layer_indices": [],
                    "projector_out_dim": 4096,
                },
                decoder_config={"embed_dim": 5120},
            )

        with self.assertRaisesRegex(RuntimeError, "num_deepstack_layers=0"):
            bridge_v8._validate_composition_runtime(
                circuit,
                encoder_config={
                    "num_deepstack_layers": 1,
                    "deepstack_layer_indices": [5],
                    "projector_out_dim": 5120,
                },
                decoder_config={"embed_dim": 5120},
            )

    def test_zero_deepstack_codegen_calls_resolved_stitch_providers(self) -> None:
        circuit = bridge_v8._load_explicit_composition_circuit("qwen36vl")
        bridge = bridge_v8._composition_bridge_contract(circuit)
        config = {
            "embed_dim": 5120,
            "num_deepstack_layers": 0,
            "multimodal_bridge_contract": bridge,
        }
        helpers = codegen_prefill_v8._emit_multimodal_prefill_bridge_helpers(
            config,
            "mrope_qk_text",
        )
        self.assertIn("ck_multimodal_mrope_positions_2d(", helpers)
        self.assertIn("mrope_qk_imrope_positions(q, k,", helpers)
        self.assertNotIn("for (int tok = 0; tok < total_tokens", helpers)

        embedding_op = {
            "op": "dense_embedding_lookup",
            "function": "embedding_forward",
            "args": [
                {"name": "token_ids", "expr": "(int32_t*)(model->bump + A_TOKEN_IDS)"},
                {"name": "token_embeddings", "expr": "(float*)(model->bump + W_TOKEN_EMBEDDINGS)"},
                {"name": "output", "expr": "(float*)(model->bump + A_EMBEDDED_INPUT)"},
                {"name": "embed_dim", "expr": "5120"},
                {"name": "aligned_embed_dim", "expr": "5120"},
                {"name": "context_window", "expr": "1400"},
            ],
        }
        api = codegen_prefill_v8.emit_multimodal_bridge_api([embedding_op], config)
        self.assertIn("return ck_multimodal_prefix_insert_f32(", api)
        self.assertIn("if (prefix_tokens > 0)", api)
        self.assertNotIn("for (int i = 0; i < count; ++i)", api)

    def test_explicit_stitch_cannot_fall_back_to_codegen_helpers(self) -> None:
        circuit = bridge_v8._load_explicit_composition_circuit("qwen36vl")
        bridge = bridge_v8._composition_bridge_contract(circuit)
        bridge["providers"].pop("position_builder")
        with self.assertRaisesRegex(RuntimeError, "position-builder provider"):
            codegen_prefill_v8._emit_multimodal_prefill_bridge_helpers(
                {
                    "embed_dim": 5120,
                    "num_deepstack_layers": 0,
                    "multimodal_bridge_contract": bridge,
                },
                "mrope_qk_text",
            )

        bridge = bridge_v8._composition_bridge_contract(circuit)
        bridge["providers"].pop("prefix_insert")
        with self.assertRaisesRegex(RuntimeError, "prefix-insert provider"):
            codegen_prefill_v8.emit_multimodal_bridge_api(
                [{
                    "op": "dense_embedding_lookup",
                    "function": "embedding_forward",
                    "args": [{"name": "token_embeddings", "expr": "weights"}],
                }],
                {
                    "embed_dim": 5120,
                    "multimodal_bridge_contract": bridge,
                },
            )

if __name__ == "__main__":
    unittest.main()
