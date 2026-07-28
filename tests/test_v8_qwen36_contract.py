#!/usr/bin/env python3
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import convert_safetensors_to_bump_v8 as converter  # type: ignore
import convert_gguf_to_bump_v8 as gguf_converter  # type: ignore


FIXTURE = ROOT / "version" / "v8" / "test_assets" / "qwen36_27b_contract.json"


def _decoder_names(layer_types):
    names = {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    }
    for layer, kind in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer}"
        names.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
        if kind == "linear_attention":
            names.update(
                {
                    f"{prefix}.linear_attn.A_log",
                    f"{prefix}.linear_attn.conv1d.weight",
                    f"{prefix}.linear_attn.dt_bias",
                    f"{prefix}.linear_attn.in_proj_a.weight",
                    f"{prefix}.linear_attn.in_proj_b.weight",
                    f"{prefix}.linear_attn.in_proj_qkv.weight",
                    f"{prefix}.linear_attn.in_proj_z.weight",
                    f"{prefix}.linear_attn.norm.weight",
                    f"{prefix}.linear_attn.out_proj.weight",
                }
            )
        else:
            names.update(
                {
                    f"{prefix}.self_attn.k_norm.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                    f"{prefix}.self_attn.q_norm.weight",
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                }
            )
    return names


def _vision_names(depth):
    names = {
        "model.visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias",
        "model.visual.pos_embed.weight",
        "model.visual.merger.norm.weight",
        "model.visual.merger.norm.bias",
        "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc1.bias",
        "model.visual.merger.linear_fc2.weight",
        "model.visual.merger.linear_fc2.bias",
    }
    for layer in range(depth):
        prefix = f"model.visual.blocks.{layer}"
        names.update(
            {
                f"{prefix}.attn.proj.bias",
                f"{prefix}.attn.proj.weight",
                f"{prefix}.attn.qkv.bias",
                f"{prefix}.attn.qkv.weight",
                f"{prefix}.mlp.linear_fc1.bias",
                f"{prefix}.mlp.linear_fc1.weight",
                f"{prefix}.mlp.linear_fc2.bias",
                f"{prefix}.mlp.linear_fc2.weight",
                f"{prefix}.norm1.bias",
                f"{prefix}.norm1.weight",
                f"{prefix}.norm2.bias",
                f"{prefix}.norm2.weight",
            }
        )
    return names


def _mtp_names():
    names = {
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
    prefix = "mtp.layers.0"
    names.update(
        {
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.mlp.down_proj.weight",
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
            f"{prefix}.self_attn.k_norm.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.self_attn.q_norm.weight",
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
        }
    )
    return names


class Qwen36ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(FIXTURE.read_text(encoding="utf-8"))
        text = cls.config["text_config"]
        interval = int(text["full_attention_interval"])
        text["layer_types"] = [
            "full_attention" if (layer + 1) % interval == 0 else "linear_attention"
            for layer in range(int(text["num_hidden_layers"]))
        ]
        cls.metadata = converter._qwen35_architecture_metadata(cls.config)
        cls.decoder = _decoder_names(text["layer_types"])
        cls.vision = _vision_names(int(cls.config["vision_config"]["depth"]))
        cls.mtp = _mtp_names()
        cls.names = cls.decoder | cls.vision | cls.mtp
        cls.headers = {
            name: converter.HeaderTensor(name, "BF16", [1], Path("unused.safetensors"))
            for name in cls.names
        }

    def test_official_config_resolves_without_new_compiler_family_branch(self):
        self.assertEqual(converter._infer_arch(self.config), "qwen35")
        self.assertEqual(len(self.metadata["layer_types"]), 64)
        self.assertEqual(self.metadata["layer_kinds"].count("recurrent"), 48)
        self.assertEqual(self.metadata["layer_kinds"].count("full_attention"), 16)
        self.assertEqual(
            [i for i, kind in enumerate(self.metadata["layer_kinds"]) if kind == "full_attention"],
            list(range(3, 64, 4)),
        )

    def test_official_mrope_vision_and_mtp_metadata_are_preserved(self):
        self.assertEqual(self.metadata["rotary_dim"], 64)
        self.assertEqual(self.metadata["mrope_sections"], [11, 11, 10])
        self.assertEqual(self.metadata["mrope_n_dims"], 64)
        self.assertTrue(self.metadata["mrope_interleaved"])
        self.assertTrue(self.metadata["has_vision_encoder"])
        self.assertEqual(self.metadata["vision_arch"], "qwen3_vl_vision")
        self.assertEqual(self.metadata["vision_depth"], 27)
        self.assertEqual(self.metadata["vision_output_size"], 5120)
        self.assertTrue(self.metadata["has_mtp_assistant"])
        self.assertEqual(self.metadata["mtp_num_layers"], 1)
        self.assertFalse(self.metadata["mtp_use_dedicated_embeddings"])

    def test_runtime_config_preserves_partial_mrope_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(self.config), encoding="utf-8"
            )
            with mock.patch.object(
                converter, "_load_safetensors_headers", return_value=self.headers
            ):
                runtime = converter._build_config(model_dir, "qwen35", None)

        self.assertEqual(runtime["head_dim"], 256)
        self.assertEqual(runtime["rotary_dim"], 64)
        self.assertEqual(runtime["mrope_n_dims"], 64)
        self.assertEqual(runtime["mrope_sections"], [11, 11, 10, 0])
        self.assertTrue(runtime["mrope_interleaved"])

    def test_explicit_attention_head_width_does_not_require_hidden_partition(self):
        self.assertEqual(
            gguf_converter.resolve_attention_head_dim(5120, 24, 256),
            256,
        )
        with self.assertRaisesRegex(
            gguf_converter.GGUFError,
            "no explicit attention key_length",
        ):
            gguf_converter.resolve_attention_head_dim(5120, 24)

    def test_invalid_layer_schedule_hard_fails(self):
        malformed = copy.deepcopy(self.config)
        malformed["text_config"]["layer_types"] = malformed["text_config"]["layer_types"][:-1]
        with self.assertRaisesRegex(SystemExit, "layer_types length"):
            converter._qwen35_architecture_metadata(malformed)

    def test_unsupported_layer_kind_hard_fails(self):
        malformed = copy.deepcopy(self.config)
        malformed["text_config"]["layer_types"][7] = "invented_attention"
        with self.assertRaisesRegex(SystemExit, "Unsupported Qwen3.5 layer types"):
            converter._qwen35_architecture_metadata(malformed)

    def test_invalid_interleaved_mrope_width_hard_fails(self):
        malformed = copy.deepcopy(self.config)
        malformed["text_config"]["rope_parameters"]["mrope_section"] = [11, 11, 9]
        with self.assertRaisesRegex(SystemExit, "M-RoPE sections"):
            converter._qwen35_architecture_metadata(malformed)

    def test_generated_tensor_inventory_matches_official_index_contract(self):
        inventory = self.config["official_tensor_inventory"]
        self.assertEqual(len(self.decoder), inventory["decoder_tensor_count"])
        self.assertEqual(len(self.vision), inventory["vision_tensor_count"])
        self.assertEqual(len(self.mtp), inventory["mtp_tensor_count"])
        self.assertEqual(len(self.names), inventory["tensor_count"])
        digest = hashlib.sha256(("\n".join(sorted(self.names)) + "\n").encode()).hexdigest()
        self.assertEqual(digest, inventory["sorted_name_sha256"])

    def test_decoder_and_vision_passes_consume_their_complete_tensor_scopes(self):
        decoder_refs = converter._qwen35_text_refs(self.config, self.headers)
        decoder_audit = converter._build_source_audit("qwen35", self.headers, decoder_refs)
        self.assertEqual(decoder_audit["verdict"], "pass")
        self.assertEqual(decoder_audit["unmapped_source_tensors"], [])

        vision = self.config["vision_config"]
        vision_config = {
            "num_layers": vision["depth"],
            "embed_dim": vision["hidden_size"],
            "intermediate_size": vision["intermediate_size"],
            "patch_dim": (
                vision["in_channels"]
                * vision["temporal_patch_size"]
                * vision["patch_size"]
                * vision["patch_size"]
            ),
            "deepstack_layer_indices": vision["deepstack_visual_indexes"],
        }
        vision_refs = converter._qwen3vl_vision_refs(vision_config, self.headers)
        vision_audit = converter._build_source_audit(
            "qwen3_vl_vision", self.headers, vision_refs
        )
        self.assertEqual(vision_audit["verdict"], "pass")
        self.assertEqual(vision_audit["unmapped_source_tensors"], [])

        mtp_ignored = [
            row
            for row in decoder_audit["ignored_source_tensors"]
            if row["source"].startswith("mtp.")
        ]
        self.assertEqual(len(mtp_ignored), 15)
        self.assertEqual(
            {row["reason"] for row in mtp_ignored},
            {"mtp_decoder_block_not_in_main_pass"},
        )


if __name__ == "__main__":
    unittest.main()
