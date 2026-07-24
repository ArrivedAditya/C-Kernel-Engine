#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8_ROOT = ROOT / "version" / "v8"
BUILD_IR_PATH = V8_ROOT / "scripts" / "build_ir_v8.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_ir_v8 = _load_module("build_ir_v8_qwen3vl_bf16_matrix_tests", BUILD_IR_PATH)
attention_resolver = _load_module(
    "resolve_attention_contracts_v8_qwen3vl_bf16_matrix_tests",
    V8_ROOT / "scripts" / "resolve_attention_contracts_v8.py",
)


class Qwen3VLBf16KernelMatrixTests(unittest.TestCase):
    def test_every_selected_numerical_provider_has_a_pytorch_oracle(self) -> None:
        template = build_ir_v8._load_builtin_template_doc("qwen3vl")
        manifest = {
            "config": {
                "model": "qwen3vl",
                "arch": "qwen3vl",
                "decode_kv_cache_dtype": "bf16",
                "decoder_norm_storage_boundary": "bf16",
                "decoder_norm_reduction_policy": "pytorch_avx2_cascade_exact",
                "decoder_qk_norm_reduction_policy": "pytorch_avx2_cascade_exact",
                "decoder_mrope_storage_boundary": "pytorch_bf16_exact",
                "decoder_prefill_projection_storage_boundary": "bf16",
                "decoder_projection_reduction_policy": "pytorch_onednn_brgemm_exact",
                "decoder_residual_storage_boundary": "bf16",
                "decoder_swiglu_storage_boundary": "pytorch_bf16_exact",
            },
            "template": template,
        }
        expected_operations = {
            "decoder.layer.mrope.bf16_pytorch",
            "decoder.layer.qkv_projection.bf16_pytorch_onednn",
            "decoder.layer.out_projection.bf16_pytorch_onednn",
            "decoder.layer.mlp_projection.bf16_pytorch_onednn",
            "decoder.logits.bf16_pytorch_onednn",
            "decoder.layer.residual.bf16_pytorch",
            "decoder.rmsnorm.bf16",
            "decoder.qk_norm.bf16",
            "decoder.swiglu.bf16",
        }

        selected_kernel_ids: set[str] = set()
        for phase in ("prefill", "decode"):
            plans = build_ir_v8._resolve_manifest_execution_contracts(manifest, phase)
            by_operation = {plan["operation"]: plan for plan in plans}
            self.assertTrue(
                expected_operations.issubset(by_operation),
                (phase, sorted(by_operation)),
            )
            selected_kernel_ids.update(
                str(by_operation[name]["kernel"]["id"])
                for name in expected_operations
            )

        attention_contracts = attention_resolver.load_json(
            attention_resolver.DEFAULT_CONTRACTS
        )
        attention_kernels = attention_resolver.load_kernel_capabilities()
        for phase in ("prefill", "decode"):
            resolved_attention = attention_resolver.resolve_contract(
                copy.deepcopy(template),
                copy.deepcopy(attention_contracts),
                copy.deepcopy(attention_kernels),
                operation="decoder.attention.bf16_pytorch",
                phase=phase,
                mode="bringup",
                source_circuit_path=V8_ROOT / "circuits" / "qwen3vl.json",
            )
            selected_kernel_ids.add(str(resolved_attention["kernel"]["id"]))

        vision_template = build_ir_v8._load_builtin_template_doc(
            "qwen3_vl_vision"
        )
        vision_manifest = {
            "config": {
                "model": "qwen3_vl_vision",
                "arch": "qwen3_vl_vision",
                "vision_patch_projection_reduction_policy":
                    "pytorch_onednn_conv3d_exact",
                "position_interpolation_policy": "align_corners_bilinear",
                "vision_position_storage_boundary": "bf16",
                "vision_layernorm_storage_boundary": "bf16",
                "vision_layernorm_reduction_policy": "pytorch_welford_exact",
                "vision_mrope_storage_boundary": "bf16",
                "vision_mrope_reduction_policy": "pytorch_mkl_exact",
                "vision_projection_storage_boundary": "bf16",
                "vision_projection_reduction_policy":
                    "pytorch_onednn_brgemm_exact",
                "vision_attention_storage_boundary": "bf16",
                "vision_residual_storage_boundary": "bf16",
                "vision_activation_storage_boundary": "bf16",
            },
            "template": vision_template,
        }
        vision_plans = build_ir_v8._resolve_manifest_execution_contracts(
            vision_manifest, "prefill"
        )
        expected_vision_operations = {
            "vision.frontend.patch_projection.pytorch_onednn_exact",
            "vision.frontend.position",
            "vision.layer.mrope.pytorch_mkl",
            "vision.layer.layernorm.pytorch_welford_exact",
            "vision.layer.qkv_projection.pytorch_onednn_exact",
            "vision.layer.attention",
            "vision.layer.out_projection.pytorch_onednn_exact",
            "vision.layer.residual",
            "vision.layer.mlp_projection.pytorch_onednn_exact",
            "vision.layer.mlp_activation",
            "vision.projector.activation.pytorch_sleef_exact",
            "vision.projector.projection.pytorch_onednn_exact",
        }
        vision_by_operation = {
            plan["operation"]: plan for plan in vision_plans
        }
        self.assertTrue(
            expected_vision_operations.issubset(vision_by_operation),
            sorted(vision_by_operation),
        )
        selected_kernel_ids.update(
            str(vision_by_operation[name]["kernel"]["id"])
            for name in expected_vision_operations
        )

        resolved_vision_attention = attention_resolver.resolve_contract(
            copy.deepcopy(vision_template),
            copy.deepcopy(attention_contracts),
            copy.deepcopy(attention_kernels),
            operation="vision_encoder.attention",
            phase="prefill",
            mode="bringup",
            source_circuit_path=V8_ROOT / "circuits" / "qwen3_vl_vision.json",
        )
        selected_kernel_ids.add(
            str(resolved_vision_attention["kernel"]["id"])
        )

        schedule = (
            template["contract"]["multimodal_bridge"]["prefill_schedules"][
                "unified_mixed"
            ]
        )
        selected_kernel_ids.update(
            {
                schedule["deepstack_injection"]["kernel_id"],
                schedule["position_transform"]["kernel_id"],
            }
        )
        selected_kernel_ids.update(
            {
                "embedding_forward_bf16_fp32",
                "feature_concat",
                "kv_cache_store_batch_bf16",
                "kv_cache_store_bf16",
                "spatial_merge_contiguous_tiled",
                "split_qkv_packed_head_major_forward",
                "vision_position_ids_2d_merge",
            }
        )

        self.assertNotIn("gemv_bf16", selected_kernel_ids)
        for kernel_id in sorted(selected_kernel_ids):
            map_path = V8_ROOT / "kernel_maps" / f"{kernel_id}.json"
            self.assertTrue(map_path.is_file(), kernel_id)
            kernel_map = json.loads(map_path.read_text(encoding="utf-8"))
            tests = kernel_map.get("tests") or {}
            unit_tests = tests.get("unit") or []
            self.assertTrue(unit_tests, f"{kernel_id} has no executable oracle test")
            for relative in unit_tests:
                test_path = ROOT / str(relative).split("::", 1)[0]
                self.assertTrue(
                    test_path.is_file(),
                    f"{kernel_id} references missing oracle {relative}",
                )


if __name__ == "__main__":
    unittest.main()
