import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "audit_kernel_allocations_v8.py"
SPEC = importlib.util.spec_from_file_location("audit_kernel_allocations_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TestKernelAllocationAudit(unittest.TestCase):
    def test_scanner_ignores_comments_and_literals(self):
        source = '''
        void clean(void) {
            // malloc(4);
            const char *text = "free(ptr)";
        }
        void debt(void) {
            void *ptr = malloc(16);
            free(ptr);
        }
        '''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.c"
            path.write_text(source, encoding="utf-8")
            calls = AUDIT.scan_source(path)
        self.assertEqual(
            [(call["function"], call["allocator"]) for call in calls],
            [("debt", "malloc"), ("debt", "free")],
        )

    def test_checked_in_debt_does_not_regress(self):
        report = AUDIT.build_report()
        baseline = json.loads(AUDIT.BASELINE.read_text(encoding="utf-8"))
        AUDIT.validate_ratchet(report, baseline)
        self.assertEqual(report["counts"]["production_allocation_calls"], 56)
        self.assertEqual(report["counts"]["mapped_allocating_providers"], 7)
        self.assertEqual(
            report["counts"]["mapped_allocating_without_scratch_contract"], 7
        )
        self.assertEqual(
            [warning["code"] for warning in report["warnings"]],
            ["production_allocator_debt", "mapped_allocator_without_scratch"],
        )

    def test_new_allocator_identity_fails_closed(self):
        report = {
            "call_site_identities": {"src/kernels/new.c::bad::malloc": 1},
            "counts": {
                "production_allocation_calls": 1,
                "mapped_allocating_providers": 0,
                "mapped_allocating_without_scratch_contract": 0,
            },
        }
        baseline = {
            "maximum_call_site_identities": {},
            "maximum_production_allocation_calls": 100,
            "maximum_mapped_allocating_providers": 100,
            "maximum_mapped_allocating_without_scratch_contract": 100,
        }
        with self.assertRaisesRegex(RuntimeError, "new kernel allocator call sites"):
            AUDIT.validate_ratchet(report, baseline)

    def test_recurrent_v8_provider_is_allocation_free_and_scratch_owned(self):
        kernel_map = json.loads(
            (
                ROOT
                / "version/v8/kernel_maps/recurrent_conv_state_update_backward.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            kernel_map["impl"]["function"],
            "recurrent_conv_state_update_backward_workspace",
        )
        self.assertEqual([entry["name"] for entry in kernel_map["scratch"]], ["d_conv_total"])
        sources = {
            param["name"]: param["source"]
            for param in kernel_map["call_abi"]["params"]
        }
        self.assertEqual(sources["d_conv_total"], "scratch:d_conv_total")
        allocating = {
            row["function"] for row in AUDIT.build_report()["mapped_allocating_providers"]
        }
        self.assertNotIn("recurrent_conv_state_update_backward_workspace", allocating)

    def test_recurrent_workspace_shape_resolves_exact_bytes(self):
        build_ir_path = ROOT / "version/v8/scripts/build_ir_v8.py"
        sys.path.insert(0, str(build_ir_path.parent))
        spec = importlib.util.spec_from_file_location("build_ir_v8_allocation_test", build_ir_path)
        assert spec is not None and spec.loader is not None
        build_ir = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_ir)
        scratch = {
            "dtype": "fp32",
            "shape": ["num_seqs", "conv_total_tokens", "ssm_conv_channels"],
            "size_resolution": "required",
        }
        size = build_ir._kernel_scratch_size_bytes(
            scratch,
            {"seq_len": 9},
            {"num_seqs": 2, "ssm_conv_history": 3, "ssm_conv_channels": 48},
        )
        self.assertEqual(size, 2 * (3 + 9) * 48 * 4)

    def test_selected_attention_and_amx_providers_are_workspace_owned(self):
        cases = {
            "attention_forward_causal_head_major_gqa_flash_strided_f16kv.json": (
                "attention_forward_causal_head_major_gqa_flash_strided_f16kv_workspace",
                "rounded_kv",
            ),
            "gemm_nt_bf16_amx_bf16_storage.json": (
                "gemm_nt_bf16_amx_bf16_storage_workspace",
                "activation_bf16",
            ),
        }
        allocating = {
            row["function"] for row in AUDIT.build_report()["mapped_allocating_providers"]
        }
        for filename, (function, scratch_name) in cases.items():
            with self.subTest(filename=filename):
                kernel_map = json.loads(
                    (ROOT / "version/v8/kernel_maps" / filename).read_text(encoding="utf-8")
                )
                self.assertEqual(kernel_map["impl"]["function"], function)
                scratch = {entry["name"]: entry for entry in kernel_map["scratch"]}
                self.assertEqual(scratch[scratch_name]["size_resolution"], "required")
                sources = {
                    param["name"]: param["source"]
                    for param in kernel_map["call_abi"]["params"]
                }
                self.assertEqual(sources[scratch_name if scratch_name == "rounded_kv" else "a_bf16"], f"scratch:{scratch_name}")
                self.assertNotIn(function, allocating)

    def test_segmented_f16_attention_maps_share_planner_owned_workspace(self):
        filenames = [
            "attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract.json",
            "attention_forward_causal_head_major_gqa_prefill_append_f16cache_single_range.json",
            "attention_forward_causal_head_major_gqa_prefill_append_f16cache_flash_auto_qtile64.json",
        ]
        function = (
            "attention_forward_causal_head_major_gqa_prefill_append_"
            "f16cache_contract_workspace"
        )
        allocating = {
            row["function"] for row in AUDIT.build_report()["mapped_allocating_providers"]
        }
        for filename in filenames:
            with self.subTest(filename=filename):
                kernel_map = json.loads(
                    (ROOT / "version/v8/kernel_maps" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(kernel_map["impl"]["function"], function)
                self.assertEqual(
                    kernel_map["scratch"],
                    [{
                        "name": "token_workspace",
                        "dtype": "fp32",
                        "shape": [2, "H", "D"],
                        "size_resolution": "required",
                        "lifetime": "kernel_call",
                        "desc": kernel_map["scratch"][0]["desc"],
                    }],
                )
                sources = {
                    param["name"]: param["source"]
                    for param in kernel_map["call_abi"]["params"]
                }
                self.assertEqual(sources["token_workspace"], "scratch:token_workspace")
                self.assertEqual(
                    sources["token_workspace_bytes"],
                    "scratch_size:token_workspace",
                )
                self.assertNotIn(function, allocating)

    def test_segmented_f16_attention_workspace_resolves_exact_bytes(self):
        build_ir_path = ROOT / "version/v8/scripts/build_ir_v8.py"
        sys.path.insert(0, str(build_ir_path.parent))
        spec = importlib.util.spec_from_file_location(
            "build_ir_v8_attention_workspace_test", build_ir_path
        )
        assert spec is not None and spec.loader is not None
        build_ir = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_ir)
        scratch = {
            "dtype": "fp32",
            "shape": [2, "H", "D"],
            "size_resolution": "required",
        }
        size = build_ir._kernel_scratch_size_bytes(
            scratch,
            {"num_heads": 16, "head_dim": 128},
            {},
        )
        self.assertEqual(size, 2 * 16 * 128 * 4)


if __name__ == "__main__":
    unittest.main()
