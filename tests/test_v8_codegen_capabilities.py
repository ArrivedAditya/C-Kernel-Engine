from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = _load("codegen_core_v8_capability_tests", SCRIPTS / "codegen_core_v8.py")
prefill = _load("codegen_prefill_v8_capability_tests", SCRIPTS / "codegen_prefill_v8.py")
resolver = _load(
    "resolve_attention_contracts_v8_capability_tests",
    SCRIPTS / "resolve_attention_contracts_v8.py",
)

GOVERNED_SYMBOLS = {
    "gemv_q4_k_q8_k",
    "gemv_q6_k_q8_k",
    "gemm_nt_q4_k_q8_k",
    "gemm_nt_q6_k_q8_k",
}


def _execution(weight: str, *, prefill_mode: bool = False) -> dict:
    is_q4 = weight == "q4_k"
    prefix = "gemm_nt" if prefill_mode else "gemv"
    return {
        "numerical_contract": f"{weight}_x_q8_k_fp32_block_order",
        "implementation": {
            "weight_storage": {
                "format": weight,
                "block_elements": 256,
                "block_bytes": 144 if is_q4 else 210,
            },
            "activation_storage": {"format": "q8_k", "block_elements": 256},
            "diagnostic_providers": {
                "fp32_activation": f"{prefix}_{weight}",
                **(
                    {"row_quantized": f"renamed_{weight}_row_provider"}
                    if prefill_mode
                    else {}
                ),
            },
        },
    }


def _decode_op(op_name: str, weight: str) -> dict:
    return {
        "idx": 1,
        "op": op_name,
        "function": f"renamed_{weight}_production",
        "layer": 0,
        "section": "body",
        "resolved_execution": _execution(weight),
        "args": [
            {"name": "y", "expr": "Y"},
            {"name": "W", "expr": "W"},
            {"name": "x_q8", "expr": "XQ"},
            {"name": "M", "expr": "M"},
            {"name": "K", "expr": "K"},
        ],
    }


def _prefill_op(op_name: str, weight: str) -> dict:
    return {
        "idx": 1,
        "op": op_name,
        "function": f"renamed_{weight}_production",
        "layer": 0,
        "section": "body",
        "resolved_execution": _execution(weight, prefill_mode=True),
        "args": [
            {"name": "A", "expr": "A"},
            {"name": "B", "expr": "B"},
            {"name": "bias", "expr": "BIAS"},
            {"name": "C", "expr": "C"},
            {"name": "M", "expr": "M", "source": "dim:_m"},
            {"name": "N", "expr": "N"},
            {"name": "K", "expr": "K"},
        ],
    }


class V8CodegenCapabilityTests(unittest.TestCase):
    def test_kernel_maps_own_q4_q6_storage_and_diagnostic_providers(self) -> None:
        kernels = resolver.load_kernel_execution_capabilities()["kernels"]
        expected = {
            "gemv_q4_k_q8_k": ("q4_k", 144, "gemv_q4_k", None),
            "gemv_q6_k_q8_k": ("q6_k", 210, "gemv_q6_k", None),
            "gemm_nt_q4_k_q8_k": ("q4_k", 144, "gemm_nt_q4_k", "gemv_q4_k_q8_k"),
            "gemm_nt_q6_k_q8_k": ("q6_k", 210, "gemm_nt_q6_k", "gemv_q6_k_q8_k"),
        }
        for kernel_id, (weight, block_bytes, fp32, row) in expected.items():
            with self.subTest(kernel=kernel_id):
                implementation = kernels[kernel_id]["implementation"]
                self.assertEqual(implementation["weight_storage"]["format"], weight)
                self.assertEqual(implementation["weight_storage"]["block_bytes"], block_bytes)
                self.assertEqual(implementation["activation_storage"]["format"], "q8_k")
                self.assertEqual(implementation["diagnostic_providers"]["fp32_activation"], fp32)
                self.assertEqual(implementation["diagnostic_providers"].get("row_quantized"), row)

    def test_missing_storage_capability_is_a_hard_failure(self) -> None:
        source = ROOT / "version" / "v8" / "kernel_maps" / "gemv_q4_k_q8_k.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        del document["implementation"]["weight_storage"]
        with tempfile.TemporaryDirectory(prefix="cke_missing_linear_storage_") as td:
            path = Path(td) / source.name
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(resolver.ContractError, "no explicit storage capability"):
                resolver.load_kernel_execution_capabilities(Path(td))

    def test_storage_capability_must_match_numerical_contract(self) -> None:
        source = ROOT / "version" / "v8" / "kernel_maps" / "gemv_q4_k_q8_k.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        document["implementation"]["weight_storage"]["format"] = "q6_k"
        with tempfile.TemporaryDirectory(prefix="cke_wrong_linear_storage_") as td:
            path = Path(td) / source.name
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(resolver.ContractError, "disagrees with its contract"):
                resolver.load_kernel_execution_capabilities(Path(td))

    def test_missing_diagnostic_provider_is_a_hard_failure(self) -> None:
        source = ROOT / "version" / "v8" / "kernel_maps" / "gemm_nt_q4_k_q8_k.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        del document["implementation"]["diagnostic_providers"]["row_quantized"]
        with tempfile.TemporaryDirectory(prefix="cke_missing_linear_diagnostic_") as td:
            path = Path(td) / source.name
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(resolver.ContractError, "no row-quantized provider"):
                resolver.load_kernel_execution_capabilities(Path(td))

    def test_decode_projection_categories_ignore_production_symbol_spelling(self) -> None:
        for weight in ("q4_k", "q6_k"):
            for op_name in ("q_proj", "k_proj", "v_proj", "logits", "out_proj", "mlp_down"):
                with self.subTest(weight=weight, op=op_name):
                    code = core.emit_op(_decode_op(op_name, weight))
                    self.assertIn(f"renamed_{weight}_production(", code)
                    self.assertIn(f"gemv_{weight}(", code)

    def test_decode_gate_up_uses_map_owned_row_layout(self) -> None:
        for weight, block_bytes in (("q4_k", 144), ("q6_k", 210)):
            with self.subTest(weight=weight):
                code = core.emit_op(_decode_op("mlp_gate_up", weight))
                self.assertIn(f"/ 256u) * {block_bytes}u", code)
                self.assertIn(f"gemv_{weight}(", code)
                self.assertIn(f"renamed_{weight}_production(", code)

    def test_prefill_projection_categories_ignore_production_symbol_spelling(self) -> None:
        for weight in ("q4_k", "q6_k"):
            with self.subTest(weight=weight, op="out_proj"):
                code = prefill.emit_prefill_op(_prefill_op("out_proj", weight), 1, {})
                self.assertIn(f"renamed_{weight}_production(", code)
                self.assertIn(f"gemm_nt_{weight}(", code)
            with self.subTest(weight=weight, op="mlp_gate_up"):
                code = prefill.emit_prefill_op(_prefill_op("mlp_gate_up", weight), 1, {})
                self.assertIn(f"renamed_{weight}_row_provider(", code)
                self.assertIn(f"gemm_nt_{weight}(", code)

    def test_prefill_override_requires_resolved_capability(self) -> None:
        op = _prefill_op("mlp_down", "q4_k")
        del op["resolved_execution"]
        with self.assertRaisesRegex(RuntimeError, "requires resolved Q4/Q6"):
            prefill._emit_prefill_gemm_fp32_override(
                op,
                1,
                debug_flag_name="debug",
                debug_input_name="input",
            )

    def test_codegen_conditions_do_not_name_governed_q4_q6_providers(self) -> None:
        for filename in ("codegen_core_v8.py", "codegen_prefill_v8.py"):
            tree = ast.parse((SCRIPTS / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.If, ast.IfExp, ast.While)):
                    continue
                literals = {
                    child.value
                    for child in ast.walk(node.test)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                }
                with self.subTest(file=filename, line=node.lineno):
                    self.assertFalse(literals & GOVERNED_SYMBOLS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
