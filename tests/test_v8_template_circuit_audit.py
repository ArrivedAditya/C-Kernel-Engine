from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "version/v8/scripts/audit_template_circuit_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_template_circuit_v8", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _op(op_id: int, layer: int, op: str, inputs: dict, outputs: dict) -> dict:
    return {
        "op_id": op_id,
        "layer": layer,
        "op": op,
        "kernel": op,
        "dataflow": {"inputs": inputs, "outputs": outputs},
    }


class TemplateCircuitAuditTests(unittest.TestCase):
    def test_mamba_in_proj_must_consume_block_rmsnorm_output(self) -> None:
        audit = _load_module()
        ir = {
            "ops": [
                _op(0, -1, "dense_embedding_lookup", {}, {"out": {"slot": "main_stream", "dtype": "fp32"}}),
                _op(1, 0, "residual_save", {"src": {"from_op": 0, "slot": "main_stream"}}, {"dst": {"slot": "residual", "dtype": "fp32"}}),
                _op(2, 0, "block_rmsnorm", {"input": {"from_op": 0, "slot": "main_stream"}}, {"output": {"slot": "layer_input", "dtype": "fp32"}}),
                _op(3, 0, "mamba_in_proj", {"x": {"from_op": 0, "slot": "main_stream"}}, {"y": {"slot": "recurrent_packed", "dtype": "fp32"}}),
            ]
        }
        errors = audit.audit_ir1(ir)
        self.assertTrue(any("mamba_in_proj.x" in e and "expected block_rmsnorm" in e for e in errors), errors)

    def test_fixed_mamba_circuit_and_lowered_buffers_pass(self) -> None:
        audit = _load_module()
        ir = {
            "ops": [
                _op(0, -1, "dense_embedding_lookup", {}, {"out": {"slot": "main_stream", "dtype": "fp32"}}),
                _op(1, 0, "residual_save", {"src": {"from_op": 0, "slot": "main_stream"}}, {"dst": {"slot": "residual", "dtype": "fp32"}}),
                _op(2, 0, "block_rmsnorm", {"input": {"from_op": 0, "slot": "main_stream"}}, {"output": {"slot": "layer_input", "dtype": "fp32"}}),
                _op(3, 0, "mamba_in_proj", {"x": {"from_op": 2, "slot": "layer_input"}}, {"y": {"slot": "recurrent_packed", "dtype": "fp32"}}),
                _op(4, 0, "mamba_in_proj_split", {"projected": {"from_op": 3, "slot": "recurrent_packed"}}, {"gate": {"slot": "recurrent_z"}, "hidden_bc": {"slot": "recurrent_conv_qkv"}, "dt": {"slot": "recurrent_g"}}),
            ]
        }
        lowered = {
            "operations": [
                {"idx": 2, "layer": 0, "op": "block_rmsnorm", "activations": {"input": {"buffer": "embedded_input"}}, "outputs": {"output": {"buffer": "layer_input"}}},
                {"idx": 3, "layer": 0, "op": "mamba_in_proj", "activations": {"x": {"buffer": "layer_input"}}, "outputs": {"y": {"buffer": "recurrent_packed"}}},
                {"idx": 4, "layer": 0, "op": "mamba_in_proj_split", "activations": {"projected": {"buffer": "recurrent_packed"}}, "outputs": {}},
            ]
        }
        self.assertEqual(audit.audit_ir1(ir), [])
        self.assertEqual(audit.audit_lowered(lowered), [])

    def test_cli_reports_generated_c_mamba_input_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c_path = Path(td) / "model_v8.c"
            c_path.write_text(
                """
    /* Op 3: gemv_bf16 (mamba_in_proj) layer=0 section=body */
    gemv_bf16(
        (float*)(model->bump + A_RECURRENT_PACKED),
        (const void*)(model->bump + W_LAYER_0_MAMBA_IN_PROJ),
        (const float*)(model->bump + A_EMBEDDED_INPUT),
        22656,
        4480
    );
""",
                encoding="utf-8",
            )
            audit = _load_module()
            errors = audit.audit_c_source(c_path)
            self.assertEqual(len(errors), 1)
            self.assertIn("expected=A_LAYER_INPUT", errors[0])


if __name__ == "__main__":
    unittest.main()
