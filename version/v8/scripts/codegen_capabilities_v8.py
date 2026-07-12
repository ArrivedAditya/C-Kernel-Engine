"""Read resolved code-generation capabilities without interpreting C symbols."""

from __future__ import annotations

from typing import Any, Dict


def resolved_quantized_linear_emission(op: Dict[str, Any]) -> Dict[str, Any] | None:
    execution = op.get("resolved_execution")
    if not isinstance(execution, dict) or not execution.get("numerical_contract"):
        return None
    implementation = execution.get("implementation")
    if not isinstance(implementation, dict):
        raise RuntimeError("resolved quantized linear operation has no implementation metadata")
    weight = implementation.get("weight_storage")
    activation = implementation.get("activation_storage")
    diagnostics = implementation.get("diagnostic_providers")
    if not all(isinstance(value, dict) for value in (weight, activation, diagnostics)):
        raise RuntimeError(
            "resolved quantized linear operation is missing map-owned storage or diagnostic providers"
        )
    result = {
        "weight_format": str(weight["format"]),
        "weight_block_elements": int(weight["block_elements"]),
        "weight_block_bytes": int(weight["block_bytes"]),
        "activation_format": str(activation["format"]),
        "activation_block_elements": int(activation["block_elements"]),
        "fp32_activation_function": str(diagnostics["fp32_activation"]),
    }
    row_provider = diagnostics.get("row_quantized")
    if row_provider is not None:
        result["row_quantized_function"] = str(row_provider)
    return result


def is_q4_q6_q8_linear(capability: Dict[str, Any] | None) -> bool:
    return bool(
        capability
        and capability["weight_format"] in {"q4_k", "q6_k"}
        and capability["activation_format"] == "q8_k"
    )
