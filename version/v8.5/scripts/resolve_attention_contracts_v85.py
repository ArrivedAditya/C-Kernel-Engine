#!/usr/bin/env python3
"""Resolve v8.5 attention semantics against kernel capabilities.

This resolver is intentionally model-name blind. It consumes a circuit,
a canonical contract registry, and a kernel capability overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


V85_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V85_ROOT.parents[1]
DEFAULT_CONTRACTS = V85_ROOT / "contracts" / "attention_reductions.json"
DEFAULT_KERNELS = V85_ROOT / "kernel_maps" / "attention_contracts.overlay.json"
VALID_STATES = {"unresolved", "observed", "validated"}
AMBIGUOUS_IDS = {"fp16", "f16", "bf16", "fp32", "f32", "fast", "strict"}


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except FileNotFoundError as exc:
        raise ContractError(f"Contract input does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ContractError(f"Expected a JSON object in {path}")
    return doc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_keys(node: Dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in node]
    if missing:
        raise ContractError(f"{context} is missing required fields: {', '.join(missing)}")


def validate_state(value: Any, context: str) -> str:
    state = str(value or "").strip()
    if state not in VALID_STATES:
        raise ContractError(
            f"{context} has invalid validation state {state!r}; expected one of {sorted(VALID_STATES)}"
        )
    return state


def validate_contract_registry(doc: Dict[str, Any]) -> None:
    require_keys(doc, ("contracts", "required_semantic_fields"), "attention contract registry")
    contracts = doc["contracts"]
    fields = doc["required_semantic_fields"]
    if not isinstance(contracts, dict) or not contracts:
        raise ContractError("attention contract registry must define at least one contract")
    if not isinstance(fields, list) or not fields:
        raise ContractError("required_semantic_fields must be a non-empty list")
    for contract_id, contract in contracts.items():
        if contract_id.lower() in AMBIGUOUS_IDS:
            raise ContractError(f"Ambiguous reduction contract ID is forbidden: {contract_id}")
        if not isinstance(contract, dict):
            raise ContractError(f"Reduction contract {contract_id} must be an object")
        require_keys(contract, fields, f"reduction contract {contract_id}")
        validate_state(contract.get("status"), f"reduction contract {contract_id}")
        partition = contract.get("partition")
        if not isinstance(partition, dict) or not partition.get("kind"):
            raise ContractError(f"reduction contract {contract_id}.partition requires kind")


def validate_kernel_overlay(doc: Dict[str, Any]) -> None:
    kernels = doc.get("kernels")
    if not isinstance(kernels, dict) or not kernels:
        raise ContractError("kernel numerical contract overlay must define kernels")
    for kernel_id, kernel in kernels.items():
        if not isinstance(kernel, dict):
            raise ContractError(f"Kernel capability {kernel_id} must be an object")
        require_keys(
            kernel,
            ("op", "mode", "base_kernel_map", "provides", "supported_reductions"),
            f"kernel capability {kernel_id}",
        )
        map_path = (REPO_ROOT / str(kernel["base_kernel_map"])).resolve()
        base_map = load_json(map_path)
        if base_map.get("id") != kernel_id:
            raise ContractError(
                f"Kernel capability {kernel_id} points to map with id {base_map.get('id')!r}: {map_path}"
            )
        base_function = (base_map.get("impl") or {}).get("function")
        provides = kernel["provides"]
        if not isinstance(provides, dict) or not provides:
            raise ContractError(f"Kernel capability {kernel_id}.provides must be a non-empty object")
        for capability, values in provides.items():
            if not isinstance(values, list) or not values:
                raise ContractError(
                    f"Kernel capability {kernel_id}.provides[{capability!r}] must be a non-empty list"
                )
        supported = kernel["supported_reductions"]
        if not isinstance(supported, dict) or not supported:
            raise ContractError(f"Kernel capability {kernel_id} must support at least one reduction")
        for reduction_id, implementation in supported.items():
            if not isinstance(implementation, dict):
                raise ContractError(f"Kernel implementation {kernel_id}/{reduction_id} must be an object")
            require_keys(
                implementation,
                ("status", "function", "explicit_selector"),
                f"kernel implementation {kernel_id}/{reduction_id}",
            )
            if implementation["function"] != base_function:
                raise ContractError(
                    f"Kernel implementation {kernel_id}/{reduction_id} names function "
                    f"{implementation['function']!r}, but kernel map names {base_function!r}"
                )
            advertised = provides.get("numerics.attention_reduction", [])
            if reduction_id not in advertised:
                raise ContractError(
                    f"Kernel implementation {kernel_id}/{reduction_id} is not advertised by provides"
                )


def circuit_path(circuit: str) -> Path:
    return V85_ROOT / "circuits" / f"{circuit}.json"


def _capability_satisfies(provides: Dict[str, Any], requires: Dict[str, Any]) -> bool:
    for capability, required in requires.items():
        available = provides.get(capability)
        if not isinstance(available, list) or required not in available:
            return False
    return True


def resolve_contract(
    circuit_doc: Dict[str, Any],
    contract_doc: Dict[str, Any],
    kernel_doc: Dict[str, Any],
    *,
    operation: str,
    phase: str,
    mode: str,
    source_circuit_path: Path | None = None,
) -> Dict[str, Any]:
    validate_contract_registry(contract_doc)
    validate_kernel_overlay(kernel_doc)
    if mode not in {"bringup", "production"}:
        raise ContractError(f"Unknown resolution mode: {mode}")

    operations = circuit_doc.get("operations")
    if not isinstance(operations, dict) or operation not in operations:
        raise ContractError(f"Circuit does not declare operation contract: {operation}")
    operation_doc = operations[operation]
    if not isinstance(operation_doc, dict):
        raise ContractError(f"Circuit operation {operation} must be an object")
    require_keys(operation_doc, ("op", "phases"), f"circuit operation {operation}")
    phases = operation_doc["phases"]
    if not isinstance(phases, dict) or phase not in phases:
        raise ContractError(f"Circuit operation {operation} does not declare phase: {phase}")
    request = phases[phase]
    if not isinstance(request, dict):
        raise ContractError(f"Circuit request {operation}.{phase} must be an object")
    require_keys(request, ("requires", "validation"), f"circuit request {operation}.{phase}")
    requires = request["requires"]
    if not isinstance(requires, dict) or not requires:
        raise ContractError(f"Circuit request {operation}.{phase}.requires must be a non-empty object")

    reduction_id = str(requires.get("numerics.attention_reduction", "")).strip()
    if reduction_id.lower() in AMBIGUOUS_IDS:
        raise ContractError(
            f"Circuit requested ambiguous reduction {reduction_id!r}; request a complete registered contract"
        )
    contracts = contract_doc["contracts"]
    if reduction_id not in contracts:
        raise ContractError(f"Unknown reduction contract requested: {reduction_id}")
    contract = contracts[reduction_id]

    kernels = kernel_doc.get("kernels")
    candidates = []
    for candidate_id, candidate in kernels.items():
        if candidate.get("op") != operation_doc["op"]:
            continue
        if str(candidate.get("mode", "")).strip() != phase:
            continue
        if not _capability_satisfies(candidate.get("provides", {}), requires):
            continue
        supported = candidate.get("supported_reductions")
        if isinstance(supported, dict) and reduction_id in supported:
            candidates.append((candidate_id, candidate))
    if not candidates:
        raise ContractError(
            f"No kernel satisfies circuit requirements for {operation}.{phase}: {json.dumps(requires, sort_keys=True)}"
        )
    if len(candidates) > 1:
        raise ContractError(
            f"Ambiguous kernel selection for {operation}.{phase}: {[item[0] for item in candidates]}"
        )
    kernel_id, kernel = candidates[0]
    supported = kernel.get("supported_reductions")
    implementation = supported[reduction_id]

    request_state = validate_state(request.get("validation"), f"circuit request {operation}.{phase}")
    contract_state = validate_state(contract.get("status"), f"reduction contract {reduction_id}")
    implementation_state = validate_state(
        implementation.get("status"), f"kernel implementation {kernel_id}/{reduction_id}"
    )
    explicit_selector = bool(implementation.get("explicit_selector", False))

    blockers = []
    if request_state != "validated":
        blockers.append(f"circuit request is {request_state}")
    if contract_state != "validated":
        blockers.append(f"contract definition is {contract_state}")
    if implementation_state != "validated":
        blockers.append(f"kernel implementation is {implementation_state}")
    if not explicit_selector:
        blockers.append("kernel uses legacy implicit selection")
    if mode == "production" and blockers:
        raise ContractError(
            f"Production contract resolution rejected {operation}.{phase}: " + "; ".join(blockers)
        )

    base_circuit = str(circuit_doc.get("base_circuit", "")).strip()
    base_path = (REPO_ROOT / base_circuit).resolve() if base_circuit else None
    if base_path is not None and not base_path.is_file():
        raise ContractError(f"Base circuit does not exist: {base_path}")

    return {
        "schema": "cke.resolved_attention_contract",
        "schema_version": 1,
        "engine_contract_version": "8.5",
        "circuit": circuit_doc.get("circuit"),
        "operation": operation,
        "phase": phase,
        "resolution_mode": mode,
        "kernel": {
            "id": kernel_id,
            "function": implementation.get("function"),
            "implementation_status": implementation_state,
            "explicit_selector": explicit_selector,
            "selector": implementation.get("selector")
        },
        "reduction": {
            "id": reduction_id,
            "definition_status": contract_state,
            "semantics": {key: contract[key] for key in contract_doc["required_semantic_fields"]}
        },
        "request_status": request_state,
        "requirements": requires,
        "production_blockers": blockers,
        "inputs": {
            "circuit": str(source_circuit_path) if source_circuit_path else None,
            "base_circuit": str(base_path) if base_path else None,
            "base_circuit_sha256": sha256_file(base_path) if base_path else None
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", required=True, help="Circuit name, for example qwen3vl")
    parser.add_argument("--operation", default="decoder.attention")
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--mode", choices=("bringup", "production"), default="bringup")
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--kernel-overlay", type=Path, default=DEFAULT_KERNELS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_circuit_path = circuit_path(args.circuit)
    try:
        result = resolve_contract(
            load_json(source_circuit_path),
            load_json(args.contracts),
            load_json(args.kernel_overlay),
            operation=args.operation,
            phase=args.phase,
            mode=args.mode,
            source_circuit_path=source_circuit_path,
        )
    except ContractError as exc:
        print(f"v8.5 contract resolution: FAIL: {exc}")
        return 2

    rendered = json.dumps(result, indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
