#!/usr/bin/env python3
"""Resolve circuit numerical requirements to one exact kernel implementation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from jsonschema import Draft202012Validator


V8_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V8_ROOT.parents[1]
DEFAULT_CONTRACTS = V8_ROOT / "contracts" / "numerical_execution.json"
DEFAULT_KERNELS = V8_ROOT / "kernel_maps"
SCHEMA_ROOT = V8_ROOT / "schemas"
CONTRACT_SCHEMA = SCHEMA_ROOT / "numerical_execution_contract_registry.schema.json"
REQUIREMENTS_SCHEMA = SCHEMA_ROOT / "numerical_required_contracts.schema.json"
CAPABILITY_SCHEMA = SCHEMA_ROOT / "numerical_kernel_capability.schema.json"
RESOLVED_SCHEMA = SCHEMA_ROOT / "resolved_numerical_execution_contract.schema.json"
VALID_STATES = {"unresolved", "observed", "validated"}
AMBIGUOUS_IDS = {"auto", "default", "fast", "strict", "fp16", "bf16", "fp32"}


class ContractError(RuntimeError):
    pass


def hard_fault(summary: str, detail: str, remediation: str) -> ContractError:
    return ContractError(
        f"HARD CONTRACT FAULT: {summary}\n  {detail}\n  Fix: {remediation}\n"
        "  Do not add ranking, fallback dispatch, or tolerance relaxation."
    )


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot load contract JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Expected JSON object in {path}")
    return value


def validate_schema(value: Dict[str, Any], path: Path, context: str) -> None:
    errors = sorted(
        Draft202012Validator(load_json(path)).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise hard_fault(
            f"{context} violates {path.name}",
            f"At {location}: {error.message}",
            "correct the versioned circuit, contract registry, or kernel capability.",
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_contract_registry(doc: Dict[str, Any]) -> None:
    validate_schema(doc, CONTRACT_SCHEMA, "numerical execution contract registry")
    for contract_id, contract in doc["contracts"].items():
        if contract_id.lower() in AMBIGUOUS_IDS:
            raise hard_fault(
                f"ambiguous numerical contract ID {contract_id!r}",
                "A dtype or policy label does not define storage, rounding, reduction, and threading.",
                "use a stable semantic contract ID with complete execution semantics.",
            )
        transform = contract.get("position_transform")
        if isinstance(transform, dict):
            if transform["position_rank"] != len(transform["axis_order"]):
                raise hard_fault(
                    f"position rank/axis mismatch in {contract_id!r}",
                    f"position_rank={transform['position_rank']} axis_order={transform['axis_order']}",
                    "make the position rank equal the number of named axes.",
                )
            section_semantics = transform["section_interpretation"]
            if transform["pairing"] == "multi_section" and section_semantics not in {
                "axis_selection",
                "interleaved_axis_selection",
            }:
                raise hard_fault(
                    f"multi-section position contract {contract_id!r} redefines rotary width",
                    "Qwen-style M-RoPE sections select position axes; rotary_width remains independent.",
                    "use axis_selection or interleaved_axis_selection and validate the full rotary width separately.",
                )
            rotary_width = transform.get("rotary_width_value")
            head_width = transform.get("head_width_value")
            mrope_width = transform.get("mrope_n_dims_value")
            if rotary_width is not None and head_width is not None and rotary_width > head_width:
                raise hard_fault(
                    f"position contract {contract_id!r} rotates beyond the head width",
                    f"rotary_width={rotary_width}, head_width={head_width}",
                    "declare the actual partial/full rotary width independently of section routing.",
                )
            if mrope_width is not None and rotary_width is not None and mrope_width != rotary_width:
                raise hard_fault(
                    f"position contract {contract_id!r} has inconsistent M-RoPE width",
                    f"mrope_n_dims={mrope_width}, required_rotary_width={rotary_width}",
                    "make mrope_n_dims equal the required full rotary width; sections only select axes.",
                )


def _validate_capability_against_contract(
    kernel_id: str,
    capability: Dict[str, Any],
    contract: Dict[str, Any],
) -> None:
    validate_schema(capability, CAPABILITY_SCHEMA, f"kernel {kernel_id} numerical capability")
    semantics = contract
    threading = semantics["threading"]
    reduction = semantics["reduction"]
    advertised = capability["arithmetic"]
    expected = {
        "partial_accumulator": reduction["partial_accumulator"],
        "merge_order": reduction["merge_order"],
        "deterministic": threading["deterministic"],
        "thread_count_changes_arithmetic_order": threading["thread_count_changes_arithmetic_order"],
        "split_strategy": threading["split_strategy"],
    }
    if advertised != expected:
        raise hard_fault(
            f"kernel {kernel_id!r} arithmetic metadata disagrees with contract {capability['contract_id']!r}",
            f"expected={expected}, advertised={advertised}",
            "correct the kernel capability or bind it to the contract it actually implements.",
        )
    partitions = capability["implementation"]["threading"]["work_partition"]
    if threading["work_partition"] not in partitions:
        raise hard_fault(
            f"kernel {kernel_id!r} cannot satisfy work partition {threading['work_partition']!r}",
            f"advertised partitions={partitions}",
            "advertise the exact partition only after validating the implementation.",
        )


def _load_operation_interface(doc: Dict[str, Any], path: Path) -> Optional[Dict[str, Any]]:
    """Build one canonical logical interface from a hardened kernel map."""
    interface_id = str(doc.get("operation_interface", "") or "").strip()
    if not interface_id:
        return None

    ports = []
    groups = (
        ("input", "inputs", "read"),
        ("weight", "weights", "read"),
        ("output", "outputs", "write"),
    )
    for role, field, expected_access in groups:
        values = doc.get(field)
        if not isinstance(values, list):
            raise hard_fault(
                f"kernel interface {interface_id!r} has no {field}",
                f"source={path}",
                "declare every logical port in the kernel map before enabling map-first resolution.",
            )
        for value in values:
            if not isinstance(value, dict):
                raise hard_fault(
                    f"kernel interface {interface_id!r} has a malformed {role} port",
                    f"source={path}, port={value!r}",
                    "declare the port as an object with complete physical semantics.",
                )
            required = (
                "name", "dtype", "shape", "layout", "access",
                "storage_class", "consumption",
            )
            missing = [key for key in required if value.get(key) in (None, "")]
            if missing:
                raise hard_fault(
                    f"kernel interface {interface_id!r} has an incomplete {role} port",
                    f"source={path}, port={value.get('name')!r}, missing={missing}",
                    "declare dtype, shape, layout, access, storage class, and consumption in the kernel map.",
                )
            # Append-style state ports (e.g., KV cache slices) are read_write
            # state outputs, not pure write outputs; all other ports keep the
            # strict per-role access contract.
            if (
                role == "output"
                and value["storage_class"] == "state"
                and value["access"] == "read_write"
            ):
                expected = "read_write"
            else:
                expected = expected_access
            if value["access"] != expected:
                raise hard_fault(
                    f"kernel interface {interface_id!r} has invalid {role} access",
                    f"source={path}, port={value['name']!r}, access={value['access']!r}",
                    f"declare {expected_access!r} access or use a separate state/in-place interface.",
                )
            ports.append({
                "role": role,
                "name": str(value["name"]),
                "dtype": str(value["dtype"]),
                "shape": copy.deepcopy(value["shape"]),
                "layout": str(value["layout"]),
                "access": str(value["access"]),
                "storage_class": str(value["storage_class"]),
                "consumption": str(value["consumption"]),
                **(
                    {"alias_of": str(value["alias_of"])}
                    if value.get("alias_of") not in (None, "")
                    else {}
                ),
                **(
                    {"may_alias": copy.deepcopy(value["may_alias"])}
                    if value.get("may_alias") not in (None, "")
                    else {}
                ),
            })

    identities = [(port["role"], port["name"]) for port in ports]
    if len(identities) != len(set(identities)):
        raise hard_fault(
            f"kernel interface {interface_id!r} has duplicate ports",
            f"source={path}, ports={identities}",
            "give every logical port one unique role/name identity.",
        )
    identity_set = {f"{role}:{name}" for role, name in identities}
    for port in ports:
        alias_of = port.get("alias_of")
        may_alias = port.get("may_alias")
        if alias_of is not None and may_alias is not None:
            raise hard_fault(
                f"kernel interface {interface_id!r} mixes required and optional aliasing",
                f"source={path}, port={port['name']!r}",
                "use alias_of for mandatory in-place storage or may_alias for optional in-place storage.",
            )
        if may_alias is not None:
            if (
                port["role"] != "output"
                or not isinstance(may_alias, list)
                or not may_alias
                or len(set(may_alias)) != len(may_alias)
                or any(target not in identity_set for target in may_alias)
                or any(not str(target).startswith("input:") for target in may_alias)
            ):
                raise hard_fault(
                    f"kernel interface {interface_id!r} has invalid optional aliases",
                    f"source={path}, port={port['name']!r}, may_alias={may_alias!r}",
                    "list one or more unique input:name ports with physically compatible storage.",
                )
            for target_id in may_alias:
                target_role, target_name = target_id.split(":", 1)
                target = next(
                    candidate
                    for candidate in ports
                    if candidate["role"] == target_role and candidate["name"] == target_name
                )
                for field in ("dtype", "shape", "layout", "storage_class"):
                    if port[field] != target[field]:
                        raise hard_fault(
                            f"kernel interface {interface_id!r} has an incompatible optional alias",
                            f"source={path}, output={port['name']!r}, target={target_id!r}, "
                            f"field={field!r}, output_value={port[field]!r}, target_value={target[field]!r}",
                            "make optionally aliased ports physically identical or remove may_alias.",
                        )
        if alias_of is None:
            continue
        if port["role"] != "output":
            raise hard_fault(
                f"kernel interface {interface_id!r} aliases a non-output port",
                f"source={path}, port={port['name']!r}, alias_of={alias_of!r}",
                "declare aliases on output ports and point them at one input or state port.",
            )
        if alias_of not in identity_set:
            raise hard_fault(
                f"kernel interface {interface_id!r} aliases an unknown port",
                f"source={path}, port={port['name']!r}, alias_of={alias_of!r}",
                "use a role:name target that exists in the same canonical interface.",
            )
        target_role, target_name = alias_of.split(":", 1)
        target = next(
            candidate
            for candidate in ports
            if candidate["role"] == target_role and candidate["name"] == target_name
        )
        for field in ("dtype", "shape", "layout", "storage_class"):
            if port[field] != target[field]:
                raise hard_fault(
                    f"kernel interface {interface_id!r} has an incompatible alias",
                    f"source={path}, output={port['name']!r}, target={alias_of!r}, "
                    f"field={field!r}, output_value={port[field]!r}, target_value={target[field]!r}",
                    "make aliased ports physically identical or declare an independent output.",
                )
    interface = {"id": interface_id, "op": str(doc.get("op", "")), "ports": ports}
    _validate_operation_interface_call_abi(doc, interface, path)
    return interface


def _validate_operation_interface_call_abi(
    doc: Dict[str, Any],
    interface: Dict[str, Any],
    path: Path,
) -> None:
    """Prove that one hardened logical interface reaches the C call boundary."""
    kernel_id = str(doc.get("id", "") or path.stem)
    call_abi = doc.get("call_abi")
    if not isinstance(call_abi, dict) or not isinstance(call_abi.get("params"), list):
        raise hard_fault(
            f"kernel {kernel_id!r} has an interface without a map-owned call ABI",
            f"source={path}, interface={interface['id']!r}",
            "declare the exact C argument binding in the same kernel map.",
        )

    logical_ports = {
        f"{port['role']}:{port['name']}": port
        for port in interface["ports"]
    }
    bindings: Dict[str, int] = {}
    param_ports: Dict[int, set[str]] = {}
    params = call_abi["params"]
    for index, param in enumerate(params):
        if not isinstance(param, dict):
            continue
        declared = param.get("ports")
        if declared is None:
            continue
        if (
            not isinstance(declared, list)
            or not declared
            or any(not isinstance(port_id, str) or not port_id for port_id in declared)
            or len(set(declared)) != len(declared)
        ):
            raise hard_fault(
                f"kernel {kernel_id!r} has malformed ABI port bindings",
                f"source={path}, parameter={param.get('name')!r}, ports={declared!r}",
                "bind the argument to one or more unique role:name logical ports.",
            )
        current = set(declared)
        unknown = sorted(current - set(logical_ports))
        if unknown:
            raise hard_fault(
                f"kernel {kernel_id!r} ABI references unknown logical ports",
                f"source={path}, parameter={param.get('name')!r}, unknown={unknown}",
                "use only ports declared by the operation interface.",
            )
        duplicates = sorted(port_id for port_id in current if port_id in bindings)
        if duplicates:
            raise hard_fault(
                f"kernel {kernel_id!r} ABI binds logical ports more than once",
                f"source={path}, parameter={param.get('name')!r}, duplicates={duplicates}",
                "give each logical port exactly one C argument owner.",
            )
        param_ports[index] = current
        for port_id in current:
            bindings[port_id] = index

    missing = sorted(set(logical_ports) - set(bindings))
    if missing:
        raise hard_fault(
            f"kernel {kernel_id!r} ABI does not represent every logical port",
            f"source={path}, interface={interface['id']!r}, missing={missing}",
            "add explicit call_abi.params[].ports bindings; do not infer them from argument names.",
        )

    for port_id, port in logical_ports.items():
        alias_of = port.get("alias_of")
        if alias_of is None:
            continue
        if bindings[port_id] != bindings[alias_of]:
            raise hard_fault(
                f"kernel {kernel_id!r} ABI splits an in-place alias across arguments",
                f"source={path}, output={port_id!r}, alias_of={alias_of!r}",
                "bind the aliased input and output to the same C pointer argument.",
            )

    for index, bound_ports in param_ports.items():
        if len(bound_ports) <= 1:
            continue
        valid_alias_group = all(
            port_id.startswith("output:")
            and logical_ports[port_id].get("alias_of") in bound_ports
            for port_id in bound_ports
            if port_id.startswith("output:")
        ) and all(
            port_id.startswith("output:")
            or any(
                logical_ports[output_id].get("alias_of") == port_id
                for output_id in bound_ports
                if output_id.startswith("output:")
            )
            for port_id in bound_ports
        )
        if not valid_alias_group:
            raise hard_fault(
                f"kernel {kernel_id!r} ABI combines unrelated logical ports",
                f"source={path}, parameter={params[index].get('name')!r}, ports={sorted(bound_ports)}",
                "only an input and its explicitly aliased output may share one C argument.",
            )

    for port_id, index in bindings.items():
        source = str(params[index].get("source", "") or "")
        if source == "null" and logical_ports[port_id]["consumption"] != "optional":
            raise hard_fault(
                f"kernel {kernel_id!r} ABI nulls a required logical port",
                f"source={path}, parameter={params[index].get('name')!r}, port={port_id!r}",
                "provide storage for required ports or mark a genuinely optional output explicitly.",
            )


def load_kernel_capabilities(
    root: Path = DEFAULT_KERNELS,
    contracts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    registry = contracts or load_json(DEFAULT_CONTRACTS)
    validate_contract_registry(registry)
    kernels: Dict[str, Any] = {}
    interfaces: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        doc = load_json(path)
        interface = _load_operation_interface(doc, path)
        if interface is not None:
            previous = interfaces.get(interface["id"])
            if previous is not None and previous != interface:
                raise hard_fault(
                    f"kernel maps disagree on operation interface {interface['id']!r}",
                    f"source={path}, expected={previous}, advertised={interface}",
                    "make every provider for an interface expose the same canonical logical ports.",
                )
            interfaces[interface["id"]] = interface

        capabilities = doc.get("numerical_capabilities")
        if not isinstance(capabilities, list) or not capabilities:
            continue
        kernel_id = str(doc.get("id", "")).strip()
        operation = str(doc.get("op", "")).strip()
        function = str((doc.get("impl") or {}).get("function", "")).strip()
        if not kernel_id or not operation or not function:
            raise hard_fault(
                "numerical kernel map has incomplete identity",
                f"file={path}, id={kernel_id!r}, op={operation!r}, function={function!r}",
                "declare id, op, and impl.function before advertising numerical capabilities.",
            )
        if kernel_id in kernels:
            raise hard_fault(
                f"duplicate kernel ID {kernel_id!r}",
                f"second provider={path}",
                "give every exact implementation a unique stable kernel ID.",
            )
        checked = []
        for capability in capabilities:
            contract_id = str((capability or {}).get("contract_id", ""))
            contract = registry["contracts"].get(contract_id)
            if contract is None:
                raise hard_fault(
                    f"kernel {kernel_id!r} advertises unknown contract {contract_id!r}",
                    f"source={path}",
                    "register the full contract before binding an implementation.",
                )
            if capability.get("function") != function:
                raise hard_fault(
                    f"kernel {kernel_id!r} capability changes function identity",
                    f"impl.function={function!r}, capability.function={capability.get('function')!r}",
                    "bind the capability to the exact public function in the kernel map.",
                )
            _validate_capability_against_contract(kernel_id, capability, contract)
            checked.append(copy.deepcopy(capability))
        kernels[kernel_id] = {
            "id": kernel_id,
            "op": operation,
            "function": function,
            "capabilities": checked,
            "source": str(path.resolve().relative_to(REPO_ROOT.resolve())),
            "source_hash": sha256_file(path),
        }
        if interface is not None:
            kernels[kernel_id]["operation_interface"] = interface["id"]
            kernels[kernel_id]["interface_call_abi"] = "validated"
    return {
        "schema": "cke.numerical_kernel_capabilities",
        "schema_version": 1,
        "engine_contract_version": "8",
        "kernels": kernels,
        "operation_interfaces": interfaces,
    }


def resolve_contract(
    circuit: Dict[str, Any],
    contracts: Dict[str, Any],
    kernels: Dict[str, Any],
    operation: str,
    phase: str,
    mode: str = "bringup",
    source_circuit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    validate_contract_registry(contracts)
    validate_schema(
        {"required_numerical_contracts": circuit.get("required_numerical_contracts")},
        REQUIREMENTS_SCHEMA,
        "circuit numerical requirements",
    )
    if mode not in {"bringup", "production"}:
        raise ContractError(f"Unknown resolution mode: {mode}")
    operation_doc = circuit["required_numerical_contracts"].get(operation)
    if not isinstance(operation_doc, dict):
        raise hard_fault(
            f"circuit has no numerical operation {operation!r}",
            f"circuit={circuit.get('name', '<unnamed>')}",
            "declare the operation and its exact semantic contract.",
        )
    request = (operation_doc.get("phases") or {}).get(phase)
    if not isinstance(request, dict):
        raise hard_fault(
            f"operation {operation!r} has no {phase!r} numerical requirement",
            "Prefill and decode may use different arithmetic contracts.",
            "declare the active phase explicitly.",
        )
    contract_id = request["contract_id"]
    required_interface = str(operation_doc.get("operation_interface", "") or "").strip()
    if contract_id.lower() in AMBIGUOUS_IDS:
        raise hard_fault(
            f"ambiguous requested contract {contract_id!r}",
            "The compiler cannot infer execution arithmetic from a dtype label.",
            "request one complete registry contract by stable ID.",
        )
    contract = contracts["contracts"].get(contract_id)
    if contract is None:
        raise hard_fault(
            f"unknown requested contract {contract_id!r}",
            f"operation={operation}.{phase}",
            "register and validate the contract before compiling the circuit.",
        )
    matches = []
    for kernel in kernels.get("kernels", {}).values():
        if kernel.get("op") != operation_doc["op"]:
            continue
        if required_interface and kernel.get("operation_interface") != required_interface:
            continue
        for capability in kernel.get("capabilities", []):
            if capability["contract_id"] == contract_id and phase in capability["phases"]:
                matches.append((kernel, capability))
    if len(matches) != 1:
        raise hard_fault(
            f"numerical requirement resolved to {len(matches)} kernels",
            f"operation={operation}.{phase}, contract={contract_id}, candidates={[item[0]['id'] for item in matches]}",
            "bind exactly one explicit kernel implementation; remove ambiguity or add the missing provider.",
        )
    kernel, capability = matches[0]
    if required_interface and kernel.get("interface_call_abi") != "validated":
        raise hard_fault(
            f"kernel {kernel['id']!r} has no validated interface-to-ABI boundary",
            f"operation={operation}.{phase}, interface={required_interface!r}",
            "load a map whose logical ports are completely bound to its map-owned call ABI.",
        )
    if mode == "production" and any(
        state != "validated"
        for state in (request["validation"], contract["status"], capability["status"])
    ):
        raise hard_fault(
            f"production resolution uses unvalidated contract {contract_id!r}",
            f"request={request['validation']}, contract={contract['status']}, kernel={capability['status']}",
            "produce parity evidence and promote every state to validated.",
        )
    source_hashes = {
        "contract_registry": sha256_json(contracts),
        "kernel_map": kernel["source_hash"],
        "circuit": sha256_json(circuit),
    }
    if source_circuit_path is not None:
        source_hashes["circuit_file"] = sha256_file(source_circuit_path)
    result = {
        "schema": "cke.resolved_numerical_execution_contract",
        "schema_version": 1,
        "engine_contract_version": "8",
        "circuit": str(circuit.get("name") or "embedded"),
        "operation": operation,
        "template_ops": copy.deepcopy(operation_doc["template_ops"]),
        "phase": phase,
        "mode": mode,
        "requirements": copy.deepcopy(request),
        "contract": {"id": contract_id, "status": contract["status"], "semantics": copy.deepcopy(contract)},
        "kernel": {
            "id": kernel["id"],
            "function": kernel["function"],
            "status": capability["status"],
            "explicit_selector": True,
            **(
                {"interface_call_abi": kernel["interface_call_abi"]}
                if required_interface
                else {}
            ),
        },
        "implementation": copy.deepcopy(capability["implementation"]),
        "checkpoint": copy.deepcopy(operation_doc["checkpoint"]),
        "source_hashes": source_hashes,
    }
    if required_interface:
        result["operation_interface"] = required_interface
    validate_schema(result, RESOLVED_SCHEMA, "resolved numerical execution contract")
    return result
