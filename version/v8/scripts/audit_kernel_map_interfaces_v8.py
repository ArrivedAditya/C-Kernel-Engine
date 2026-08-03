#!/usr/bin/env python3
"""Report and ratchet v8 kernel-map interface migration debt."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict


V8_ROOT = Path(__file__).resolve().parents[1]
KERNEL_MAPS = V8_ROOT / "kernel_maps"
BUILD_IR = V8_ROOT / "scripts" / "build_ir_v8.py"
BASELINE = V8_ROOT / "contracts" / "kernel_interface_migration_baseline.json"
NON_MAP_FILES = {
    "KERNEL_REGISTRY.json",
    "KERNEL_SOURCES.json",
    "kernel_bindings.json",
    "kernel_bindings.overlay.json",
}
LEGACY_CONTRACT_KEYS = {
    "contract_schema_version",
    "numerical_contract",
    "reference",
}


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _map_op_conditionals(path: Path) -> Dict[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "map_op_to_kernel"
    )
    conditionals = sum(isinstance(node, ast.If) for node in ast.walk(function))
    inline = sum(isinstance(node, ast.IfExp) for node in ast.walk(function))
    operation_specific = sum(
        isinstance(node, ast.If)
        and any(
            isinstance(name, ast.Name) and name.id == "op"
            for name in ast.walk(node.test)
        )
        for node in ast.walk(function)
    )
    # The resolved-plan presence and phase guard are the map-first entry path.
    contract_path = 2
    return {
        "total_if_statements": conditionals,
        "contract_path_if_statements": contract_path,
        "legacy_selection_if_statements": conditionals - contract_path,
        "operation_specific_if_statements": operation_specific,
        "inline_conditional_expressions": inline,
    }


def _has_complete_interface_abi(doc: Dict[str, Any]) -> bool:
    expected = {
        f"{role}:{port.get('name')}"
        for role, field in (
            ("input", "inputs"),
            ("weight", "weights"),
            ("output", "outputs"),
        )
        for port in doc.get(field, [])
        if isinstance(port, dict) and port.get("name")
    }
    declared = [
        port_id
        for param in (doc.get("call_abi") or {}).get("params", [])
        if isinstance(param, dict)
        for port_id in param.get("ports", [])
    ]
    return bool(expected) and len(declared) == len(set(declared)) and set(declared) == expected


def _validate_selection(doc: Dict[str, Any], path: Path) -> bool:
    selection = doc.get("selection")
    if selection is None:
        return False
    if not isinstance(selection, dict):
        raise RuntimeError(f"provider selection must be an object: {path}")
    status = selection.get("status")
    if status not in {"production", "candidate", "diagnostic", "deprecated"}:
        raise RuntimeError(f"invalid provider selection status in {path}: {status!r}")
    priority = selection.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise RuntimeError(f"provider selection priority must be an integer: {path}")
    if not str(selection.get("equivalence_group", "") or "").strip():
        raise RuntimeError(f"provider selection requires equivalence_group: {path}")
    phases = selection.get("phases")
    if not isinstance(phases, list) or not phases or any(
        phase not in {"init", "prefill", "decode", "training", "backward"}
        for phase in phases
    ):
        raise RuntimeError(f"provider selection has invalid phases: {path}")
    return True


def build_report() -> Dict[str, Any]:
    maps = []
    for path in sorted(KERNEL_MAPS.glob("*.json")):
        if path.name in NON_MAP_FILES:
            continue
        maps.append((path, _load(path)))

    governed = [item for item in maps if item[1].get("numerical_capabilities")]
    hardened = [item for item in governed if item[1].get("operation_interface")]
    crossvalidated = [item for item in hardened if _has_complete_interface_abi(item[1])]
    pending = [item for item in governed if not item[1].get("operation_interface")]
    legacy = [item for item in maps if not item[1].get("numerical_capabilities")]
    interface_ready = [item for item in maps if item[1].get("operation_interface")]
    interface_abi_ready = [item for item in interface_ready if _has_complete_interface_abi(item[1])]
    legacy_interface_ready = [item for item in legacy if item[1].get("operation_interface")]
    selection_managed = [item for item in maps if _validate_selection(item[1], item[0])]
    production_selected = [
        item for item in selection_managed
        if item[1]["selection"]["status"] == "production"
    ]
    legacy_contract_shaped = [
        item
        for item in legacy
        if any(key in item[1] for key in LEGACY_CONTRACT_KEYS)
    ]
    map_owned_abi = [item for item in maps if item[1].get("call_abi")]

    return {
        "schema": "cke.v8.kernel_interface_migration_report",
        "schema_version": 1,
        "counts": {
            "kernel_maps": len(maps),
            "resolver_governed_maps": len(governed),
            "interface_hardened_maps": len(hardened),
            "interface_abi_crossvalidated_maps": len(crossvalidated),
            "contract_pending_maps": len(pending),
            "legacy_maps": len(legacy),
            "legacy_contract_shaped_maps": len(legacy_contract_shaped),
            "map_owned_call_abi": len(map_owned_abi),
            "legacy_call_abi": len(maps) - len(map_owned_abi),
            "all_interface_ready_maps": len(interface_ready),
            "all_interface_abi_ready_maps": len(interface_abi_ready),
            "legacy_interface_ready_maps": len(legacy_interface_ready),
            "selection_managed_maps": len(selection_managed),
            "production_selected_maps": len(production_selected),
        },
        "selection": _map_op_conditionals(BUILD_IR),
        "interface_hardened_ids": [item[1]["id"] for item in hardened],
        "interface_abi_crossvalidated_ids": [
            item[1]["id"] for item in crossvalidated
        ],
        "contract_pending_ids": [item[1]["id"] for item in pending],
        "legacy_contract_shaped_ids": [item[1]["id"] for item in legacy_contract_shaped],
        "legacy_interface_ready_ids": [item[1]["id"] for item in legacy_interface_ready],
        "selection_managed_ids": [item[1]["id"] for item in selection_managed],
    }


def validate_ratchet(report: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    counts = report["counts"]
    selection = report["selection"]
    checks = (
        (
            counts["interface_hardened_maps"]
            >= baseline["minimum_interface_hardened_maps"],
            "interface-hardened map count regressed",
        ),
        (
            counts["interface_abi_crossvalidated_maps"]
            >= baseline["minimum_interface_abi_crossvalidated_maps"],
            "interface-to-ABI cross-validation count regressed",
        ),
        (
            counts["contract_pending_maps"]
            <= baseline["maximum_contract_pending_maps"],
            "contract-pending map debt increased",
        ),
        (
            counts["legacy_maps"] <= baseline["maximum_legacy_maps"],
            "legacy map debt increased",
        ),
        (
            selection["legacy_selection_if_statements"]
            <= baseline["maximum_legacy_selection_conditionals"],
            "map_op_to_kernel legacy conditional count increased",
        ),
        (
            selection["operation_specific_if_statements"]
            <= baseline["maximum_operation_specific_conditionals"],
            "map_op_to_kernel operation-specific conditional count increased",
        ),
        (
            counts["map_owned_call_abi"]
            >= baseline["minimum_map_owned_call_abi"],
            "map-owned call ABI count regressed",
        ),
        (
            counts["legacy_interface_ready_maps"]
            >= baseline.get("minimum_legacy_interface_ready_maps", 0),
            "legacy interface-ready map count regressed",
        ),
        (
            counts["selection_managed_maps"]
            >= baseline.get("minimum_selection_managed_maps", 0),
            "selection-managed map count regressed",
        ),
    )
    failures = [message for passed, message in checks if not passed]
    if failures:
        raise RuntimeError("; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the complete JSON report")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when migration debt regresses from the checked-in baseline",
    )
    args = parser.parse_args()

    report = build_report()
    if args.check:
        validate_ratchet(report, _load(BASELINE))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        counts = report["counts"]
        selection = report["selection"]
        print(
            "kernel interfaces: "
            f"hardened={counts['interface_hardened_maps']} "
            f"interface_abi={counts['interface_abi_crossvalidated_maps']} "
            f"contract_pending={counts['contract_pending_maps']} "
            f"legacy={counts['legacy_maps']} "
            f"map_abi={counts['map_owned_call_abi']} "
            f"legacy_interface={counts['legacy_interface_ready_maps']} "
            f"selection_managed={counts['selection_managed_maps']} "
            f"legacy_if={selection['legacy_selection_if_statements']} "
            f"op_if={selection['operation_specific_if_statements']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
