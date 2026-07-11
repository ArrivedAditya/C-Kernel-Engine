#!/usr/bin/env python3
"""Fail when cleaned generic DSL functions contain model-specific literals."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = REPO_ROOT / "version" / "v8" / "dsl_policy.json"


class DSLPolicyError(RuntimeError):
    pass


def _function_nodes(
    tree: ast.AST, requested: set[str]
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in requested:
                continue
            if node.name in nodes:
                raise DSLPolicyError(f"ambiguous compiler function name: {node.name}")
            nodes[node.name] = node
    return nodes


def scan_source(source: str, functions: list[str], forbidden: list[str], *, path: str) -> list[dict[str, Any]]:
    tree = ast.parse(source, filename=path)
    available = _function_nodes(tree, set(functions))
    findings: list[dict[str, Any]] = []
    forbidden_lc = tuple(item.lower() for item in forbidden)
    for function in functions:
        node = available.get(function)
        if node is None:
            raise DSLPolicyError(f"policy function not found: {path}:{function}")
        for child in ast.walk(node):
            if not isinstance(child, ast.Constant) or not isinstance(child.value, str):
                continue
            value = child.value.lower()
            matched = sorted({item for item in forbidden_lc if item in value})
            if matched:
                findings.append(
                    {
                        "path": path,
                        "function": function,
                        "line": child.lineno,
                        "literals": matched,
                        "value": child.value,
                    }
                )
    return findings


def audit(policy_path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema") != "cke.v8_dsl_policy" or policy.get("schema_version") != 1:
        raise DSLPolicyError("unsupported DSL policy schema")
    forbidden = policy.get("forbidden_model_literals")
    compiler_functions = policy.get("compiler_functions")
    if not isinstance(forbidden, list) or not forbidden or not all(isinstance(item, str) and item for item in forbidden):
        raise DSLPolicyError("forbidden_model_literals must be a non-empty string list")
    if not isinstance(compiler_functions, dict) or not compiler_functions:
        raise DSLPolicyError("compiler_functions must be a non-empty object")

    findings: list[dict[str, Any]] = []
    checked = 0
    for relative_path, functions in compiler_functions.items():
        if not isinstance(relative_path, str) or not isinstance(functions, list) or not functions:
            raise DSLPolicyError("each compiler policy entry requires a path and non-empty function list")
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise DSLPolicyError(f"compiler file not found: {relative_path}")
        findings.extend(
            scan_source(path.read_text(encoding="utf-8"), functions, forbidden, path=relative_path)
        )
        checked += len(functions)
    return {"status": "fail" if findings else "pass", "checked_functions": checked, "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.policy)
    except (DSLPolicyError, OSError, ValueError, SyntaxError) as exc:
        report = {"status": "error", "error": str(exc)}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
