#!/usr/bin/env python3
"""Measure whether stored Q/K/V drift is amplified by authoritative PyTorch SDPA.

This is a causal-attribution aid, not a parity gate.  Both the reference and
subject tensors pass through the same PyTorch operation.  The report therefore
separates sensitivity to an existing forward perturbation from disagreement in
a CK backward implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "xray_attention_sensitivity.schema.json"


class SensitivityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_report(report: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise SensitivityError(f"report schema failure at {location}: {error.message}")


def _load_raw(path: Path, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if dtype == "fp32":
        values = np.fromfile(path, dtype=np.float32)
    elif dtype == "fp16":
        values = np.fromfile(path, dtype=np.float16).astype(np.float32)
    elif dtype == "bf16":
        raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32)
        values = (raw << 16).view(np.float32)
    else:
        raise SensitivityError(f"unsupported exported dtype: {dtype}")
    expected = math.prod(shape)
    if values.size != expected:
        raise SensitivityError(
            f"{path}: found {values.size} values, expected {expected} for shape {shape}"
        )
    return np.ascontiguousarray(values.reshape(shape), dtype=np.float32)


def _metrics(reference: np.ndarray, subject: np.ndarray, axes: list[str]) -> dict[str, Any]:
    if reference.shape != subject.shape:
        raise SensitivityError(f"shape mismatch: {reference.shape} != {subject.shape}")
    ref = reference.astype(np.float64, copy=False)
    got = subject.astype(np.float64, copy=False)
    delta = got - ref
    absolute = np.abs(delta)
    flat = int(np.argmax(absolute)) if absolute.size else 0
    coordinate = np.unravel_index(flat, absolute.shape) if absolute.size else ()
    l2 = float(np.linalg.norm(delta.reshape(-1)))
    rmse = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    return {
        "l2": l2,
        "rmse": rmse,
        "max_abs": float(absolute.reshape(-1)[flat]) if absolute.size else 0.0,
        "mean_abs": float(np.mean(absolute)) if absolute.size else 0.0,
        "exact_elements": int(np.count_nonzero(reference == subject)),
        "total_elements": int(reference.size),
        "byte_exact": bool(np.array_equal(reference, subject)),
        "finite": bool(np.isfinite(reference).all() and np.isfinite(subject).all()),
        "worst_coordinate": {axis: int(index) for axis, index in zip(axes, coordinate)},
    }


def _combined_l2(metrics: dict[str, dict[str, Any]]) -> float:
    return math.sqrt(sum(float(value["l2"]) ** 2 for value in metrics.values()))


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def analyze_attention_sensitivity(
    reference: dict[str, np.ndarray],
    subject: dict[str, np.ndarray],
    *,
    storage_dtype: str = "bf16",
    query_start: int = 0,
    query_count: int | None = None,
    probe_seed: int = 0,
    causal: bool = False,
    threads: int = 1,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise SensitivityError("PyTorch is required for attention sensitivity analysis") from exc

    for name in ("q", "k", "v"):
        if name not in reference or name not in subject:
            raise SensitivityError(f"missing {name!r} tensor")
        if reference[name].shape != subject[name].shape:
            raise SensitivityError(
                f"{name} reference/subject shape mismatch: "
                f"{reference[name].shape} != {subject[name].shape}"
            )
        if reference[name].ndim != 3:
            raise SensitivityError(f"{name} must use [head, token, channel] layout")

    q_shape = reference["q"].shape
    k_shape = reference["k"].shape
    v_shape = reference["v"].shape
    if k_shape != v_shape:
        raise SensitivityError(f"K/V shape mismatch: {k_shape} != {v_shape}")
    if q_shape[2] != k_shape[2]:
        raise SensitivityError(f"Q/K/V channel mismatch: {q_shape[2]} != {k_shape[2]}")
    if q_shape[0] % k_shape[0] != 0:
        raise SensitivityError(f"query heads {q_shape[0]} must be divisible by KV heads {k_shape[0]}")
    if query_start < 0 or query_start >= q_shape[1]:
        raise SensitivityError(f"query_start {query_start} is outside token dimension {q_shape[1]}")
    if query_count is None:
        query_count = q_shape[1] - query_start
    if query_count <= 0 or query_start + query_count > q_shape[1]:
        raise SensitivityError(
            f"query range [{query_start}, {query_start + query_count}) is outside {q_shape[1]} tokens"
        )
    if causal and (query_start != 0 or query_count != q_shape[1]):
        raise SensitivityError(
            "bounded query ranges are not position-equivalent for causal SDPA; use the full query range"
        )
    if storage_dtype not in {"bf16", "fp16", "fp32"}:
        raise SensitivityError(f"unsupported storage dtype: {storage_dtype}")

    torch.set_num_threads(max(1, threads))
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[storage_dtype]
    q_slice = slice(query_start, query_start + query_count)

    probe_rng = np.random.default_rng(probe_seed)
    probe_values = probe_rng.standard_normal(
        (1, q_shape[0], query_count, q_shape[2]), dtype=np.float32
    )
    probe = torch.from_numpy(probe_values).to(torch_dtype)

    def execute(values: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        q = torch.from_numpy(values["q"][:, q_slice, :].copy()).to(torch_dtype)
        k = torch.from_numpy(values["k"].copy()).to(torch_dtype)
        v = torch.from_numpy(values["v"].copy()).to(torch_dtype)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        output = functional.scaled_dot_product_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            is_causal=causal,
            enable_gqa=q_shape[0] != k_shape[0],
        )
        gradients = torch.autograd.grad(
            output, (q, k, v), grad_outputs=probe, retain_graph=False, create_graph=False
        )
        return (
            output.detach().float().cpu().numpy()[0],
            {
                "q": gradients[0].detach().float().cpu().numpy(),
                "k": gradients[1].detach().float().cpu().numpy(),
                "v": gradients[2].detach().float().cpu().numpy(),
            },
        )

    active_reference = {
        "q": reference["q"][:, q_slice, :],
        "k": reference["k"],
        "v": reference["v"],
    }
    reference_output, reference_gradients = execute(reference)
    cases: dict[str, dict[str, Any]] = {}
    substitutions = {
        "q_only": {"q"},
        "k_only": {"k"},
        "v_only": {"v"},
        "qkv": {"q", "k", "v"},
    }
    tensor_axes = {
        "q": ["head", "query", "channel"],
        "k": ["kv_head", "key", "channel"],
        "v": ["kv_head", "key", "channel"],
    }
    for case_name, replaced in substitutions.items():
        values = {
            name: subject[name] if name in replaced else reference[name]
            for name in ("q", "k", "v")
        }
        active_subject = {
            "q": values["q"][:, q_slice, :],
            "k": values["k"],
            "v": values["v"],
        }
        input_metrics = {
            name: _metrics(active_reference[name], active_subject[name], tensor_axes[name])
            for name in ("q", "k", "v")
        }
        output, gradients = execute(values)
        output_metrics = _metrics(
            reference_output, output, ["head", "query", "channel"]
        )
        gradient_metrics = {
            name: _metrics(reference_gradients[name], gradients[name], tensor_axes[name])
            for name in ("q", "k", "v")
        }
        input_l2 = _combined_l2(input_metrics)
        gradient_l2 = _combined_l2(gradient_metrics)
        if input_l2 == 0.0:
            observation = "NO_SURVIVING_STORAGE_DELTA"
        elif output_metrics["l2"] == 0.0 and gradient_l2 == 0.0:
            observation = "NO_OBSERVED_DOWNSTREAM_EFFECT"
        else:
            observation = "SENSITIVITY_OBSERVED"
        cases[case_name] = {
            "observation": observation,
            "input_delta": input_metrics,
            "input_delta_l2": input_l2,
            "output_delta": output_metrics,
            "forward_amplification_l2": _ratio(float(output_metrics["l2"]), input_l2),
            "vjp_delta": gradient_metrics,
            "vjp_delta_l2": gradient_l2,
            "backward_amplification_l2": _ratio(gradient_l2, input_l2),
        }

    return {
        "schema": "cke.xray_attention_sensitivity",
        "schema_version": 1,
        "status": "observed",
        "interpretation": {
            "scope": "same PyTorch operator applied to reference and CK-perturbed stored inputs",
            "proves": "which stored forward perturbations are amplified by the downstream operator",
            "does_not_prove": "parity or correctness of a CK backward provider",
            "threshold_policy": "pass/fail thresholds belong in a parity profile, not this report",
        },
        "configuration": {
            "operation": "scaled_dot_product_attention",
            "causal": causal,
            "storage_dtype": storage_dtype,
            "q_shape": list(q_shape),
            "kv_shape": list(k_shape),
            "query_start": query_start,
            "query_count": query_count,
            "probe_seed": probe_seed,
            "threads": max(1, threads),
            "torch_version": torch.__version__,
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch_mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        },
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ("reference", "subject"):
        for name in ("q", "k", "v"):
            parser.add_argument(f"--{side}-{name}", type=Path, required=True)
    parser.add_argument("--q-shape", type=int, nargs=3, metavar=("HEADS", "TOKENS", "DIM"), required=True)
    parser.add_argument("--kv-shape", type=int, nargs=3, metavar=("HEADS", "TOKENS", "DIM"), required=True)
    parser.add_argument("--exported-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--storage-dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    q_shape = tuple(args.q_shape)
    kv_shape = tuple(args.kv_shape)
    reference = {
        name: _load_raw(getattr(args, f"reference_{name}"), args.exported_dtype, q_shape if name == "q" else kv_shape)
        for name in ("q", "k", "v")
    }
    subject = {
        name: _load_raw(getattr(args, f"subject_{name}"), args.exported_dtype, q_shape if name == "q" else kv_shape)
        for name in ("q", "k", "v")
    }
    report = analyze_attention_sensitivity(
        reference,
        subject,
        storage_dtype=args.storage_dtype,
        query_start=args.query_start,
        query_count=args.query_count,
        probe_seed=args.probe_seed,
        causal=args.causal,
        threads=args.threads,
    )
    report["artifacts"] = {
        side: {
            name: {
                "path": str(getattr(args, f"{side}_{name}").resolve()),
                "sha256": _sha256(getattr(args, f"{side}_{name}")),
            }
            for name in ("q", "k", "v")
        }
        for side in ("reference", "subject")
    }
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status={report['status']} cases={len(report['cases'])}")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
