"""Import observed GGML convolution edges and compare operation packets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from .xray_operation_bundle_v8 import BundleError, load_bundle, sha256, write_bundle
except ImportError:
    from xray_operation_bundle_v8 import BundleError, load_bundle, sha256, write_bundle


_ROLES = ("conv_image", "conv_im2col", "conv_weight", "conv_result")
_LAYOUT_ROLES = ("conv_layout_0", "conv_layout_1")
_GGML_DTYPES = {0: ("fp32", np.dtype("<f4")), 1: ("fp16", np.dtype("<f2"))}
_AXES = {
    "conv_image": ("batch", "channel", "height", "width"),
    "conv_im2col": ("batch", "out_height", "out_width", "reduction"),
    "conv_weight": ("out_channel", "reduction"),
    "conv_result": ("row", "out_channel"),
}
_EDGES = {
    "conv_image": "im2col.src[1]",
    "conv_im2col": "mul_mat.src[0].im2col",
    "conv_weight": "mul_mat.src[1]",
    "conv_result": "mul_mat.output",
}
_CANONICAL_TRANSFORMS = {
    "conv_image": "reverse_ggml_axes",
    "conv_im2col": "reverse_ggml_axes",
    "conv_weight": "squeeze_outer_singletons_to_out_channel_reduction",
    "conv_result": "squeeze_outer_singletons_then_transpose_to_row_out_channel",
}


def load_oracle_capture(directory: Path, capture_id: int) -> tuple[dict, dict]:
    rows = []
    try:
        for line in (directory / "meta.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("capture_id") == capture_id:
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError("oracle capture metadata is missing or malformed") from exc
    by_role = {row.get("role"): row for row in rows}
    if (len(rows) != len(by_role) or not set(_ROLES).issubset(by_role) or
        not set(by_role).issubset(set(_ROLES) | set(_LAYOUT_ROLES))):
        raise BundleError("oracle convolution capture is incomplete or duplicated")
    tensors = {}
    provenance = {}
    for role in (*_ROLES, *(role for role in _LAYOUT_ROLES if role in by_role)):
        row = by_role[role]
        expected_edge = _EDGES.get(role, "layout_descendant_of_mul_mat")
        if row.get("status") != "observed" or row.get("source_edge") != expected_edge:
            raise BundleError(f"{role} is unavailable or has the wrong source edge")
        storage = _GGML_DTYPES.get(row.get("type"))
        if storage is None:
            raise BundleError(f"{role} has unsupported storage dtype")
        dtype_name, dtype = storage
        ne, nb = row.get("ne"), row.get("nb")
        if (not isinstance(ne, list) or not isinstance(nb, list) or len(ne) != 4 or
            len(nb) != 4 or any(type(value) is not int or value <= 0 for value in ne) or
            any(type(value) is not int or value < dtype.itemsize for value in nb)):
            raise BundleError(f"{role} has invalid physical geometry")
        filename = f"conv_{capture_id}_{role}.raw"
        if row.get("raw_file") != filename:
            raise BundleError(f"{role} has an unsafe or unexpected raw path")
        raw_path = directory / filename
        try:
            raw = raw_path.read_bytes()
        except OSError as exc:
            raise BundleError(f"{role} raw capture is missing") from exc
        if len(raw) > (256 << 20) or len(raw) < dtype.itemsize + sum((dim - 1) * stride for dim, stride in zip(ne, nb)):
            raise BundleError(f"{role} raw capture is truncated or oversized")
        provenance[role] = {
            "source_edge": expected_edge, "producer_op": row.get("op"),
            "physical_shape_ggml": ne, "physical_strides_bytes_ggml": nb,
            "raw_sha256": sha256(raw_path), "raw_size_bytes": len(raw),
            "canonical_transform": _CANONICAL_TRANSFORMS.get(role, "unresolved_raw_layout"),
        }
        if role in _LAYOUT_ROLES:
            continue
        physical = np.ndarray(tuple(reversed(ne)), dtype=dtype,
                              buffer=raw, strides=tuple(reversed(nb))).copy()
        if role == "conv_weight":
            if ne[2:] != [1, 1]:
                raise BundleError("conv_weight has unexpected rank")
            physical = physical.reshape(ne[1], ne[0])
        elif role == "conv_result":
            if ne[2:] != [1, 1]:
                raise BundleError("conv_result has unexpected rank")
            physical = physical.reshape(ne[1], ne[0]).T.copy()
        tensors[role] = (np.ascontiguousarray(physical), dtype_name, _AXES[role])
    image = tensors["conv_image"][0]
    patches = tensors["conv_im2col"][0]
    weights = tensors["conv_weight"][0]
    result = tensors["conv_result"][0]
    geometry = by_role["conv_im2col"].get("convolution")
    if not isinstance(geometry, dict) or geometry.get("groups") != 1 or geometry.get("is_2d") != 1:
        raise BundleError("observed 2D convolution geometry is missing or unsupported")
    for field, minimum in (("kernel", 1), ("stride", 1), ("padding", 0), ("dilation", 1)):
        values = geometry.get(field)
        if not isinstance(values, list) or len(values) != 2 or any(
            type(value) is not int or value < minimum for value in values
        ):
            raise BundleError(f"invalid convolution {field}")
    kw, kh = geometry["kernel"]
    sw, sh = geometry["stride"]
    pw, ph = geometry["padding"]
    dw, dh = geometry["dilation"]
    ow = (image.shape[3] + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    oh = (image.shape[2] + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    if (patches.shape[0] != image.shape[0] or weights.shape[1] != patches.shape[-1] or
        patches.shape[1:] != (oh, ow, image.shape[1] * kh * kw) or
        result.shape != (math.prod(patches.shape[:3]), weights.shape[0])):
        raise BundleError("oracle convolution operands have inconsistent geometry")
    provenance["conv_im2col"]["convolution"] = geometry
    return tensors, provenance


def import_oracle_capture(directory: Path, capture_id: int, runtime: Path,
                          output: Path, contract: str) -> Path:
    tensors, provenance = load_oracle_capture(directory, capture_id)
    identity_path = directory / "capture-identity.json"
    identity = None
    if identity_path.is_file():
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleError("capture identity is malformed") from exc
        if not isinstance(identity, dict) or identity.get("selected_capture_id") != capture_id:
            raise BundleError("capture identity selects a different convolution")
        for name in ("image", "projector", "shim"):
            path = Path(str(identity.get(f"{name}_path", "")))
            if not path.is_file() or sha256(path) != identity.get(f"{name}_sha256"):
                raise BundleError(f"{name} artifact changed since capture")
        loaded = Path(str(identity.get("loaded_mtmd_path", "")))
        if (not loaded.is_file() or loaded.resolve() != runtime.resolve() or
            sha256(loaded) != identity.get("loaded_mtmd_sha256")):
            raise BundleError("loaded MTMD library identity changed")
        if (identity.get("processed_width") != tensors["conv_image"][0].shape[3] or
            identity.get("processed_height") != tensors["conv_image"][0].shape[2]):
            raise BundleError("processed image geometry differs from capture")
    return write_bundle(
        output, operation="conv_patch_projection", provider="ggml_conv_2d",
        contract=contract, runtime=runtime,
        context={"capture_id": capture_id, "tensor_provenance": provenance,
                 "capture_kind": "observed_graph_edges",
                 "runtime_identity_source": ("proc_self_maps_at_capture" if identity else
                                             "caller_supplied_artifact_not_loaded_proof"),
                 "capture_identity": identity},
        tensors=tensors,
    )


def compare_packets(cke_path: Path, oracle_path: Path, max_abs_limit: float) -> dict:
    if not math.isfinite(max_abs_limit) or max_abs_limit < 0:
        raise BundleError("numerical limit must be finite and nonnegative")
    cke_meta, cke = load_bundle(cke_path)
    oracle_meta, oracle = load_bundle(oracle_path)
    report = {"operation": "conv_patch_projection", "gates": {},
              "first_observed_failure": None, "first_proven_local_failure": None,
              "evidence_scope": "packet_numerics_only",
              "connected_execution_certified": False}
    if cke_meta["operation"] != report["operation"] or oracle_meta["operation"] != report["operation"]:
        raise BundleError("packets describe different operations")
    for role in _ROLES:
        if role not in cke or role not in oracle:
            report["gates"][role] = {"status": "UNAVAILABLE"}
            report["first_observed_failure"] = report["first_observed_failure"] or role
            continue
        left, right = cke[role], oracle[role]
        ls, rs = cke_meta["tensors"][role], oracle_meta["tensors"][role]
        if ls["axes"] != rs["axes"] or ls["shape"] != rs["shape"] or ls["dtype"] != rs["dtype"]:
            report["gates"][role] = {"status": "ALIGNMENT_UNRESOLVED"}
            report["first_observed_failure"] = report["first_observed_failure"] or role
            continue
        diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
        worst = float(diff.max())
        passed = worst <= max_abs_limit
        report["gates"][role] = {"status": "PASS" if passed else "FAIL",
                                  "max_abs": worst,
                                  "worst_index": [int(value) for value in np.unravel_index(
                                      int(diff.argmax()), diff.shape)]}
        if not passed:
            report["first_observed_failure"] = report["first_observed_failure"] or role
    input_roles = _ROLES[:3]
    inputs_match = all(report["gates"][role]["status"] == "PASS" for role in input_roles)
    identical_inputs = inputs_match and all(np.array_equal(cke[role], oracle[role])
                                            for role in input_roles)
    if identical_inputs and report["gates"]["conv_result"]["status"] == "FAIL":
        report["first_proven_local_failure"] = "conv_result"
    report["operand_construction"] = "PASS" if inputs_match else "FAIL_OR_INCOMPLETE"
    report["identical_input_replay_ready"] = bool(identical_inputs)
    report["local_arithmetic"] = ("FAIL" if report["first_proven_local_failure"] else
                                  "NOT_ISOLATED" if not identical_inputs else "PASS")
    statuses = {gate["status"] for gate in report["gates"].values()}
    report["status"] = ("INCOMPLETE" if statuses & {"UNAVAILABLE", "ALIGNMENT_UNRESOLVED"} else
                        "FAIL" if "FAIL" in statuses else "PASS")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cke-bundle", type=Path, required=True)
    parser.add_argument("--oracle-capture", type=Path, required=True)
    parser.add_argument("--oracle-runtime", type=Path, required=True)
    parser.add_argument("--capture-id", type=int, default=0)
    parser.add_argument("--oracle-bundle-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-abs", type=float, required=True)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    try:
        if args.capture_id < 0 or not math.isfinite(args.max_abs) or args.max_abs < 0:
            raise BundleError("capture index and numerical limit must be valid")
        oracle = import_oracle_capture(args.oracle_capture, args.capture_id,
                                       args.oracle_runtime, args.oracle_bundle_dir, args.contract)
        result = compare_packets(args.cke_bundle, oracle, args.max_abs)
    except BundleError as exc:
        result = {"status": "INCOMPLETE", "error": str(exc)}
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
