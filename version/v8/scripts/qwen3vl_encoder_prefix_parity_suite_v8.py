#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
NUMERIC_PARITY = SCRIPT_DIR / "numeric_parity_qwen3vl_mmproj_v8.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*command: str) -> str | None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _checked_in_llama_commit() -> str | None:
    tree = _git_output("git", "-C", str(REPO_ROOT), "ls-tree", "HEAD", "llama.cpp")
    if not tree:
        return None
    fields = tree.split()
    return fields[2] if len(fields) == 4 and fields[:2] == ["160000", "commit"] else None


def _oracle_commit(root: Path) -> str:
    root = root.resolve()
    if not (root / "ggml" / "include" / "ggml.h").is_file():
        raise ValueError(f"llama.cpp source is missing under {root}")
    toplevel = _git_output("git", "-C", str(root), "rev-parse", "--show-toplevel")
    commit = _git_output("git", "-C", str(root), "rev-parse", "HEAD")
    if toplevel is None or Path(toplevel).resolve() != root or not commit:
        raise ValueError(f"llama.cpp source revision is unresolved under {root}")
    return commit


def _sanitize_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    text = text.strip("._-")
    return text or "sample"


def _load_image_specs(summary_json: Path | None, image_paths: list[Path], limit: int | None) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    if summary_json is not None:
        data = json.loads(summary_json.read_text(encoding="utf-8"))
        for idx, item in enumerate(data.get("results", []), 1):
            image = item.get("image")
            if not image:
                continue
            sample_id = str(item.get("id") or Path(str(image)).stem or idx)
            specs.append({"id": sample_id, "image": str(image)})
    for image in image_paths:
        specs.append({"id": image.stem, "image": str(image)})
    if limit is not None and limit > 0:
        specs = specs[:limit]
    return specs


def _load_manifest_specs(manifest_path: Path, limit: int | None) -> list[dict[str, str]]:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("image manifest must contain a non-empty samples list")
    specs: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for index, sample in enumerate(samples, 1):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be an object")
        inputs = sample.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict):
            raise ValueError(f"sample {index} must contain exactly one image input")
        raw_path = inputs[0].get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"sample {index} has no image path")
        sample_id = str(sample.get("id") or f"case-{index:03d}")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate image sample ID: {sample_id}")
        seen_ids.add(sample_id)
        image = Path(raw_path).expanduser()
        if not image.is_absolute():
            image = manifest_path.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"sample {index} image is missing: {image}")
        image_sha256 = _sha256_file(image)
        pinned_sha256 = inputs[0].get("sha256")
        if pinned_sha256 is not None and pinned_sha256 != image_sha256:
            raise ValueError(f"sample {index} image SHA-256 differs from manifest")
        if image_sha256 in seen_hashes:
            raise ValueError(
                f"sample {index} duplicates image content from {seen_hashes[image_sha256]}"
            )
        seen_hashes[image_sha256] = sample_id
        specs.append({"id": sample_id, "image": str(image), "image_sha256": image_sha256})
        if limit is not None and len(specs) >= limit:
            break
    return specs


def _metric_value(sample: dict[str, Any], name: str) -> float | None:
    metrics = sample.get("metrics")
    if not isinstance(metrics, dict) or name not in metrics:
        return None
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _shape_status(sample: dict[str, Any], embed_dim: int) -> tuple[bool, int | None]:
    grid = sample.get("grid")
    values = sample.get("num_values")
    if not isinstance(grid, list) or len(grid) != 2:
        return False, None
    if (any(type(value) is not int or value <= 0 for value in grid)
            or type(values) is not int or values <= 0 or type(embed_dim) is not int or embed_dim <= 0):
        return False, None
    gx, gy, got = grid[0], grid[1], values
    expected = gx * gy * int(embed_dim)
    raw = sample.get("raw_num_values")
    raw_ok = isinstance(raw, dict) and all(type(raw.get(side)) is int and raw[side] == got
                                           for side in ("ck", "llama"))
    return got == expected and raw_ok, expected


def _sample_from_report(spec: dict[str, str], report_path: Path, embed_dim: int) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("numeric parity report must be a JSON object")
    sample = {
        "id": spec["id"],
        "image": spec["image"],
        "reported_image": report.get("image_path"),
        "report": str(report_path),
        "status": report.get("status"),
        "invocation_id": report.get("invocation_id"),
        "artifact_identity": report.get("artifact_identity"),
        "artifact_identity_kind": report.get("artifact_identity_kind"),
        "input_provenance": report.get("input_provenance"),
        "preprocess_evidence": report.get("preprocess_evidence"),
        "source_image_sha256": report.get("source_image_sha256"),
        "gguf": report.get("gguf"),
        "grid": report.get("merged_grid"),
        "height": report.get("height"),
        "width": report.get("width"),
        "num_values": report.get("num_values"),
        "raw_num_values": report.get("raw_num_values"),
        "metrics": report.get("metrics", {}),
        "timings_sec": report.get("timings_sec", {}),
        "worst_rows": (report.get("row_diagnostics", {}) or {}).get("worst_rows", [])[:3],
    }
    shape_ok, expected = _shape_status(sample, embed_dim)
    sample["shape_ok"] = shape_ok
    sample["expected_values"] = expected
    return sample


def _verify_artifacts(sample: dict[str, Any], args: argparse.Namespace, env: dict[str, str]) -> str | None:
    identities = sample.get("artifact_identity")
    if sample.get("artifact_identity_kind") != "selected_file_hashes_after_execution":
        return "artifact identity method is missing or unsupported"
    model_library = args.runtime_dir / "libqwen3vl_mmproj_v8.so"
    selected_engine = env.get("CK_ENGINE_SO")
    if selected_engine is None:
        adjacent_engine = args.runtime_dir / "libckernel_engine.so"
        selected_engine = str(adjacent_engine if adjacent_engine.is_file() else REPO_ROOT / "build" / "libckernel_engine.so")
    llama_bin = Path(env.get("CK_LLAMA_CPP_ROOT", str(REPO_ROOT / "llama.cpp"))) / "build" / "bin"
    required = {
        "mmproj_gguf": args.gguf,
        "ck_model_library": model_library,
        "ck_generated_source": args.runtime_dir / "qwen3_vl_mmproj_v8.c",
        "ck_weights": args.runtime_dir / "weights.bump",
        "ck_manifest": args.runtime_dir / "weights_manifest.map",
        "llama_shim_library": args.runtime_dir / "libmtmd_clip_shim.so",
        "ck_engine_library": Path(selected_engine),
        "llama_mtmd_library": llama_bin / "libmtmd.so",
    }
    required.update({
        name: llama_bin / filename
        for name, filename in (
            ("llama_ggml_base_library", "libggml-base.so"),
            ("llama_ggml_library", "libggml.so"),
            ("llama_ggml_cpu_library", "libggml-cpu.so"),
        )
        if (llama_bin / filename).is_file()
    })
    if not isinstance(identities, dict) or set(identities) != set(required):
        return "incomplete model/oracle artifact identities"
    for role, expected_path in required.items():
        item = identities[role]
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            return f"invalid {role} artifact identity"
        path = Path(item["path"]).resolve()
        if path != expected_path.resolve():
            return f"{role} path differs from selected artifact"
        if not path.is_file() or path.stat().st_size <= 0 or item.get("size_bytes") != path.stat().st_size:
            return f"{role} artifact missing or changed size"
        if item.get("sha256") != _sha256_file(path):
            return f"{role} artifact hash mismatch"
    return None


def _run_one(
    *,
    spec: dict[str, str],
    index: int,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    sample_name = f"{index:02d}_{_sanitize_id(spec['id'])}"
    report_path = args.output_dir / "reports" / f"{sample_name}.json"
    log_path = args.output_dir / "logs" / f"{sample_name}.log"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    reuse = args.reuse_reports and report_path.exists()
    if not reuse:
        invocation_id = uuid.uuid4().hex
        if report_path.exists():
            archive = report_path.parent / "archive"
            archive.mkdir(parents=True, exist_ok=True)
            report_path.replace(archive / f"{sample_name}.{invocation_id}.json")
        if log_path.exists():
            archive = log_path.parent / "archive"
            archive.mkdir(parents=True, exist_ok=True)
            log_path.replace(archive / f"{sample_name}.{invocation_id}.log")
        cmd = [
            sys.executable,
            str(NUMERIC_PARITY),
            "--gguf",
            str(args.gguf),
            "--output-dir",
            str(args.runtime_dir),
            "--image-path",
            spec["image"],
            "--threads",
            str(args.threads),
            "--ck-threads",
            str(args.ck_threads),
            "--report",
            str(report_path),
            "--invocation-id",
            invocation_id,
        ]
        if getattr(args, "independent_preprocess", False):
            cmd.append("--independent-preprocess")
        if args.image_min_tokens is not None:
            cmd.extend(["--image-min-tokens", str(args.image_min_tokens)])
        if args.image_max_tokens is not None:
            cmd.extend(["--image-max-tokens", str(args.image_max_tokens)])
        start = time.perf_counter()
        with log_path.open("wb") as log_file:
            completed = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        fresh_report = report_path.is_file()
    else:
        elapsed = 0.0
        completed = None
        fresh_report = True
        invocation_id = None

    if fresh_report:
        try:
            sample = _sample_from_report(spec, report_path, args.embed_dim)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            sample = {"id": spec["id"], "image": spec["image"], "shape_ok": False,
                      "execution_error": f"invalid numeric parity report: {exc}"}
    else:
        sample = {"id": spec["id"], "image": spec["image"], "shape_ok": False,
                  "execution_error": "numeric parity process produced no fresh report"}
    if completed is not None and completed.returncode != 0:
        sample["execution_error"] = (
            f"numeric parity process exited {completed.returncode}; retained report at {report_path}"
            if sample.get("report") else _log_failure_reason(log_path, completed.returncode)
        )
    if fresh_report and not sample.get("execution_error"):
        if reuse:
            sample["evidence_error"] = "reused report is not current execution evidence"
        elif sample.get("invocation_id") != invocation_id:
            sample["evidence_error"] = "report invocation ID differs from current execution"
        else:
            artifact_error = _verify_artifacts(sample, args, env)
            if artifact_error:
                sample["evidence_error"] = artifact_error
    if fresh_report and not sample.get("execution_error"):
        try:
            expected_hash = _sha256_file(Path(spec["image"]))
        except OSError as exc:
            expected_hash = None
            sample["evidence_error"] = f"cannot verify current source image: {exc}"
        if "evidence_error" not in sample:
            if spec.get("image_sha256") and expected_hash != spec["image_sha256"]:
                sample["evidence_error"] = "source image changed after manifest selection"
            elif Path(str(sample.get("reported_image") or "")).resolve() != Path(spec["image"]).resolve():
                sample["evidence_error"] = "report image path does not match the current case"
            elif sample.get("source_image_sha256") != expected_hash:
                sample["evidence_error"] = "source image hash does not match the current case"
            elif getattr(args, "independent_preprocess", False) and sample.get("input_provenance") != "independently_preprocessed_from_shared_decoded_rgb8":
                sample["evidence_error"] = "report did not execute independent preprocessing"
            elif str(sample.get("gguf")) != str(args.gguf):
                sample["evidence_error"] = "report used a different mmproj path"
    sample["log"] = str(log_path)
    sample["suite_elapsed_sec"] = elapsed
    return sample


def _log_failure_reason(path: Path, returncode: int) -> str:
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
    except OSError:
        lines = []
    detail = lines[-1] if lines else "child process produced no diagnostic output"
    return f"numeric parity process exited {returncode}: {detail}"


def _evaluate_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for sample in samples:
        sid = str(sample.get("id", "sample"))
        for key in ("execution_error", "evidence_error"):
            if sample.get(key):
                failures.append(f"{sid}: {sample[key]}")
        if sample.get("status") != "complete":
            failures.append(f"{sid}: encoder report is not a complete measurement")
        if getattr(args, "independent_preprocess", False):
            preprocessing = sample.get("preprocess_evidence")
            if not isinstance(preprocessing, dict) or preprocessing.get("verdict") != "pass":
                failures.append(f"{sid}: independent preprocessing did not pass")
        if not sample.get("shape_ok"):
            failures.append(f"{sid}: shape mismatch, got {sample.get('num_values')} expected {sample.get('expected_values')}")
        required_metrics = ("cosine", "rmse", "mean_abs", "max_abs")
        values = {name: _metric_value(sample, name) for name in required_metrics}
        if any(value is None for value in values.values()):
            failures.append(f"{sid}: missing, malformed, or nonfinite encoder parity metrics")
            continue
        cosine, rmse, mean_abs, max_abs = (values[name] for name in required_metrics)
        if not (-1.000001 <= cosine <= 1.000001) or any(value < 0 for value in (rmse, mean_abs, max_abs)):
            failures.append(f"{sid}: invalid encoder parity metric values")
            continue
        if cosine < float(args.min_cosine):
            failures.append(f"{sid}: cosine {cosine:.9f} < {args.min_cosine:.9f}")
        if rmse > float(args.max_rmse):
            failures.append(f"{sid}: rmse {rmse:.9f} > {args.max_rmse:.9f}")
        if args.max_abs is not None and max_abs > float(args.max_abs):
            failures.append(f"{sid}: max_abs {max_abs:.9f} > {args.max_abs:.9f}")
    return failures


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    def shown(value: Any, digits: int) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "n/a"

    lines = [
        "# Qwen3-VL Encoder Prefix Parity",
        "",
        f"- selected: {summary['selected_count']}",
        f"- completed: {summary['completed_count']}",
        f"- passing: {summary['passing_count']}",
        f"- threads: {summary['threads']}",
        f"- image_max_tokens: {summary.get('image_max_tokens')}",
        f"- input_provenance: {'independent_preprocessing' if summary.get('independent_preprocess') else 'shared_processed_tensor'}",
        f"- min_cosine: {shown(summary['aggregate']['min_cosine'], 9)}",
        f"- max_rmse: {shown(summary['aggregate']['max_rmse'], 6)}",
        f"- max_abs: {shown(summary['aggregate']['max_abs'], 6)}",
        f"- failures: {len(summary['failures'])}",
        "",
        "| sample | preprocessing | grid | values | cosine | rmse | mean_abs | max_abs | ck_s | llama_s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in summary["samples"]:
        grid = sample.get("grid")
        if not isinstance(grid, list) or len(grid) != 2:
            grid = ["?", "?"]
        timings = sample.get("timings_sec") if isinstance(sample.get("timings_sec"), dict) else {}
        preprocessing = sample.get("preprocess_evidence")
        lines.append(
            "| {sid} | {preprocess} | {gx}x{gy} | {values} | {cos} | {rmse} | {mean} | {max_abs} | {ck:.1f} | {llama:.1f} |".format(
                sid=sample.get("id"),
                preprocess=(preprocessing.get("verdict", "missing") if isinstance(preprocessing, dict)
                            else "missing" if summary.get("independent_preprocess") else "shared"),
                gx=grid[0],
                gy=grid[1],
                values=sample.get("num_values"),
                cos=shown(_metric_value(sample, "cosine"), 9),
                rmse=shown(_metric_value(sample, "rmse"), 6),
                mean=shown(_metric_value(sample, "mean_abs"), 6),
                max_abs=shown(_metric_value(sample, "max_abs"), 6),
                ck=float(timings.get("ck_encode", 0.0)),
                llama=float(timings.get("llama_encode", 0.0)),
            )
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL encoder prefix parity over a small image set.")
    parser.add_argument("--gguf", type=Path, required=True, help="Path to mmproj-Qwen3VL-*.gguf")
    parser.add_argument("--summary-json", type=Path, default=None, help="OCR summary JSON containing result image paths")
    parser.add_argument("--manifest", type=Path, help="Image corpus manifest with samples and image inputs")
    parser.add_argument("--image", type=Path, action="append", default=[], help="Extra image path; may be repeated")
    parser.add_argument("--limit", type=int, help="Maximum selected images; defaults to all manifest images or 10 ad hoc images")
    parser.add_argument("--require-images", type=int, help="Fail unless at least this many images are selected")
    parser.add_argument("--expected-llama-commit", help="Immutable oracle revision; defaults to the checked-in gitlink")
    parser.add_argument("--output-dir", type=Path, default=Path("build/qwen3vl_encoder_prefix_parity"))
    parser.add_argument("--runtime-dir", type=Path, default=None, help="Reusable generated mmproj runtime directory")
    parser.add_argument("--image-min-tokens", type=int, default=None)
    parser.add_argument("--image-max-tokens", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("CK_NUM_THREADS", "20") or "20"))
    parser.add_argument("--ck-threads", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=16384)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--max-rmse", type=float, default=0.03)
    parser.add_argument("--max-abs", type=float, default=None)
    parser.add_argument("--reuse-reports", action="store_true", help="Reuse existing per-image reports instead of recomputing")
    parser.add_argument("--independent-preprocess", action="store_true", help="Require independent CKE/llama.cpp RGB preprocessing for each image")
    parser.add_argument("--no-fail", action="store_true", help="Write reports but return success even when thresholds fail")
    parser.add_argument("--show-private-details", action="store_true", help="Print image IDs, paths, and failure details to the console")
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.independent_preprocess and (args.reuse_reports or args.no_fail):
        parser.error("independent preprocessing cannot use --reuse-reports or --no-fail")
    if args.manifest is not None and (args.summary_json is not None or args.image):
        parser.error("--manifest cannot be combined with --summary-json or --image")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.require_images is not None and args.require_images <= 0:
        parser.error("--require-images must be positive")

    args.output_dir = args.output_dir.resolve()
    args.runtime_dir = (args.runtime_dir or (args.output_dir / "runtime")).resolve()
    args.ck_threads = int(args.ck_threads or args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)

    limit = args.limit if args.limit is not None else (None if args.manifest else 10)
    try:
        manifest_sha256 = _sha256_file(args.manifest) if args.manifest else None
        specs = (_load_manifest_specs(args.manifest, limit) if args.manifest is not None
                 else _load_image_specs(args.summary_json, args.image, limit))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        (args.output_dir / "input_error.log").write_text(str(exc) + "\n", encoding="utf-8")
        parser.error(str(exc) if args.show_private_details else "image input validation failed; see private input_error.log")
    if not specs:
        raise SystemExit("no images selected; pass --summary-json or --image")
    if args.require_images is not None and len(specs) < args.require_images:
        parser.error(f"selected {len(specs)} images; --require-images needs {args.require_images}")

    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.ck_threads)
    env["OMP_NUM_THREADS"] = str(args.ck_threads)
    expected_llama_commit = args.expected_llama_commit or _checked_in_llama_commit()
    llama_root = Path(env.get("CK_LLAMA_CPP_ROOT", str(REPO_ROOT / "llama.cpp")))
    llama_commit = None
    if args.manifest is not None or args.independent_preprocess:
        if not expected_llama_commit:
            parser.error("oracle source pin is unavailable; supply --expected-llama-commit")
        try:
            llama_commit = _oracle_commit(llama_root)
        except ValueError as exc:
            parser.error(str(exc))
        if llama_commit != expected_llama_commit:
            parser.error(f"llama.cpp revision {llama_commit} differs from required {expected_llama_commit}")

    samples: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, 1):
        detail = f" {spec['id']} -> {spec['image']}" if args.show_private_details else ""
        print(f"[{index}/{len(specs)}] encoder parity{detail}", flush=True)
        try:
            samples.append(_run_one(spec=spec, index=index, args=args, env=env))
        except (OSError, ValueError, TypeError) as exc:
            samples.append({"id": spec["id"], "image": spec["image"], "shape_ok": False,
                            "execution_error": str(exc)})

    failures = _evaluate_samples(samples, args)
    if llama_commit is not None:
        try:
            current_llama_commit = _oracle_commit(llama_root)
        except ValueError:
            current_llama_commit = None
        if current_llama_commit != llama_commit:
            failures.append("llama.cpp source revision changed or disappeared during encoder parity execution")
    if args.manifest is not None:
        try:
            current_manifest_sha256 = _sha256_file(args.manifest)
        except OSError:
            current_manifest_sha256 = None
        if current_manifest_sha256 != manifest_sha256:
            failures.append("image manifest changed or disappeared during encoder parity execution")
    measured = [sample for sample in samples if all(_metric_value(sample, name) is not None
                for name in ("cosine", "rmse", "mean_abs", "max_abs"))]
    aggregate = {
        "min_cosine": min((_metric_value(sample, "cosine") for sample in measured), default=None),
        "max_rmse": max((_metric_value(sample, "rmse") for sample in measured), default=None),
        "max_abs": max((_metric_value(sample, "max_abs") for sample in measured), default=None),
        "all_shapes_ok": all(bool(sample.get("shape_ok")) for sample in samples),
    }
    summary = {
        "gguf": str(args.gguf),
        "manifest_sha256": manifest_sha256,
        "expected_llama_commit": expected_llama_commit,
        "llama_commit": llama_commit,
        "status": "pass" if not failures else "fail",
        "sample_count": len(samples),
        "selected_count": len(specs),
        "completed_count": sum(sample.get("status") == "complete" for sample in samples),
        "passing_count": sum(not _evaluate_samples([sample], args) for sample in samples),
        "required_images": args.require_images,
        "independent_preprocess": bool(args.independent_preprocess),
        "threads": args.threads,
        "ck_threads": args.ck_threads,
        "image_min_tokens": args.image_min_tokens,
        "image_max_tokens": args.image_max_tokens,
        "embed_dim": args.embed_dim,
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_rmse": args.max_rmse,
            "max_abs": args.max_abs,
        },
        "aggregate": aggregate,
        "failures": failures,
        "samples": samples,
    }
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(report_path, summary)

    console_summary = {
        "selected_count": summary["selected_count"],
        "completed_count": summary["completed_count"],
        "passing_count": summary["passing_count"],
        "aggregate": aggregate,
        "failure_count": len(failures),
    }
    if args.show_private_details:
        console_summary.update({"failures": failures, "summary": str(summary_path), "report": str(report_path)})
    print(json.dumps(console_summary, indent=2))
    return 0 if (not failures or args.no_fail) else 1


if __name__ == "__main__":
    raise SystemExit(main())
