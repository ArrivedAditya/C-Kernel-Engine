#!/usr/bin/env python3
"""Export an allowlisted public view of private vision-parity evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any


ARTIFACT_ROLES = (
    "mmproj_gguf",
    "ck_generated_source",
    "ck_model_library",
    "ck_engine_library",
    "llama_shim_library",
    "llama_mtmd_library",
    "llama_ggml_cpu_library",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ALIAS_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {label}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid {label}")
    return float(value)


def _artifact_hashes(samples: list[dict[str, Any]]) -> dict[str, list[str]]:
    observed: dict[str, set[str]] = {role: set() for role in ARTIFACT_ROLES}
    for sample in samples:
        report = sample.get("artifact_identity")
        if not isinstance(report, dict):
            raise ValueError("missing artifact identity")
        for role in ARTIFACT_ROLES:
            item = report.get(role)
            digest = item.get("sha256") if isinstance(item, dict) else None
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ValueError(f"missing {role} hash")
            observed[role].add(digest)
    return {role: sorted(hashes) for role, hashes in observed.items()}


def _build_encoder_public_summary(private: dict[str, Any], corpus_alias: str) -> dict[str, Any]:
    if not ALIAS_RE.fullmatch(corpus_alias):
        raise ValueError("invalid corpus alias")
    samples = private.get("samples")
    failures = private.get("failures")
    aggregate = private.get("aggregate")
    if not isinstance(samples, list) or not samples or not all(isinstance(s, dict) for s in samples):
        raise ValueError("missing sample evidence")
    if not isinstance(failures, list) or not isinstance(aggregate, dict):
        raise ValueError("missing verdict evidence")
    selected = _count(private.get("selected_count"), "selected count")
    completed = _count(private.get("completed_count"), "completed count")
    passing = _count(private.get("passing_count"), "passing count")
    if selected != len(samples) or not (0 <= passing <= completed <= selected):
        raise ValueError("inconsistent case counts")
    status = private.get("status")
    if status not in ("pass", "fail") or (status == "pass") != (passing == selected and not failures):
        raise ValueError("inconsistent overall verdict")
    oracle_commit = private.get("llama_commit")
    if not isinstance(oracle_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", oracle_commit):
        raise ValueError("missing oracle source revision")
    metrics = {
        name: _number(aggregate.get(name), name)
        for name in ("min_cosine", "max_rmse", "max_abs")
    }
    elapsed = sum(_number(sample.get("suite_elapsed_sec"), "elapsed time") for sample in samples)
    if elapsed < 0:
        raise ValueError("invalid elapsed time")
    return {
        "schema_version": 1,
        "lane": "independent_encoder_prefix" if private.get("independent_preprocess") is True else "shared_input_encoder_prefix",
        "corpus_alias": corpus_alias,
        "status": status,
        "selected_count": selected,
        "completed_count": completed,
        "passing_count": passing,
        "failure_count": len(failures),
        "independent_preprocess": private.get("independent_preprocess") is True,
        "aggregate": metrics,
        "total_elapsed_sec": elapsed,
        "oracle_source_commit": oracle_commit,
        "bridge_reported_artifact_sha256": _artifact_hashes(samples),
    }


def _build_decoder_public_summary(private: dict[str, Any], corpus_alias: str) -> dict[str, Any]:
    rows = private.get("rows")
    provenance = private.get("provenance")
    timing = private.get("timing")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("missing decoder rows")
    if not isinstance(provenance, dict) or not isinstance(timing, dict):
        raise ValueError("missing decoder provenance")
    selected = _count(private.get("requested"), "selected count")
    completed = _count(private.get("completed"), "completed count")
    passing = _count(private.get("passed"), "passing count")
    failed = _count(private.get("failed"), "failure count")
    if completed != len(rows) or completed != passing + failed or completed > selected:
        raise ValueError("inconsistent decoder counts")
    status = private.get("status")
    expected_status = "fail" if failed else "pass" if completed == selected else "incomplete"
    if status != expected_status:
        raise ValueError("inconsistent decoder verdict")
    scope = private.get("certification_scope")
    if scope not in ("shared_prefix_decoder", "localization"):
        raise ValueError("invalid decoder certification scope")
    oracle_commit = provenance.get("llama_commit")
    cke_commit = provenance.get("cke_commit")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value)
           for value in (oracle_commit, cke_commit)):
        raise ValueError("missing decoder source revision")
    identities: dict[str, set[str]] = {"decoder": set(), "engine": set()}
    for row in rows:
        for role, key in (("decoder", "decoder_sha256"), ("engine", "engine_sha256")):
            digest = row.get(key)
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ValueError(f"missing {role} hash")
            identities[role].add(digest)
    elapsed = _number(timing.get("total_sec"), "total elapsed time")
    if elapsed < 0:
        raise ValueError("invalid total elapsed time")
    return {
        "schema_version": 1,
        "lane": scope,
        "corpus_alias": corpus_alias,
        "status": status,
        "selected_count": selected,
        "completed_count": completed,
        "passing_count": passing,
        "failure_count": failed,
        "total_elapsed_sec": elapsed,
        "cke_source_commit": cke_commit,
        "oracle_source_commit": oracle_commit,
        "bridge_reported_artifact_sha256": {role: sorted(hashes) for role, hashes in identities.items()},
    }


def _build_ocr_public_summary(private: dict[str, Any], corpus_alias: str) -> dict[str, Any]:
    aggregate = private.get("aggregate")
    rows = private.get("rows")
    if not isinstance(aggregate, dict) or not isinstance(rows, list):
        raise ValueError("missing OCR aggregate or rows")
    selected = _count(aggregate.get("requested"), "selected count")
    completed = _count(aggregate.get("completed"), "completed count")
    passing = _count(aggregate.get("passing"), "passing count")
    errors = _count(aggregate.get("errors"), "error count")
    if not selected or not (0 <= passing <= completed <= selected) or errors > selected - completed:
        raise ValueError("inconsistent OCR case counts")
    if len(rows) != completed:
        raise ValueError("incomplete OCR row evidence")
    expected_status = "pass" if passing == selected else "fail" if completed == selected else "incomplete"
    expected_execution = "complete" if completed == selected else "incomplete"
    if (private.get("quality_status") != expected_status
            or private.get("status") != expected_execution):
        raise ValueError("inconsistent OCR verdict")
    identities: dict[str, set[str]] = {"encoder": set(), "decoder": set()}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "complete":
            raise ValueError("invalid OCR case row")
        evidence = row.get("execution_evidence")
        if (not isinstance(evidence, dict)
                or evidence.get("evidence_kind") != "bridge_reported_paths_and_artifact_hashes"
                or evidence.get("loaded_engine_verified") is not False
                or evidence.get("prefix_source") != "encoder"):
            raise ValueError("invalid OCR execution evidence")
        hashes = evidence.get("model_library_sha256")
        for role in identities:
            digest = hashes.get(role) if isinstance(hashes, dict) else None
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ValueError(f"missing {role} generated-library hash")
            identities[role].add(digest)
    accuracy = _number(aggregate.get("field_accuracy"), "field accuracy")
    nonempty_accuracy = _number(aggregate.get("nonempty_field_accuracy"), "nonempty field accuracy")
    elapsed = _number(aggregate.get("total_wall_sec"), "total elapsed time")
    if not (0 <= accuracy <= 1 and 0 <= nonempty_accuracy <= 1 and elapsed >= 0):
        raise ValueError("invalid OCR aggregate metrics")
    return {
        "schema_version": 1,
        "lane": "independent_image_to_ocr_task_quality",
        "corpus_alias": corpus_alias,
        "status": expected_status,
        "selected_count": selected,
        "completed_count": completed,
        "passing_count": passing,
        "error_count": errors,
        "unattempted_count": selected - completed - errors,
        "field_accuracy": accuracy,
        "nonempty_field_accuracy": nonempty_accuracy,
        "total_elapsed_sec": elapsed,
        "runtime_identity_kind": "bridge_reported_paths_and_artifact_hashes",
        "loaded_engine_verified": False,
        "bridge_reported_artifact_sha256": {role: sorted(hashes) for role, hashes in identities.items()},
    }


def build_public_summary(private: dict[str, Any], corpus_alias: str) -> dict[str, Any]:
    if not ALIAS_RE.fullmatch(corpus_alias):
        raise ValueError("invalid corpus alias")
    if private.get("schema") == "cke.multimodal_ocr_corpus_certification":
        return _build_ocr_public_summary(private, corpus_alias)
    if "samples" in private:
        return _build_encoder_public_summary(private, corpus_alias)
    if "rows" in private:
        return _build_decoder_public_summary(private, corpus_alias)
    raise ValueError("unknown vision-parity summary")


def build_count_free_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    validated = build_public_summary(private, "private")
    return {
        "schema_version": validated["schema_version"],
        "lane": validated["lane"],
        "status": validated["status"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-summary", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--corpus-alias", help="Required unless --count-free is used")
    parser.add_argument(
        "--count-free", action="store_true",
        help="Publish only the scoped lane and verdict, without corpus identifiers, counts, metrics, or timings",
    )
    args = parser.parse_args(argv)
    if not args.count_free and not args.corpus_alias:
        parser.error("--corpus-alias is required without --count-free")
    os.umask(0o077)
    try:
        if args.private_summary.resolve() == args.public_summary.resolve():
            raise ValueError("public and private summary paths must differ")
        args.public_summary.unlink(missing_ok=True)
        private = json.loads(args.private_summary.read_text(encoding="utf-8"))
        if not isinstance(private, dict):
            raise ValueError("invalid private summary")
        public = (
            build_count_free_public_summary(private)
            if args.count_free else build_public_summary(private, args.corpus_alias)
        )
        args.public_summary.parent.mkdir(parents=True, exist_ok=True)
        args.public_summary.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
        args.public_summary.chmod(0o600)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        parser.error("public summary export failed; inspect the private evidence locally")
    print("public summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
