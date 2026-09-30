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


def _artifact_hashes(samples: list[dict[str, Any]]) -> dict[str, str]:
    identities: list[dict[str, str]] = []
    for sample in samples:
        report = sample.get("artifact_identity")
        if not isinstance(report, dict):
            raise ValueError("missing artifact identity")
        selected: dict[str, str] = {}
        for role in ARTIFACT_ROLES:
            item = report.get(role)
            digest = item.get("sha256") if isinstance(item, dict) else None
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ValueError(f"missing {role} hash")
            selected[role] = digest
        identities.append(selected)
    if not identities or any(item != identities[0] for item in identities[1:]):
        raise ValueError("artifact identities differ across samples")
    return identities[0]


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
    identities = []
    for row in rows:
        selected_hashes = {}
        for role, key in (("decoder", "decoder_sha256"), ("engine", "engine_sha256")):
            digest = row.get(key)
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ValueError(f"missing {role} hash")
            selected_hashes[role] = digest
        identities.append(selected_hashes)
    if any(item != identities[0] for item in identities[1:]):
        raise ValueError("decoder artifact identities differ across rows")
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
        "bridge_reported_artifact_sha256": identities[0],
    }


def build_public_summary(private: dict[str, Any], corpus_alias: str) -> dict[str, Any]:
    if not ALIAS_RE.fullmatch(corpus_alias):
        raise ValueError("invalid corpus alias")
    if "samples" in private:
        return _build_encoder_public_summary(private, corpus_alias)
    if "rows" in private:
        return _build_decoder_public_summary(private, corpus_alias)
    raise ValueError("unknown vision-parity summary")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-summary", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--corpus-alias", required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.private_summary.resolve() == args.public_summary.resolve():
            raise ValueError("public and private summary paths must differ")
        args.public_summary.unlink(missing_ok=True)
        private = json.loads(args.private_summary.read_text(encoding="utf-8"))
        if not isinstance(private, dict):
            raise ValueError("invalid private summary")
        public = build_public_summary(private, args.corpus_alias)
        args.public_summary.parent.mkdir(parents=True, exist_ok=True)
        args.public_summary.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
        args.public_summary.chmod(0o600)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        parser.error("public summary export failed; inspect the private evidence locally")
    print("public summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
