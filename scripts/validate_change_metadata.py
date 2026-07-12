#!/usr/bin/env python3
"""Validate durable engineering and content handoff metadata."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


SUBSTANTIVE_TYPES = {"feat", "fix", "refactor", "perf", "revert"}
COMMIT_FIELDS = {
    "Why": ("Why", "Context", "Reason"),
    "What": ("What",),
    "Validation": ("Validation", "Test", "Tests"),
    "Evidence": ("Evidence",),
    "Docs": ("Docs", "Documentation"),
    "Nightly": ("Nightly", "CI"),
    "Content": ("Content", "Content-Handoff", "Content Handoff"),
}
PR_SECTIONS = (
    "Why",
    "What changed",
    "Evidence",
    "Validation",
    "Regression coverage",
    "Documentation",
    "Content handoff",
)
PLACEHOLDER_RE = re.compile(r"^\s*(?:<[^>]+>|tbd|todo|n/?a)\s*$", re.IGNORECASE)


def _clean_commit(message: str) -> str:
    return "\n".join(line for line in message.splitlines() if not line.startswith("#"))


def _field_values(message: str) -> dict[str, str]:
    values: dict[str, str] = {}
    aliases = {
        alias.lower(): canonical
        for canonical, names in COMMIT_FIELDS.items()
        for alias in names
    }
    for line in message.splitlines()[2:]:
        match = re.match(r"^([A-Za-z][A-Za-z -]*):\s*(.*)$", line)
        if match and match.group(1).lower() in aliases:
            values[aliases[match.group(1).lower()]] = match.group(2).strip()
    return values


def validate_commit(message: str) -> list[str]:
    message = _clean_commit(message).strip()
    if not message:
        return []
    header = message.splitlines()[0]
    if header.startswith(("Merge ", "Revert \"", "fixup! ", "squash! ")):
        return []
    match = re.match(r"^([a-z]+)(?:\([^)]+\))?!?: ", header)
    if not match or match.group(1) not in SUBSTANTIVE_TYPES:
        return []

    values = _field_values(message)
    errors = []
    for field in COMMIT_FIELDS:
        value = values.get(field, "")
        if not value or PLACEHOLDER_RE.match(value):
            errors.append(f"missing or placeholder {field}: field")

    content = values.get("Content", "")
    if content and not PLACEHOLDER_RE.match(content):
        if content.lower().startswith("not publishable;"):
            if len(content.partition(";")[2].strip()) < 12:
                errors.append("Content: not publishable requires a concrete reason")
        else:
            for key in ("Angle=", "Claims=", "Caveats=", "Sources="):
                if key not in content:
                    errors.append(f"Content: must include {key[:-1]}")
    return errors


def _sections(body: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1).strip().lower()] = body[match.end():end].strip()
    return sections


def validate_pr(body: str) -> list[str]:
    body = re.sub(r"<!--[\s\S]*?-->", "", body)
    sections = _sections(body)
    errors = []
    for required in PR_SECTIONS:
        value = sections.get(required.lower(), "")
        if not value or PLACEHOLDER_RE.match(value):
            errors.append(f"missing or empty PR section: {required}")

    content = sections.get("content handoff", "")
    if content and not PLACEHOLDER_RE.match(content):
        if re.search(r"(?mi)^Not publishable:\s*.{12,}$", content):
            return errors
        for field in ("Audience", "Angle", "Claims", "Caveats", "Sources"):
            if not re.search(rf"(?mi)^-?\s*{field}:\s*\S", content):
                errors.append(f"Content handoff must include {field}:")
    return errors


def _git_messages(base: str, head: str) -> list[tuple[str, str]]:
    hashes = subprocess.check_output(
        ["git", "rev-list", "--no-merges", f"{base}..{head}"], text=True
    ).splitlines()
    return [
        (commit, subprocess.check_output(["git", "show", "-s", "--format=%B", commit], text=True))
        for commit in hashes
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    commit = subparsers.add_parser("commit")
    commit.add_argument("--file", type=Path)
    commit.add_argument("--message")
    pr = subparsers.add_parser("pr")
    pr.add_argument("--body-file", type=Path, required=True)
    commits = subparsers.add_parser("commits")
    commits.add_argument("--base", required=True)
    commits.add_argument("--head", required=True)
    args = parser.parse_args()

    failures: list[str] = []
    if args.mode == "commit":
        if bool(args.file) == bool(args.message):
            parser.error("commit requires exactly one of --file or --message")
        message = args.file.read_text() if args.file else args.message
        failures = validate_commit(message)
    elif args.mode == "pr":
        failures = validate_pr(args.body_file.read_text())
    else:
        for commit_hash, message in _git_messages(args.base, args.head):
            failures.extend(
                f"{commit_hash[:12]}: {error}" for error in validate_commit(message)
            )

    if failures:
        print("Change metadata validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "Content metadata should let an agent explain: problem -> diagnosis -> "
            "fix -> measured delta -> regression guard -> limitation -> next step.",
            file=sys.stderr,
        )
        return 1
    print("Change metadata validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
