<!--
PR metadata is validated by scripts/validate_change_metadata.py (Change Metadata workflow).
Keep every section below; empty or placeholder sections fail the gate.
Validate locally before opening or editing the PR:
  make validate-pr-metadata BODY=path/to/pr-body.md
-->

## Why

<!-- What problem or evidence gap required this change? -->

## What changed

<!-- Describe the implementation and ownership boundary. -->

## Evidence

<!-- Record measurements, numerical deltas, artifact paths, and the comparison baseline. -->

## Validation

<!-- Exact commands and results. Do not describe unavailable hardware as passing. -->

## Regression coverage

<!-- State the unit, stitched, nightly, and end-to-end guards added or exercised. -->

## Documentation

<!-- Link updated documentation, or explain why documentation is unnecessary. -->

## Content handoff

<!--
This is a durable prompt for an Antsand/content agent scanning PRs and git history.
Provide enough evidence to explain the progression methodically:
problem -> diagnosis -> fix -> measured delta -> regression guard -> limitation -> next step.
Link detailed documentation and machine-readable artifacts instead of duplicating them here.
-->

- Audience: <!-- Who benefits from this evidence? -->
- Angle: <!-- What is the defensible engineering story? -->
- Claims: <!-- Which measured claims may be published? -->
- Caveats: <!-- Hardware, dataset, parity, or reproducibility limits. -->
- Sources: <!-- Commit, code, docs, logs, JSON/CSV, and profiler artifacts. -->

<!-- For non-publishable work, replace the five lines above with: -->
<!-- Not publishable: concrete reason this change has no useful external story -->
