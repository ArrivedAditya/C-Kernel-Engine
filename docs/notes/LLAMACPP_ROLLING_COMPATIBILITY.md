# llama.cpp Rolling Compatibility

CK uses two distinct llama.cpp lanes:

1. The repository gitlink is the authoritative numerical oracle. It is pinned,
   reviewed, and deterministic.
2. The rolling compatibility workflow probes current upstream `HEAD`. It
   detects API, patch, build, numerical, and performance changes, but never
   updates the authoritative pin.

Run the metadata and patch-compatibility probe locally:

```bash
.venv/bin/python scripts/run_llamacpp_rolling_compat.py --resolve-only
```

Run the full rolling quick-parity probe:

```bash
.venv/bin/python scripts/run_llamacpp_rolling_compat.py
```

Evidence is written under `build/llamacpp_rolling/`. The JSON report records
the CK commit, pinned and rolling llama.cpp commits, the upstream commit delta,
patch applicability, and the rolling quick-parity result.

A rolling failure is actionable compatibility evidence. It does not authorize
changing tolerances, expected values, or the pinned oracle. Updating the pin
requires a separate reviewed change with pinned and rolling results compared on
the same hardware, model, prompts, thread settings, and numerical gates.
