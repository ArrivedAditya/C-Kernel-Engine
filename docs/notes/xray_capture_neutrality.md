# X-ray capture neutrality

## Why X-ray needs a neutrality contract

X-ray is intended to explain a numerical difference, but checkpoint export is additional work inside the observed process. It performs environment checks, copies or reads activation memory, formats paths, opens files, writes data, and closes files between model operations. That work changes timing and cache state. Correct deterministic kernels should still return the same values, but an existing race, unsafe scratch-buffer lifetime, uninitialized read, or scheduling-dependent reduction can become visible only when instrumentation changes timing.

Consequently, a captured tensor is evidence only if the captured execution reproduces an uncaptured execution on the same causal token history. A plausible tensor dump is not sufficient.

## Acceptance sequence

The persistent trajectory harness applies the following sequence automatically whenever CKE checkpoint capture is requested:

1. **Control A — uncaptured execution.** Establish the reference predictions and full logits.
2. **Control B — uncaptured forced replay.** Replay Control A's tokens, or the declared external teacher tokens, so every causal prefix is identical. Compare all logits bit-for-bit.
3. **Repeatability gate.** If Controls A and B differ, reject X-ray attribution with `uncaptured_runtime_is_not_repeatable`. Capture is not attempted because observer interference cannot be distinguished from production nondeterminism.
4. **Aggregate capture.** Replay the identical tokens with all requested checkpoints enabled.
5. **Neutrality gate.** Compare every captured-run logit bit-for-bit with Control B. Accept the aggregate artifacts only when they are identical.
6. **Isolated-boundary fallback.** If a multi-boundary hidden capture is non-neutral, run one replay per boundary. Each replay is independently compared with Control B. Artifacts are accepted only if every isolated replay is neutral.
7. **Fail closed.** Rejected aggregate artifacts remain labelled as rejected, but are never returned in the accepted `artifacts` list. A rejected capture causes a distinct nonzero CLI result even if CKE and the external oracle otherwise agree.

The fallback is intentionally unavailable for KV capture and binary parity capture because those formats can represent coupled state that cannot be reconstructed by blindly splitting a comma-separated operation list.

## Why the comparison is bit-exact

The neutrality question differs from CKE-versus-llama.cpp parity. Different engines may legitimately use different floating-point reduction orders, so their logits are compared with numerical metrics and token decisions. Two executions of the same CKE runtime, provider map, thread count, input tokens, and forced history have a stronger contract: diagnostic instrumentation should not change any output bit.

Using a numerical tolerance for observer neutrality could hide precisely the small perturbation that later becomes amplified by a deep recurrent model. If a production provider is intentionally nondeterministic, that fact must first be measured and declared as its own contract; X-ray must not silently absorb it into a tolerance.

## Report schema

The capture report uses `cke.xray.capture-neutrality.v1` and records:

- the full-logit repeatability result for two uncaptured controls;
- the aggregate captured-versus-control comparison;
- the first differing generated step and logit metrics;
- whether isolated fallback ran;
- acceptance or rejection for every requested boundary;
- accepted and rejected artifact lists; and
- the final accepted mode: `aggregate`, `isolated_boundaries`, or none.

## Qwen3.6 qualification finding

The real Qwen3.6-27B Q4_K_M qualification rejected capture before instrumentation. Two uncaptured 24-thread runs on the same forced trajectory first differed at generated step 76. Their top-1 token still matched, but the full logits were not bit-identical (cosine 0.999641, RMSE 0.069543, maximum absolute difference 0.326646).

This corrects the earlier provisional interpretation that exporting many layer-63 boundaries necessarily perturbed CKE. The current evidence establishes production-path run-to-run numerical nondeterminism. It does not yet identify its source. The next investigation should hold the token history fixed and vary only thread count and provider/scheduling selection to isolate whether the source is a parallel reduction, recurrent-state update, scratch-buffer race, or another provider contract.
