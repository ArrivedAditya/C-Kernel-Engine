# Qwen3-VL BF16 Numerical Contract Handoff

## Scope

This change introduces a model-blind, fail-closed numerical execution contract path. It does not claim complete Qwen3-VL BF16 parity.

The circuit declares the required semantic contract and checkpoint ABI. A kernel map advertises one exact implementation and its arithmetic/threading behavior. The resolver accepts exactly one matching kernel ID and function; zero or multiple matches stop compilation. No benchmark, ranking, model-name condition, or fallback participates in selection.

The first registered contract covers `gemm_nt_bf16` prefill. Additional attention, normalization, residual, activation, projector, and position-transform contracts must be added only after their boundaries are measured against the backend oracle.

## Proven Baseline

The private large-form diagnostic established a BF16 storage-boundary mismatch:

| Checkpoint | Before relative RMSE | BF16-matched position relative RMSE |
| --- | ---: | ---: |
| Position output | 0.002781 | 0.0000328 |
| Layer 0 output | 0.004467 | 0.003836 |
| Final prefix | 0.048227 | 0.046911 |

This proves that explicit storage/rounding semantics matter, but it does not justify a Qwen-specific position bypass. The private image and raw tensors are intentionally excluded.

## Contract ABI

Resolved GraphIR retains:

- required and resolved contract IDs
- exact kernel ID and public function
- storage, compute, and accumulator dtypes
- rounding points
- reduction order, partial accumulator, and merge order
- work partition and split strategy
- determinism and thread-count arithmetic behavior
- semantic checkpoint ID, producer, logical layout, and named axes

The same metadata is copied into LoweredIR and call IR by the existing contract propagation path. A complete example is in `version/v8/contracts/examples/bf16_gemm_resolved_graph_ir.json`.

## Position Contracts

Position transforms explicitly distinguish split-half, pairwise/interleaved, and multi-section pairing. They also declare rotary width, head width, position rank, axis order, frequency/intermediate precision, rounding, and threading.

For Qwen3-VL, multi-section metadata must use `axis_selection`. Sections cannot redefine rotary width. When concrete values are present, `mrope_n_dims` must equal the required full rotary width and the rotary width cannot exceed head width.

## Parity Profiles And Bisection

Tolerances are not circuit data. `version/v8/parity_profiles/qwen3vl_pytorch_bf16_v1.json` keeps backend mappings and dtype thresholds separate from execution semantics.

`plan_parity_bisection_v8.py` consumes bounded checkpoint results. A sparse layer-8 pass/layer-16 failure requests only layers 9 through 15. Once the first layer is identified, the profile can request norm, Q/K/V, RoPE, attention, projection, residual, and MLP boundaries for that block.

Comparators must reject comparisons when checkpoint ID, producer, logical layout, named axes, resolved contract ID, kernel ID, or function differ.

## Validation

Focused results:

- numerical execution contract tests: 10 passed
- hidden-export extent tests: 2 passed
- existing attention contract tests: 22 passed
- full attention tests: 7 passed
- FP16 split-KV local checks: 11 passed; llama.cpp oracle row unavailable because this checkout has no llama.cpp headers/libraries

`make test-numerical-contracts` therefore stops on the external llama.cpp availability gate in this workspace. This is a recorded environment limitation, not a relaxed test.

## Next Contract Work

1. Use sparse BF16 checkpoints on the public synthetic fixture and the private large-form diagnostic.
2. Register contracts only at the first measured divergent semantic edge.
3. Add capability metadata to the exact kernel that implements the measured arithmetic.
4. Require zero/one/many provider tests plus stitched GraphIR/LoweredIR/call-IR tests.
5. Compare mixed-prefill logits and teacher-forced tokens before promoting a contract to `validated`.
6. Promote validated checks into `test-bf16` and conditional BF16 nightly coverage.

AVX2 should consume this metadata after rebasing its draft diagnostics. Its canonical tensor loading, llama.cpp adapters, metrics, and reports remain complementary and should not define a competing contract schema.
