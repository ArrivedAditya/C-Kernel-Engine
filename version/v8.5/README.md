# CKE v8.5 Numerical Contracts

v8.5 is an additive contract-validation layer over the v8 inference pipeline.
It does not change v8 code generation or runtime dispatch by default.

The compiler pipeline is:

```text
weights + circuits + kernel maps -> deterministic DSL lowering -> generated C
```

The purpose of this lane is to separate three decisions that v8 currently
mixes together:

1. A circuit declares operations, edges, and required semantics.
2. A kernel map declares which complete numerical contracts it implements.
3. Lowering resolves the request or fails before C code is generated.

The first supported contract family is attention reduction. A reduction ID is
a complete semantic profile, not a loose dtype label. For example,
`f16_online_fp32_merge` specifies query rounding, score accumulation, online
softmax state, value accumulation, partitioning, and partial merge behavior.
The full-attention `f16_kv_fp32_online` contract separately specifies FP32 Q,
FP16-rounded K/V, and FP32 online-softmax accumulation.

## Layout

- `contracts/attention_reductions.json`: canonical semantic definitions.
- `circuits/*.json`: graph/circuit declarations and semantic requirements.
- `kernel_maps/attention_contracts.overlay.json`: capabilities of existing
  kernel IDs.
- `scripts/resolve_attention_contracts_v85.py`: model-name-blind resolver.
- `tests/test_attention_contracts_v85.py`: schema, selection, and failure tests.

## Validation states

- `unresolved`: candidate semantics; usable only for attribution work.
- `observed`: supported by focused evidence but not the full production gate.
- `validated`: kernel, route, stitched circuit, and end-to-end parity pass.

Production resolution requires `validated` at the contract definition,
circuit request, and kernel implementation levels. It also requires an
explicit runtime selector. There is no silent fallback.

## Usage

```bash
make test-numerical-contracts

python3 version/v8.5/scripts/resolve_attention_contracts_v85.py \
  --circuit qwen3vl --phase decode --mode bringup --pretty
```

The Qwen3-VL decode route uses an explicit v8.5 C reduction selector. It
resolves in `bringup` mode and intentionally fails in `production` mode until
the circuit request is promoted from `observed` to `validated` by stitched
and end-to-end parity. The generic reduction and C route are validated against
llama.cpp at KV 511, 512, and 1058.

## Migration rule

Move one operator family at a time:

1. Record the current semantic profile without changing runtime behavior.
2. Add leaf and public-route numerical tests.
3. Add an explicit C contract selector.
4. Validate generated one-layer and full-model parity.
5. Promote all three validation states to `validated`.
6. Only then let v8.5 production lowering select the implementation.

Circuits never name kernel IDs. They declare `op + requires`. The kernel-map
compiler must find exactly one provider. Zero matches and multiple matches are
both compile-time errors.
