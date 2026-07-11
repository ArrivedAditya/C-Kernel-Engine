# v8 DSL Zero-Hardcoding Refactor

## Target

The generic v8 compiler consumes weights/configuration, circuits, kernel maps,
and numerical contracts. It must not select model mathematics from model-family
names. Circuits own topology and required semantics; kernel maps own supported
functions and execution capabilities; code generation emits the resolved call
IR without reselecting it.

## First Inventory

The initial AST inventory found 44 explicit family predicates in generic
compiler modules:

- `build_ir_v8.py`: 21
- `codegen_core_v8.py`: 6
- `codegen_prefill_v8.py`: 13
- `codegen_v8.py`: 4

The first migration removed three runtime numerical-policy branches for
Qwen3.5, Qwen3-VL vision, and Gemma3. Their defaults now live in validated
`contract.runtime_defaults` sections of the corresponding circuits. The
remaining 41 sites are tracked as cleanup work, not accepted exceptions.

No legacy allowlist is used. `version/v8/dsl_policy.json` lists only compiler
functions already cleaned and covered by the zero-hardcoding gate. Each
migration expands that enforced set. Removing a named function from the policy
or naming a nonexistent function is a test failure.

## Gates

```bash
make test-v8-dsl-policy
make test-v8-dsl
```

The lightweight policy gate runs in `v8-regression-fast` and as a dedicated
nightly row. It currently checks:

- AST rejection of model-family literals in cleaned generic functions.
- Circuit identity invariance.
- JSON key-order determinism.
- Explicit runtime override stability.
- Unknown circuit runtime policy hard failures.
- Policy-scope weakening hard failures.

The aggregate gate also runs numerical contract resolution and template
dataflow audits, including zero-provider, ambiguous-provider, incompatible
threading/reduction, and invalid M-RoPE cases.

## Migration Order

1. Replace family-based runtime/config defaults with validated circuit fields.
2. Replace family-based kernel selection with exact kernel-map resolution.
3. Replace multimodal family checks with declared runtime capabilities and
   explicit circuit operations.
4. Replace model-specific diagnostic modes with semantic checkpoint metadata.
5. Remove model-specific LoweredIR and codegen branches.
6. Expand the AST policy to the complete generic compiler modules.

Every group is characterized before removal, then validated with stage-level
negative tests, generated-IR checks, compilation, family regression, and the
applicable llama.cpp or PyTorch parity gate.
