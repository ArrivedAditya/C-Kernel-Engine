# v8 Contract Hard-Fail Policy

Numerical and execution contracts are release gates. A contract error means the
circuit requirement, canonical contract, or executable kernel map is incomplete
or incompatible.

## Required response to a contract failure

1. Identify the first invalid or unresolved operation.
2. Correct the circuit semantics, canonical contract, or kernel capability.
3. Add or update leaf, public-route, stitched, and E2E evidence.
4. Re-run `make test-numerical-contracts` and the affected model-family sweep.

## Forbidden responses

Agents and contributors must not:

- add a fallback provider;
- add a silent default;
- ignore an unknown schema field;
- relax a numerical tolerance;
- downgrade a hard fault to a warning;
- add an environment variable or CLI flag that bypasses resolution;
- restore a model-specific attention selection in codegen;
- claim a route is validated without named reference evidence.

Threading belongs in execution capabilities. If worker partitioning changes an
accumulation or merge order, the numerical contract must define that order too.
A performance planner may rank only providers that already satisfy the complete
semantic contract.

The supported v8 text circuits and Qwen3-VL routes must resolve attention before
GraphIR construction. GraphIR records the required contract and resolved
provider. LoweredIR and call-ready IR may carry that decision but may not replace
or reinterpret it.
