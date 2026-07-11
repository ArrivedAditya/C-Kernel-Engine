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
The compiler must bind the exact kernel ID and public function named by the
kernel map; it must not rank or guess functions from ISA, shape, or benchmark
data. Dimensions and thread counts are function inputs. A named public function
may dispatch internally only among implementations proven to satisfy its
numerical contract.

Quantized and reduced-precision linear kernels must name a scalar contract
oracle. The required evidence ladder is external backend versus scalar oracle,
optimized public function versus scalar oracle, and threadpool dispatch versus
scalar oracle. Independent-output partitioning must not change reduction order.
Split-K requires a distinct contract with partial dtype and merge order.
`make test-numerical-contracts` must execute representative kernel numerics for
declared reduction contracts; schema and resolver checks alone are insufficient.

The supported v8 text circuits and Qwen3-VL routes must resolve attention before
GraphIR construction. GraphIR records the required contract and resolved
provider. LoweredIR and call-ready IR may carry that decision but may not replace
or reinterpret it.
