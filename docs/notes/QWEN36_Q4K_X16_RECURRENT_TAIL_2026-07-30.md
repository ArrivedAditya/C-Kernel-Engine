# Qwen3.6 Q4_K x16 recurrent-tail optimization

Date: 2026-07-30

## Scope

The experimental AVX-512/VNNI x16 provider processes complete four-row
Qwen3.6 prefill groups with its matrix kernel. Residual rows must retain the
independently certified llama.cpp-compatible GEMV reduction order.

For the common 23-token prompt, the recurrent-gate projection has shape
`M=23, N=6144, K=5120`: twenty rows use the matrix provider and three rows use
the exact GEMV provider.

## Change

The three residual rows now share one thread-pool dispatch. Each worker keeps
its assigned x16 packed-weight range and evaluates the three rows sequentially
with the existing exact GEMV function. No dot-product or reduction order
changed.

The optimization is deliberately restricted to `N=6144`. Applying the same
schedule to the much wider `N=34816` gate/up projection was slower because its
per-worker weight range does not remain usefully cache-resident.

`CK_DISABLE_Q4K_X16_BATCHED_TAIL=1` restores one dispatch per residual row for
diagnosis and same-host A/B measurement.

## Measured result

On the 24-thread Xeon test host, alternating real-model A/B runs measured the
recurrent-gate projection at a median 78.545 ms before and 58.133 ms after,
a 26.0% projection-level improvement. This removes about 20.4 ms from the
measured prompt, approximately 0.6% of total prefill time. Whole-prefill
measurements contain additional host/cache noise and are not used to claim a
larger direct gain.

The complete real-model first-token logit array was byte-exact between the old
and new schedules. The production oracle also exercises the full
`M=23, N=6144, K=5120` shape against llama.cpp when AVX-512/VNNI is available.

This does not promote the experimental x16 provider globally and does not
affect decode, AVX2, Qwen3.5, or other model shapes.
