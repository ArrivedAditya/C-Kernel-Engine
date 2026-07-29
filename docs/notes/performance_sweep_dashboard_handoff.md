# Performance Sweep Dashboard Handoff

The source of truth for a future performance page is the structured output of
`benchmarks/sweep_q4k_llama_performance.py`, not hand-entered HTML.

The first page may be one static HTML file adjacent to the nightly-results
view. It should render a table with these columns:

- CPU marketing name and raw model string
- visible physical/logical cores and ISA
- kernel family and GEMM/GEMV phase
- explicit `M x N x K` shape
- provider ID, packed layout, M tile, and threads
- CKE steady-state time
- llama.cpp steady-state time and ratio
- PyTorch or oneDNN time only when the reference is relevant; otherwise `N/A`
- exactness status
- engine/reference provenance and run timestamp

Filters can be added later for CPU family, model shape, kernel, ISA, packing,
and thread count. Raw reports must remain append-only; a generated view may
select the newest result for each configuration and highlight regressions.

Do not display or collect hostnames, IP addresses, Kubernetes node names,
usernames, private paths, prompts, images, or model data. CPU/compiler/commit
fields are measurement provenance, never an allowlist for runtime execution.

The implemented first scope is deliberately Qwen3.6 Q4_K x Q8_K prefill. Its
default hot-shape suite measures the MLP gate/up and recurrent gate projections
at the observed 33-token prompt and the configured 1034-token context. The
runner emits both structured JSON and a flat CSV sweep table. This is the
initial performance investigation, not a runtime dispatch database.

Qwen3.6 decode uses the related Q4_K x Q8_K GEMV path and should be added as a
separate provider sweep rather than pretending that `M=1` is a prefill GEMM.
Only after the Qwen3.6 hot paths are understood should future runners extend the
same schema to Q5_K x Q8_K, Q6_K x Q8_K, Q8_0 x Q8_0, FP32, BF16, and other
models. A mixed `Q4 x Q5` row should only exist if an actual kernel uses Q5
activations; GGUF Q4_K_M normally mixes weight types while retaining Q8_K
activations.

## Initial Qwen3.6 finding

The first sweep on an Intel Xeon Gold 6542Y (24 physical cores, 48 logical
CPUs) compared the real 33-token Qwen3.6 MLP gate/up and recurrent-gate
dimensions at 12 and 24 threads. Every reported CKE candidate matched the
certified four-row CKE reference exactly, and the complete production operation
matched the independent llama.cpp graph bit-for-bit.

The native VNNI provider was about 2.4x faster than CKE's forced-AVX2 provider,
so ISA dispatch is working and VNNI is valuable. It still measured roughly 2x
slower than llama.cpp in the least noisy same-run samples. Source and
disassembly attribution explain why:

- CKE's VNNI provider uses 256-bit YMM `vpdpbusd` with a 4M x 8N tile.
- llama.cpp's `q4_K_8x8` packed format enters an AVX-512 implementation on this
  CPU and processes 16 token rows x 16 output columns with ZMM registers.
- CKE's existing 8M provider uses the older packed-meta/AVX2 arithmetic; it is
  not an AVX-512 VNNI equivalent.
- The real 33-token operation consists of 32 complete prefill rows plus one
  decode-order GEMV residual. Provider rows therefore measure M=32 and record
  `requested_M=33`; only the production oracle measures the complete M=33
  operation.

The next candidate should be an AVX-512 VNNI 16M x 16N (with smaller exact
tails), using the existing 4M provider as the fast internal exactness reference
and llama.cpp as the external oracle. It must enter the sweep as a new provider;
do not replace production dispatch until the table demonstrates exactness and
a stable win at both Qwen3.6 hot shapes.
