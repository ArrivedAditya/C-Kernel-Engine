# Instella-MoE V8 bring-up (2026-08-02)

This note records the fail-closed CKE bring-up for AMD's
`amd/Instella-MoE-16B-A3B-Think`. Tensor mapping and synthetic generated-C
lowering are complete; this is not yet a real-weight parity claim.

## Upstream evidence

- Model card: <https://huggingface.co/amd/Instella-MoE-16B-A3B-Think>
- AMD architecture article: <https://rocm.blogs.amd.com/artificial-intelligence/instella-moe/README.html>
- Published configuration SHA-256:
  `0ffd49ba9560130ffac0e0cb9a1715469ba45176bccb214ab7ce74004aee2c3a`
- Published safetensors index SHA-256:
  `01833f14b5968713b6f3b795ab05a9520c592f35e0881fa394c3a08390f7a0d0`
- Reference modeling file SHA-256:
  `225f68ab938171f8d7e4b5dbd4dc25a34fbbcf9ac4428ee6f3c4c150490b0c4a`
- Reference configuration file SHA-256:
  `0bba36fa1ffd0abdfa92c15caaa66cc36a854c81a5dca274cdf5e165a93adcfb`

The index was audited without downloading the 31.7 GB weight payload. It has
5,344 tensors in six shards, 27 decoder layers, one dense prefix layer, and
exactly 64 routed experts with gate/up/down weights in each of layers 1-26.

## Why this is not stock DeepSeek-V3 or Kimi

The Hugging Face config intentionally advertises `model_type=deepseek_v3`, but
the architecture class is `InstellaMoEForCausalLM`. Instella adds two graph
semantics that prevent safe reuse of a stock MLA circuit:

1. Gated MLA multiplies attention output by a learned sigmoid gate before the
   output projection.
2. FarSkip carries both a main residual stream and a routed-free residual
   stream between decoder layers. The shared expert result feeds both streams,
   while the routed expert result feeds only the main stream.

It also uses partial interleaved RoPE with YaRN scaling. Treating it as Kimi or
ordinary DeepSeek would map most tensor names while silently executing the
wrong model.

## Implemented in the first patch

- Architecture-first detection so the DeepSeek compatibility label cannot
  override the Instella class.
- A strict, declared safetensors index contract and published-index audit.
- Tensor-role mapping for embeddings, MLA, attention gates, dense layer 0,
  routed/shared experts, norms, and logits.
- Runtime config metadata for gated attention, FarSkip, MoE topology, MLA
  dimensions, interleaved RoPE, and YaRN.
- An observed BF16 shared-SwiGLU/FarSkip reference provider that computes the
  shared expert once and emits both residual streams with the reference
  parenthesization.
- Synthetic converter, topology-failure, contract-inspector, numerical-contract,
  and PyTorch boundary tests.

## Synthetic circuit status

The following pieces now exist:

- `instella_moe.json`, an executable circuit with separate dense, first-FarSkip,
  and continuing-FarSkip layer kinds.
- Circuit-owned `graph_slots` and `activation_bindings`; the compiler has no
  Instella architecture branch.
- Generic explicit-position YaRN FP32/BF16 cache providers. Generated
  initialization does not yet bind their position input and full YaRN
  parameters, so this boundary remains fail-closed.
- A BF16 FarSkip composite provider with a kernel-map-owned call ABI.
- A tiny two-layer model test that converts safetensors, lowers prefill and
  decode to error-free call IR, generates strict-contract C, and passes a C11
  syntax compile.

The contract inspector reports `bringup_required` for the missing generated
interleaved-YaRN binding. Synthetic C compilation proves graph and ABI
structure, not numerical readiness. Release support additionally requires
real-weight PyTorch X-ray parity.

## Next certification sequence

1. Bind explicit positions and all YaRN parameters through the circuit and
   kernel map to `yarn_rope_init`, then assert that provider in call-ready IR.
2. Add a deterministic PyTorch boundary fixture for the complete gated-MLA
   layer (the YaRN and FarSkip component fixtures already exist).
3. Download/convert the six real shards with cgroup/storage headroom checks.
4. Run first-token and teacher-forced X-ray attribution against the AMD/Hugging
   Face implementation, followed by coherent generation and performance work.

## Remaining DSL debt

Instella itself is mechanically stitched: the circuit owns ports and persistent
stream bindings, kernel maps own call ABIs, and providers own math. DeepStack
still uses the older branch subgraph declaration plus centralized allocation and
slice-offset compatibility code. The new activation-binding mechanism is the
route for migrating those buffer names, but that migration is intentionally a
separate regression-sensitive patch; this Instella work does not claim it is
already complete.
