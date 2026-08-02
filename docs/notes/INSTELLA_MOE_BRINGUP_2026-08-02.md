# Instella-MoE V8 bring-up (2026-08-02)

This note records the first fail-closed CKE bring-up for AMD's
`amd/Instella-MoE-16B-A3B-Think`. It is a tensor-contract and graph-semantics
patch, not an end-to-end support claim.

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

## Deliberate fail-closed boundary

Full conversion/execution remains blocked until all three items below exist:

- `instella_moe.json`: an executable V8 circuit, not a renamed Kimi circuit.
- Persistent two-stream FarSkip dataflow and lifetime planning.
- A certified partial-interleaved YaRN MLA positional contract/provider.

The contract inspector therefore returns `bringup_required` and names those
missing capabilities. This prevents a superficially successful conversion from
producing numerically invalid text.

## Next certification sequence

1. Add tiny deterministic PyTorch fixtures for gated MLA, interleaved YaRN, and
   two consecutive FarSkip layers.
2. Implement the two-stream slots in the DSL/planner and the exact Instella
   circuit.
3. Lower both prefill and decode and compare every new boundary with the
   fixtures.
4. Download/convert the six real shards only after the synthetic circuit is
   complete.
5. Run first-token and teacher-forced X-ray attribution against the AMD/Hugging
   Face implementation, followed by coherent generation and performance work.

