# Wall-Clock Performance Baseline - 2026-06-30

This file captures the current C-Kernel-Engine speed position against
llama.cpp using wall-clock timing only.  The machine used for the latest local
checks is the 5th Gen Xeon OpenShift environment.  We do not assume `perf`,
VTune, Advisor, or sudo access for this baseline.

## Revision

| Field | Value |
|---|---|
| Branch base | `nemotron-mamba2-reference` / PR #77 stack |
| Patch scope | Qwen3-VL OCR bridge contract, staged-decode reporting, and wall-clock benchmark harness |
| Expected speed impact | Low; this patch makes bridge/runtime timing measurable; hot-kernel speed work remains separate |
| Timing method | Runtime-reported prompt/decode timing and CK per-op wall-clock CSV |

## Model-Level Speed Snapshot

These rows combine documented direct CK-vs-llama runs and recent local CKE
runtime reports.  Treat them as a baseline ledger, not a single perfectly
controlled benchmark suite.

| Model | Quant | Threads | Source | llama.cpp prompt | CKE prompt | CKE/llama prompt | llama.cpp decode | CKE decode | CKE/llama decode | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5 0.8B | Q4_K_M | 24 | direct comparison log | 96.0 tok/s | 28.7 tok/s | 0.30x | 45.9 tok/s | 28.0 tok/s | 0.61x | Older direct comparison; useful speed reference. |
| Qwen3 0.6B | Q8_0 | local cached v8 | runtime report | n/a | 237.2 tok/s | n/a | n/a | 39.0 tok/s | n/a | CKE-only smoke after correctness work. |
| Qwen3.5 0.8B | Q4_K_M | local cached v8 | runtime report | n/a | 20.2 tok/s | n/a | n/a | 18.1 tok/s | n/a | CKE-only smoke; slower prompt path remains visible. |
| Gemma3 270M | Q5_K_M | local cached v8 | runtime report | n/a | 26.8 tok/s | n/a | n/a | 27.4 tok/s | n/a | CKE-only smoke. |
| Gemma4 E4B | Q4_K_M | 24 | local chat run | n/a | about 10.6 tok/s | n/a | n/a | about 9.1 tok/s | n/a | Coherent long C-code answer; still slow. |
| GLM-4 9B | Q4_K_M | 24 | local chat run | n/a | 16.8 tok/s | n/a | n/a | 8.2 tok/s | n/a | Coherent long C/Python/SQL answer after producer/consumer fix. |
| Nemotron Nano 9B v2 | Q4_K_M | 24 | local chat run | n/a | 9.9 tok/s | n/a | n/a | 7.5 tok/s | n/a | Coherent text after Mamba/state fixes. |

## Prefill Micro-Profile Snapshot

Recent wall-clock prefill profiles show where CKE is still losing most of the
time.  These are CKE-only profile rows from `build/v8_prefill_profile_*.csv`;
they are useful for patch-to-patch comparisons even without hardware counters.

| Model/profile | Threads | Shape | Total | Rate | Main hot path |
|---|---:|---|---:|---:|---|
| Qwen3.5 Q4_K_M, scalar opt-out | 12 | p128 | 958.9 ms | 133.5 tok/s | Q4_K x Q8_K `mlp_gate_up` at 353.5 ms |
| Qwen3.5 Q4_K_M, VNNI default | 12 | p128 | 616.4 ms | 207.6 tok/s | Q4_K x Q8_K `mlp_gate_up` at 165.2 ms |
| Gemma4 E4B Q4_K_M, scalar opt-out | 12 | p64 | 4084.4 ms | 15.7 tok/s | Q4_K x Q8_K `mlp_gate_up` at 1767.6 ms |
| Gemma4 E4B Q4_K_M, VNNI default | 12 | p64 | 3042.5 ms | 21.0 tok/s | Q4_K x Q8_K `mlp_gate_up` at 1108.2 ms |

Current profile files:

```text
build/v8_prefill_profile_qwen35-0.8b-q4_k_m_p128_t12.csv
build/v8_prefill_profile_qwen35-0.8b-q4_k_m_p128_t24.csv
build/v8_prefill_profile_qwen35-0.8b-q4_k_m_p512_t24.csv
build/v8_prefill_profile_gemma4-e4b-q4_k_m_p64_t12.csv
build/v8_prefill_profile_gemma4-e4b-q4_k_m_p512_t24.csv
```

## Current Read

| Area | Status |
|---|---|
| Decode on small Qwen-family models | Often close enough to be usable; sometimes near llama.cpp, but not consistently faster. |
| Qwen3.5 decode | Usually behind llama.cpp, roughly 0.6x in the documented direct comparison. |
| Prompt/prefill | Main gap.  CKE is often 0.1x-0.3x of llama.cpp on larger or hybrid models. |
| Gemma4 text | Correctness is much stronger now, but speed remains weak. |
| Qwen3-VL OCR bridge | Correctness improved; speed still needs a separate controlled CK-vs-llama vision sweep. |

## Aggregated CKE Wall-Clock Hot Spots

These rankings come from summing `time_us` in the p512/t24 CKE prefill profile
CSVs.  They are the best no-sudo proxy for "where time is going" on this
machine.

### Gemma4 E4B Q4_K_M p512/t24

| Rank | Kernel/op | Time | Share |
|---:|---|---:|---:|
| 1 | `gemm_nt_q4_k_q8_k` / `mlp_gate_up` | 6965.2 ms | 31.9% |
| 2 | `geglu_forward_exact` / `geglu` | 4036.4 ms | 18.5% |
| 3 | `gemma4_per_layer_embed_forward` / `gemma4_per_layer_embed` | 3008.0 ms | 13.8% |
| 4 | `attention_forward_causal_head_major_gqa_flash_strided_sliding_gemma4` / `attn_sliding` | 1735.7 ms | 8.0% |
| 5 | `gemm_nt_q4_k_q8_k` / `mlp_down` | 1457.0 ms | 6.7% |

Top three share: about 64.2% of measured prefill time.

### Qwen3.5 0.8B Q4_K_M p512/t24

| Rank | Kernel/op | Time | Share |
|---:|---|---:|---:|
| 1 | `gemm_nt_q4_k_q8_k` / `mlp_gate_up` | 499.5 ms | 21.5% |
| 2 | `ssm_conv1d_forward` / `recurrent_ssm_conv` | 327.2 ms | 14.1% |
| 3 | `attention_forward_causal_head_major_gqa_flash_strided` / `attn` | 288.6 ms | 12.4% |
| 4 | `gemm_nt_q5_k` / `recurrent_qkv_proj` | 174.6 ms | 7.5% |
| 5 | `swiglu_forward` / `silu_mul` | 154.3 ms | 6.6% |

Top three share: about 48.0% of measured prefill time.

## No-Sudo Measurement Plan

Use wall-clock and existing CKE per-op CSVs:

1. Fix the run shape:
   - same model file
   - same prompt tokens or same text prompt
   - same context length
   - same max decode tokens
   - same thread count
   - same temperature, preferably `0.0`
2. Run llama.cpp and CKE with warm cache where possible.
3. Record:
   - prompt tok/s
   - decode tok/s
   - total elapsed time
   - generated token count
   - stop reason
4. For CKE, also save `profile_decode.csv` or prefill profile CSV when available.
5. Compare patch-to-patch before comparing across machines.

Recommended sweep dimensions:

| Dimension | Values |
|---|---|
| Threads | 1, 2, 4, 8, 12, 24, 48 |
| Prompt tokens | 64, 128, 512 |
| Decode tokens | 64, 128, 256 |
| Models | Qwen3.5 0.8B, Gemma4 E4B, GLM-4 9B, Nemotron 9B, Qwen3-VL |

## Next Speed Targets

| Priority | Target | Why |
|---:|---|---|
| 1 | Packed/repacked Q4_K/Q6_K x Q8_K batched prefill GEMM | This is still the largest prompt/prefill gap versus llama.cpp. |
| 2 | Gemma4 `geglu_forward_exact` and per-layer embed/prepare | Gemma4 profiles show large scalar/glue overhead. |
| 3 | Head-major/final logits paths | Large vocab projections dominate decode for some models. |
| 4 | Threadpool/orchestrator tuning | Useful only after hot kernels are not decode-style row loops. |
| 5 | Vision bridge timing split | Needed for Qwen3-VL/Gemma4V OCR: encoder, projector, replay, mixed prefill, decode. |

## Rule For Future Updates

Every performance patch should update this baseline or add a dated sibling file
with:

- git head
- local patch name
- exact command shape
- prompt/decode tok/s
- top three CKE profile rows by wall-clock time
- whether output quality/parity stayed acceptable
