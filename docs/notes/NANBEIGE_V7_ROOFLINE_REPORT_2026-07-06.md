# Nanbeige v7 AVX2 Roofline Report - 2026-07-06

## Scope

Model:
`Nanbeige4.1-3B.Q4_K_M.gguf`

Runtime:
v7 native CLI, Intel Core i7-14700T, AVX2/FMA, no AVX-512/AMX, `OMP_NUM_THREADS=1`.

Change tested:
Nanbeige prefill now uses last-only logits through the generic `llama` template path used by
this GGUF. The old full prefill logits path computed
`[prompt_tokens x vocab]` logits, then copied the last row. Native generation consumes only
the last prefill logits row, so last-only logits removes unnecessary footer GEMM work.

## Direct Runtime Scaling

Prompt: `The quick brown fox`
Generation: 32 max tokens, 31 decoded tokens, 14 prompt tokens after template.

| CK threads | Total wall ms | Prefill ms | Decode ms | Decode tok/s |
|---:|---:|---:|---:|---:|
| 1 | 31965.0 | 4491.2 | 27473.8 | 1.1 |
| 8 | 3125.2 | 668.8 | 2456.4 | 12.6 |
| 16 | 2857.0 | 576.4 | 2280.5 | 13.6 |
| 20 | 2774.8 | 579.5 | 2195.4 | 14.1 |
| 24 | 2666.3 | 486.1 | 2180.2 | 14.2 |
| 28 | 2935.1 | 608.0 | 2327.0 | 13.3 |

Best observed total: 24 CK threads. Decode saturates around 16-24 threads; 28 regresses.

Prior full-logits baseline from the same profiling session:

| Build | Total wall ms | Prefill ms | Decode ms | Decode tok/s |
|---|---:|---:|---:|---:|
| full prefill logits | 7790.3 | 3540.5 | 4249.8 | 7.3 |
| last-only prefill logits | 2666.3 | 486.1 | 2180.2 | 14.2 |
| last-only logits, Q4 packed paths gated | 2589.3 | 432.3 | 2157.0 | 14.4 |

Observed end-to-end speedup: about 2.9x on this prompt. Most of that comes from removing
full prefill logits work and using the better 24-thread point.

llama.cpp comparison on the same GGUF and CPU-only AVX2 path:

| Framework | Threads | Prompt tokens | Prompt ms | Prompt tok/s | Decode tokens | Decode ms | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CK v7 | 24 | 11 | 432.3 | 25.4 | 31 | 2157.0 | 14.4 |
| llama.cpp `llama-bench` | 24 | 14 | 135.4 | 103.4 | 31 | 1908.7 | 16.2 |

Decode is now close to llama.cpp on this tiny prompt. Prefill remains materially slower, which
points at quantized small-M GEMM/layout rather than memory bandwidth alone.

## Q4_K Packed Dispatch Probe

The Q4_K hot shape from Nanbeige MLP gate/up was benchmarked directly:
`M=14, N=20992, K=2560`.

| Path | Time ms |
|---|---:|
| serial baseline | 9.891 |
| thread pool baseline | 9.816 |
| packed-M | 7.373 |
| packed-N | 5.730 |
| packed-x8 | 3.263 |

The packed-x8 layout is about 3.0x faster than the baseline for this isolated shape when the
weights are already packed. Lazy packing during the first prefill call is not acceptable for
smoke/pre-commit runs because it adds one-time conversion cost inside inference. Current v7/v8
behavior keeps these packed Q4_K paths opt-in/profile-gated until packing is moved to model
initialization or a persistent sidecar cache.

## CK Profile Hotspots

Best direct run: 24 CK threads.

| Kernel | Time ms | Percent |
|---|---:|---:|
| `gemv_q4_k_q8_k` | 1283.17 | 48.3% |
| `gemv_q6_k_q8_k` | 640.16 | 24.1% |
| `gemm_nt_q4_k_q8_k` | 364.73 | 13.7% |
| `attention_forward_decode_head_major_gqa_flash_f16kv` | 204.96 | 7.7% |
| `swiglu_forward` | 60.15 | 2.3% |
| `gemm_nt_q6_k_q8_k` | 50.46 | 1.9% |

Top op/kernel pairs:

| Mode | Op | Kernel | Time ms | Percent |
|---|---|---|---:|---:|
| decode | `mlp_gate_up` | `gemv_q4_k_q8_k` | 787.43 | 29.6% |
| decode | `mlp_down` | `gemv_q6_k_q8_k` | 324.66 | 12.2% |
| decode | `logits` | `gemv_q6_k_q8_k` | 297.81 | 11.2% |
| prefill | `mlp_gate_up` | `gemm_nt_q4_k_q8_k` | 225.03 | 8.5% |
| decode | `mlp_down` | `gemv_q4_k_q8_k` | 220.37 | 8.3% |
| decode | `attn` | decode attention | 204.96 | 7.7% |

After the last-logits change, logits prefill is no longer a major hotspot. The remaining
roofline target is quantized decode GEMV and prefill MLP GEMM.

## VTune Hotspots

VTune warning:
microarchitecture performance insights were unavailable because the sampling driver is not
enabled. Hotspot attribution was still collected.

| Function | CPU time |
|---|---:|
| `gemv_q4_k_q8_k_parallel_simd` | 24.308 s |
| `gemv_q4_k_q8_k_avx2` | 11.229 s |
| `gemv_q6_k_q8_k_parallel_simd` | 10.051 s |
| `worker_main` | 4.392 s |
| `__intel_avx_rep_memset` | 0.800 s |
| `gemm_nt_q6_k_q8_k` | 0.450 s |
| `attention_forward_decode_head_major_gqa_flash_f16kv` | 0.340 s |
| `ck_threadpool_dispatch_n` | 0.290 s |

## Advisor Roofline

Advisor summary:

| Metric | Value |
|---|---:|
| Program elapsed | 6.571 s |
| CPU threads | 24 |
| Total CPU time | 53.140 s |
| Time in vectorized loops | 44.620 s |
| Time in scalar loops | 7.026 s |
| GFLOPS | 4.23 |
| GINTOPS | 49.21 |
| Mixed ops/s | 53.44 |
| DRAM bandwidth | 39.13 GB/s |
| L1 bandwidth | 6973.12 GB/s |
| L2 bandwidth | 3241.25 GB/s |
| L3 bandwidth | 847.73 GB/s |
| Float arithmetic intensity | 0.086 |
| Int arithmetic intensity | 1.003 |
| Mixed arithmetic intensity | 1.090 |
| ISA used | AVX2, AVX, SSE4, SSE2 |

Vectorization hotspots:

| Routine | Vectorized | Self time |
|---|---:|---:|
| `gemv_q4_k_q8_k_parallel_simd` | yes | 23.727 s |
| `gemv_q4_k_q8_k_avx2` | yes | 10.217 s |
| `gemv_q6_k_q8_k_parallel_simd` | yes | 9.657 s |
| `worker_main` | no | 4.251 s |

Interpretation:
the core compute loops are vectorized, but arithmetic intensity is low and DRAM bandwidth is
about 39 GB/s. The next speed work should reduce quantized GEMV memory traffic and unpack cost,
not focus on attention first.

## Next Targets

1. Prepack Q4_K/Q6_K weights at model load or into a persistent sidecar cache, then route prefill
   GEMM through packed layouts without first-token conversion cost.
2. Repacked Q4_K/Q6_K decode layouts for `gemv_q4_k_q8_k` and `gemv_q6_k_q8_k`.
3. Fused/decode MLP gate-up plus SwiGLU where layout allows it.
4. Q6 logits GEMV, which remains about 11% of direct CK profile time.
5. Keep thread cap around 20-24 on this CPU; 28 threads regressed.

Artifacts:

- `build/nanbeige_v7_roofline/last_threads_24.csv`
- `build/nanbeige_v7_roofline/last_vtune_hotspots.txt`
- `build/nanbeige_v7_roofline/last_advisor_roofline.txt`
- `/home/antshiv/.cache/ck-engine-v7/models/Nanbeige4.1-3B.Q4_K_M/advisor_summary.json`
