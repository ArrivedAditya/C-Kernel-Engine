# Qwen3-VL OCR Q4/Q8 Prefill Speed Notes - 2026-07-02

This note records the Xeon/OpenShift Qwen3-VL OCR speed investigation so the same
ideas can be retested on CPUs with different cache/core ratios, especially Ryzen
or newer Xeon parts with larger effective cache per active core.

## Baseline Context

Target workload:

```text
Model: Qwen3-VL 8B Instruct GGUF Q4_K_M + Q8_0 mmproj
Image: SDPR form, 1400 px PPM
Prompt: Extract visible form fields as compact JSON.
Image tokens: 1024
Text+visual mixed prefill: 1028 tokens
Threads: 20 for the current best Xeon run
```

Historical steady-state wall-clock on this OpenShift Xeon node:

| Stage | Approx Time | Notes |
|---|---:|---|
| Original CK OCR path | ~281 s | Before staged bridge and encoder/kernel fixes. |
| After staged bridge + attention/FP16 work | ~101 s | Encoder still dominated by Q8/attention work. |
| After raw `gemm_nt_q8_0` CK threadpool | ~78.3 s | Encoder dropped to ~34.6 s. |
| After Q4 x16 SwiGLU output-loop cleanup | ~76.7 s | Mixed prefill dropped modestly. |
| llama.cpp OCR report baseline | ~55.8 s/sample | 40-sample SDPR report; likely noisy and not a theoretical roofline. |

The current CK gap is specific and measurable: decoder mixed prefill remains
largely Q4_K/Q6_K x Q8_K projection work, not bridge/compiler overhead.

## Current Best CK Split

Latest representative real-image run after Q8 threadpool and Q4 x16 output-loop cleanup:

```text
steady_state_ms:        76695
encoder_execute_ms:     34171
mixed_prefill_ms:       42499
decode_1_token_ms:         24
```

Decoder mixed-prefill top ops:

| Op | Time |
|---|---:|
| `gemm_nt_q4_k_q8_k_gateup_swiglu_x16` / `mlp_gate_up_swiglu` | ~20.9 s |
| `mlp_down` total | ~9.1 s |
| `out_proj` | ~3.6 s |
| `q_proj` | ~3.6 s |
| attention | ~1.7 s |

## Q4_K Gate/Up Microbench Results

Representative shape for the hot fused gate/up path:

```text
M = 1028
D = 12288
K = 4096
weights ~= 54 MiB Q4_K gate+up matrix
output ~= 48 MiB FP32
intermediate gate/up scratch equivalent ~= 96 MiB if unfused
```

The packed x16 benchmark uses the real CK threadpool and compares against the
existing unfused reference path.

Useful command:

```bash
CK_NUM_THREADS=20 LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH \
  build/bench_q4k_gateup_swiglu \
  --M 1028 --D 12288 --K 4096 --tile-m 8 --mode x16 --warmup 1 --iters 3
```

Measured Xeon/OpenShift examples before the final output-loop cleanup were noisy:

| Threads | x16 Time | Observation |
|---:|---:|---|
| 12 | ~535 ms | Underuses available physical cores for this shape. |
| 16 | ~444 ms | Strong. |
| 20 | ~428 ms | Best observed point in that sweep. |
| 24 | ~574 ms | Regressed in that noisy sweep. |

After the vectorization-friendly final SwiGLU loop, a clean sequential CK-threadpool
run on the same real OCR shape produced near-tied results:

| Threads | x16 Time | Parity |
|---:|---:|---|
| 16 | ~352.4 ms | rel diff ~5.7e-7, cosine 1.0 |
| 20 | ~351.2 ms | rel diff ~5.7e-7, cosine 1.0 |
| 24 | ~352.7 ms | rel diff ~5.7e-7, cosine 1.0 |

The updated result is better interpreted as: use physical cores, avoid SMT for this
kernel, and do not overfit the cap to one noisy OpenShift run. The limiter is still
core-to-cache/core-to-memory ratio, not just visible thread count.

## Experiments Kept vs Rejected

Kept:

- Raw `gemm_nt_q8_0` CK-threadpool dispatch for encoder branch/projector GEMMs.
- Q4_K x16 fused gate/up path as an opt-in path.
- Q4_K x16 final SwiGLU output loop made vectorization-friendly:
  - same math: `gate / (1 + expf(-gate)) * up`
  - removes scalar helper call from the hot lane loop
  - real-shape standalone timing improved into the ~351-353 ms region on 16-24 physical threads
  - previous model-level run showed a modest gain of about 1.6 s on the tested SDPR image

Rejected on this Xeon, but should be retested on larger-cache CPUs:

- Dual gate/up accumulator helper that shares one Q8 activation traversal while
  updating both gate and up accumulators.
- It was numerically clean, but slower on this Xeon:
  - vectorized-output-loop x16 best: ~428 ms
  - dual-accumulator x16: ~442 ms
- Likely reason: register/instruction pressure outweighed saved Q8 traversal.
- Retest on Ryzen/X3D or other CPUs with different cache hierarchy and register
  scheduling behavior before permanently discarding the idea.

## Generic Q4 Projection x16 Opt-In Check

A generic packed-meta x16 projection dispatch is available behind:

```bash
CK_ENABLE_Q4K_PACKED_META_X16_PREFILL=1
```

It is intentionally opt-in. On the generated fast OCR clean-text asset, enabling
it together with gate/up x16 did not improve mixed prefill:

| Run | Encoder | Mixed Prefill | Decode | Text |
|---|---:|---:|---:|---|
| gate/up x16 only | 2740.7 ms | 4279.4 ms | 960.2 ms | `CK OCR TEST\nTOTAL 42` |
| gate/up x16 + generic projection x16 | 2729.1 ms | 4308.5 ms | 878.6 ms | `CK OCR TEST\nTOTAL 42` |

Standalone projection microbenchmarks still show small wins for some q/out-style
shapes, so keep the path for CPU-family sweeps, but do not enable it by default.
The next default-quality speed work is a better projection/down microkernel or
conversion-time prepacking, not simply routing every Q4 projection through x16.

## 2026-07-03 VNNI Horizontal-Sum Fix

A VTune software-hotspots run on the packed x16 gate/up path showed the hot work
was not SiLU/`expf`; it was the Q4_K/Q8_K VNNI dot/accumulation loop. The shared
`hsum256_epi32` helper in `gemm_kernels_q4k_q8k_vnni.c` used two
`_mm_hadd_epi32` reductions. Switching it to the standalone research harness'
shuffle/add reduction closed the gap between the research x16 path and the actual
shared-library x16 path.

Focused gate/up benchmark, real OCR shape (`M=1028, D=12288, K=4096`, 20 threads):

| Variant | Time | Parity |
|---|---:|---|
| library x16 before hsum fix | ~452 ms | rel diff ~5.7e-7, cosine 1.0 |
| library x16 after hsum fix | ~348 ms | rel diff ~5.7e-7, cosine 1.0 |

Large-prefix OCR A/B on generated clean-text image (`image_tokens=1024`, gate/up
x16 enabled, generic projection x16 disabled):

| Run | Encoder | Mixed Prefill | Decode | Top improvement |
|---|---:|---:|---:|---|
| before hsum fix | 61444.7 ms | 42908.0 ms | 1239.9 ms | `mlp_gate_up_swiglu` 21086.0 ms |
| after hsum fix | 60881.0 ms | 39578.6 ms | 1324.1 ms | `mlp_gate_up_swiglu` 17390.7 ms |

Net: mixed prefill improved by about 3.3 s on this large-prefix OCR check. This
reduces the dominant Q4 gate/up cost, but CK is still slower than llama.cpp due
to remaining encoder cost plus Q4/Q6 down/projection work.

## Retest Matrix for Other CPUs

Run this matrix on Ryzen, AVX2-only i7, and any larger-cache Xeon/EPYC host:

| Variable | Values |
|---|---|
| Threads | physical cores only, then SMT separately |
| Tile M | 1, 2, 4, 8, 16 if supported |
| Scheduler | CK threadpool first; OpenMP only as research comparison |
| Shape | `M=128/512/1028`, `D=12288`, `K=4096` |
| Kernel variants | unfused, x16, x16 output-loop cleanup, dual accumulator |

Record:

- wall time median and min over several iterations
- relative diff and cosine against reference
- physical-core count and SMT status
- L1/L2/L3 sizes and cache per active worker
- NUMA placement, if applicable

## Interpretation

llama.cpp is faster on the current OCR report, but it is probably not at the
hardware theoretical peak either. It is ahead because its mature layouts,
packing, and executor scheduling keep the hot quantized projection loops closer
to the practical cache/memory roofline.

CK is now close enough that the next improvements should be kernel-specific:

1. Better Q4_K/Q6_K packed prefill kernels for mixed prefill.
2. Load/conversion-time prepacking instead of lazy runtime packing.
3. Cache-derived active-thread defaults with env overrides.
4. Separate results by CPU family; do not tune only for this OpenShift Xeon.
