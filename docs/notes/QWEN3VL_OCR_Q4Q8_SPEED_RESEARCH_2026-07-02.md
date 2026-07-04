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

Encoder activation-policy follow-up: forcing Qwen3-VL vision `branch_fc1` and
`branch_fc2` from fp32 activation to Q8_0 activation made the encoder much
faster, but it broke OCR output on the clean-text image. Split tests showed both
`branch_fc1=q8_0` and `branch_fc2=q8_0` individually degraded the answer. Keep
the branch FC path fp32-activation until a parity-clean fp32 x Q8_0 kernel or
more precise activation policy is available.

## 2026-07-03 Opt-In FP32 x Q8_0 M4N4 Encoder GEMM

Because Qwen3-VL branch FC layers are not correctness-clean with Q8_0
activations, the safe speed path is a better fp32-activation x Q8_0-weight GEMM.
An opt-in AVX512 M4xN4 kernel behind `CK_ENABLE_Q80_FP32_M4N4=1` keeps four FP32
activation rows and four Q8_0 weight rows live together, reducing repeated
weight dequant/reduction work while preserving the fp32 activation contract.

Focused benchmark (`M=1028, N=4096, K=4096`):

| Variant | Time | Parity |
|---|---:|---|
| existing row-GEMV fp32 x Q8_0 | ~3084 ms | reference |
| opt-in M4N4 fp32 x Q8_0 | ~366 ms | max diff 0, cosine 1.0 |

Large-prefix Qwen3-VL OCR with `CK_ENABLE_Q80_FP32_M4N4=1` plus gate/up x16:

| Run | Encoder | Mixed Prefill | Decode | Output |
|---|---:|---:|---:|---|
| hsum + gate/up x16 baseline | 60881.0 ms | 39578.6 ms | 1324.1 ms | `CK OCR TEST\nTOTAL 42` |
| + fp32 x Q8_0 M4N4 | 37419.3 ms | 38479.5 ms | 1299.6 ms | `CK OCR TEST\nTOTAL 42` |

Net: the heavy 1024-visual-token clean OCR check improved from about 101.8 s to
about 77.2 s steady-state while preserving the OCR answer. The new encoder top
bottleneck is visual attention (`attention_forward_full_head_major_gqa_flash_strided`,
about 15.0 s), followed by Q8_0/Q8_0 MLP/QKV and remaining fp32 branch FC work.


## 2026-07-03 Q4_K Gate/Up x16 Chunk4 Reuse

After the fp32 x Q8_0 M4N4 encoder fix, the largest decoder-side mixed-prefill
projection remained `mlp_gate_up_swiglu`. The x16 packed Q4_K path still loaded
the same Q8_K activation block once per output lane. An opt-in chunked reuse path
behind `CK_Q4K_X16_CHUNK4=1` processes up to four x16 lanes per activation load,
so the Q8_K row stays hot while several Q4_K output lanes consume it.

Focused gate/up benchmark (`M=1028, D=12288, K=4096`, 20 CK threads):

| Variant | Time | Parity |
|---|---:|---|
| x16 baseline | ~403 ms | reference |
| x16 chunk4 | ~274 ms | rel diff ~1.0e-6, cosine 1.0 |

Large-prefix Qwen3-VL OCR with `CK_ENABLE_Q80_FP32_M4N4=1`,
`CK_ENABLE_Q4K_GATEUP_SWIGLU_X16=1`, and `CK_Q4K_X16_CHUNK4=1`:

| Run | Encoder | Mixed Prefill | Decode | Steady | Output |
|---|---:|---:|---:|---:|---|
| M4N4 + gate/up x16 | 37419.3 ms | 38479.5 ms | 1299.6 ms | 77198.3 ms | `CK OCR TEST\nTOTAL 42` |
| + Q4 chunk4 reuse | 37474.0 ms | 35169.9 ms | 1285.7 ms | 73929.6 ms | `CK OCR TEST\nTOTAL 42` |

Net: chunk4 is a correctness-clean opt-in improvement for this workload, reducing
mixed prefill by about 3.3 s and the steady OCR run by about 4.4%. The next
measured bottleneck moves to Qwen3-VL encoder full attention
(`attention_forward_full_head_major_gqa_flash_strided`, about 15.0 s). A quick
attention-thread-cap sweep did not produce a stable scheduler-only win, and the
existing tiled/ggml full-attention path is much slower at the real encoder shape
(`T=4232, H=16, D=72`).


## 2026-07-03 AVX512 Full-Attention QBlock4 Reuse

After Q4 chunk4 reuse, the largest encoder-side cost was full visual attention
at the real Qwen3-VL encoder shape (`T=4232, H=16, D=72`). The existing full
attention path computes one query row at a time, so each nearby query rereads the
same K/V rows. An opt-in AVX512 path behind `CK_ATTENTION_QBLOCK4=1` processes
four query rows together for non-causal full attention with `head_dim=72`,
keeping K/V rows hotter across the four query accumulators. The path is guarded
by shape and ISA checks and falls back to the existing kernel otherwise.

Focused exact-shape attention benchmark (`T=4232, H=16, KV=16, D=72`, 20 CK
threads):

| Variant | Time | Parity |
|---|---:|---|
| existing full attention | ~622.9 ms | reference |
| opt-in qblock4 | ~598.0 ms | max diff 0, cosine 1.0 |

Large-prefix Qwen3-VL OCR with `CK_ENABLE_Q80_FP32_M4N4=1`,
`CK_ENABLE_Q4K_GATEUP_SWIGLU_X16=1`, `CK_Q4K_X16_CHUNK4=1`, and
`CK_ATTENTION_QBLOCK4=1`:

| Run | Encoder | Mixed Prefill | Decode | Steady | Output |
|---|---:|---:|---:|---:|---|
| chunk4 baseline | 37908.6 ms | 34832.3 ms | 1370.4 ms | 74111.4 ms | `CK OCR TEST\nTOTAL 42` |
| + attention qblock4 | 33803.1 ms | 34260.0 ms | 1328.2 ms | 69391.3 ms | `CK OCR TEST\nTOTAL 42` |

Net: qblock4 is a correctness-clean opt-in model-level win on this Xeon/OpenShift
node, saving about 4.7 s on the clean OCR workload. The next measured bottleneck
moves back to decoder mixed prefill, especially `mlp_gate_up_swiglu` through
`gemm_nt_q4_k_q8_k_gateup_swiglu_x16` at about 12.2 s in the latest profile.

## 2026-07-04 Speed Profile Flag Collapse

The tuned Xeon/OpenShift OCR stack can now be enabled with one high-level speed
profile instead of listing every low-level kernel flag:

```bash
CK_SPEED_PROFILE=qwen3vl_ocr_xeon_avx512
```

Equivalent short alias:

```bash
CK_QWEN3VL_OCR_FAST=1
```

`CK_PROFILE` is intentionally not used here because it already means
profiling/timing instrumentation in CK scripts and Make targets. Speed policy is
kept separate as `CK_SPEED_PROFILE`, so OCR tuning can coexist with `CK_PROFILE=1`
or `--profile` runs.

The speed profile defaults these settings when they are not explicitly set:

| Setting | Profile default |
|---|---|
| `CK_ENABLE_Q80_FP32_M4N4` | `1` |
| `CK_ENABLE_Q4K_GATEUP_SWIGLU_X16` | `1` |
| `CK_Q4K_GATEUP_SWIGLU_X16_THREAD_CAP` | `20` |
| `CK_Q4K_X16_CHUNK4` | `1` |
| `CK_ATTENTION_QBLOCK4` | `1` |
| `CK_NUM_THREADS` | `20` in `ck_run_v8.py` if unset |
| `OMP_NUM_THREADS` | `1` in `ck_run_v8.py` if unset |

Explicit low-level env values still override the profile, so individual kernels
can be disabled for A/B testing, for example `CK_ATTENTION_QBLOCK4=0`.

Validation on this Xeon node after rebasing onto `accec0fa`:

| Check | Result |
|---|---|
| Native AVX512 build | passed |
| AVX2-only build with `AVX_FLAGS='-mavx2 -mfma'` | passed; AVX512 qblock4 compiles out |
| Q4 gate/up profile-only microbench | rel diff ~1.0e-6, cosine 1.0 |
| Attention profile-only microbench | max diff 0, cosine 1.0 |
| Rebased `CK_SPEED_PROFILE` clean OCR E2E | `CK OCR TEST\nTOTAL 42`; encoder 33865.4 ms, mixed 33591.0 ms, gen 1461.5 ms |

Clean OCR E2E default-vs-profile check, same image/prompt/settings and both
`--force-compile`:

| Stage | Default ms | Speed profile ms | Speedup | Delta ms |
|---|---:|---:|---:|---:|
| Encoder execute | 60300.1 | 34328.0 | 1.76x | 25972.1 |
| Mixed prefill | 48907.7 | 33295.4 | 1.47x | 15612.3 |
| Generation | 1395.6 | 1331.8 | 1.05x | 63.8 |
| Steady total | 110603.4 | 68955.3 | 1.60x | 41648.2 |

Correctness output for both runs:

```text
CK OCR TEST
TOTAL 42
```

This is still a profile, not a universal default. Promote it to automatic behavior
only after CPU-family sweeps confirm that the same choices are neutral or
positive on AVX2-only laptops, Ryzen/EPYC, and other Xeon cache/core ratios.

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

## 2026-07-04 Q4_K x8 Large-M Projection/Down Profile

The mixed-prefill profile after the encoder and gate/up fixes showed the next
Q4-heavy costs were ordinary projection/down GEMMs rather than the fused
`mlp_gate_up_swiglu` path:

| Op | Baseline Time | Notes |
|---|---:|---|
| `mlp_down` / `gemm_nt_q4_k_q8_k` | ~5.3 s | Q4 projection/down family |
| `out_proj` / `gemm_nt_q4_k_q8_k` | ~3.8 s | decoder output projection |
| `q_proj` / `gemm_nt_q4_k_q8_k` | ~3.3 s | decoder Q projection |

The dispatch matrix benchmark was extended to include the generic packed-meta
x16 path and Qwen3-VL OCR-shaped rows. At `M=1028`, `N=4096`, `K=4096/11008`,
the existing x8/x16 packed paths were faster than the current threadpool
fallback in isolation:

| Shape | Pool | x8 | x16 reuse | Best Read |
|---|---:|---:|---:|---|
| `qwen3vl_proj` (`1028x4096x4096`) | ~104.5 ms | ~78.5 ms | ~71.3 ms | x16 best, x8 still useful |
| `qwen3vl_down` (`1028x4096x11008`) | ~289.4 ms | ~173.5 ms | ~179.8 ms | x8 best |

A full OCR A/B then raised only the x8 max-M gate for the Qwen3-VL OCR speed
profile (`CK_Q4K_PACKED_META_X8_MAX_M=2048`). The answer stayed correct
(`CK OCR TEST\nTOTAL 42`) and mixed prefill improved:

| Run | Encoder | Mixed Prefill | Decode | Output |
|---|---:|---:|---:|---|
| profile baseline | 34910.5 ms | 33006.0 ms | 1329.1 ms | `CK OCR TEST\nTOTAL 42` |
| + x8 max-M 2048 | 38408.5 ms | 31648.5 ms | 1390.2 ms | `CK OCR TEST\nTOTAL 42` |

Kernel/op deltas from that run:

| Kernel/Op | Delta |
|---|---:|
| `gemm_nt_q4_k_q8_k` / `out_proj` | -863 ms |
| `gemm_nt_q4_k_q8_k` / `mlp_down` | -593 ms |
| `gemm_nt_q4_k_q8_k` / `q_proj` | -574 ms |
| `gemm_nt_q4_k_q8_k` / `k_proj` | -117 ms |
| `gemm_nt_q4_k_q8_k` / `v_proj` | -72 ms |
| `gemm_nt_q4_k_q8_k_gateup_swiglu_x16` / `mlp_gate_up_swiglu` | +820 ms |

Net: this is a real mixed-prefill win despite gate/up noise in the full run. The
setting is profile-scoped, not global: normal Q4_K x8 dispatch remains capped at
`M<=64`, while `CK_SPEED_PROFILE=qwen3vl_ocr_*` uses `M<=2048` unless explicitly
overridden.

Final wrapper verification, with only `CK_SPEED_PROFILE=qwen3vl_ocr_xeon_avx512`
set in the benchmark command after applying profile defaults consistently:

| Run | Encoder | Mixed Prefill | Decode | Output |
|---|---:|---:|---:|---|
| profile baseline | 34910.5 ms | 33006.0 ms | 1329.1 ms | `CK OCR TEST\nTOTAL 42` |
| profile + x8 max-M default | 38107.6 ms | 30695.1 ms | 1390.6 ms | `CK OCR TEST\nTOTAL 42` |

Final mixed-prefill delta: about -2.31 s on this generated OCR check. The main
wins were Q4 `mlp_down` (-1.02 s), `out_proj` (-0.91 s), and `q_proj` (-0.72 s).

## 2026-07-04 Attention Thread-Cap Sweep

After exposing the nested encoder profile in the OCR benchmark JSON, the fresh
Qwen3-VL OCR profile showed encoder full attention as the largest single
remaining encoder op. The speed profile already enables the AVX512 qblock4 path
with `CK_ATTENTION_QBLOCK4=1`, so the next low-risk A/B was active-thread
capping for attention only.

Generated clean OCR image, `CK_SPEED_PROFILE=qwen3vl_ocr_xeon_avx512`, 20 CK
threads, 1024 image tokens:

| Attention Cap | Encoder | Mixed Prefill | Encoder Attention | Decoder Attention | Output |
|---:|---:|---:|---:|---:|---|
| default/profile | 37992.6 ms | 31277.2 ms | 12748.6 ms | 1732.3 ms | `CK OCR TEST\nTOTAL 42` |
| 16 | 37900.0 ms | 30919.4 ms | 12516.7 ms | 2185.8 ms | `CK OCR TEST\nTOTAL 42` |
| 12 | 39307.2 ms | 31345.7 ms | 13496.2 ms | 2605.7 ms | `CK OCR TEST\nTOTAL 42` |

Cap 16 is the best measured point in this small sweep. It is now part of the
Qwen3-VL OCR speed profile defaults as `CK_ATTENTION_THREAD_CAP=16`, while
normal runs and explicit user env settings remain unchanged. This is a modest
contention/cache tuning win, not a replacement for a better encoder attention
microkernel.
