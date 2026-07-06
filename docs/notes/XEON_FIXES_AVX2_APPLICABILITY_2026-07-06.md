# Xeon Qwen3-VL Fixes vs AVX2 Applicability - 2026-07-06

## Scope

This note maps the recent Qwen3-VL/Xeon performance fixes in local `main` to
the AVX2 path. There is no local branch literally named `xeon`; the relevant
history is the July 2026 `perf(v8)` Qwen3-VL OCR speed stack and the local
VTune/Advisor artifacts under `build/`.

## What The Xeon Stack Fixed

The speed work is mostly performance/profile policy, not the Qwen3-VL long-token
accuracy issue.

Relevant commits:

| Commit | Area | Result from local notes |
|---|---|---|
| `884e90af` | Thread large F16 vision GEMM | Moves vision FP16 GEMM to CK threadpool for large shapes. |
| `484e7f48` | Q4_K/Q8_K VNNI hsum | Gate/up x16 focused benchmark improved from about 452 ms to 348 ms. |
| `a2e40265` | FP32 activation x Q8_0 encoder GEMM | Encoder-heavy OCR dropped from about 101.8 s to 77.2 s with OCR answer preserved. |
| `6014ddea` | Q4 x16 chunk4 reuse | Mixed prefill improved by about 3.3 s on large OCR. |
| `03d84f40` | `CK_SPEED_PROFILE=qwen3vl_ocr_xeon_avx512` | Collapsed per-kernel flags into one profile; default-vs-profile improved about 1.60x on Xeon. |
| `e41140d1` | Q4 x8 profile gate widened | Mixed prefill improved by about 2.3 s on generated OCR. |
| `2102a9ef` | Attention thread cap | Profile-scoped attention cap set to 16. |
| `945d8f49` | AVX512 qblock8 attention + Q4 x8 M-reuse | Focused encoder attention improved about 1.38x; OCR stayed correct on clean sample. |
| `9cfd80ed` | Q6_K thread cap | Q6 row-split benchmark improved from 161.41 ms to 142.18 ms. |
| `b34e7438` | Qblock fast exp | Profile-scoped fast exp inside AVX512 qblock attention only. |

Best logged Xeon OCR profile result in the note is about `61.6 s` steady-state
for the large clean OCR check, with expected output:

```text
CK OCR TEST
TOTAL 42
```

The older llama.cpp SDPR reference in the notes is about `55.8 s/sample`, so the
logged Xeon profile was roughly 10 percent behind that reference, not yet a
clear win.

## Already Useful On AVX2

These changes are architecture-neutral or have AVX2 code paths already present:

| Fix | AVX2 status | Evidence |
|---|---|---|
| `CK_SPEED_PROFILE` plumbing | Applies directly | `include/ck_speed_profiles.h` enables the profile aliases. |
| Thread defaults/caps | Applies directly | `ck_run_v8.py`, OCR bench wrappers, attention cap, and Q6 cap are not ISA-specific. |
| Q4 packed-meta x8/x8-M-reuse dispatch policy | Likely applies | `ck_parallel_prefill_v8.c` routes profile Q4 large-prefill shapes through x8/x8reuse gates. The underlying file has AVX2 guarded code for packed-meta dot paths. |
| Q6_K profile thread cap | Applies directly | Shape-specific cap for `M>=512,N=4096,K=12288` is independent of AVX512. |
| Focused benchmarks/perf pipeline | Applies directly | `bench_qwen3vl_encoder_attention`, `bench_q4k_dispatch_matrix`, and `qwen3vl_ocr_perf_pipeline.py` all build/run on AVX2 fallback. |
| Local F16C/AVX F16 GEMM adaptation | Applies to AVX2 | Current workspace adds F16C conversion and AVX/FMA dot loops in `gemm_kernels_f16.c`. This is the clearest AVX2-port of a Xeon-side win. |

## Xeon-Only Or Mostly Xeon-Only Today

These compile out or lose their main acceleration on AVX2:

| Fix | Why it does not directly transfer |
|---|---|
| FP32 x Q8_0 M4N4 encoder GEMM | Current fast body is guarded by `#if defined(__AVX512F__)`; AVX2 falls back to row-loop GEMV. |
| Qblock4/qblock8 encoder attention | Current qblock bodies are guarded by `#if defined(__AVX512F__)`; AVX2 uses the generic threaded full-grid path and AVX2 per-query attention. |
| Qblock fast exp | Only used inside the AVX512 qblock paths. |
| VNNI horizontal-sum win | The exact `_mm256_dpbusd_epi32` path requires AVX512-VNNI/VL. AVX2 has to emulate int8 dot products with unpack/madd pairs. |
| Q4 x16 chunk4 VNNI reuse | The chunk4 body is AVX512-VNNI-only and falls back to the existing m-reuse function otherwise. |

## AVX2 Port Candidates

Priority for AVX2 should be:

1. Keep and validate the local F16C/AVX F16 GEMM patch.
   This attacks the `ck_gemm_f16_input_fp16_work` hotspot visible in the current
   AVX2 VTune output.

2. Add an AVX2 version of FP32 x Q8_0 M4N4 or M4N2.
   The Xeon M4N4 result is the largest encoder-side single win, but the current
   implementation is AVX512-only. An AVX2 version should reuse 4 activation rows
   and 2 or 4 Q8 rows, using 8-wide float lanes.

3. Add AVX2 qblock full-attention for Qwen3-VL encoder shape.
   The shape is fixed and friendly: full attention, `head_dim=72`, aligned head
   dim around 80. A qblock4 AVX2 path is more realistic than qblock8 because ymm
   registers are fewer and narrower.

4. Port Q4_K/Q8_K packed-meta reuse to true AVX2 dot kernels.
   Use the existing packed layout and shape gates, but replace VNNI dpbusd with
   AVX2 unpack/subtract/maddubs/madd or a parity-clean scalar-assisted variant.

5. Keep thread caps as tunable, not universal defaults.
   The Xeon notes repeatedly show 16/20/24 thread results shifting by kernel and
   machine. AVX2 should sweep `1,8,16,20,24` before accepting a default.

## Accuracy Caveat

The inspected Xeon commits are speed/profile commits. They do not explain the
Qwen3-VL long-token drift we saw locally.

Current AVX2 diagnostic state from the parity work:

| Check | Current read |
|---|---|
| Normal Qwen decoder | Good parity in the available v8 checks. |
| Qwen3-VL multimodal decoder | First visible greedy token mismatch around generated step 23, but numeric drift exists at step 0. |
| Encoder/bridge vs llama.cpp mtmd | Prefix shape/token count matches; values are close on OCR-sized images but not exact. |
| First concrete decoder dump failure | Layer-0 attention/residual boundary before MLP (`ffn_inp`) in the wide step-23 dump. |

So for AVX2, do not treat the Xeon speed stack as an accuracy fix. Keep the
parity harness enabled while porting speed kernels.

## Recommended AVX2 Validation Command

Use the profile for measurement, but remember AVX512-only pieces compile out:

```bash
CK_SPEED_PROFILE=qwen3vl_ocr_xeon_avx512 \
CK_NUM_THREADS=20 \
OMP_NUM_THREADS=1 \
.venv/bin/python benchmarks/qwen3vl_ocr_perf_pipeline.py \
  --emit-vtune \
  --json-out build/qwen3vl_ocr_perf_pipeline_avx2.json \
  --md-out build/qwen3vl_ocr_perf_pipeline_avx2.md \
  --raw-json-out build/qwen3vl_ocr_perf_raw_avx2.json
```

For focused AVX2 port work, start with:

```bash
CK_NUM_THREADS=20 LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH \
  build/bench_q8_0_fp32_gemm --M 1028 --N 4096 --K 4096 --warmup 1 --iters 3

CK_NUM_THREADS=20 LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH \
  build/bench_qwen3vl_encoder_attention --threads 20 --warmup 1 --iters 3

CK_NUM_THREADS=20 LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH \
  build/bench_q4k_dispatch_matrix
```

## Bottom Line

The Xeon agent fixed a large part of Qwen3-VL OCR speed by combining profile
policy, thread caps, packed Q4/Q6 prefill routing, AVX512 Q8 encoder GEMM, and
AVX512 qblock attention. The profile and dispatch policy are already in `main`
and are safe to use for AVX2 measurement. The largest remaining AVX2 work is not
applying commits mechanically; it is writing AVX2 equivalents for the AVX512
encoder GEMM and qblock-attention kernels, while keeping the long-token parity
harness as a gate.
