# Qwen3.5 CK vs llama.cpp parity worklog

This log records the empirical path used to chase Qwen3.5 long-generation
drift on the Xeon AVX-512 host. Keep it current when changing kernels or
lowering policy so later runs can reproduce the same evidence.

## Repro command

Use the local llama.cpp build and the cached Qwen3.5 run directory:

```bash
/usr/bin/env \
  LD_LIBRARY_PATH=/opt/app-root/src/Software/llama.cpp/build/bin:/opt/app-root/src/Programs/intel/oneapi_2025/compiler/2025.3/lib:/opt/app-root/src/Programs/intel/oneapi_2025/2025.3/lib \
  .venv/bin/python version/v8/scripts/compare_multitoken_logits_v8.py \
  --model-dir /opt/app-root/src/.cache/ck-engine-v8/models/unsloth--Qwen3.5-0.8B-GGUF \
  --gguf /opt/app-root/src/.cache/ck-engine-v8/models/unsloth--Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --tokens 248045,846,198,32002,5673,741,198,33963,728,449,3010,314,351,11,5526,321,9834,1970,30,248046,198,248045,74455,198,248068,271,248069,271,8160,369,264 \
  --max-new-tokens 40 --ctx-len 1034 --top-k 16 --threads 24 \
  --append-on-divergence stop --summary \
  --json-out /tmp/qwen35_sourcefix_step40.json
```

After rebuilding `build/libckernel_engine.so`, copy it into the run directory:

```bash
cp build/libckernel_engine.so \
  /opt/app-root/src/.cache/ck-engine-v8/models/unsloth--Qwen3.5-0.8B-GGUF/libckernel_engine.so
```

## Findings

- Baseline after the earlier tokenizer/SwiGLU fixes still failed at step 0:
  `ck_next=4434`, `llama_next=14542`, cosine about `0.99918`, top-k overlap
  `15/16`.
- Tightened Q8_K activation quantization to llama-style nearest-int rounding.
  The dedicated Q8_K quantizer parity test now compares dispatch/SSE/AVX/AVX2/
  AVX512 against scalar reference.
- Changed Q4_K/Q8_K and Q6_K/Q8_K scalar references to llama-style eight-lane
  int32 accumulation before float reduction. Q4 and Q6 unit parity tests pass.
- Routed Q4_K/Q8_K VNNI through the reference path for now. This favors
  correctness while the production VNNI accumulation order is still under
  investigation.
- Q5_K generic-dot and shared-Q8 probes did not fix Qwen3.5 step-0 drift.
- Strongest diagnostic probe before source changes was:
  `CK_DEBUG_Q4K_Q8_CONTRACT=1 CK_V8_DEBUG_MLP_GATE_UP_FP32_LAYER=0`, which
  passed 40 tokens on the old generated model.
- Encoding that behavior in source required two pieces:
  - Qwen3.5 `recurrent_gate_proj` defaults to Q8_K activation.
  - Qwen3.5 layer-0 `mlp_gate_up` defaults to FP32 adapter.
- A first source attempt changed `recurrent_gate_proj` to Q8_K without inserting
  the needed quantize op; that produced hard corruption (`cosine=0.600999`).
- The lowerer now scans a full normed section for Q8 consumers, so recurrent
  blocks insert `quantize_input_0` after `attn_norm` even when the immediate
  next op is FP32 QKV.
- Current clean source-generated model moves the first divergence from step 0
  to step 2:
  `ck_next=42726`, `llama_next=8755`, cosine about `0.999169`, RMSE about
  `0.136147`, top-k overlap `16/16`.
- Rejected probes after the step-2 baseline:
  - `CK_DEBUG_Q5K_FP32_FALLBACK=1`: regressed to step 0
    (`ck_next=4434`, `llama_next=14542`, RMSE about `0.154663`).
  - `CK_DEBUG_Q5K_GENERIC_DOT=1`: regressed to step 0
    (RMSE about `0.158204`).
  - `CK_DEBUG_Q5K_SHARED_Q8_QUANT=1`: unchanged at step 2.
  - `CK_V7_DEBUG_MLP_DOWN_FP32_LAYER=0`: regressed to step 0
    (RMSE about `0.183399`).
  - `CK_V7_DEBUG_OUTPROJ_FP32_LAYER=0` and
    `CK_V8_DEBUG_ATTN_PROJ_FP32_LAYER=0`: unchanged at step 2.
  - `CK_DELTANET_FORCE_REF=1` / `CK_STRICT_PARITY=1`: regressed to step 0,
    so the current AVX512 DeltaNet path is not the obvious cause.
  - `CK_RMSNORM_EXACT=1`, `CK_SWIGLU_EXACT=1`, `CK_DEBUG_Q8K_REF=1`,
    `CK_DEBUG_Q4K_Q8_REF=1`: unchanged at step 2.
  - `CK_V8_DEBUG_LM_HEAD_FP32=1`: still step 2, with slightly better RMSE
    (`0.132172`) but no top-1 parity.
  - `CK_DEBUG_Q8_0_Q8_0_REF=1`: regressed to step 0; the Q8_0 AVX512 dot is
    not the current drift source.
  - `CK_DEBUG_Q6K_Q8K_SIMD=1`: unchanged at step 2. Q6_K/Q8_K final-head
    SIMD-vs-reference order is not enough to move the current borderline
    ranking.
  - Removing the source default `mlp_gate_up_fp32_layers=[0]`: regressed to
    step 0. Restore and keep that default.

## Current state

The hard first-token flip is reduced, but full 40-token parity is not closed.
The remaining step-2 divergence is a borderline ranking drift with complete
top-k overlap, not tokenizer/KV corruption.

Do not broaden FP32 fallback globally. Most broad reference/FP32 probes either
do nothing or regress to step 0, which means the current step-2 parity is partly
dependent on matching the production decode path rather than simply replacing
more kernels with scalar fallbacks. The useful next target is finer layer/logit
attribution around the step-2 prefix, with emphasis on remaining quantized
projection ordering after layer 0 and final-head ranking.

## Validation commands run

```bash
.venv/bin/python -m py_compile version/v8/scripts/build_ir_v8.py version/v8/scripts/codegen_core_v8.py
.venv/bin/python unittest/test_q8_k_quantize_parity.py
.venv/bin/python unittest/test_q4_k_q8_k_matvec.py
.venv/bin/python unittest/test_q6k_q8k_parity.py
make build/libckernel_engine.so
```
