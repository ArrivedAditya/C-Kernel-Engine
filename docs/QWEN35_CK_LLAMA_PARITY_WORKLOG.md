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

## Post-PR #24 pull verification and scoped recurrent probes

After pulling `origin/main` at `53839e89`, the tree was clean and the quick
validation set passed:

```bash
.venv/bin/python -m py_compile version/v8/scripts/build_ir_v8.py version/v8/scripts/codegen_core_v8.py version/v8/scripts/compare_multitoken_logits_v8.py
.venv/bin/python unittest/test_q8_k_quantize_parity.py
.venv/bin/python unittest/test_q6k_q8k_parity.py
```

The regenerated Qwen3.5 `libmodel.so` still has the same production baseline:

```text
status=fail step=2 prefix_len=33 ck_next=42726 llama_next=8755 cosine=0.999169 rmse=0.136147 topk_overlap=16/16
```

Scoped recurrent projection probes were added as disabled-by-default codegen
instrumentation:

```bash
CK_V8_DEBUG_RECURRENT_PROJ_FP32_LAYER=<layer>
CK_V8_DEBUG_RECURRENT_PROJ_FP32_OP=recurrent_gate_proj|recurrent_alpha_proj|recurrent_beta_proj|all
```

Probe results on the step-2 prefix:

- Layer-0 `recurrent_gate_proj` FP32 input fallback regressed to step 0:
  `ck_next=4434`, `llama_next=14542`, cosine `0.998873`, RMSE `0.167207`,
  top-k overlap `14/16`.
- Layer-0 `recurrent_alpha_proj` FP32 input fallback regressed to step 0:
  cosine `0.999044`, RMSE `0.152741`, top-k overlap `16/16`.
- Layer-0 `recurrent_beta_proj` kept step 2 but worsened RMSE to `0.167373`.
- Layer-1 recurrent projection fallback with `OP=all` regressed to step 0:
  cosine `0.999032`, RMSE `0.153540`, top-k overlap `14/16`.
- `CK_DEBUG_Q5K_FP32_FALLBACK=1` regressed to step 0.
- `CK_DEBUG_Q5K_GENERIC_DOT=1` regressed to step 0.
- `CK_DEBUG_Q5K_SHARED_Q8_QUANT=1` was identical to baseline.
- Fresh `CK_DEBUG_Q8_0_Q8_0_REF=1` regressed to step 0.

Conclusion: the production path remains the best tested path. The remaining
step-2 mismatch is not fixed by broad recurrent FP32 fallback, Q5_K FP32/generic
fallback, or Q8_0 reference mode. The next useful probe should be a narrower
layer/op output attribution around layer-0 residual/MLP/final logits, not a
broader reference fallback.

## Narrower layer-0/final-head attribution

The multi-token parity runner now records margin fields for borderline top-1
cases:

- `ck_top1_margin`
- `llama_top1_margin`
- `ck_llama_winner_delta_in_ck`
- `llama_winner_delta_in_llama`

Fresh baseline with margin reporting:

```text
status=fail step=2 prefix_len=33 ck_next=42726 llama_next=8755 cosine=0.999169 rmse=0.136147 ck_margin=0.050137 llama_margin=0.014244 topk_overlap=16/16
```

This confirms the failure is a swapped top-2 ranking. CK prefers `42726` over
`8755` by about `0.050`, while llama prefers `8755` over `42726` by only about
`0.014`.

Additional targeted probes:

- `CK_V8_DEBUG_LM_HEAD_FP32=1` improves global RMSE to about `0.132172`, but
  still fails at step 2 and increases CK's wrong top-1 margin to about
  `0.053362`. Final-head Q8_K activation quantization is not the whole issue.
- `CK_V8_DEBUG_LM_HEAD_FP32=1 CK_DEBUG_Q5K_SHARED_Q8_QUANT=1` is identical to
  final-head FP32 alone.
- `CK_V8_DEBUG_LM_HEAD_FP32=1 CK_DEBUG_Q6K_Q8K_SIMD=1` is identical to
  final-head FP32 alone because the final-head debug path uses `gemv_q6_k`.
- First full-attention layer `CK_V8_DEBUG_ATTN_PROJ_FP32_LAYER=3
  CK_V8_DEBUG_ATTN_PROJ_FP32_OP=q_gate_proj` is unchanged at step 2.
- First full-attention layer `OP=out_proj` regresses to step 0.
- First full-attention layer `OP=all` regresses to step 0.
- Extending the layer-0 MLP gate/up FP32 treatment to layer 1 with
  `CK_V8_DEBUG_MLP_GATE_UP_FP32_LAYER=1` regresses to step 0.

Conclusion: do not generalize the layer-0 MLP gate/up FP32 default, do not make
full-attention projection FP32 by default, and do not switch final-head logits
to FP32 input as a correctness fix. The remaining drift is already present in
the final hidden vector, but the tested broad layer-1/full-attention/final-head
knobs either do nothing or make the earlier step-0 ranking worse.

## Hidden/layer boundary attribution

Added disabled-by-default CK hidden export instrumentation:

```bash
CK_DEBUG_EXPORT_HIDDEN=/tmp/qwen35_ck_hidden_prefix31
```

The generated model now writes raw float32 dumps for:

- `after_attn`: after the attention residual add.
- `layer_out`: after the MLP residual add.
- `final_hidden`: after final RMSNorm.

The dump naming convention is:

```text
tok_<token_index>_layer_<layer>_<name>.f32
```

Added helper scripts:

- `version/v8/scripts/export_ck_hidden_v8.py`
- `version/v8/scripts/export_llama_hidden_v8.py`
- `version/v8/scripts/compare_hidden_vectors_v8.py`

For the 31-token Qwen3.5 replay prefix, llama.cpp `result_norm` and CK
`final_hidden` have matching shape (`1024` float32 values) but still differ:

```text
cosine=0.998619420 rmse=0.228622310 mean_abs=0.180518913 max_abs=0.967097282
```

Layer boundary comparison against llama.cpp `l_out-N` shows the residual stream
is still close before final norm, but drift accumulates through the stack:

```text
layer=00 layer_rmse=0.000584642
layer=08 layer_rmse=0.003003950
layer=16 layer_rmse=0.005816995
layer=23 layer_rmse=0.015478137
final    final_rmse=0.228622310
```

The final RMSNorm ratio check shows the norm itself is consistent: the
CK/llama scale ratio is effectively constant (`1.0027507`, std about
`5e-8`). The apparent final-hidden jump is the expected amplification of the
layer-23 residual drift by the final RMSNorm scale, not a separate final-norm
layout bug.

Splitting each layer into attention residual and MLP residual at the same
prefix shows the first drift starts in layer-0 MLP, while attention is initially
essentially exact:

```text
layer=00 attn_rmse=0.000000294 layer_rmse=0.000584642 mlp_delta_rmse=0.000584625
layer=01 attn_rmse=0.001277270 layer_rmse=0.001718792 mlp_delta_rmse=0.001613365
layer=16 attn_rmse=0.005361738 layer_rmse=0.005816995 mlp_delta_rmse=0.003202434
layer=23 attn_rmse=0.013048312 layer_rmse=0.015478137 mlp_delta_rmse=0.007171859
```

Conclusion: the next probe should instrument MLP internals (`attn_post_norm`,
gate/up projections, SwiGLU output, and down projection) rather than KV cache,
template handling, final RMSNorm, or broad FP32 fallbacks.

## MLP internal attribution

Extended `CK_DEBUG_EXPORT_HIDDEN` to also dump:

- `post_attn_norm`
- `mlp_gate`
- `mlp_up`
- `mlp_swiglu`

At the same 31-token prefix, layer 0 now isolates the first meaningful drift to
the Q4_K x Q8_K gate/up projection feeding SwiGLU:

```text
layer=00 post_attn_norm_rmse=0.000007283
layer=00 gate_rmse=0.006835827
layer=00 up_rmse=0.004497923
layer=00 swiglu_rmse=0.000858409
layer=00 ffn_out_rmse=0.000584625
```

Selected later-layer gate/up/SwiGLU drift:

```text
layer=01 gate_rmse=0.016662135 up_rmse=0.010981179 swiglu_rmse=0.002851053
layer=03 gate_rmse=0.040134239 up_rmse=0.018443348 swiglu_rmse=0.005236901
layer=16 gate_rmse=0.024762276 up_rmse=0.014746895 swiglu_rmse=0.005745402
layer=23 gate_rmse=0.032057280 up_rmse=0.019168103 swiglu_rmse=0.012375210
```

Conclusion: the remaining Qwen3.5 long-generation drift is now narrowed to
quantized MLP gate/up projection accumulation/layout, with SwiGLU mostly
amplifying that input difference. The next corrective patch should target
`gemv_q4_k_q8_k`/Q8_K activation quantization parity for the gate/up path, not
KV cache or final RMSNorm.

## Scoped MLP gate/up Q8_K contract probe

Added a scoped codegen probe for the MLP gate/up projection:

```bash
CK_V8_DEBUG_MLP_GATE_UP_Q8_CONTRACT_LAYER=0
CK_V8_DEBUG_MLP_GATE_UP_Q8_CONTRACT=1
```

This lets the generated `mlp_gate_up` path quantize the saved FP32 activation
to Q8_K and call `gemv_q4_k_q8_k`/`gemv_q6_k_q8_k`, even when the layer is
otherwise covered by the FP32 adapter. It is intentionally opt-in and is not a
production default.

At the 31-token prefix, forcing only layer 0 through the Q8_K contract improves
local hidden parity:

```text
baseline layer=00 gate_rmse=0.006835827 up_rmse=0.004497923 swiglu_rmse=0.000858409 layer_out_rmse=0.000584642
layer0-q8 layer=00 gate_rmse=0.000001400 up_rmse=0.000000987 swiglu_rmse=0.000000201 layer_out_rmse=0.000000302

baseline layer=01 gate_rmse=0.016782489 up_rmse=0.011081605 swiglu_rmse=0.002876791 layer_out_rmse=0.001725532
layer0-q8 layer=01 gate_rmse=0.009773169 up_rmse=0.006565175 swiglu_rmse=0.001707974 layer_out_rmse=0.001080668
```

But it does not improve production greedy parity. It regresses the multi-token
runner from the current step-2 divergence back to step 0:

```text
baseline: status=fail step=2 ck_next=42726 llama_next=8755 cosine=0.999169 rmse=0.136147 topk_overlap=16/16
layer0-q8: status=fail step=0 ck_next=4434 llama_next=14542 cosine=0.999011 rmse=0.157136 topk_overlap=15/16
```

The final-head probe gives the same warning: `CK_V8_DEBUG_LM_HEAD_FP32=1`
slightly improves RMSE at the current step-2 divergence (`0.132172`) but does
not recover top-1 parity, and combining it with the layer-0 Q8_K contract still
fails at step 0.

Conclusion: Q8_K gate/up contract is useful attribution instrumentation, but it
should not be made a default yet. The next target is the later-layer/final-head
borderline ranking drift after the improved layer-0 hidden parity, not a broad
gate/up fallback.

## Llama replay contract attribution

The earlier step-2 divergence was partly a parity-harness contract mismatch.
For recurrent Qwen3.5, llama.cpp chat behavior is best approximated by batching
the initial prompt/prefix and then decoding generated continuation tokens
sequentially. The old runner only supported full-batched or full-sequential
replay:

```text
full-batched:    status=fail step=2 prefix_len=33 ck_next=42726 llama_next=8755 rmse=0.136147 topk_overlap=16/16
full-sequential: status=fail step=0 prefix_len=31 ck_next=14542 llama_next=4434 rmse=0.134203 topk_overlap=16/16
hybrid:          status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791 rmse=0.167313 topk_overlap=15/16
```

The helper now uses `--prefix-decode-mode` for `--tokens-before`, and the
multi-token runner supports `--llama-decode-mode hybrid`. In `auto` mode,
`sequential_decode` CK contracts use hybrid llama replay.

The remaining mismatch is still real. At the step-5 hybrid divergence, the
competing logits are:

```text
id=1791  ck=20.545345 llama=20.547802 diff=-0.002457
id=11290 ck=20.693890 llama=20.240915 diff=+0.452974
```

Final hidden at the same prefix remains directionally close but numerically
different enough to flip that borderline final-head ranking:

```text
final_hidden cosine=0.998118609 rmse=0.260651554 mean_abs=0.202567363 max_abs=0.879841700
```

Conclusion: do not treat the old step-2 failure as a kernel regression by
itself. The next real fix target is the hidden-stream accumulation before the
step-5 hybrid divergence, then final-head/logit attribution on that corrected
contract.

## Step-5 recurrent attribution

Added `CK_DEBUG_EXPORT_HIDDEN` coverage for recurrent internals in generated
v8 decode code. This is diagnostic only: runtime math is unchanged unless the
environment variable is set.

At the step-5 hybrid divergence prefix, layer-boundary drift is gradual rather
than a KV-position or recurrent-state collapse. Exact RMSNorm does not move the
result:

```text
CK_RMSNORM_EXACT=1:
status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
cosine=0.998818 rmse=0.167313 topk_overlap=15/16
```

Layer 0 recurrent internals are essentially exact until the SSM output
projection:

```text
layer 0 linear_attn_qkv_mixed rmse=2.05e-7
layer 0 conv_output_silu      rmse=1.89e-8
layer 0 q_conv_predelta       rmse=1.19e-6
layer 0 k_conv_predelta       rmse=2.53e-6
layer 0 gate                  rmse=6.67e-8
layer 0 attn_output           rmse=6.97e-8
layer 0 final_output          rmse=1.45e-6
layer 0 linear_attn_out       rmse=1.10e-4
layer 0 layer_out             rmse=6.72e-4
```

The first nontrivial jump is therefore `recurrent_out_proj` (`ssm_out`), which
currently lowers to `gemv_q5_k` with Q5_K weights and FP32 activation adapter.
Debug variants show this is not solved by broad fallbacks:

```text
normal Q5_K:       L0 linear_attn_out rmse=0.000109637
shared Q8 quant:   L0 linear_attn_out rmse=0.000109637
generic Q5 dot:    L0 linear_attn_out rmse=0.000109633
FP32 dequant path: L0 linear_attn_out rmse=0.001397091
```

Conclusion: DeltaNet/state update is not the source of the current step-5
drift. The next focused target is Q5_K `ssm_out` projection parity against
llama.cpp's production CPU path, then re-run the hybrid multi-token runner.

## Step-5 Q5 ssm_out and all-layer attribution

Added a diagnostic-only `CK_DEBUG_EXPORT_HIDDEN` export for the special
`mlp_down` Q4/Q6 x Q8 generated branch. This branch returned before the generic
hidden export, so previous layer scans could not directly compare CK
`mlp_down` with llama.cpp `ffn_out`.

Q5_K `ssm_out` was tested against llama.cpp with production repack enabled and
disabled. The layer-0 `linear_attn_out` mismatch is effectively unchanged by
repack mode:

```text
llama repack:    L0 linear_attn_out rmse=0.000109637074
llama no-repack: L0 linear_attn_out rmse=0.000109637936
```

CK Q5_K debug variants also do not materially improve `ssm_out`; broad FP32
dequant is worse:

```text
normal Q5_K:       L0 linear_attn_out rmse=0.000109637
shared Q8 quant:   L0 linear_attn_out rmse=0.000109637
generic Q5 dot:    L0 linear_attn_out rmse=0.000109633
FP32 dequant path: L0 linear_attn_out rmse=0.001397091
```

All-layer boundary comparison at the step-5 prefix shows gradual accumulation
rather than a single early recurrent-state failure. Representative layers:

```text
layer 0  mlp_down rmse=0.000666019 layer_out rmse=0.000671901
layer 1  mlp_down rmse=0.001783894 layer_out rmse=0.001823632
layer 14 mlp_down rmse=0.003055496 layer_out rmse=0.005089171
layer 20 mlp_down rmse=0.004561305 layer_out rmse=0.012175406
layer 23 mlp_down rmse=0.010093235 layer_out rmse=0.019746876
```

Layer-scoped FP32 for layer-23 `mlp_down` does not fix final hidden/logit drift:

```text
layer 23 mlp_down FP32: rmse=0.009868351 max_abs=0.148512125
layer 23 layer_out FP32: rmse=0.019657867 max_abs=0.157317400
final_hidden FP32: cosine=0.998114085 rmse=0.260965042
```

The 100-token hybrid runner still diverges at the same point:

```text
status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
cosine=0.998818 rmse=0.167313 ck_margin=0.129669 llama_margin=0.123436
topk_overlap=15/16
```

Conclusion: the next target is no longer broad Q5_K `ssm_out`. The remaining
failure is accumulated hidden-stream drift amplified by final norm/final-head
borderline ranking. Next useful probes are final RMSNorm/logit attribution and
layer-scoped quantized projection parity where the drift grows late, especially
the Q4/Q6 MLP/output projections, without making broad FP32 fallbacks default.

Follow-up diagnostics:

```text
all MLP-down FP32:
  layer 23 layer_out rmse=0.018691192 max_abs=0.073951483
  final_hidden rmse=0.253469587
  token parity worsens to step=0 ck_next=4434 llama_next=14542

LM-head FP32:
  status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
  cosine=0.998821 rmse=0.167931 topk_overlap=15/16
```

Conclusion: broad MLP-down FP32 and final-head FP32 are not production fixes.
The remaining mismatch is a hidden-stream numerical accumulation issue. The
next patch should attribute the recurrent/full-attention block inputs and the
Q4/Q6 MLP gate/up/down projections around the late layers, then tighten only
the kernel/order that demonstrably moves the hybrid step-5 logits closer.

Cross-model smoke checks after diagnostic changes:

```text
Qwen3-0.6B cached v8:
  prompt: "Hello! Give one short sentence."
  response: "Hello, how can I assist you today?"
  prompt eval: 237.18 tok/s, decode: 39.03 tok/s, stop=eos

Gemma-3-270m-it cached v8:
  prompt: "Hello! Give one short sentence."
  response: "Hi there! How can I help you today?"
  prompt eval: 26.75 tok/s, decode: 27.37 tok/s, stop=eos

Qwen3.5-0.8B cached v8:
  prompt: "Hello! Give one short sentence."
  response is coherent and stops on eos, but decode remains slower
  prompt eval: 20.23 tok/s, decode: 18.14 tok/s
```

Qwen2 and Nanbeige were not present in the local v8 cache during this pass, so
they were not re-run without triggering downloads.

Layer-23 full-attention projection FP32 was also tested:

```text
CK_V8_DEBUG_ATTN_PROJ_FP32_LAYER=23 CK_V8_DEBUG_ATTN_PROJ_FP32_OP=all:
  status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
  cosine=0.998637 rmse=0.184371 topk_overlap=15/16
```

This worsens the metric relative to baseline, so full-attention projection FP32
is not a fix.

## Step-5 late-layer Q4/Q6 projection attribution

Added diagnostic-only exports for the special generated full-attention
`q_proj`/`k_proj`/`v_proj`/`out_proj` branches and added
`version/v8/scripts/compare_hidden_dirs_v8.py` so CK hidden dumps can be
compared against llama.cpp graph dumps without one-off scripts.

At the current step-5 divergent prefix, Q4 and Q8 reference modes do not move
the failure:

```text
CK_DEBUG_Q4K_Q8_REF=1:
  status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
  cosine=0.998818 rmse=0.167313

CK_DEBUG_Q8K_REF=1:
  status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
  cosine=0.998818 rmse=0.167313
```

The Q6_K x Q8_K standalone parity harness now compiles and passes:

```text
vec_dot_q6_k_q8_k: PASS
gemv_q6_k_q8_k: PASS
gemm_nt_q6_k_q8_k: PASS
vs_fp32_accuracy: PASS
```

Layer-output drift grows gradually across the model and then jumps most at the
last layer:

```text
layer 0  layer_out rmse=0.000671901
layer 14 layer_out rmse=0.005089171
layer 19 layer_out rmse=0.011063572
layer 22 layer_out rmse=0.015108324
layer 23 layer_out rmse=0.019746876 max_abs=0.176826477
```

Exact layer-23 sub-tensor attribution:

```text
after_attn      rmse=0.015569128 max_abs=0.054430604
post_attn_norm  rmse=0.064206322 max_abs=0.224881053
mlp_down/ffn_out rmse=0.010093235 max_abs=0.168021202
layer_out       rmse=0.019746876 max_abs=0.176826477
final_hidden    rmse=0.260651554 max_abs=0.879841700
```

The large final-hidden RMSE is not a separate final RMSNorm bug. CK and
llama.cpp both amplify layer-23 output by about 13x through final RMSNorm, so
the `0.0197` layer-output drift naturally becomes about `0.26` at
`result_norm`.

llama.cpp no-repack mode does not explain the mismatch. It changes borderline
logits but does not make CK match the no-repack hidden stream:

```text
llama repack:    layer23 layer_out rmse=0.019746876
llama no-repack: layer23 layer_out rmse=0.019932953
```

Current conclusion: the remaining step-5 mismatch is not Q4 SIMD/VNNI, not
Q8_K SIMD quantization, not Q6_K SIMD, not final head, and not final RMSNorm.
It is accumulated hidden-stream drift that becomes borderline-logit-visible.
The next useful target is a controlled cross-input projection probe: run CK
layer-23 Q6_K `ffn_down` with llama's `ffn_swiglu` input, and conversely compare
CK's `ffn_swiglu` input against llama's, so we can separate upstream
hidden-stream drift from Q6_K down-projection math without making broad FP32
fallbacks.

## Layer-23 controlled projection probes

Added `version/v8/scripts/probe_ck_q6_projection_v8.py`, a diagnostic-only
external-input projection probe. It initializes the generated CK model, resolves
generated weight offsets such as `W_LAYER_23_FFN_DOWN`, quantizes a supplied
FP32 input vector to Q8_K with CK's own `quantize_row_q8_k`, and runs the CK
Q4_K/Q6_K x Q8_K GEMV kernel against the loaded CK weights.

Probe self-checks against CK's own hidden dumps reproduce generated-model
outputs to float noise:

```text
CK post_attn_norm -> CK layer23 ffn_gate:
  rmse=0.000000268 max_abs=0.000001907

CK post_attn_norm -> CK layer23 ffn_up:
  rmse=0.000000119 max_abs=0.000000954

CK ffn_swiglu -> CK layer23 ffn_down:
  rmse=0.000000000 max_abs=0.000000000
```

The cross-input probes also match llama.cpp when using llama.cpp's exact inputs:

```text
llama attn_post_norm -> CK layer23 Q4_K ffn_gate vs llama ffn_gate:
  rmse=0.000000218 max_abs=0.000001907

llama attn_post_norm -> CK layer23 Q4_K ffn_up vs llama ffn_up:
  rmse=0.000000090 max_abs=0.000000954

llama ffn_swiglu -> CK layer23 Q6_K ffn_down vs llama ffn_out:
  rmse=0.000000040 max_abs=0.000000238
```

Conclusion: layer-23 Q4_K gate/up projection math and Q6_K down-projection math
are not the source of the current user-visible divergence. When controlled with
llama's inputs, CK matches llama. The remaining drift is already present in the
incoming hidden stream before layer-23 FFN. The next target should move one
stage earlier: layer-22 recurrent output/state update or the residual stream
entering layer 23, not Q4/Q6 projection accumulation.

## Layer-18..22 recurrent attribution

Added recurrent-label support to `version/v8/scripts/compare_hidden_dirs_v8.py`
and compared CK hidden dumps against llama.cpp dumps for token 35 at layers
18..22. The important result is that no single recurrent block creates a large
new jump. Recurrent core outputs remain close, while the residual stream drift
is already present and grows gradually:

```text
layer 18 attn_output rmse=0.000189929 final_output rmse=0.005203748 layer_out rmse=0.009478236
layer 20 attn_output rmse=0.000254188 final_output rmse=0.005859494 layer_out rmse=0.012175406
layer 21 attn_output rmse=0.000147115 final_output rmse=0.004915851 layer_out rmse=0.013586523
layer 22 attn_output rmse=0.000073131 final_output rmse=0.006740517 layer_out rmse=0.015108324
```

Layer 22 token 31..35 is internally close to llama.cpp:

```text
token 31 layer22 layer_out rmse=0.015393589
token 32 layer22 layer_out rmse=0.014333183
token 33 layer22 layer_out rmse=0.016999627
token 34 layer22 layer_out rmse=0.015404123
token 35 layer22 layer_out rmse=0.015108324
```

This clears layer-22 recurrent core/state update as a standalone large-error
source for the current step-5 divergence. It is still part of the accumulated
hidden stream, but the evidence now points to many small residual/projection
differences rather than one broken recurrent op.

Additional attribution checks:

```text
CK_DELTANET_FORCE_REF=1:
  worsens to step=0 divergence, ck_next=4434, llama_next=14542

CK_V8_DEBUG_LM_HEAD_FP32=1:
  still step=5 divergence, ck_next=11290, llama_next=1791

llama.cpp --no-repack:
  still prefers token 1791 at step 5, with a smaller llama margin
```

The parity runners now expose `--llama-no-repack` so no-repack comparisons can
be reproduced directly instead of requiring a hand-written helper command.

Current conclusion: Qwen3.5 step-5 mismatch is still real, but the latest
evidence clears layer-22 recurrent math, layer-23 Q4/Q6 projections, final
RMSNorm, final head, DeltaNet reference-vs-AVX, and llama CPU repack as single
smoking guns. The next useful fix target is earlier gradual residual-stream
drift, starting from the first layers where layer-output RMSE begins to grow,
with scoped projection/norm probes rather than broad FP32 fallbacks.

## Layer-0 attribution

Moved the comparison earlier than layer 18. At token 35, the residual drift
starts in layer 0 and then grows gradually across later layers:

```text
layer 0 layer_out rmse=0.000671901
layer 1 layer_out rmse=0.001823632
layer 2 layer_out rmse=0.002421916
layer 17 layer_out rmse=0.008501958
```

Layer-0 projection probes with llama.cpp inputs clear the quantized projection
paths:

```text
llama ffn_swiglu -> CK W_LAYER_0_FFN_DOWN Q6_K:
  rmse=0.000000007 max_abs=0.000000060

llama attn_post_norm -> CK W_LAYER_0_FFN_GATE Q4_K:
  rmse=0.000000072 max_abs=0.000000477

llama attn_post_norm -> CK W_LAYER_0_FFN_UP Q4_K:
  rmse=0.000000039 max_abs=0.000000179

llama final_output -> CK W_LAYER_0_SSM_OUT Q5_K:
  rmse=0.000000026 max_abs=0.000000060
```

Added Q5_K support to `probe_ck_q6_projection_v8.py` so recurrent `SSM_OUT`
can be checked directly.

Layer-0 recurrent internals are also very close before the final residual:

```text
z rmse=0.000000184 max_abs=0.000001431
alpha rmse=0.000000242 max_abs=0.000000477
gate rmse=0.000000067 max_abs=0.000000238
attn_output rmse=0.000000070 max_abs=0.000002246
final_output rmse=0.000001453 max_abs=0.000051141
```

Added `probe_ck_recurrent_norm_gate_v8.py` and verified CK gated RMSNorm with
llama `attn_output` and llama `z` reproduces llama `final_output`:

```text
recurrent_norm_gate_forward:
  rmse=0.000000012 max_abs=0.000000238
```

A narrow AVX-512 DeltaNet FMA experiment was tested and then reverted because
the multi-token parity result was unchanged:

```text
status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
cosine=0.998818 rmse=0.167313 topk_overlap=15/16
```

Current conclusion: the old hard corruption loop is not back. The remaining
Qwen3.5 issue is a small recurrent hidden-stream numerical drift that appears
from layer 0 and is amplified by normal downstream projections/norms. The
quantized Q4/Q5/Q6 projection kernels checked so far match llama.cpp with
controlled inputs. Next target should be a more direct DeltaNet fused-vs-CK
probe, especially comparing llama's fused `__fgdn_ar__` result layout/state
against CK's `attn_output`/state, rather than changing projection kernels.

## DeltaNet state and q/k norm probe

Added diagnostic state export for CK `recurrent_core` and added
`probe_ck_deltanet_v8.py` for direct CK DeltaNet calls with external tensors.

The raw CK state dump does not match llama's raw `new_state` dump because the
state layout is transposed by row/column per head:

```text
raw state compare:
  cosine=0.005202360 rmse=0.041935225 max_abs=3.336471231

CK [head,row,col] transposed to llama [head,col,row]:
  cosine=0.999999940 rmse=0.000002794 max_abs=0.000334039
```

With llama's q/k/v/g/beta and previous state fed directly into CK
`gated_deltanet_autoregressive_forward`, CK reproduces llama's layer-0
DeltaNet output and state:

```text
out:
  rmse=0.000000001 max_abs=0.000000045

state_llama_layout:
  rmse=0.000000001 max_abs=0.000000238
```

This clears CK DeltaNet math and state update for the tested layer-0 token.

The first visible upstream growth point is recurrent q/k L2 normalization:

```text
conv_output_silu:
  rmse=0.000000019 max_abs=0.000000477
q_conv:
  rmse=0.000000023 max_abs=0.000000477
k_conv:
  rmse=0.000000016 max_abs=0.000000238
q_conv_predelta:
  rmse=0.000001194 max_abs=0.000029743
k_conv_predelta:
  rmse=0.000002527 max_abs=0.000044525
```

I tested changing CK recurrent q/k L2 norm from `sqrt(sum_sq + eps)` to
llama.cpp's `max(sqrt(sum_sq), eps)` semantics. That matched llama's fully
sequential replay better, but it regressed the production-like hybrid target
from the known step-5 mismatch back to step 0:

```text
unsafe q/k norm experiment:
  status=fail step=0 ck_next=4434 llama_next=14542
```

That runtime change was reverted. The baseline production-like result is back
to the prior state:

```text
status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
cosine=0.998818 rmse=0.167313 topk_overlap=15/16
```

Current conclusion: this does not look like a restored hard-corruption bug.
It looks like an old, small decode-vs-prefill/hybrid numerical mismatch in
Qwen3.5 recurrent q/k normalization or its interaction with llama's chunked
prefill path. The next safe target is to compare CK production prefill, not
only sequential `ck_model_decode`, against llama hybrid before changing q/k
norm math.

## CK replay mode attribution

Added explicit CK replay modes to `compare_multitoken_logits_v8.py`:

- `--ck-prefill-mode auto`: follows the model runtime contract.
- `--ck-prefill-mode sequential`: feeds all tokens through `ck_model_decode`.
- `--ck-prefill-mode batched`: feeds the full prefix through
  `ck_model_embed_tokens` + `ck_model_forward`.
- `--ck-prefill-mode hybrid`: batches the initial prompt and decodes generated
  tokens one at a time.

For Qwen3.5, the runtime contract currently reports
`prefill_policy=sequential_decode`, but all three CK modes produced the same
production-like result against llama hybrid:

```text
auto:
  status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
  cosine=0.998818 rmse=0.167313 topk_overlap=15/16

batched:
  status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
  cosine=0.998818 rmse=0.167313 topk_overlap=15/16

hybrid:
  status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
  cosine=0.998818 rmse=0.167313 topk_overlap=15/16
```

llama sequential still diverges immediately from the production-like llama
hybrid target:

```text
llama sequential:
  status=fail step=0 prefix_len=31 ck_next=14542 llama_next=4434
  cosine=0.999262 rmse=0.134203 topk_overlap=16/16
```

Disabling llama CPU tensor repacking did not move the first production-like
divergence:

```text
llama hybrid --llama-no-repack:
  status=fail step=5 prefix_len=36 ck_next=11290 llama_next=1791
  cosine=0.998772 rmse=0.169180 topk_overlap=15/16
```

I also tried making llama-style q/k L2 norm a runtime env switch. Even with the
environment unset, the extra branch changed the hot kernel enough to flip a
borderline step-0 token after rebuild, so that diagnostic branch was removed.
The default rebuilt engine is back to the known step-5 baseline above.

Current conclusion: the remaining Qwen3.5 issue is not caused by comparing the
wrong CK prefill path. It is a shared CK numerical drift visible in sequential,
batched, and hybrid CK replay. The next safe target is narrower hidden-stream
attribution around the step-5 prefix, with attention to recurrent q/k norm
interaction and layer-0/layer-1 residual drift, but without adding branches to
hot math kernels or changing the default q/k norm formula.

## Step-5 hidden attribution

Generated a richer llama hidden dump for the step-5 prefix and compared layers
0-1 against CK hidden exports. Layer 0 shows the first measurable difference at
q/k L2 normalization:

```text
layer 0 linear_attn_qkv_mixed:
  rmse=0.000000205 max_abs=0.000002384
layer 0 q_conv:
  rmse=0.000000023 max_abs=0.000000477
layer 0 k_conv:
  rmse=0.000000016 max_abs=0.000000238
layer 0 q_conv_predelta:
  rmse=0.000001194 max_abs=0.000029743
layer 0 k_conv_predelta:
  rmse=0.000002527 max_abs=0.000044525
layer 0 attn_output:
  rmse=0.000000070 max_abs=0.000002246
layer 0 final_output:
  rmse=0.000001453 max_abs=0.000051141
layer 0 linear_attn_out:
  rmse=0.000109637 max_abs=0.000807405
layer 0 layer_out:
  rmse=0.000671901 max_abs=0.002632155
```

Offline formula check using CK `q_conv`/`k_conv` inputs confirms llama's q/k
norm output is reproduced by `x / max(sqrt(sum_sq), eps)`, while CK's default
`x / sqrt(sum_sq + eps)` accounts for the observed q/k predelta delta:

```text
q default formula vs llama:
  rmse=0.000001198 max_abs=0.000029862
q llama formula vs llama:
  rmse=0.000000020 max_abs=0.000000238

k default formula vs llama:
  rmse=0.000002527 max_abs=0.000044495
k llama formula vs llama:
  rmse=0.000000023 max_abs=0.000000209
```

However, switching the hot kernel to llama's formula globally still regresses
the prompt boundary to step 0. This means q/k norm semantics are a real local
parity difference, but changing them globally is not yet a safe production fix.
The likely explanation is that CK's sequential prompt-state evolution and
llama's hybrid/chunked prompt-state evolution currently have compensating
differences. The next safe target is to compare recurrent state and q/k norm
across the prompt boundary, especially token 30 -> generated token 31, before
applying any formula change.
