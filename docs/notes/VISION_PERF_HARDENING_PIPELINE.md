# Vision Performance Hardening Pipeline

This runbook describes the repeatable CK workflow for closing vision-model performance gaps without breaking model correctness. It is intentionally model-agnostic: Qwen3-VL OCR is the first concrete contract, but the same loop applies to Gemma4V, Qwen3.5-VL, Kimi-VL, GLM-VL, and future encoder/decoder bridge models.

## Core Rule

Do not accept a speed patch unless the correctness contract still passes.

Correctness can be one of:

- exact generated text, e.g. `CK OCR TEST
TOTAL 42`
- top-1/top-k logits parity
- layer-boundary tensor parity
- OCR field/JSON score
- model-specific semantic smoke output

If correctness fails, stop performance work and debug stitching/kernel parity first.

## Deterministic Loop

1. Pick one fixed workload.
   - Same model artifact.
   - Same prompt/image/tokens/context.
   - Same thread count and environment flags.

2. Run the benchmark with profiling enabled.
   - For Qwen3-VL OCR, use `benchmarks/qwen3vl_ocr_perf_pipeline.py`.
   - Store raw benchmark JSON and pipeline summary JSON/Markdown.

3. Check correctness.
   - If expected text/logits/parity fails, reject the speed change.

4. Aggregate hotspots by section/op/kernel.
   - Encoder CSV.
   - Decoder mixed-prefill CSV.
   - Generation/decode CSV if available.

5. Compare against a known baseline.
   - Report encoder delta, mixed-prefill delta, generation delta, and total steady-state delta.

6. Choose only the next largest correctness-clean bottleneck.
   - Do not tune broad thread policy before identifying the actual hot op.
   - Do not change activation quantization policy unless parity proves it is safe.

7. Make one focused change.
   - Kernel implementation.
   - Prepacking/layout.
   - Tile size.
   - Thread cap for that kernel/shape.

8. Re-run the exact same pipeline.
   - Keep if correctness passes and model-level timing improves.
   - Reject if faster but wrong.
   - Keep as research only if microbench improves but model-level timing does not.

## Current Qwen3-VL OCR Contract

Optimized flags currently used for the heavy 1024 visual-token smoke:

```bash
CK_ENABLE_Q80_FP32_M4N4=1 \
CK_ENABLE_Q4K_GATEUP_SWIGLU_X16=1 \
CK_Q4K_X16_CHUNK4=1 \
CK_ATTENTION_QBLOCK4=1 \
CK_Q4K_GATEUP_SWIGLU_X16_THREAD_CAP=20 \
CK_NUM_THREADS=20 \
OMP_NUM_THREADS=1
```

Pipeline command:

```bash
.venv/bin/python -B benchmarks/qwen3vl_ocr_perf_pipeline.py \
  --image version/v8/test_assets/v8_ocr_clean_text.ppm \
  --threads 20 \
  --image-tokens 1024 \
  --context-len 1536 \
  --max-tokens 8 \
  --enable-q80-m4n4 \
  --enable-q4-gateup-x16 \
  --enable-q4-chunk4 \
  --enable-attention-qblock4 \
  --emit-vtune
```

Expected text:

```text
CK OCR TEST
TOTAL 42
```

Current optimized reference from this Xeon/OpenShift host:

- Encoder: about 33.8 s
- Decoder mixed prefill: about 34.3 s
- Generation: about 1.3 s
- Steady total: about 69.4 s

The current pipeline recommendation after Q4 chunk4 reuse and attention qblock4 is:

```text
decoder mlp_gate_up_swiglu / gemm_nt_q4_k_q8_k_gateup_swiglu_x16
```

That op is still the largest measured mixed-prefill hotspot at about 12.2 s in
the latest profile. The next work should improve Q4_K/Q8_K prefill reuse or
prepacking, not enable the opt-in AVX512 paths globally without shape and host
sweeps.

## Local and Nightly Coverage

The lightweight gate for the current Q4_K opt-in speed path is:

```bash
CK_Q4K_X16_CHUNK4=1 CK_NUM_THREADS=4 make bench-q4k-gateup-swiglu-x16-chunk4-quick
```

The chunk4 benchmark emits `speedup_x16`, so nightly can render it as a
performance lane.

The full OCR pipeline is the model-level proof:

```bash
CK_ENABLE_Q80_FP32_M4N4=1 \
CK_ENABLE_Q4K_GATEUP_SWIGLU_X16=1 \
CK_Q4K_X16_CHUNK4=1 \
CK_ATTENTION_QBLOCK4=1 \
CK_Q4K_GATEUP_SWIGLU_X16_THREAD_CAP=20 \
CK_NUM_THREADS=20 \
OMP_NUM_THREADS=1 \
make qwen3vl-ocr-perf-pipeline
```

Keep the full pipeline on high-memory runners or dedicated Xeon boxes with the
Qwen3-VL artifacts cached. It is too expensive for every default CI pass, but it
is the contract that proves the quick kernel gates translate into end-to-end OCR
speed without text corruption.

`CK_ATTENTION_QBLOCK4=1` is intentionally validated through the full OCR pipeline
for now. The existing standalone `unittest/test_attention_full.py` flash tests
currently fail their strict PyTorch tolerance on this AVX2 laptop even without
qblock4 enabled, so promoting that suite as a qblock4 nightly gate would add
noise rather than signal. Add a dedicated qblock4 parity microbench before making
that dispatch a default nightly correctness lane.

## Current Known Rejections

These were faster but not correctness-clean:

- `branch_fc1=q8_0`
- `branch_fc2=q8_0`
- `branch_fc1=q8_0` plus `branch_fc2=q8_0`

Reason: Qwen3-VL OCR output changed even though encoder time improved. Keep Qwen3-VL branch FC as fp32 activation unless a future parity-clean policy or kernel proves otherwise.

## VTune/Advisor Agent Instructions

Run the same pipeline workload under VTune/Advisor. Capture:

- hotspots
- memory access
- microarchitecture exploration
- threading analysis
- vectorization/advisor roofline if available

Answer these questions for the current top op:

- Is it memory-bandwidth limited or compute limited?
- What are L1/L2/L3 miss rates?
- Are VNNI/FMA units well utilized?
- Are reductions or scalar unpack/metadata paths hot?
- Is thread scaling worse after physical cores?
- Is there false sharing or output cache-line contention?
- Is the kernel rereading weights or activations unnecessarily across M/N tiles?

For Q4_K/Q8_K gate/up, test at least:

```bash
CK_NUM_THREADS=20 build/bench_q4k_gateup_swiglu   --M 1028 --D 12288 --K 4096   --threads 20 --tile-m 8   --iters 5 --warmup 2 --mode x16
```

Thread/tile sweeps should include:

- threads: 12, 16, 20, 24, 32, 48
- tile_m: 4, 8, 16 where supported
- physical cores before SMT

## Generalizing Beyond Qwen3-VL

The generic version of this pipeline should take a model contract file with:

- runner command
- input assets/prompts/token IDs
- correctness policy
- expected output or parity target
- profile artifact paths
- baseline JSON
- enabled feature flags

The process is the same for every CK model family because the CK architecture is kernel stitching plus explicit circuit contracts. The only model-specific part should be the contract, not the profiling/decision loop.
