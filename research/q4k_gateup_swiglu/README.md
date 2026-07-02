# Q4_K Gate/Up + SwiGLU Standalone Research Harness

This folder contains standalone experiments for the Qwen3-VL OCR MLP hot path:

```text
M = 79 tokens
D = 12288 hidden/intermediate half-dim
K = 4096 input dim
weights = ~54 MiB Q4_K gate_up matrix
```

The goal is to tune cache-aware Q4_K x Q8_K gate/up + SwiGLU kernels before promoting them into a default v8 runtime path. The C file links against the main CK engine for block definitions, Q8_K activation quantization, the current reference GEMM/SwiGLU path, and optionally the CK threadpool. It can run the same x16 packed-panel microkernel through either OpenMP or CK threadpool scheduling.

The v8 prefill codegen now has an experimental opt-in integration guarded by:

```bash
CK_ENABLE_Q4K_GATEUP_SWIGLU_X16=1
```

Keep this disabled by default. The current implementation lazily packs x16 panels at runtime and keeps that cache alive until process exit. The next production step is conversion/load-time prepacking plus shape and ISA gating.

Build:

```bash
make --no-print-directory build/bench_q4k_gateup_swiglu_omp_standalone
```

Representative run on Xeon Gold 6542Y, physical cores only:

```bash
/usr/bin/env KMP_BLOCKTIME=0 OMP_NUM_THREADS=16 taskset -c 0-23 \
  build/bench_q4k_gateup_swiglu_omp_standalone \
  --threads 16 --tile-m 8 --iters 10 --warmup 2 --mode x16 --scheduler ck
```

Useful sweeps:

```bash
for tm in 1 2 4 8; do
  for t in 1 2 4 8 12 16 20 24 32 48; do
    cpus=0-23; [ "$t" -gt 24 ] && cpus=0-47
    /usr/bin/env KMP_BLOCKTIME=0 OMP_NUM_THREADS=$t taskset -c $cpus \
      build/bench_q4k_gateup_swiglu_omp_standalone \
      --threads $t --tile-m $tm --iters 4 --warmup 1 --mode x16 --scheduler ck
  done
done
```

VTune software hotspots, no hardware PMU required:

```bash
/opt/app-root/src/Programs/intel/oneapi_2025/vtune/2025.7/bin64/vtune \
  -collect hotspots \
  -knob sampling-mode=sw \
  -knob enable-characterization-insights=false \
  -knob enable-stack-collection=true \
  -r build/vtune_q4_gate_omp_x16_t8_16_sw \
  -- /usr/bin/env KMP_BLOCKTIME=0 OMP_NUM_THREADS=16 taskset -c 0-23 \
     build/bench_q4k_gateup_swiglu_omp_standalone \
     --threads 16 --tile-m 8 --iters 8 --warmup 1 --mode x16 --scheduler ck
```

Current finding on this Xeon:

```text
Best OpenMP region: tile_m=8, 16 physical threads, ~26-32 ms/layer depending on runtime noise
Best CK-threadpool region: tile_m=8, 20 physical threads, ~24.7 ms/layer in one sweep
Current hot-op region: ~37-40 ms/layer
Parity: rel_diff ~4.3e-7, cosine 1.0
SMT/48 threads is worse.
```

This is research/perf infrastructure, not a default production path. Do not enable it globally until model-level sweeps prove a material win on the target hardware.


Scheduler comparison:

```bash
for sched in omp ck; do
  /usr/bin/env KMP_BLOCKTIME=0 OMP_NUM_THREADS=16 CK_NUM_THREADS=16 taskset -c 0-23 \
    build/bench_q4k_gateup_swiglu_omp_standalone \
    --threads 16 --tile-m 8 --iters 8 --warmup 2 --mode x16 --scheduler $sched
done
```

Measured example:

```text
scheduler=omp: ~26.3 ms/layer
scheduler=ck:  ~25.2 ms/layer
```


Deterministic cache model:

```bash
python3 research/q4k_gateup_swiglu/cache_model.py --M 79 --D 12288 --K 4096
```

This computes `tile_m` and `active_threads` from L1/L2/L3 and physical-core topology. Benchmark sweeps validate the model, but defaults should come from this deterministic cache calculation rather than a single noisy OpenShift timing run.

## 2026-07-02 Qwen3-VL OCR x16 Notes

A companion summary is tracked at:

```text
docs/notes/QWEN3VL_OCR_Q4Q8_SPEED_RESEARCH_2026-07-02.md
```

Important result from the Xeon/OpenShift run: more active threads were not always
better. For the real Qwen3-VL OCR gate/up shape (`M=1028, D=12288, K=4096`),
20 physical threads beat 24 and 48. This matches the cache/core-ratio model from
the BC server validation package: the bottleneck is effective cache and memory
bandwidth per active worker, not just total visible hardware threads.

The dual gate/up accumulator idea was numerically clean but slower on this Xeon
(~442 ms versus ~428 ms for the best x16 variant), likely due to register and
instruction pressure. Keep it as a retest candidate for Ryzen/X3D, EPYC, and
larger-cache Xeon systems before discarding it globally.
