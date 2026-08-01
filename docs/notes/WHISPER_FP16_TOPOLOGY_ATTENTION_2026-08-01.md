# Whisper FP16 topology and attention optimization

## Scope

This change optimizes generated Whisper FP16 encoder execution without changing the FP32 provider or making FP16 the default. Provider selection remains circuit and kernel-map driven.

## Changes

- Detect hybrid CPU topology from Linux `thread_siblings_list` data.
- Use SMT-capable performance cores for FP16 audio workers when SMT and singleton cores coexist.
- Respect an explicit `CK_NUM_THREADS` setting.
- Record worker affinity and thread count in the end-to-end report.
- Resolve FP16 Whisper encoder attention to a bounded, tiled online-softmax provider.
- Keep FP32 Whisper encoder attention on its existing packed-K contract.

## Same-host evidence

Whisper Base on the 11-second JFK fixture improved from repeated 1.53-1.58 second runs to 1.25-1.29 seconds. The 25-token transcript remained exact. The new attention provider measured 15.58 ms per encoder layer versus 26.87 ms for the prior full-score provider.

VTune measured 42.79 billion retired instructions versus 48.71 billion previously, and effective physical-core utilization increased from 3.31 to 3.89 cores. FP16 GEMM is now the largest remaining hotspot, followed by Conv1D and tiled attention.

Generated FP16 runtimes transcribed the same fixture coherently with Tiny, Base, and Small. These are focused single-fixture promotion results, not broad transcript-corpus certification.

## Remaining work

- Extend the AVX2 4-token by 2-output FP16 GEMM tile to a separately certified AVX-512 implementation.
- Improve Conv1D channel and frame partitioning.
- Certify clean, noisy, multilingual, timestamp, and long-audio fixtures.
- Compare each checkpoint with same-host PyTorch and whisper.cpp runs.

## AVX2 FP16 GEMM follow-up

The AVX2 FP16 provider now computes four token rows and two output rows together. This reuses each loaded weight vector across four activations while retaining the previous eight-lane FMA chain and horizontal reduction independently for every output.

Full production-shape baseline comparisons are byte-exact for Tiny, Base, and Small projections and MLPs. The independent llama.cpp production-graph oracle also remains bit-exact at 1, 16, 20, and 24 threads.

Alternating same-library Whisper runs improved the encoder by about 4 percent for Tiny, 5-6 percent for Base, and 7 percent for Small. Their transcripts remained unchanged. The AVX2 tile is compiled out on AVX-512 so it cannot alter that ISA's 16-lane reduction contract.

The durable Base-shape benchmark measures 1.28-1.70x isolated provider speedups on this host. VTune attributes 97.6 percent of the provider-only CPU time to `ck_gemm_f16_input_fp16_work`; with P-core affinity and no competing BLAS pool, the Base MLP-up median is approximately 8.36 ms. Intel Advisor 2026.0 could not ingest the Python-driven trace on this host and reported `_advi_dynamic_regions_table` followed by no data, so no Advisor roofline claim is made.

## AVX2 stride-2 Conv1D follow-up

Whisper's second encoder stem reads every other input frame. The previous AVX2 implementation issued an indexed gather for each input-channel and kernel-tap tile. The optimized implementation loads two contiguous eight-float vectors and compacts their even lanes before executing the unchanged per-lane FMA sequence.

The full Base stem shape, 512 input channels by 512 output channels by 3000 input frames and 1500 output frames, is byte-identical to the prior gather implementation. A corrected alternating benchmark, with provider selection performed once before dispatch rather than inside the inner loop, improves the median from 47.40 milliseconds to 23.54 milliseconds, approximately 2.0x.

Alternating FP16 E2E encoder runs improve from 0.899/0.907 seconds to 0.883/0.871 seconds, about 2.9 percent on average, and produce the same transcript. The disposable test runtime replaced the original oneDNN-linked engine, making its decoder-prefill timing non-comparable; no whole-run speedup is claimed from this experiment.

## Native profiler boundary

The Python Whisper runner remains useful for artifact preparation, backend rotation, transcript comparison, and machine-readable reports. It must not be the sampled process for an Advisor roofline claim: Python startup, NumPy, subprocess orchestration, and report assembly obscure the generated C runtime and previously prevented Advisor from producing a usable result.

The native profiling path should reuse the loader conventions in `version/v8/src/ck_cli_v8.c`, while exposing an audio-specific executable rather than routing audio through the text-only chat loop. That executable should:

1. Load the generated encoder and decoder `libmodel.so` files once.
2. Initialize both from their manifests.
3. Read the WAV bytes in C and call `ck_model_run_audio_wav_window` directly.
4. Pass the named encoder activation to the generated decoder cross-attention ABI.
5. Run warmups before measurement and repeat the same phase without reloading weights.
6. Expose separate encoder, decoder-prefill, decode, and provider-only regions.
7. Emit the runtime hashes, compiler flags, thread affinity, transcript tokens, and phase timings used by the Python comparison report.

VTune and Advisor should attach to that native executable. Provider roofline work should use an even narrower native C harness that repeatedly invokes one production-shape kernel after setup. This keeps initialization and Python outside the measured region and makes utilization, memory traffic, instruction count, and thread imbalance attributable to the actual CKE provider.
