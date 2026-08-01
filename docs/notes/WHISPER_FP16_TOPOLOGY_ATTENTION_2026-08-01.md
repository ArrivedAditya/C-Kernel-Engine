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

- Improve FP16 GEMM packing and activation reuse.
- Improve Conv1D channel and frame partitioning.
- Certify clean, noisy, multilingual, timestamp, and long-audio fixtures.
- Compare each checkpoint with same-host PyTorch and whisper.cpp runs.
