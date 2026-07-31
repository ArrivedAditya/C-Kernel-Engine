# Qwen3.6 exact batched-prefill optimization

## Scope

This patch accelerates the opt-in Qwen3.6 Q4_K_M batched-prefill graph without
changing its numerical trajectory:

1. DeltaNet prefill partitions independent value heads once per prompt. Each
   worker advances its assigned heads through every token in order, preserving
   the certified fused-recurrent arithmetic and avoiding one dispatch per row.
2. The existing AVX-512 VNNI x16 Q4_K provider is promoted to every measured
   Qwen3.6 Q4 prefill shape, including recurrent output and MLP-down. Packing
   remains at model initialization rather than the first measured prompt.

The default `sequential_decode` prefill policy is unchanged. Batched execution
still requires `CK_V8_FORCE_BATCHED_PREFILL=1` until long-context accumulated
logit drift against llama.cpp is closed.

## Measurements

Host: 14-CPU cgroup, ICX native AVX-512/VNNI build, Qwen3.6-27B Q4_K_M,
29-token templated prompt:

`Give an example of C, Python, SQL code in detail.`

| Configuration | Prompt time | Prompt rate |
|---|---:|---:|
| Batched baseline | 6682.61 ms | 4.34 tok/s |
| Exact head-parallel DeltaNet | 4014.20 ms | 7.22 tok/s |
| Exact DeltaNet + all measured Q4 x16 shapes | 2588.18 ms | 11.20 tok/s |

The final result is 61.3% less prompt time and 2.58x the baseline throughput.

At the real `M=28, N=5120, K=17408` vectorized MLP-down portion, x16 measured
3.049 ms versus 5.650 ms for x8, with exact provider output. The Qwen-shaped
DeltaNet operator measured 7.74 ms versus 63.50 ms for the serial provider,
with bit-exact CKE output and state.

## Correctness gates

- Independent oracle: llama.cpp `f3e182816421c648188b5eab269853bf1531d950`.
- Qwen3.6-shaped DeltaNet serial/parallel output and state: bit-exact.
- llama.cpp fused DeltaNet operator oracle: pass at 1 and 14 threads on the
  measurement host, and at 1, 16, and 20 threads on the AVX2 integration host.
- llama.cpp Q4_K production oracle: all six promoted x16 shapes pass, including
  the residual row at `M=29`.
- Targeted Python contracts/performance tests: 86 passed.
- Persistent CK/llama trajectories:
  - short prompt: 8/8 greedy tokens match;
  - 353-token prefix: 3/3 greedy tokens match;
  - 1K-token prefix: 3/3 greedy tokens match.

The 1K first-step logit comparison remains cosine `0.865948879`, RMSE
`2.014045`, max absolute difference `19.671235`. This is a pre-existing
long-context numerical-accumulation issue, not introduced by the scheduling or
x16-provider changes, and remains the blocker for making batched prefill the
default.

## Reproduction artifacts

The following paths identify transient artifacts on the measurement host; they
are evidence provenance rather than repository fixtures:

- `/dev/shm/qwen36-exact-head-allx16.log`
- `/dev/shm/qwen36-exact-head-allx16-trajectory8.json`
- `/dev/shm/qwen36-exact-head-allx16-trajectory353.json`
- `/dev/shm/qwen36-exact-head-allx16-trajectory1000.json`
