# Qwen3.6 parallel DeltaNet decode

## Change

The exact llama.cpp-compatible grouped DeltaNet decode kernel now exposes a
head-range entry point. The v8 runtime wrapper partitions independent heads
over the existing persistent CKE threadpool. It does not split a head or
change its dot-product, FMA, gate, state-update, or output reduction order.
Non-AVX2 builds retain the serial provider.

The production decode kernel map selects
`gated_deltanet_llama_avx2_parallel_forward`. Prefill continues to use the
separately certified 64-token chunked provider.

## Exactness

The production llama.cpp fused-operator test now includes the Qwen3.6 shape:

- 48 value heads
- 16 Q/K groups
- state dimension 128
- 1 and 14 CKE threads

The parallel output and recurrent state were bit-exact to llama.cpp and to the
serial CKE provider. The persistent eight-step greedy X-ray trajectory also
retained top-1 agreement with no divergence:

- report: `/dev/shm/qwen36-parallel-trajectory.json`
- report SHA-256:
  `d2dc952da0a150868d6dbfa75685947f1da5c6d53e980a2b102232ea02693196`
- llama layer profile SHA-256:
  `3b43deda9469c2670236185466b44410296e8290501110c9520ae4f8492dffbe`

The trajectory's existing quantized-logit differences from llama.cpp remain;
this optimization does not add or amplify them.

## Performance

Production-shape kernel sweep:

| Threads | Serial | Parallel | Speedup |
|---:|---:|---:|---:|
| 1 | 2.3045 ms | 2.1623 ms | 1.066x |
| 2 | 2.1023 ms | 1.1134 ms | 1.888x |
| 4 | 2.2090 ms | 0.6115 ms | 3.612x |
| 8 | 2.3797 ms | 0.4765 ms | 4.995x |
| 14 | 2.3765 ms | 0.2778 ms | 8.555x |
| 24 | 2.3012 ms | 0.1944 ms | 11.838x |

On the quota-matched 14-thread Qwen3.6-27B Q4_K_M end-to-end prompt
`Give an example of C, Python, SQL code in detail.`:

| Provider | Decode time | Decode rate |
|---|---:|---:|
| Serial recurrent core | 380.84 ms/token | 2.63 tokens/s |
| Parallel recurrent core | 276.58 ms/token | 3.62 tokens/s |

That is a 27.4% latency reduction and a 37.6% throughput increase. The saved
104.26 ms/token agrees with the prior X-ray attribution of approximately
113.6 ms/token to the serial recurrent core. CKE decode is substantially closer
to the quota-matched llama.cpp result of 5.5 tokens/s, but it is not yet equal;
Q4 and Q6 decode projections are the next dominant families.

Do not attribute the A/B prompt-rate difference to this patch: the prefill
provider did not change, so that variation is cache and scheduler noise.
