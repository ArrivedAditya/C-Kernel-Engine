# Qwen3.6 Q6_K M4 prefill provider

## Scope

This change targets the Q6_K MLP-down projections used by the locally
certified Qwen3.6-27B Q4_K_M artifact. It does not change decode dispatch.
Automatic selection is deliberately limited to short, wide prefill:

- `4 <= M <= 63`
- MLP-down shapes with `N >= 4096` and `K >= 8192`
- Qwen3.6 recurrent-QKV with `N = 10240` and `K = 5120`

Longer prefill, strict-parity mode, forced-reference mode, and other shapes
remain on the established Q6_K provider.

## Arithmetic contract

The AVX2 M4 kernel unpacks each Q6_K block once and reuses it for up to four
independent Q8_K token rows. Each row retains the llama.cpp-compatible integer
accumulation, block-scale FMA order, and horizontal FP32 reduction tree.

The production oracle explicitly forces the M4 provider and compares every
output bit with the pinned llama.cpp commit `f3e18281`. It runs at 1, 16, 20,
and 24 threads through `test-q6k-q8k-llama-production`. The oracle build must
disable AVX-VNNI for this AVX2 reduction contract. A source checkout, headers,
and shared library from different llama.cpp builds are not valid evidence.

## Measurements

Same host, ICX build, 24 CKE threads:

| Test | Established provider | M4 provider | Result |
|---|---:|---:|---:|
| Synthetic Qwen shape (`M=23,N=5120,K=17408`) | 5.57 ms | 4.09 ms | 1.36x kernel speedup |
| Synthetic recurrent QKV (`M=23,N=10240,K=5120`) | 3.29 ms | 2.41 ms | 1.37x kernel speedup |
| Real 23-token Qwen3.6 prefill | 3.799 s | 3.662 s | 3.6% end-to-end prefill win |
| Real-model Q6 MLP-down attribution | 1217.0 ms | 1130.8 ms | 7.1% Q6 MLP-down win |
| Real-model recurrent QKV, same-host A/B | 388.3 ms | 160.9 ms | 58.6% recurrent-QKV win |
| Real-model prefill, same-host A/B | 3725.8 ms | 3381.9 ms | 9.2% observed end-to-end win |

The real-model first-token comparison retained the same top-1 and complete
top-5 set as llama.cpp. Its maximum logit difference was `0.482825`, equal to
the pre-existing Qwen3.6 numerical envelope.

The end-to-end A/B total includes normal host noise in unrelated operators.
The directly attributable recurrent-QKV saving was 227.5 ms, or 6.1% of the
old measured prefill total.

On a separate Core i7 AVX2 host with GCC, the same synthetic production shape
measured 7.79 ms for the established provider and 5.75 ms for M4, a 1.355x
speedup. The clean pinned llama.cpp AVX2 oracle was bit-exact for all five
decode/prefill shapes at 1, 16, 20, and 24 threads. An initially selected
source tree at `dc659243` had local edits and an older AVX-VNNI-enabled shared
library; that invalid mixed-provenance oracle differed for both the established
and M4 providers by up to 4.58e-5. Rebuilding the pinned AVX2 oracle removed
the difference.

## Regression gates

- `make test-q6k-q8k-llama-production` checks M4 bit-exactness independently
  against llama.cpp.
- `make test-qwen36-q6k-m4-performance` checks full-output exactness against
  the established CKE provider, verifies automatic dispatch at `M=63` and
  `M=64`, and requires a 1.05x same-host speedup.
- The performance target is registered in the nightly runner's `bench`
  category.
- Boundary checks at `M=63` and `M=64` compare every output byte with the
  independent-row provider. The 353- and 1K-token lanes remain outside the M4
  dispatch scope, so this patch does not change their established numerical
  trajectory.
