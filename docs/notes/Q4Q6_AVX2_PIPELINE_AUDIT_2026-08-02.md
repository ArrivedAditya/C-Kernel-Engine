# Q4_K/Q6_K AVX2 Pipeline Audit

Date: 2026-08-02

This audit separates quantized dot-product arithmetic from matrix scheduling. It
uses CKE's global persistent threadpool; the benchmark does not create another
pool. The llama.cpp comparison is loaded from the same-host GGML kernel-test
library and every reported leaf result must be bit-exact.

## Reproduction

```bash
make bench-q4q6-pipeline-quick

CK_LLAMA_KERNEL_TEST_LIB="$PWD/llama.cpp/libggml_kernel_test.so" \
CK_LLAMA_GGML_CPU_LIB="$PWD/llama.cpp/build/bin/libggml-cpu.so.0" \
CK_NUM_THREADS=8 taskset -c 0,2,4,6,8,10,12,14 \
  build/bench_q4q6_pipeline --leaf-iters 200000 --jobs 16384 --pool-iters 5
```

Use CPUs `16-27` with 12 threads for the E-core lane and CPUs
`0,2,4,6,8,10,12,14,16-27` with 20 threads for the mixed physical-core lane on
the audited host. CPU numbering is machine-specific and must not be copied to a
different topology without checking sysfs or `lscpu -e`.

## Leaf Results

K is 4096, or 16 quantization blocks. Measurements below use one P-core.

| Exact dot provider | ns/call | Relative to llama.cpp |
| --- | ---: | ---: |
| CKE Q4_K x Q8_K, before | 183.66 | 1.87x slower |
| CKE Q4_K x Q8_K, after | 106.78 | 1.09x slower |
| llama.cpp Q4_K x Q8_K | 97.71 | 1.00x |
| CKE Q6_K x Q8_K | 121.83 | 1.34x faster |
| llama.cpp Q6_K x Q8_K | 163.34 | 1.00x |

The old Q4 dispatcher called `getenv` for every output row. It retired about
3,392 instructions and 330 branches per call. Caching that diagnostic switch
and decoding packed scales with AVX2 reduced the result to about 1,696
instructions and 33 branches per call. llama.cpp retired about 1,609
instructions and 24 branches. Cache misses were negligible in both cases.

The remaining Q4 leaf gap is therefore instruction and shuffle efficiency, not
external memory bandwidth. Q6 already beats the compared llama.cpp leaf and
must not be rewritten merely to make Q4 and Q6 look structurally alike.

## Topology Results

For 16,384 independent K=4096 leaf jobs, static scheduling produced these best
wall times after the Q4 fix:

| Lane | Q4 static | Q6 static |
| --- | ---: | ---: |
| 8 P-cores | 0.376 ms | 0.249 ms |
| 12 E-cores | 0.464 ms | 0.396 ms |
| 8 P + 12 E physical cores | 0.421 ms | 0.240 ms |

CKE's threadpool callback contract assigns work statically by worker index.
It does not currently let a fast worker pull work from a slow worker. In the
mixed lane, P-cores completed equal-sized Q4 partitions in roughly 0.09 ms and
E-cores in roughly 0.20 ms.

The benchmark-only dynamic queue proved that heterogeneous workers can consume
different job counts, but an atomic claim every four dot products cost more
than it saved. Dynamic production scheduling should use coarse tiles or
topology-weighted static ranges. Per-leaf atomics must not be promoted.

### Coarse dynamic scheduling

A sustained 262,144-output run used 256-output claims. Homogeneous lanes were
neutral, while the mixed lane improved substantially:

| Lane | Q4 dynamic/static | Q6 dynamic/static |
| --- | ---: | ---: |
| 8 P-cores | 0.98x | 1.00x |
| 12 E-cores | 0.99x | 0.99x |
| 8 P + 12 E physical cores | 1.52x | 1.55x |

On the mixed lane, worker-time spread fell from 1.7--2.2 ms to 0.08--0.13 ms.
Every output remained bit-exact because each job owns one output value and its
K reduction order does not change.

VTune 2026 threading analysis independently confirmed the sustained Q4 result
with 16,777,216 outputs and nine measured dispatches:

| Schedule | Best dispatch | Effective CPU utilization | Poor-utilization wait |
| --- | ---: | ---: | ---: |
| Equal static | 187.6 ms | 47% | 11.47 CPU-seconds |
| Dynamic, 256 outputs/claim | 156.3 ms | 64% | 0.96 CPU-seconds |

Static P-core workers waited as much as 95 ms at the dispatch boundary while
slower workers completed equal partitions. Dynamic workers consumed roughly
1.28--1.35 million jobs on P-cores and 0.48--0.52 million on E-cores, then
finished within approximately 0.1 ms of dispatch return.

The profiling results are in `build/vtune-q4-static-*` and
`build/vtune-q4-dynamic-*`. Those generated directories are local evidence and
are not committed.

The same coarse ownership model is available through the production persistent
pool for the independent Q4 packed-x8, Q6 2D prefill, and Q8 row providers.
It is selected through the typed engine ABI and command-line policy:

```bash
version/v8/scripts/cks-v8-run run <model> --gemm-schedule auto
build/ck-cli-v8 --model <name> --gemm-schedule static
```

`auto` is the default and currently uses dynamic claims for providers whose
jobs write independent output tiles. `static` is the reproducible comparison
override and `dynamic` explicitly requests the same balanced policy. Grain
sizes remain provider-owned constants rather than user environment settings.
Split attention and other ordered reductions do not use this scheduler.

On the audited Qwen3-VL OCR image, the first-token mixed-prefill stage measured
24.71 seconds static and 22.09 seconds dynamic with identical reported logits,
an initial 10.6% improvement. A following static run took 31.19 seconds,
demonstrating substantial thermal/load variance. This is evidence that the
production route works and can help, but it is not sufficient to enable it by
default or claim a stable end-to-end ratio in isolation. The later alternating
whole-corpus matrix below provides the promotion evidence.

A subsequent private-corpus certification ran the complete native CLI, Python
CKE, and pinned llama.cpp paths with a 128-token ceiling. The image reached its
natural stop after 12 generated tokens; all 12 pre-EOS tokens matched exactly
across all three paths. Under that sustained hot load, encoder execution was
17.12 seconds and dynamic mixed prefill was 20.25 seconds. This confirms full
end-to-end control flow and persistent decode after dynamic prefill, but it is
not a forced 128-step decode stress test.

A three-image, one-token prefill comparison then ran all static cases followed
by all dynamic cases. Dynamic mixed prefill was 5.1% faster in aggregate and
10.8% faster by median. Per-image speedups were 0.90x, 1.11x, and 1.14x, so the
first image regressed. Encoder plus mixed-prefill time was effectively neutral
(0.998x) because the later dynamic batch also had slower encoder times, even
though this scheduler does not control the encoder's dominant Q8 work. Outputs
matched for all three cases. The run-order thermal bias means this is not a
whole-corpus speed claim; promotion needs paired alternating order and
temperature telemetry.

The final alternating 40-image matrix removed that run-order bias. All 80 CKE
runs completed, and static versus dynamic generated text matched on all 40
images. Dynamic steady compute averaged 48.95 seconds/image versus 53.51 for
static, an 8.4% mean improvement; the paired median improvement was 9.0%.
Dynamic encoder time averaged 21.84 seconds versus 22.87, while mixed prefill
averaged 27.09 seconds versus 30.63. The latest same-host llama.cpp lane averaged
47.65 seconds of internal compute, placing dynamic CKE about 2.7% behind for
this one-token OCR workload. Process-wall ratios are not directly equivalent
because the runtimes account for startup differently.

Small sub-millisecond shapes remain noisy and sometimes regress. The production
default is consequently scoped: `auto` selects dynamic claiming only inside the
audited independent-output GEMM providers. Ordered reductions, split attention,
and providers outside this audit retain their existing deterministic schedule.
`--gemm-schedule static` remains available for diagnosis and comparative gates.

## Production Provider Results

The first leaf comparison was not the production Qwen3-VL boundary. The
generated decoder invokes
`gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch`, which selects the
packed VNNI x8/4M implementation, and invokes the Q6 2D/M4 provider for
MLP-down. The vision encoder separately invokes
`gemm_nt_q8_0_q8_0_contract`. A one-row `vec_dot` result cannot be used as an
end-to-end speed estimate for any of these multi-row providers.

Measured with 20 physical P/E workers on the OCR mixed-prefill shape:

| Production boundary | Shape | Static | Dynamic | Speedup |
| --- | --- | ---: | ---: | ---: |
| Q4_K x Q8_K packed VNNI | 1035 x 4096 x 4096 | 34.48 ms | 22.05 ms | 1.56x |
| Q6_K x Q8_K 2D/M4 | 1035 x 4096 x 11008 | 144.52 ms | 118.72 ms | 1.22x |

The Q8_0 x Q8_0 vision provider now uses the same persistent dynamic scheduler
for independent output rows. Static-reference parity passes for small, medium,
and MLP test shapes. In full encoder measurements, a first cool dynamic run
improved encoder time from 18.80 to 17.07 seconds (1.10x), while a repeated hot
run took 18.83 seconds and lost that gain. Q8 therefore remains experimental;
there is no stable Q8 speed claim yet.

A profile-enabled same-image decoder comparison measured mixed prefill at
24.46 seconds static and 20.27 seconds dynamic (1.21x). Gate/up improved from
8.28 to 6.80 seconds, MLP-down from 5.36 to 4.04 seconds, and Q/K/V/output
projection groups improved by roughly 1.38x to 1.74x. These are production
operation timings, not substituted leaf timings.

## VTune and Advisor Findings

VTune memory-access collection on the actual Q4 production provider measured:

| Schedule | Wall time | Average DRAM bandwidth | DRAM-bound time |
| --- | ---: | ---: | ---: |
| Static | 3.825 s | 2.40 GB/s | 0.2% |
| Dynamic | 2.487 s | 2.98 GB/s | 0.1% |

The platform STREAM-like maximum used by the analysis is approximately
39 GB/s. Dynamic scheduling increases useful work issuance and bandwidth, but
the provider still consumes less than 8% of measured platform bandwidth. It is
not close to becoming DRAM-bandwidth-bound.

Advisor trip-count and integer-operation analysis measured approximately
452 GINTOP/s for the dynamic Q4 provider. In its main
`gemm_q4_packed_vnni_x8_q8k_4m_job`, about 21.18 CPU-seconds remained in scalar
outer/control work and 19.82 CPU-seconds in the AVX2 body. Dynamic queue claims
accounted for only about 0.19 CPU-seconds. The next optimization target is
therefore the job itself: scale decoding, pointer/address generation, output
tile reuse, and reduction/control instructions around VNNI. Reducing queue
overhead or adding DRAM prefetch alone cannot close the remaining gap.

On the i7-14700T host, turbostat also showed the power tradeoff. Dynamic Q4
raised aggregate busy time from 45.6% to 68.3%, while average busy frequency
fell from 3.65 to 3.12 GHz. The kernel still improved by about 1.57x, but under
a sustained full-model load the package power limit can convert higher E-core
occupancy into lower P-core frequency. Alternating hot/cool E2E runs are
required before default promotion.

All dynamic jobs own disjoint output rows or tiles and preserve each row's K
reduction order. The Q8 dispatcher passed static-reference parity, the dynamic
threadpool suite passed all production-provider cases, and the llama-backed
FP16 attention matrix remained 41/41 exact at 1, 16, 20, and 24 threads.
After adding Q8 scheduling, a fresh private image gate also passed the complete
native CLI/Python CKE/pinned llama.cpp comparison: 12/12 pre-EOS greedy tokens,
no first divergence, GCC runtime provenance, and llama.cpp commit
`f3e182816421c648188b5eab269853bf1531d950`.
After removing nested OpenMP from the Q8 outer-pool path, the same three-way
gate was repeated for the first generated token and remained exact.

## Interpretation

The earlier full-model finding that Q4 was roughly 2.1x behind a reference was
serious, but it did not imply that the CPU lacked compute or memory bandwidth.
The isolated benchmark found a hot-path diagnostic lookup and inefficient scale
unpacking. After removing those costs, the exact Q4 leaf is within about 9% of
llama.cpp on the audited P-core, while production prefill still depends on
multi-row packing, tile reuse, and thread partitioning.

The leaf benchmark remains a provider diagnostic rather than an end-to-end
claim. Default promotion instead rests on the production-kernel parity gates,
the full numerical regression lane, and the alternating 40-image same-host
matrix above. That matrix covers the actual Qwen3-VL encoder and mixed-prefill
path while controlling run-order bias; it is the evidence for making `auto`
the default.
