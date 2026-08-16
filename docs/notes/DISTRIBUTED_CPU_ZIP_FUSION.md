# Distributed CPU: Zip Fusion Beyond Tensor Parallelism

> **Canonical documentation:** [Distributed CPU: Zip Fusion Beyond Tensor
> Parallelism](../site/distributed-cpu.html) ([published
> HTML](https://c-kernel-engine.github.io/C-Kernel-Engine/distributed-cpu.html)).
> This Markdown file is a subordinate engineering and experiment-planning
> note. It must not replace or contradict the canonical HTML page.

Status: research design. CKE does not yet provide distributed execution as a
production feature. Nothing here is a measured result, a supported
configuration, or a committed performance claim. Every quantitative figure in
this note is idealized arithmetic unless explicitly marked otherwise.

## Thesis

CKE's distributed-execution research direction is **whole-graph algebraic
scheduling on CPU-first economics**. Instead of exposing named parallel
strategies (TP, CP, EP, PP) and letting the user compose them, the scheduler
asks, per graph region:

```text
For this exact graph region,
what representation lets N CPU nodes execute
the longest independent local path
before any global synchronization?
```

The unit of distribution is the whole quantized operator subgraph, including
work that most frameworks treat as implementation detail: packed-weight
unpack, scale decode, quantized dot products, local accumulation, SwiGLU,
RMSNorm partial statistics, attention-head work, output projection, expert
routing, and Memory-Tetris residency.

The goal is to make a cluster behave as **one pool of compute that re-shapes
per graph region**, not as N machines each holding 1/N of the model.

## Prior Art, Candidly

Head-shard and context-shard attention are not new, and CKE did not invent
them:

- **Megatron Core** combines tensor, pipeline, context and expert parallelism
  and treats context parallelism as a first-class way to partition long
  sequences.
- **DeepSpeed** provides pipeline, tensor and expert parallelism; its MoE API
  can distribute experts differently per MoE layer.
- **vLLM** supports combinations such as data-parallel attention with expert-
  or tensor-parallel MoE layers.
- **DeepSpeed-Ulysses** partitions along sequence length and uses all-to-all
  communication for attention.
- **Unified Sequence Parallelism** combines Ulysses with Ring Attention.
- **LoongTrain** explicitly combines head parallelism and context parallelism
  as a 2-D attention decomposition.
- **ATTENTION2D** similarly partitions attention along query and key/value
  dimensions.

The potentially distinctive claim is narrower than any of these: on CPU-first
economics (abundant DRAM, measured heterogeneous nodes, explicit storage
tiers), use whole-graph algebra to minimize global synchronization boundaries
per layer — and prove it experimentally. The value CKE could add is the
combination, not any single decomposition.

## The Zip Schedule

Per layer, each node runs a fused 1-D lane. For a two-node, Qwen-like
attention block:

```text
                  replicated hidden state x
                         │
             ┌───────────┴───────────┐
          Node A                  Node B
       unpack W_A               unpack W_B
       QKV shard                QKV shard
       heads 0..H/2             heads H/2..H
       RoPE / QK norm           RoPE / QK norm
       attention                attention
       local Wo shard           local Wo shard
       partial y_A              partial y_B
             └──────────┬────────────┘
                 small reduction
                        │
                 next hidden state
```

Key points:

- Heads are **not** concatenated into one giant intermediate tensor. Each node
  applies its local output-projection shard to its own attention output and
  produces a hidden-state-sized partial. The only communication boundary is
  the final reduction `y = partial_A + partial_B`.
- For attention at long context, the decomposition **unzips** from 1-D to
  2-D: head groups on one axis, context shards on the other. Example: 6 nodes
  = 3 context shards × 2 head groups; each node computes large local
  attention tiles over 1/3 of the context and 1/2 of the heads.
- Cross-context merging uses **online-softmax sufficient statistics** —
  per-partition local max, local normalization sum, and local weighted
  output — never the raw attention matrix. The network sees three small
  vectors per query block instead of a 1M × 1M score matrix.
- After attention the schedule **zips back** to 1-D for the MLP lane:
  gate/up shard → local SwiGLU → local down-projection partial → one merge.

Payload sizes at the boundaries (idealized arithmetic):

```text
hidden state, 8192-dim BF16, 1 token:   8192 × 2 B  ≈  16 KB
hidden state, 32-token batch:                          ≈ 512 KB
online-softmax stats per query block:    local max + local sum
                                         + local weighted output
```

GQA note: modern Qwen-like models may have, conceptually, 32 query heads
against 8 KV heads. Query-head groups must be partitioned so the needed KV
heads are local or cheaply replicated. Because CPU RAM is plentiful,
replicating K/V-head metadata or even some KV data may be preferable to
forcing communication.

## The Metric: Global Synchronization Boundaries per Layer

The useful metric is **global synchronization boundaries per layer**, not
tensor-parallel degree — and the honest baseline is stronger than a
strawman. Established TP schedules already pair column-parallel with
row-parallel linears, so a conventional layer pays roughly **two
collectives**, not one per operator:

```text
QKV GEMM (column-parallel) → attention (head-local) → output GEMM (row-parallel)
→ collective → MLP gate/up (column-parallel) → SwiGLU → MLP down (row-parallel)
→ collective
```

A hyper-fused zip schedule aims for one merge per fused region:

```text
NODE A                         NODE B
------------------------------------------------
unpack W_A                     unpack W_B
QKV shard                      QKV shard
local heads                    local heads
attention                      attention
output projection partial      output projection partial
local reduction                local reduction
          \                   /
           tiny global merge            ← boundary 1

gate/up shard                  gate/up shard
SwiGLU local                   SwiGLU local
down projection partial        down projection partial
          \                   /
             one merge                  ← boundary 2
```

"We support TP=2" says nothing about how often nodes stop to agree. The real
contrast with paired TP is not the nominal collective count but the
**unresolved span**: paired TP keeps activations sharded between its two
boundaries, while the zip schedule pushes the whole operator subgraph —
quantized unpack, scale decode, local accumulation, RoPE / QK norm, SwiGLU,
norm statistics, projection partials — into the shard layout, and asks
experimentally whether `Shard → Shard → Shard → PartialSum` can stay
unresolved for longer portions of the graph than existing schedules allow.
The interesting output of the experimental program is a
**topology-versus-context map**, not a single tok/s figure:

```text
small context:        1 CPU node
medium:               TP=2
long:                 head parallel + TP
very long (128K–1M):  head parallel × context parallel
                      (+ perhaps TP inside projections)
```

## Communication Must Be Measured

The idealized serialization of 512 KB at a nominal 400 Gb/s (≈ 50 GB/s) link
is about 10 µs. That figure is a **division, not a measurement**. Real
end-to-end synchronization adds RDMA/collective latency, software overhead,
PCIe traversal, memory access and barrier costs.

The thesis stands on the **ratio**, not the absolute number: if each node
performs milliseconds of useful attention/GEMM work between boundaries,
hundreds of KB of communication is not scary — and at 1M context the
compute/communication ratio can become particularly favorable. But this is a
research hypothesis. It must be demonstrated on real NICs and a real switch,
not assumed from link-sheet bandwidth.

## CPU-First Economics

GPU frameworks shard because HBM is scarce; state often *cannot* be
replicated. CPU nodes can carry 150 GB, 256 GB, 512 GB or eventually TBs of
DRAM, so CKE can deliberately **spend memory to eliminate communication**:

- replicate the hidden input;
- replicate norm parameters;
- replicate routing metadata;
- keep multiple packed layouts;
- cache hot experts on several nodes.

This inverts the usual optimization objective. Memory-Tetris residency adds
another dimension the mainstream parallelism literature usually does not
center: tensor residency itself is part of scheduling. Per tensor, the
planner could choose among compute-from-resident, fetch from local NVMe,
fetch from remote DRAM over RDMA, route the activation to a remote resident
expert, recompute, or replicate for future reuse. See
[MEMORY_TETRIS_NVME_DESIGN.md](MEMORY_TETRIS_NVME_DESIGN.md).

## Heterogeneous Topology

Megatron-style systems generally assume fairly regular accelerator groups.
CKE's lab fleet is explicitly heterogeneous: a P3-class Intel i7-14700T node
(active), a Ryzen 9950X3D node (planned), and Xeon AMX hardware (future).
Shards should be assigned **proportionally to measured throughput** — e.g.
Node A takes 30% of a tensor shard and Node B 70% because X-Ray measured them
that way — rather than assuming equal ranks. The target is distributed graph
compilation over heterogeneous CPU resources, with 200/400G RDMA-class links
as the fabric assumption to be validated.

## Quantitative Sizing Model (THEORETICAL)

The HTML page carries the full worked version with figures; this note records
the model itself. Everything in this section is an **OPTIMISTIC LOWER BOUND**
— measured time will be ≥ predicted — and exists to be validated or
falsified by the validation ladder, not to advertise a number.

**Shard fractions.** Let `P_i` be node `i`'s X-Ray-measured CKE throughput
for the relevant region — and note `P_i` is really `P_i(r, phase, dtype)`:
never nominal FLOPS, and not one constant per node. The same Xeon/Ryzen
ratio does not apply everywhere; that is exactly where this differs from a
static `--tensor-split 0.2,0.2,0.6` — the shard size is operator- and
phase-dependent:

```text
s_i(r) = P_i(r) / Σ_j P_j(r)

P_i(decode)    ≈ packed-weight stream bandwidth (the model's quant format,
                 e.g. an MXFP4-class format for the target model class)
P_i(prefill)   ≈ measured GEMM throughput (ISA-dependent: AVX2 / AVX-512 / AMX)
P_i(attention) ≈ f(BW_KV, F_attention) at long context
```

Decode is vector × matrix: every packed weight byte is read once and used
once, so the region is bandwidth dominated. The arithmetic intensity is
format-dependent, not a universal constant:

```text
AI_decode ≈ 2 × P_active / B_packed weights
FP32: 0.5 FLOP/B · BF16: 1 FLOP/B · ideal 4-bit: ≈ 4 FLOP/B
(e.g. 100B params ≈ 50 GB packed → 200 GFLOP / 50 GB ≈ 4 FLOP/B,
 before scale/block metadata, unpack overhead and activation traffic)
```

Prefill is token-matrix × weight-matrix: each byte is reused across T
tokens, so the region is GEMM/compute dominated. The same nodes can
therefore deserve different shard fractions — and even different
topologies — per phase.

**Fresh bytes.** The bandwidth variable is `B_fresh`, not total weight
bytes:

```text
B_fresh = B_required − B_cache hits/reuse
```

This is why a large-cache node can outperform what its nominal two-channel
DRAM bandwidth suggests on some regions. Execution model: compute the
current tile in parallel with fetching the next tile.

**Region and token time.** For a region with `B_fresh,r` fresh bytes and
`F_r` FLOPs, with `C_r(N)` the measured, node-count-dependent collective /
synchronization time:

```text
T_r     = max( B_fresh,r / Σ_i BW_i,r , F_r / Σ_i F_i,r ) + C_r(N)

Headline equation — the whole hypothesis in one line:
T_token = Σ_r [ max( B_fresh,r / Σ_i BW_i,r , F_r / Σ_i F_i,r ) + C_r(N) ]
tok/s   = 1000 / T_token(ms)
```

The `max()` ignores overlap penalties, stragglers, collective algorithms and
software overhead — hence the lower-bound label.

**Worked example (parameterized, no real-model attribution).** 50 tok/s
target → 20 ms/token; 93 layers → ≈ 215 µs/layer budget. Hidden 8192-dim
BF16 → 16 KB per hidden-state reduction; 2 reductions/layer → 186 × 16 KB
≈ 3 MB/token before collective-algorithm overhead. Local stream budget per
layer at B_i = 80 GB/s: 80 GB/s × 215 µs ≈ 17.2 MB — cache-resident current
tile plus DRAM streaming the next tile, not "fit the model in cache". A
100 tok/s variant halves every budget and doubles the sync share, making it
a stress test of the communication path.

**The latency arithmetic (why 3 MB is not the story).** The reductions do
not overlap each other; each layer boundary waits for its collective, so:

```text
T_collective/token = 186 × L_collective

L =  2 µs → 0.372 ms  ( 1.9% of 20 ms)
L =  5 µs → 0.93  ms  ( 4.7%)
L = 10 µs → 1.86  ms  ( 9.3%)
L = 20 µs → 3.72  ms  (18.6%)
L = 50 µs → 9.3   ms  (46.5%)
At 100 tok/s (10 ms/token) every share doubles.
```

The experimental gate: **can CKE get its real small-vector reduction into
the latency regime the target token budget requires?** Measured L answers
this, not the link sheet.

**1M context: decode ≠ prefill.** Over the same heads × context grid,
decode has one query (Q: 1×d against K,V: L×d); each context shard returns
per-head statistics (m, l, o) — bytes-to-KB, independent of L. Prefill runs
T simultaneous queries, so the sufficient statistics carry the query
dimension and the merge payload scales with T — tiled/streamed, priced like
a GEMM-shaped region. Any 1M-context claim that does not name its phase is
incomplete.

**Falsifiability rule (hypothetical example numbers).** If one node decodes
at 0.20 tok/s and two deliver 0.38, the model is supported. If two deliver
0.21, an assumption is falsified, and the per-region breakdown shows which
term — bandwidth, GEMM, or sync — was wrong. The headline question is not
"CPU vs GPU" but: **can transformer latency scale approximately with
aggregate calibrated CPU capability until synchronization becomes the
ceiling?**

**Validation ladder evidence pack (published at every rung):** predicted vs
measured region time; predicted vs measured tok/s; collective-time fraction;
scaling efficiency; effective stream GB/s and prefill GEMM throughput;
watts; hardware cost per added tok/s. Rungs: one node (calibration) → two
identical nodes (purest scaling test) → heterogeneous pair (proportional
shards) → larger fleet (find the ceiling).

## Open Questions

- What is the real end-to-end cost of one hidden-state reduction on the lab
  fabric (RDMA setup, collective latency, barriers), and how does it scale
  with node count?
- At what context length does unzipping to 2-D attention actually beat a
  1-D head-parallel schedule?
- How expensive is GQA KV replication in practice, and when does it beat
  communicating KV?
- How do stragglers on heterogeneous nodes interact with barrier-heavy
  schedules, and does proportional sharding actually equalize lane time?
- Can switch congestion at 6+ nodes keep the compute/communication ratio
  favorable, or does the fabric become the boundary condition?
- Which Memory-Tetris residency choices (local DRAM, remote DRAM, NVMe,
  recompute, replicate) survive contact with measured latencies?

## Experimental Plan and Promotion Gates

Ordered experiments, each with a promotion gate:

1. **Single-node baseline.** Fully resident model, measured kernel times per
   graph region. Gate: reproducible baseline numbers recorded to CKE's
   evidence standard.
2. **Two-node TP reference.** A conventional TP=2 implementation, measuring
   the fraction of wall time spent in collectives. Gate: collective-time
   fraction measured and published, whatever the number is.
3. **Hyper-fused schedule.** Same two nodes, fused 1-D lanes, measuring
   synchronization boundaries per layer and total sync time. Gate: fewer
   boundaries and lower collective-time fraction than experiment 2, with
   identical numerics.
4. **Head × context 2-D at long context.** 6-node 3×2 grid with
   online-softmax merging. Gate: favorable ratio versus 1-D head-parallel at
   the same context, measured not estimated.
5. **Numerical parity.** Every distributed schedule must match the
   single-node baseline under the existing zero-tolerance numerical contract
   rules. Token parity alone is insufficient; selected layer/checkpoint
   outputs must also match. Gate: parity holds across thread counts, node
   counts and schedules.

**Hypothesis to test (not a result):** a fused schedule might cut collective
time from roughly 25% of wall time (a plausible conventional-TP figure on
this class of hardware) to roughly 5–10%. These numbers are the hypothesis
the experiments are designed to confirm or refute.

**Failure modes to measure:** barrier stalls, stragglers on heterogeneous
nodes, GQA KV replication cost, and switch congestion.

## Canonical Documentation Contract

The canonical [Distributed CPU HTML page](../site/distributed-cpu.html)
presents this work as a research design, not current production support.
Future documentation changes must update the HTML source and generated page
first. This note may retain lower-level experiment plans, but it must link
back to the HTML representation and remain subordinate to it.

Avoid claiming that CKE invented tensor/head/context parallelism (it did
not), that idealized bandwidth divisions are measurements, or that any
distributed schedule is implemented. The defensible story is that whole-graph
algebraic scheduling on CPU-first economics may reduce global synchronization
boundaries per layer, and the hardware gates will determine whether the
compute/communication ratio actually holds.
