# Memory Tetris: Lifetime-Aware DRAM over NVMe

> **Canonical documentation:** [Memory Tetris: DRAM as a Tensor Cache over
> NVMe](../site/memory-tetris.html) ([published
> HTML](https://c-kernel-engine.github.io/C-Kernel-Engine/memory-tetris.html)).
> This Markdown file is a subordinate engineering and experiment-planning
> note. It must not replace or contradict the canonical HTML page.

Status: proposed architecture and hardware research plan. CKE does not yet
provide this storage-tier planner as a production feature.

## Definition

Memory Tetris is CKE's proposed lifetime-aware reuse of a fixed DRAM arena.
When a tensor reaches its declared last use, its arena slot may become the
final destination for a future tensor loaded asynchronously from NVMe or
another node. The runtime overlaps that transfer with current computation.

The intended abstraction is:

```text
DRAM = explicitly managed tensor working set
NVMe = capacity-oriented backing tier
CKE  = planner that schedules ownership, transfer and consumption
```

This is not SSD-backed virtual memory and does not claim that SSD has DRAM
latency or bandwidth.

## Example

Assume that four layer-weight groups fit in the streaming arena:

```text
time T0: [ L1 ][ L2 ][ L3 ][ L4 ][ scratch ][ persistent state ]
time T1: [free][ L2 ][ L3 ][ L4 ][ scratch ][ persistent state ]
time T2: [ L5 ][ L2 ][ L3 ][ L4 ][ scratch ][ persistent state ]
```

After `L1` completes and its slot is proven dead, an asynchronous read may
place `L5` directly into that slot. Other live tensors are not shifted. The
planner changes slot ownership and waits for the transfer-completion event
before allowing the `L5` consumer to run.

With double buffering, the desired timeline is:

```text
CPU:  [ compute L17 ][ compute L18 ][ compute L19 ]
I/O:       [ load L18 ][ load L19 ][ load L20 ]
```

Storage traffic is hidden only when the next tensor arrives before its first
use. For a 500 MB tensor, an idealized transfer takes approximately 71 ms at
7 GB/s and 36 ms at 14 GB/s. These are bandwidth calculations, not latency or
end-to-end performance guarantees.

## Required Contracts

Every tier-managed tensor needs machine-readable metadata:

```text
tensor_id
source_artifact and source_offset
arena_slot and final_address_offset
size and alignment
first_use and last_use
mutability and dirty state
storage tier and residency state
prefetch deadline and completion event
reuse distance and observed reuse count
checksum or artifact provenance
```

The planner must fail closed when:

- the previous slot owner is still live;
- a read or writable alias remains active;
- the slot is pinned by a kernel or transfer;
- the source extent, destination extent or alignment is invalid;
- provenance does not match the converted model manifest;
- a transfer is incomplete at the consumer dependency;
- mutable state would be evicted without an explicit writeback policy; or
- numerical execution would select a different arithmetic contract merely
  because the tensor came from another tier.

Weights are the safest first target because converted weights are immutable.
KV caches, recurrent state, gradients, optimizer state and checkpoints require
separate persistence, coherence and writeback contracts.

## I/O Reality

An ordinary buffered `read()` does not guarantee direct PCIe DMA into the
final userspace arena. The kernel page cache and an additional copy may be
involved. Candidate implementations must be measured independently:

1. buffered asynchronous reads;
2. `mmap()` plus controlled readahead and page residency;
3. aligned `O_DIRECT` reads into final arena slots; and
4. registered `io_uring` buffers where the kernel and filesystem support them.

The fastest mechanism is not assumed in advance. Alignment constraints,
queue depth, filesystem behavior, cache reuse and CPU overhead all affect the
result. The planner interface should describe a transfer source and completion
event without hard-coding one Linux I/O mechanism into model circuits.

## Dense and MoE Policies

Dense models provide a predictable sliding window. The initial implementation
should prefetch future layer weights in circuit order and retain only the
number of layers allowed by the DRAM budget.

MoE models require a cache policy as well as look-ahead:

```text
DRAM: common weights, hot experts, active experts, KV/state and staging slots
NVMe: cold experts, future layers and inactive immutable weights
```

Routing decisions determine immediate expert demand. Historical hit rate can
guide residency, but a missing selected expert is a hard dependency: compute
must wait for verified residency rather than silently substituting an expert.

The same transfer abstraction should later support a remote node as a source.
The planner can then compare resident reuse, local NVMe transfer, recomputation
and network transfer without encoding distributed behavior in individual
kernels.

## X-Ray and Performance Evidence

The feature is not successful merely because a larger model runs. X-Ray must
record, per tensor and per layer:

```text
requested bytes and completed bytes
source tier and destination slot
request, first-byte, completion and first-use timestamps
prefetch lead time
exposed wait at first use
achieved read bandwidth
slot reuse and residency duration
cache hit, miss and eviction reason
temporary-copy bytes
checksum/provenance result
```

Aggregate reports should include:

- model bytes resident in DRAM;
- model bytes served by NVMe per token or prompt;
- I/O time hidden behind compute;
- exposed storage stall time;
- useful bandwidth versus device baseline;
- hot-expert hit rate for MoE;
- page-cache and direct-I/O mode; and
- token parity against the fully resident baseline.

Numerical certification compares the tiered runtime with the same generated
runtime and provider schedule using fully resident weights. Token parity alone
is insufficient: selected layer/checkpoint outputs and artifact hashes must
also match according to their existing numerical contracts.

## Implementation Sequence

1. Add immutable tensor source extents and storage-tier metadata to the model
   manifest and validated IR.
2. Add a fixed, aligned streaming-slot allocator alongside persistent and
   scratch arena classes.
3. Implement synchronous final-slot reads as the correctness baseline.
4. Add asynchronous prefetch and explicit completion dependencies.
5. Certify deterministic dense-layer sliding windows with one and two slots.
6. Add double/triple buffering and tune look-ahead from measured kernel time.
7. Add X-Ray transfer and stall reporting.
8. Add MoE expert residency with deterministic cache-policy fixtures.
9. Add mutable-state tiers only after explicit writeback and recovery design.
10. Generalize the transfer source to remote-node tensors for distributed
    execution.

## Hardware Test Plan

Start with the installed P3 Gen4 NVMe and a fully resident model baseline.
Record the exact CPU, memory, filesystem, mount options, SSD firmware,
temperature and free-space state for every run.

### Device qualification

- Measure direct and buffered sequential read bandwidth at multiple block
  sizes and queue depths.
- Measure cold-cache and warm-cache behavior separately.
- Measure sustained bandwidth after the drive leaves its SLC cache.
- Record latency distributions, CPU overhead and thermal throttling.
- Do not publish the provisional approximately 6.9 GB/s P3 result as a CKE
  benchmark until the command, artifact size and cache state are recorded.

### Runtime gates

- Fully resident versus one-slot synchronous streaming.
- One-slot versus double- and triple-buffered asynchronous streaming.
- Layer sizes from 64 MB through 1 GB.
- Compute windows shorter than, equal to and longer than transfer time.
- Identical numerical output with 1, 8 and production thread counts.
- Forced short read, stale artifact, checksum failure and slot-overlap tests.
- Dense model first, then a public MoE model with controlled routing.

### Promotion criteria

- No undeclared staging copy.
- No read from an incomplete or stale slot.
- Fully resident numerical contract remains unchanged.
- X-Ray accounts for all tiered bytes and exposed wait.
- Asynchronous mode beats the synchronous baseline reproducibly.
- The selected model exceeds available DRAM or demonstrates a measured
  residency benefit; otherwise storage streaming adds complexity without
  value.

Gen5 should be evaluated only after Gen4 establishes the implementation's
ability to overlap I/O. A nominal 14 GB/s-class device cannot fix an incorrect
schedule, insufficient look-ahead, small random reads or excessive copies.

## Canonical Documentation Contract

The canonical [Memory Tetris HTML page](../site/memory-tetris.html) presents
this work as research in progress, not current production support. Future
documentation changes must update the HTML source and generated page first.
This note may retain lower-level implementation details and experiment plans,
but it must link back to the HTML representation and remain subordinate to it.

The canonical page includes:

- a time-versus-arena diagram showing a dead layer slot becoming the direct
  destination for a future layer;
- an overlapped CPU/I/O timeline with hidden and exposed transfer regions;
- a dense sliding-window example and a separate hot/cold MoE expert example;
- an I/O-path comparison for buffered read, `mmap`, `O_DIRECT` and registered
  `io_uring` buffers;
- an X-Ray panel mockup listing bytes, bandwidth, prefetch lead and exposed
  stall per tensor; and
- a hardware-results table that remains empty or marked unverified until the
  reproducible Gen4 experiments above are complete.

Avoid claims that SSD is RAM, that PCIe bandwidth equals application
bandwidth, or that DMA always lands directly in a CKE arena. The defensible
story is that CKE's explicit tensor lifetimes may let it manage a predictable
DRAM working window more effectively than demand paging, and the hardware
gates will determine how much I/O can actually be hidden.
