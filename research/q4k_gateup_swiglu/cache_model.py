#!/usr/bin/env python3
"""
Deterministic cache model for Q4_K gate_up + SwiGLU research kernels.

This does not try to learn from benchmark timing. It derives a conservative
starting tile/thread plan from the CPU cache hierarchy and the M/D/K shape.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

QK_K = 256
Q4K_BLOCK_BYTES = 144
Q8K_BLOCK_BYTES = 292  # sizeof(block_q8_K): d + qs[256] + bsums[16] + padding/alignment in CK builds
CACHE_LINE = 64


def read_int(path: Path, default: int) -> int:
    try:
        s = path.read_text().strip().upper()
    except OSError:
        return default
    mult = 1
    if s.endswith("K"):
        mult = 1024
        s = s[:-1]
    elif s.endswith("M"):
        mult = 1024 * 1024
        s = s[:-1]
    try:
        return int(s) * mult
    except ValueError:
        return default


def cache_info(cpu: int = 0) -> dict[str, int]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache")
    out = {"l1d": 48 * 1024, "l2": 2 * 1024 * 1024, "l3": 60 * 1024 * 1024, "line": 64}
    if not base.exists():
        return out
    for idx in base.glob("index*"):
        try:
            level = idx.joinpath("level").read_text().strip()
            typ = idx.joinpath("type").read_text().strip()
        except OSError:
            continue
        size = read_int(idx / "size", 0)
        line = read_int(idx / "coherency_line_size", CACHE_LINE)
        if level == "1" and typ == "Data":
            out["l1d"] = size
        elif level == "2" and typ == "Unified":
            out["l2"] = size
        elif level == "3" and typ == "Unified":
            out["l3"] = size
        out["line"] = line or out["line"]
    return out


def physical_cores() -> int:
    cores = set()
    try:
        for cpu_dir in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
            topo = cpu_dir / "topology"
            pkg = (topo / "physical_package_id").read_text().strip()
            core = (topo / "core_id").read_text().strip()
            cores.add((pkg, core))
    except OSError:
        return 1
    return max(1, len(cores))


def floor_multiple(x: int, m: int) -> int:
    return max(m, (x // m) * m)


def choose_tile_m(l1d: int, l2: int, k: int, n_lanes: int, max_tile_m: int) -> int:
    row_bytes = (k // QK_K) * Q4K_BLOCK_BYTES
    paired_panel = 2 * n_lanes * row_bytes
    # Keep the paired gate/up panel comfortably in L2. 1/8 L2 is conservative
    # enough to leave room for Q8 rows, stacks, output lines, and prefetch noise.
    l2_panel_budget = l2 // 8
    if paired_panel > l2_panel_budget:
        # Reduce lanes in future kernels; for this x16 experiment, keep tile_m small.
        return 1

    q8_row = (k // QK_K) * Q8K_BLOCK_BYTES
    # Let Q8 token rows occupy up to 1/4 L2. They are reused across the current
    # panel and should not evict the whole weight panel.
    max_by_l2 = max(1, (l2 // 4) // max(1, q8_row))
    # L1 cannot hold the full paired x16 panel; use it for active block slices.
    # Keep output writes for tile_m*n_lanes under about half L1d.
    max_by_l1_out = max(1, (l1d // 2) // max(1, n_lanes * 4))
    return max(1, min(max_tile_m, max_by_l2, max_by_l1_out))


def choose_threads(l2: int, l3: int, physical: int, weight_bytes: int) -> int:
    # Deterministic cache rule:
    # Each active core should own enough of the weight stream to make progress,
    # but not so many cores that the per-core share is far below L2 and all cores
    # fight over the same L3-sized matrix. Target a per-core share near 1.25x L2.
    target_share = max(1, int(1.25 * l2))
    by_share = max(1, (weight_bytes + target_share - 1) // target_share)

    # L3 is shared. Keep total active cores below physical cores and avoid SMT by
    # construction. Leave some L3 headroom for activations/output/runtime.
    l3_usable = int(0.85 * l3)
    if weight_bytes > l3_usable:
        # If weights exceed usable L3, avoid full-core saturation by reserving
        # about 1/6 of physical cores for lower contention.
        by_l3_pressure = max(1, physical - max(1, physical // 6))
    else:
        by_l3_pressure = physical

    threads = min(physical, max(1, by_share), by_l3_pressure)
    # Keep thread count aligned to groups of 4 for balanced static work chunks.
    if threads >= 8:
        threads = floor_multiple(threads, 4)
    return max(1, threads)


def model(m: int, d: int, k: int, lanes: int, max_tile_m: int) -> dict[str, object]:
    c = cache_info()
    phys = physical_cores()
    row_bytes = (k // QK_K) * Q4K_BLOCK_BYTES
    q8_row = (k // QK_K) * Q8K_BLOCK_BYTES
    weight_bytes = 2 * d * row_bytes
    packed_x16_bytes = ((2 * d + lanes - 1) // lanes) * (k // QK_K) * (2 + 2 + 8 + 8 + QK_K // 2) * lanes
    paired_panel = 2 * lanes * row_bytes
    tile_m = choose_tile_m(c["l1d"], c["l2"], k, lanes, max_tile_m)
    threads = choose_threads(c["l2"], c["l3"], phys, weight_bytes)
    return {
        "shape": {"M": m, "D": d, "K": k, "lanes": lanes},
        "cache": {"L1d": c["l1d"], "L2_per_core": c["l2"], "L3_shared": c["l3"], "line": c["line"], "physical_cores": phys},
        "bytes": {
            "q4k_row": row_bytes,
            "gate_up_weights": weight_bytes,
            "packed_x16_estimate": packed_x16_bytes,
            "q8_row": q8_row,
            "q8_activation_rows": m * q8_row,
            "paired_gate_up_x16_panel": paired_panel,
            "gate_up_scratch_fp32": m * 2 * d * 4,
            "swiglu_output_fp32": m * d * 4,
        },
        "recommendation": {"tile_m": tile_m, "active_threads": threads, "avoid_smt": True},
        "rationale": {
            "tile_m": "largest token tile that keeps Q8 rows/output within L2/L1 budgets while the paired x16 Q4 panel fits L2",
            "threads": "physical-core cap from weight_bytes / (1.25*L2), reduced if the weight stream exceeds ~85% of shared L3",
        },
    }


def human(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=79)
    ap.add_argument("--D", type=int, default=12288)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--max-tile-m", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    r = model(args.M, args.D, args.K, args.lanes, args.max_tile_m)
    if args.json:
        print(json.dumps(r, indent=2))
        return 0
    print("Q4_K gate_up+SwiGLU deterministic cache model")
    print(f"shape: M={args.M} D={args.D} K={args.K} lanes={args.lanes}")
    print("cache:")
    for k, v in r["cache"].items():
        print(f"  {k}: {human(v) if k != 'physical_cores' else v}")
    print("working set:")
    for k, v in r["bytes"].items():
        print(f"  {k}: {human(v)}")
    print("recommendation:")
    print(f"  tile_m: {r['recommendation']['tile_m']}")
    print(f"  active_threads: {r['recommendation']['active_threads']}")
    print(f"  avoid_smt: {r['recommendation']['avoid_smt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
