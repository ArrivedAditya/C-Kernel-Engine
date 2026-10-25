#ifndef CK_TOPOLOGY_H
#define CK_TOPOLOGY_H

/* Generic CPU topology discovery for dispatch decisions.
 *
 * Everything is derived at runtime from the process's allowed CPU set
 * (sched_getaffinity) and per-CPU capacity exported by the kernel
 * (/sys/devices/system/cpu/cpuN/cpu_capacity, with cpufreq max-frequency as
 * fallback). Nothing is hardcoded to a specific machine, and CPUs outside
 * the process's allowed set are never considered. On non-Linux platforms or
 * when no capacity information exists, every allowed CPU is treated as equal
 * (homogeneous), which is also the x86 result.
 */

/* Number of CPUs the process may run on (affinity-aware). */
int ck_topology_allowed_cpus(void);

/* Nonzero when the allowed CPUs have meaningfully different performance
 * capacity (max/min capacity ratio > 1.15, e.g. big.LITTLE). Zero on
 * homogeneous systems and when capacity is unknown. */
int ck_topology_heterogeneous(void);

/* Fill `cpus` with the allowed CPU ids in the top capacity tier (the "big"
 * cores), ascending. Returns the number written (never more than max_cpus).
 * On homogeneous/unknown systems the top tier is every allowed CPU. */
int ck_topology_big_cpus(int *cpus, int max_cpus);

#endif /* CK_TOPOLOGY_H */
