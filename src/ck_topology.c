/**
 * @file ck_topology.c
 * @brief Generic CPU topology discovery (affinity-aware, capacity-based)
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * Discovery is cached process-wide (pthread_once) because sysfs content and
 * the process affinity mask are effectively fixed for the lifetime of an
 * inference process, and dispatch paths query this on every GEMV.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ck_topology.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

/* Bound on discovered CPUs; CPU_SETSIZE is 1024 on Linux. */
#define CK_TOPOLOGY_MAX_CPUS 1024

/* Capacity ratio above which the allowed set counts as heterogeneous
 * (GB10 Cortex-X925 vs Cortex-A725 is ~1.4; SMT siblings are ~1.0). */
#define CK_TOPOLOGY_HETEROGENEOUS_RATIO_X100 115

/* A CPU belongs to the top tier when its capacity is within 90% of the max
 * (per-core capacity values differ slightly between otherwise identical
 * big cores, e.g. 997 vs 1024). */
#define CK_TOPOLOGY_TOP_TIER_PCT 90

typedef struct {
    int n_allowed;
    int heterogeneous;
    int n_big;
    int big_cpus[CK_TOPOLOGY_MAX_CPUS];
} ck_topology_state_t;

static ck_topology_state_t g_topology;
static pthread_once_t g_topology_once = PTHREAD_ONCE_INIT;

#ifdef __linux__
static long read_cpu_capacity(int cpu)
{
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
    FILE *f = fopen(path, "r");
    if (f) {
        long v = -1;
        if (fscanf(f, "%ld", &v) != 1) v = -1;
        fclose(f);
        if (v > 0) return v;
    }
    /* Fallback: cpufreq maximum frequency in kHz. */
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
    f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}
#endif

static void topology_discover(void)
{
    ck_topology_state_t *t = &g_topology;
    t->n_allowed = 1;
    t->heterogeneous = 0;
    t->n_big = 1;
    t->big_cpus[0] = 0;

#ifdef __linux__
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;  /* keep the conservative single-CPU default */
    }

    long capacities[CK_TOPOLOGY_MAX_CPUS];
    int ids[CK_TOPOLOGY_MAX_CPUS];
    int n = 0;
    long cap_min = 0, cap_max = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE && n < CK_TOPOLOGY_MAX_CPUS; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        ids[n] = cpu;
        capacities[n] = read_cpu_capacity(cpu);
        if (capacities[n] > 0) {
            if (cap_max == 0 || capacities[n] > cap_max) cap_max = capacities[n];
            if (cap_min == 0 || capacities[n] < cap_min) cap_min = capacities[n];
        }
        ++n;
    }
    if (n == 0) {
        return;
    }
    t->n_allowed = n;

    if (cap_min <= 0 || cap_max * 100 <= cap_min * CK_TOPOLOGY_HETEROGENEOUS_RATIO_X100) {
        /* Homogeneous or capacity unknown: every allowed CPU is top tier. */
        t->heterogeneous = 0;
        t->n_big = n;
        for (int i = 0; i < n; ++i) t->big_cpus[i] = ids[i];
        return;
    }

    t->heterogeneous = 1;
    int nb = 0;
    for (int i = 0; i < n && nb < CK_TOPOLOGY_MAX_CPUS; ++i) {
        if (capacities[i] > 0 && capacities[i] * 100 >= cap_max * CK_TOPOLOGY_TOP_TIER_PCT) {
            t->big_cpus[nb++] = ids[i];
        }
    }
    if (nb == 0) {
        t->n_big = n;
        for (int i = 0; i < n; ++i) t->big_cpus[i] = ids[i];
        return;
    }
    t->n_big = nb;
#endif
}

static void topology_init(void)
{
    topology_discover();
}

int ck_topology_allowed_cpus(void)
{
    pthread_once(&g_topology_once, topology_init);
    return g_topology.n_allowed;
}

int ck_topology_heterogeneous(void)
{
    pthread_once(&g_topology_once, topology_init);
    return g_topology.heterogeneous;
}

int ck_topology_big_cpus(int *cpus, int max_cpus)
{
    pthread_once(&g_topology_once, topology_init);
    if (!cpus || max_cpus <= 0) return 0;
    int n = g_topology.n_big;
    if (n > max_cpus) n = max_cpus;
    for (int i = 0; i < n; ++i) {
        cpus[i] = g_topology.big_cpus[i];
    }
    return n;
}
