"""Emit the serialized, KV-only sequence-switching ABI."""


def emit_sequence_state_api() -> str:
    return r'''
/* Execution is serialized. Handles are generation-tagged IDs, never pointers. */
typedef struct CKSequenceStateV8 {
    uint8_t *kv;
    int pos;
    int rope_pos;
    uint64_t generation;
    int occupied;
} CKSequenceStateV8;

/* One default state and two caller-backed extra states. The caller owns and
 * budgets each 64-byte-aligned KV arena until destroy/model_free completes. */
static CKSequenceStateV8 g_sequence_states[3];
static CKSequenceStateV8 *g_active_sequence = NULL;
static uint64_t g_sequence_generation = 0;

static uint64_t ck_sequence_new_id(unsigned slot) {
    if (g_sequence_generation >= (UINT64_MAX >> 2)) return 0;
    return (++g_sequence_generation << 2) | (uint64_t)(slot + 1);
}

static CKSequenceStateV8 *ck_sequence_find(uint64_t handle) {
    unsigned encoded = (unsigned)(handle & 3u);
    if (encoded < 1 || encoded > 3) return NULL;
    CKSequenceStateV8 *state = &g_sequence_states[encoded - 1];
    return state->occupied && state->generation == handle ? state : NULL;
}

static void ck_sequence_ensure_default(void) {
    if (g_sequence_states[0].occupied || !g_model) return;
    CKSequenceStateV8 *state = &g_sequence_states[0];
    state->kv = g_model->bump + A_KV_CACHE;
    state->pos = g_model->pos;
    state->rope_pos = g_model->rope_pos;
    state->generation = ck_sequence_new_id(0);
    state->occupied = state->generation != 0;
    g_active_sequence = state->occupied ? state : NULL;
}

static void ck_sequence_release_all(void) {
    memset(g_sequence_states, 0, sizeof(g_sequence_states));
    g_active_sequence = NULL;
}

CK_EXPORT int ck_model_sequence_state_requirements(size_t *bytes, size_t *alignment) {
    if (!bytes || !alignment || KV_CACHE_SIZE <= 0) return -1;
    *bytes = (size_t)KV_CACHE_SIZE;
    *alignment = 64;
    return 0;
}

CK_EXPORT uint64_t ck_model_sequence_state_default(void) {
    if (!g_model) return 0;
    ck_sequence_ensure_default();
    return g_sequence_states[0].generation;
}

CK_EXPORT int ck_model_sequence_state_create(void *arena, size_t arena_bytes,
                                              uint64_t *out) {
    if (!g_model || !out) return -1;
    *out = 0;
    ck_sequence_ensure_default();
    if (!g_active_sequence) return -1;
    if (!arena || ((uintptr_t)arena & 63u) || arena_bytes < (size_t)KV_CACHE_SIZE)
        return -2;
    uintptr_t start = (uintptr_t)arena;
    if (start > UINTPTR_MAX - (size_t)KV_CACHE_SIZE) return -2;
    uintptr_t end = start + (size_t)KV_CACHE_SIZE;
    for (unsigned i = 0; i < 3; ++i) {
        CKSequenceStateV8 *existing = &g_sequence_states[i];
        if (!existing->occupied) continue;
        uintptr_t other = (uintptr_t)existing->kv;
        if (other <= UINTPTR_MAX - (size_t)KV_CACHE_SIZE &&
            start < other + (size_t)KV_CACHE_SIZE && other < end) return -2;
    }
    for (unsigned i = 1; i < 3; ++i) {
        CKSequenceStateV8 *state = &g_sequence_states[i];
        if (state->occupied) continue;
        uint64_t id = ck_sequence_new_id(i);
        if (!id) return -3;
        memset(arena, 0, (size_t)KV_CACHE_SIZE);
        state->kv = (uint8_t *)arena;
        state->pos = 0;
        state->rope_pos = 0;
        state->generation = id;
        state->occupied = 1;
        *out = id;
        return 0;
    }
    return -3;
}

CK_EXPORT int ck_model_sequence_state_activate(uint64_t handle) {
    if (!g_model) return -1;
    ck_sequence_ensure_default();
    CKSequenceStateV8 *state = ck_sequence_find(handle);
    if (!state) return -1;
    if (state == g_active_sequence) return 0;
    if (g_active_sequence) {
        g_active_sequence->pos = g_model->pos;
        g_active_sequence->rope_pos = g_model->rope_pos;
    }
    g_model->kv_cache = (float *)state->kv;
    g_model->kv_cache_f16 = (uint16_t *)state->kv;
    g_model->pos = state->pos;
    g_model->rope_pos = state->rope_pos;
    g_active_sequence = state;
    return 0;
}

CK_EXPORT int ck_model_sequence_state_destroy(uint64_t handle) {
    if (!g_model) return -1;
    ck_sequence_ensure_default();
    CKSequenceStateV8 *state = ck_sequence_find(handle);
    if (!state || state == &g_sequence_states[0]) return -1;
    if (state == g_active_sequence &&
        ck_model_sequence_state_activate(g_sequence_states[0].generation) != 0) return -1;
    memset(state, 0, sizeof(*state));
    return 0;
}
'''
