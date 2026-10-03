"""Emit the serialized, KV-only sequence-switching ABI."""


def emit_sequence_state_api() -> str:
    return r'''
/* These handles isolate KV and positions while execution remains serialized.
 * They are not a batched execution interface and cannot be used concurrently. */
typedef struct CKSequenceStateV8 {
    struct CKSequenceStateV8 *next;
    uint8_t *kv;
    int pos;
    int rope_pos;
} CKSequenceStateV8;

static CKSequenceStateV8 g_default_sequence;
static CKSequenceStateV8 *g_sequence_states = NULL;
static CKSequenceStateV8 *g_active_sequence = NULL;
/* The first certification milestone admits two non-default sequences. */
static int g_extra_sequence_count = 0;

static void ck_sequence_ensure_default(void) {
    if (g_sequence_states || !g_model) return;
    memset(&g_default_sequence, 0, sizeof(g_default_sequence));
    g_default_sequence.kv = g_model->bump + A_KV_CACHE;
    g_default_sequence.pos = g_model->pos;
    g_default_sequence.rope_pos = g_model->rope_pos;
    g_sequence_states = &g_default_sequence;
    g_active_sequence = &g_default_sequence;
}

static void ck_sequence_release_all(void) {
    CKSequenceStateV8 *state = g_sequence_states;
    while (state) {
        CKSequenceStateV8 *next = state->next;
        if (state != &g_default_sequence) {
            free(state->kv);
            free(state);
        }
        state = next;
    }
    g_sequence_states = NULL;
    g_active_sequence = NULL;
    g_extra_sequence_count = 0;
    memset(&g_default_sequence, 0, sizeof(g_default_sequence));
}

CK_EXPORT void *ck_model_sequence_state_default(void) {
    if (!g_model) return NULL;
    ck_sequence_ensure_default();
    return &g_default_sequence;
}

CK_EXPORT int ck_model_sequence_state_create(void **out) {
    if (!g_model || !out) return -1;
    *out = NULL;
    ck_sequence_ensure_default();
    if (g_extra_sequence_count >= 2) return -3;
    CKSequenceStateV8 *state = calloc(1, sizeof(*state));
    if (!state) return -2;
    state->kv = calloc(1, KV_CACHE_SIZE);
    if (!state->kv) { free(state); return -2; }
    state->next = g_sequence_states;
    g_sequence_states = state;
    g_extra_sequence_count++;
    *out = state;
    return 0;
}

CK_EXPORT int ck_model_sequence_state_activate(void *handle) {
    if (!g_model || !handle) return -1;
    ck_sequence_ensure_default();
    CKSequenceStateV8 *state = g_sequence_states;
    while (state && state != handle) state = state->next;
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

CK_EXPORT int ck_model_sequence_state_destroy(void *handle) {
    if (!g_model || !handle || handle == &g_default_sequence) return -1;
    ck_sequence_ensure_default();
    CKSequenceStateV8 **link = &g_sequence_states;
    while (*link && *link != handle) link = &(*link)->next;
    if (!*link) return -1;
    CKSequenceStateV8 *state = *link;
    if (state == g_active_sequence &&
        ck_model_sequence_state_activate(&g_default_sequence) != 0) return -1;
    *link = state->next;
    free(state->kv);
    free(state);
    g_extra_sequence_count--;
    return 0;
}
'''
