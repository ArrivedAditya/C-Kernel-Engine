"""Emit a narrow, generated two-sequence decode step from an admitted IR split."""

from __future__ import annotations

from typing import Any


def emit_two_row_batch_api(contract: dict[str, Any]) -> str:
    b = contract["buffers"]
    size = contract["sizes"]
    weight_q, weight_k = contract["weights"]
    gemm = contract["gemm_function"]
    # Each section is packed by row. The input section is the GEMM's M=2 A
    # matrix; Q/K sections are its two independently addressed output rows.
    input_bytes = size["input"]
    residual_bytes = size["residual"]
    q_bytes = size["q"]
    k_bytes = size["k"]
    total = 2 * (input_bytes + residual_bytes + q_bytes + k_bytes)
    return f'''
/* Generated two-row decode: shared Q/K projections, sequence-local suffix. */
static int ck_batch_ranges_overlap(const void *left, size_t left_bytes,
                                   const void *right, size_t right_bytes) {{
    uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
    if (!a || !b || a > UINTPTR_MAX - left_bytes ||
        b > UINTPTR_MAX - right_bytes) return -1;
    return a < b + right_bytes && b < a + left_bytes;
}}

CK_EXPORT int ck_model_batch_decode_workspace(size_t *bytes, size_t *alignment) {{
    if (!bytes || !alignment) return -1;
    *bytes = {total}u;
    *alignment = 64u;
    return 0;
}}

CK_EXPORT int ck_model_decode_batch2(const CKModelBatchDecodeRowV8 *rows,
                                      size_t count, void *workspace,
                                      size_t workspace_bytes) {{
    if (!g_model || !rows || count != 2 || !workspace) return -1;
    if (((uintptr_t)workspace & 63u) || workspace_bytes < {total}u) return -3;
    if (ck_model_cancel_requested() || getenv("CK_STOP_OP") ||
        getenv("CK_DEBUG_IMPORT_HIDDEN") || getenv("CK_V8_DEBUG_ATTN_PROJ_FP32_LAYER")) return -2;
    CKSequenceStateV8 *states[2];
    for (size_t i = 0; i < 2; ++i) {{
        const CKModelBatchDecodeRowV8 *row = &rows[i];
        states[i] = ck_sequence_find(row->sequence_handle);
        if (!states[i] || !row->logits || row->row_offset != i ||
            row->token_count != 1 || row->token < 0 || row->token >= VOCAB_SIZE ||
            row->position < 0 || row->position >= MAX_SEQ_LEN ||
            row->position != (states[i] == g_active_sequence ? g_model->pos : states[i]->pos))
            return -2;
    }}
    if (states[0] == states[1] || rows[0].logits == rows[1].logits) return -2;
    const size_t output_bytes = (size_t)VOCAB_SIZE * sizeof(float);
    if (ck_batch_ranges_overlap(workspace, {total}u, rows, 2 * sizeof(*rows)) != 0 ||
        ck_batch_ranges_overlap(workspace, {total}u, g_model->bump, g_model->bump_size) != 0 ||
        ck_batch_ranges_overlap(rows, 2 * sizeof(*rows),
                                g_model->bump, g_model->bump_size) != 0 ||
        ck_batch_ranges_overlap(rows[0].logits, output_bytes,
                                rows[1].logits, output_bytes) != 0) return -2;
    for (size_t j = 0; j < 3; ++j) {{
        CKSequenceStateV8 *live = &g_sequence_states[j];
        if (!live->occupied) continue;
        if (ck_batch_ranges_overlap(workspace, {total}u, live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0 ||
            ck_batch_ranges_overlap(rows, 2 * sizeof(*rows), live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0 ||
            ck_batch_ranges_overlap(rows[0].logits, output_bytes, live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0 ||
            ck_batch_ranges_overlap(rows[1].logits, output_bytes, live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0) return -2;
    }}
    for (size_t i = 0; i < 2; ++i) {{
        if (
            ck_batch_ranges_overlap(rows[i].logits, output_bytes, rows,
                                    2 * sizeof(*rows)) != 0 ||
            ck_batch_ranges_overlap(rows[i].logits, output_bytes, workspace, {total}u) != 0 ||
            ck_batch_ranges_overlap(rows[i].logits, output_bytes, g_model->bump,
                                    g_model->bump_size) != 0) return -2;
    }}
    uint8_t *base = (uint8_t *)workspace;
    float *inputs = (float *)base;
    uint8_t *residual = base + {2 * input_bytes}u;
    float *q = (float *)(residual + {2 * residual_bytes}u);
    float *k = (float *)((uint8_t *)q + {2 * q_bytes}u);

    /* Neither prefix updates position or KV. Snapshot the only call-local
     * values the suffix consumes before admitting the second prefix. */
    for (size_t i = 0; i < 2; ++i) {{
        if (ck_model_sequence_state_activate(rows[i].sequence_handle) != 0) return -2;
        ck_batch_decode_prefix(g_model, rows[i].token);
        memcpy((uint8_t *)inputs + i * {input_bytes}u,
               g_model->bump + {b['input']}, {input_bytes}u);
        memcpy(residual + i * {residual_bytes}u,
               g_model->bump + {b['residual']}, {residual_bytes}u);
    }}

    /* Both calls use map-declared Q5_1/Q8_1 M=2 arithmetic. Each packed
     * weight block is unpacked once and used for both independent rows. */
    {gemm}(inputs, g_model->bump + {weight_q}, NULL,
           q, 2, {contract['q_dim']}, {contract['input_dim']});
    {gemm}(inputs, g_model->bump + {weight_k}, NULL,
           k, 2, {contract['k_dim']}, {contract['input_dim']});

    for (size_t i = 0; i < 2; ++i) {{
        if (ck_model_sequence_state_activate(rows[i].sequence_handle) != 0) return -2;
        memcpy(g_model->bump + {b['input']},
               (uint8_t *)inputs + i * {input_bytes}u, {input_bytes}u);
        memcpy(g_model->bump + {b['residual']},
               residual + i * {residual_bytes}u, {residual_bytes}u);
        memcpy(g_model->bump + {b['q']},
               (uint8_t *)q + i * {q_bytes}u, {q_bytes}u);
        memcpy(g_model->bump + {b['k']},
               (uint8_t *)k + i * {k_bytes}u, {k_bytes}u);
        g_ck_skip_decode_logits = 0;
        ck_batch_decode_suffix(g_model, rows[i].token);
        memcpy(rows[i].logits, g_model->logits, VOCAB_SIZE * sizeof(float));
    }}
    return 0;
}}
'''
