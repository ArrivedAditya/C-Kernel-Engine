#include <stddef.h>
#include <stdint.h>
#include "ckernel_engine.h"

__attribute__((visibility("default")))
int ck_test_real_expert(
    const float *hidden, const int *indices, const float *routing_weights,
    const uint16_t *gate, const uint16_t *up, const uint16_t *down,
    float *output, int rows, int hidden_dim, int intermediate_dim,
    int experts, int top_k, void *workspace, size_t workspace_bytes)
{
    return moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_workspace(
        hidden, indices, routing_weights, gate, up, down, output,
        rows, hidden_dim, intermediate_dim, experts, top_k,
        workspace, workspace_bytes);
}

__attribute__((visibility("default")))
int ck_test_noop_expert(
    const float *hidden, const int *indices, const float *routing_weights,
    const uint16_t *gate, const uint16_t *up, const uint16_t *down,
    float *output, int rows, int hidden_dim, int intermediate_dim,
    int experts, int top_k, void *workspace, size_t workspace_bytes)
{
    (void)hidden; (void)indices; (void)routing_weights;
    (void)gate; (void)up; (void)down; (void)output;
    (void)rows; (void)hidden_dim; (void)intermediate_dim;
    (void)experts; (void)top_k; (void)workspace; (void)workspace_bytes;
    return 0;
}
