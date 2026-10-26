#include <stddef.h>
#include <stdint.h>
#include "ckernel_engine.h"

__attribute__((visibility("default")))
int ck_test_real_shared_farskip(
    const float *hidden, const float *routed, const float *residual,
    const uint16_t *gate, const uint16_t *up, const uint16_t *down,
    float *main_output, float *routed_free_output, int rows,
    int hidden_dim, int intermediate_dim, void *workspace,
    size_t workspace_bytes)
{
    return farskip_swiglu_shared_combine_pytorch_onednn_3_12_bf16_storage_workspace(
        hidden, routed, residual, gate, up, down,
        main_output, routed_free_output, rows, hidden_dim,
        intermediate_dim, workspace, workspace_bytes);
}

__attribute__((visibility("default")))
void ck_test_old_shared_farskip(
    const float *hidden, const float *routed, const float *residual,
    const uint16_t *gate, const uint16_t *up, const uint16_t *down,
    float *main_output, float *routed_free_output, int rows,
    int hidden_dim, int intermediate_dim)
{
    farskip_swiglu_shared_combine_pytorch_bf16_storage(
        hidden, routed, residual, gate, up, down,
        main_output, routed_free_output, rows, hidden_dim, intermediate_dim);
}
