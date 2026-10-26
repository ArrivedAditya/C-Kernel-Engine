#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ckernel_engine.h"

static int gemm_calls;
static int wrong_group;
static int expected_rows = 3;
void gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage_workspace(
    const float *a, const void *weights, const float *bias, float *out,
    int rows, int n, int k, uint16_t *input_stage, size_t input_bytes,
    uint16_t *output_stage, size_t output_bytes,
    uint16_t *bias_stage, size_t bias_bytes)
{
    (void)bias; (void)input_stage; (void)input_bytes;
    (void)output_stage; (void)output_bytes; (void)bias_stage; (void)bias_bytes;
    if (rows != expected_rows) wrong_group = 1;
    ++gemm_calls;
    const uint16_t *w = weights;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                const uint32_t bits = (uint32_t)w[(size_t)col * k + p] << 16;
                float value;
                memcpy(&value, &bits, sizeof(value));
                sum += a[(size_t)row * k + p] * value;
            }
            out[(size_t)row * n + col] = sum;
        }
    }
}
void swiglu_forward_pytorch_bf16_storage(
    const float *input, float *output, int rows, int intermediate_dim)
{
    for (int row = 0; row < rows; ++row)
        for (int i = 0; i < intermediate_dim; ++i)
            output[(size_t)row * intermediate_dim + i] =
                input[(size_t)row * 2 * intermediate_dim + i] *
                input[(size_t)row * 2 * intermediate_dim + intermediate_dim + i];
}
int main(void)
{
    enum { R=3, H=32, I=32, WORKSPACE=R*(16*I+8*H)+8*I,
           REQUIRED=R*(12*I+4*H+4*I)+8*I };
    float hidden[R*H], routed[R*H], residual[R*H];
    float main_output[R*H], free_output[R*H];
    uint16_t gate[I*H] = {0}, up[I*H] = {0}, down[H*I] = {0};
    for (int i = 0; i < H; ++i) {
        gate[i*H+i] = up[i*H+i] = down[i*I+i] = 0x3f80;
        for (int row = 0; row < R; ++row) {
            const int p = row*H+i;
            hidden[p] = (float)(row+1);
            routed[p] = (float)(row+1);
            residual[p] = 1.0f;
            main_output[p] = free_output[p] = -777.0f;
        }
    }
    void *raw = malloc(WORKSPACE+64);
    if (!raw) return 1;
    void *scratch = (void *)(((uintptr_t)raw+63)&~(uintptr_t)63);
    int status =
        farskip_swiglu_shared_combine_pytorch_onednn_3_12_bf16_storage_workspace(
            hidden,routed,residual,gate,up,down,main_output,free_output,
            R,H,I,scratch,REQUIRED-1);
    if (status != -1 || gemm_calls) return 2;
    for (int p = 0; p < R*H; ++p)
        if (main_output[p] != -777.0f || free_output[p] != -777.0f) return 3;
    status =
        farskip_swiglu_shared_combine_pytorch_onednn_3_12_bf16_storage_workspace(
            hidden,routed,residual,gate,up,down,main_output,free_output,
            R,H,I,scratch,WORKSPACE);
    if (status || gemm_calls != 3 || wrong_group) return 4;
    for (int row = 0; row < R; ++row) {
        const float shared = (float)((row+1)*(row+1));
        for (int i = 0; i < H; ++i) {
            const int p=row*H+i;
            if (free_output[p] != 1.0f+shared ||
                main_output[p] != 1.0f+(float)(row+1)+shared) return 5;
        }
    }
    gemm_calls = 0;
    expected_rows = 1;
    status =
        farskip_swiglu_shared_combine_pytorch_onednn_3_12_bf16_storage_workspace(
            hidden,routed,residual,gate,up,down,main_output,free_output,
            1,H,I,scratch,WORKSPACE);
    if (status || gemm_calls != 3 || wrong_group ||
        free_output[0] != 2.0f || main_output[0] != 3.0f) return 6;
    puts("PASS: grouped and single-row projections, two output streams, rejected workspace leaves outputs unchanged");
    free(raw);
    return 0;
}
