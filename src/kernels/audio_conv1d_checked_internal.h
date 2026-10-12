#ifndef CK_AUDIO_CONV1D_CHECKED_INTERNAL_H
#define CK_AUDIO_CONV1D_CHECKED_INTERNAL_H

#include <stddef.h>

typedef struct {
    const float *input;
    const float *weight;
    const float *bias;
    float *staged;
    size_t input_stride;
    size_t input_channels;
    size_t output_channels;
    size_t input_frames;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    size_t dilation;
    size_t output_frames;
    size_t channel_begin;
    size_t channel_end;
} ck_audio_conv1d_job;

typedef void (*ck_audio_conv1d_compute_fn)(const ck_audio_conv1d_job *, void *);

int ck_audio_conv1d_checked_execute(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t dilation,
    size_t output_frames, ck_audio_conv1d_compute_fn compute,
    void *compute_context);

void audio_conv1d_compute_avx2(const ck_audio_conv1d_job *job, void *unused);

#endif
