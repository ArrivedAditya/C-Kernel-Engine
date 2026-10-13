#include "ckernel_audio_conv_checked.h"
#include "ck_threadpool.h"
#include "audio_conv1d_checked_internal.h"

#include <limits.h>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
typedef struct {
    const ck_audio_conv1d_job *job;
    ck_threadpool_t *pool;
    int workers;
} ck_audio_conv1d_parallel_context;

static void audio_conv1d_channel_range(int begin, int end, void *opaque)
{
    const ck_audio_conv1d_job *source = opaque;
    ck_audio_conv1d_job range = *source;
    range.channel_begin = (size_t)begin;
    range.channel_end = (size_t)end;
    audio_conv1d_compute_avx2(&range, NULL);
}

static void audio_conv1d_compute_parallel(const ck_audio_conv1d_job *job,
                                           void *opaque)
{
    const ck_audio_conv1d_parallel_context *context = opaque;
    const int channels = (int)job->output_channels;
    int workers = context->workers;
    if (workers > ck_threadpool_capacity(context->pool))
        workers = ck_threadpool_capacity(context->pool);
    if (workers > channels) workers = channels;
    /* Dispatch overhead dominates tiny independent work. Both paths use the
     * identical per-channel AVX2 implementation and ordered reduction. */
    if (workers < 2 || channels < 8 || job->output_frames < 16) {
        audio_conv1d_compute_avx2(job, NULL);
        return;
    }
    const int grain = (channels + workers - 1) / workers;
    ck_threadpool_parallel_for_n(context->pool, workers, 0, channels, grain,
                                 audio_conv1d_channel_range, (void *)job);
}
#endif

static int audio_conv1d_parallel_checked_impl(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t dilation,
    size_t output_frames, void *worker_pool, size_t worker_count)
{
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
    if (!audio_conv1d_checked_avx2_fma_available())
        return CK_AUDIO_CONV_CHECKED_UNSUPPORTED;
    if (!worker_pool || worker_count == 0 || worker_count > INT_MAX)
        return CK_AUDIO_CONV_CHECKED_INVALID;
    ck_audio_conv1d_parallel_context context = {
        .pool = (ck_threadpool_t *)worker_pool,
        .workers = (int)worker_count,
    };
    return ck_audio_conv1d_checked_execute(input, input_elements, input_stride,
        weight, weight_elements, bias, bias_elements, output, output_elements,
        output_stride, scratch, scratch_elements, input_channels,
        output_channels, input_frames, kernel_size, stride, padding, dilation,
        output_frames, audio_conv1d_compute_parallel, &context);
#else
    (void)input; (void)input_elements; (void)input_stride;
    (void)weight; (void)weight_elements; (void)bias; (void)bias_elements;
    (void)output; (void)output_elements; (void)output_stride;
    (void)scratch; (void)scratch_elements; (void)input_channels;
    (void)output_channels; (void)input_frames; (void)kernel_size;
    (void)stride; (void)padding; (void)dilation; (void)output_frames;
    (void)worker_pool; (void)worker_count;
    return CK_AUDIO_CONV_CHECKED_UNSUPPORTED;
#endif
}

int audio_conv1d_checked_channel_major_f32_avx2_parallel(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t output_frames,
    void *worker_pool, size_t worker_count)
{
    return audio_conv1d_parallel_checked_impl(input, input_elements,
        input_stride, weight, weight_elements, bias, bias_elements, output,
        output_elements, output_stride, scratch, scratch_elements,
        input_channels, output_channels, input_frames, kernel_size, stride,
        padding, 1, output_frames, worker_pool, worker_count);
}

int audio_conv1d_dilated_checked_channel_major_f32_avx2_parallel(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t dilation,
    size_t output_frames, void *worker_pool, size_t worker_count)
{
    return audio_conv1d_parallel_checked_impl(input, input_elements,
        input_stride, weight, weight_elements, bias, bias_elements, output,
        output_elements, output_stride, scratch, scratch_elements,
        input_channels, output_channels, input_frames, kernel_size, stride,
        padding, dilation, output_frames, worker_pool, worker_count);
}
