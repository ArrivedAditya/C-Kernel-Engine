#ifndef CKERNEL_AUDIO_H
#define CKERNEL_AUDIO_H

#ifdef __cplusplus
extern "C" {
#endif

#define CK_AUDIO_WHISPER_SAMPLE_RATE 16000
#define CK_AUDIO_WHISPER_N_FFT 400
#define CK_AUDIO_WHISPER_HOP_LENGTH 160
#define CK_AUDIO_WHISPER_POWER_BINS 201

int audio_whisper_stft_power_reference_f32(
    const float *samples,
    int n_samples,
    float *power,
    int n_frames);

int audio_whisper_log_mel_from_power_reference_f32(
    const float *power,
    const float *mel_filters,
    int n_mels,
    int n_frames,
    float *log_mel);

int audio_whisper_log_mel_reference_f32(
    const float *samples,
    int n_samples,
    const float *mel_filters,
    int n_mels,
    float *power_scratch,
    float *log_mel,
    int n_frames);

#ifdef __cplusplus
}
#endif

#endif
