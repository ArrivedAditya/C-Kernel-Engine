#ifndef CK_SAMPLER_V8_H
#define CK_SAMPLER_V8_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Sample one token from mutable logits.
 *
 * random_value must be in [0, 1). The function uses an allocation-free O(V)
 * categorical path when top_p >= 1 and an exact O(V log V) nucleus path when
 * 0 < top_p < 1. Non-positive temperature or top_p selects greedy argmax.
 */
int ck_sample_top_p_v8(
    float *logits,
    int vocab_size,
    float temperature,
    float top_p,
    float random_value);

#ifdef __cplusplus
}
#endif

#endif
