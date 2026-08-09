#include "ck_sampler_v8.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    float probability;
    int32_t token_id;
} CKSampleCandidateV8;

static int sample_argmax(const float *logits, int vocab_size) {
    int best = 0;
    float best_value = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_value) {
            best = i;
            best_value = logits[i];
        }
    }
    return best;
}

static int compare_candidates_descending(const void *left, const void *right) {
    const CKSampleCandidateV8 *a = (const CKSampleCandidateV8 *)left;
    const CKSampleCandidateV8 *b = (const CKSampleCandidateV8 *)right;
    if (a->probability > b->probability) return -1;
    if (a->probability < b->probability) return 1;
    if (a->token_id < b->token_id) return -1;
    if (a->token_id > b->token_id) return 1;
    return 0;
}

static int sample_in_token_order(
    const float *probabilities,
    int vocab_size,
    float random_value) {
    float total = 0.0f;
    int fallback = 0;
    for (int i = 0; i < vocab_size; ++i) {
        total += probabilities[i];
        if (probabilities[i] > 0.0f) fallback = i;
    }
    const float target = random_value * total;
    float cumulative = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cumulative += probabilities[i];
        if (cumulative > target) return i;
    }
    return fallback;
}

int ck_sample_top_p_v8(
    float *logits,
    int vocab_size,
    float temperature,
    float top_p,
    float random_value) {
    if (!logits || vocab_size <= 0 || !isfinite(temperature) ||
        !isfinite(top_p) || !isfinite(random_value) ||
        random_value < 0.0f || random_value >= 1.0f) {
        return -1;
    }
    if (temperature <= 0.0f || top_p <= 0.0f) {
        return sample_argmax(logits, vocab_size);
    }

    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    if (!isfinite(max_logit)) return -1;

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] = expf((logits[i] - max_logit) / temperature);
        sum += logits[i];
    }
    if (!(sum > 0.0f) || !isfinite(sum)) return -1;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] /= sum;
    }

    /* top_p=1 is ordinary categorical sampling and requires no ordering. */
    if (top_p >= 1.0f) {
        return sample_in_token_order(logits, vocab_size, random_value);
    }

    CKSampleCandidateV8 *candidates = (CKSampleCandidateV8 *)malloc(
        (size_t)vocab_size * sizeof(*candidates));
    if (!candidates) return sample_argmax(logits, vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        candidates[i].probability = logits[i];
        candidates[i].token_id = i;
    }
    qsort(
        candidates,
        (size_t)vocab_size,
        sizeof(*candidates),
        compare_candidates_descending);

    float nucleus_mass = 0.0f;
    int nucleus_size = 0;
    do {
        nucleus_mass += candidates[nucleus_size].probability;
        ++nucleus_size;
    } while (nucleus_size < vocab_size && nucleus_mass < top_p);

    const float target = random_value * nucleus_mass;
    float cumulative = 0.0f;
    int result = candidates[nucleus_size - 1].token_id;
    for (int i = 0; i < nucleus_size; ++i) {
        cumulative += candidates[i].probability;
        if (cumulative > target) {
            result = candidates[i].token_id;
            break;
        }
    }
    free(candidates);
    return result;
}
