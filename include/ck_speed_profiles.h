#ifndef CK_SPEED_PROFILES_H
#define CK_SPEED_PROFILES_H

#include <stdlib.h>
#include <string.h>

static inline int ck_env_value_truthy(const char *v)
{
    return (v && v[0] && v[0] != '0') ? 1 : 0;
}

static inline int ck_speed_profile_qwen3vl_ocr_fast(void)
{
    const char *alias = getenv("CK_QWEN3VL_OCR_FAST");
    if (ck_env_value_truthy(alias)) return 1;

    /* CK_PROFILE is already used for timing/profiling instrumentation.
     * Keep speed policy separate so OCR tuning can coexist with CK_PROFILE=1.
     */
    const char *profile = getenv("CK_SPEED_PROFILE");
    if (!profile || !profile[0]) return 0;
    return strcmp(profile, "qwen3vl_ocr_xeon_avx512") == 0 ||
           strcmp(profile, "qwen3vl_ocr_fast") == 0 ||
           strcmp(profile, "qwen3vl_ocr") == 0;
}

static inline int ck_env_truthy_or_qwen3vl_ocr_profile(const char *name)
{
    const char *env = getenv(name);
    if (env) return ck_env_value_truthy(env);
    return ck_speed_profile_qwen3vl_ocr_fast();
}

#endif /* CK_SPEED_PROFILES_H */
