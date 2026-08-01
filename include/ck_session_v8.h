#ifndef CK_SESSION_V8_H
#define CK_SESSION_V8_H

#include <stddef.h>
#include <stdint.h>

#include "ck_model_abi_v8.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CK_SESSION_ABI_V8_VERSION UINT32_C(1)

typedef struct CKSessionV8 CKSessionV8;

enum CKSessionStatusV8 {
    CK_SESSION_V8_OK = 0,
    CK_SESSION_V8_ERROR_INVALID_ARGUMENT = -1,
    CK_SESSION_V8_ERROR_ABI = -2,
    CK_SESSION_V8_ERROR_LOAD = -3,
    CK_SESSION_V8_ERROR_INIT = -4,
    CK_SESSION_V8_ERROR_CAPABILITY = -5,
    CK_SESSION_V8_ERROR_BUSY = -6,
    CK_SESSION_V8_ERROR_RUNTIME = -7,
    CK_SESSION_V8_ERROR_BUFFER_TOO_SMALL = -8
};

enum CKSessionStopReasonV8 {
    CK_SESSION_STOP_NONE = 0,
    CK_SESSION_STOP_EOS = 1,
    CK_SESSION_STOP_TOKEN_LIMIT = 2,
    CK_SESSION_STOP_CANCELLED = 3,
    CK_SESSION_STOP_CALLBACK = 4,
    CK_SESSION_STOP_RUNTIME_ERROR = 5
};

enum CKSessionRequestFlagsV8 {
    CK_SESSION_REQUEST_RAW_PROMPT = UINT32_C(1) << 0,
    CK_SESSION_REQUEST_IGNORE_EOS = UINT32_C(1) << 1,
    CK_SESSION_REQUEST_TIMESTAMPS = UINT32_C(1) << 2,
    /* Requests reset generated recurrent/KV state by default. Set this only
     * when the caller deliberately continues the preceding request. */
    CK_SESSION_REQUEST_CONTINUE_STATE = UINT32_C(1) << 3
};

typedef struct CKSessionConfigV8 {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *model_library_path;
    const char *weights_path;
    const char *manifest_path;
    int32_t context_length;
    int32_t num_threads;
    uint64_t required_capabilities;
    uint64_t reserved[8];
} CKSessionConfigV8;

typedef struct CKSessionGenerateRequestV8 {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *system_text;
    const char *user_text;
    int32_t max_tokens;
    /* Sampling is a host policy applied after the generated model's mandatory
     * generation policy (for example Whisper timestamp masking). */
    float temperature;
    float top_p;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t reserved[8];
} CKSessionGenerateRequestV8;

typedef struct CKSessionGenerateResultV8 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t prompt_tokens;
    int32_t generated_tokens;
    int32_t stop_reason;
    int32_t reserved0;
    double prefill_time_ms;
    double decode_time_ms;
    uint64_t reserved[8];
} CKSessionGenerateResultV8;

/* Return zero to continue generation. A non-zero result requests a clean stop. */
typedef int (*ck_session_token_callback_v8)(
    void *user_data,
    int32_t token_id,
    const char *utf8_text,
    size_t utf8_length,
    int32_t sequence_index);

uint32_t ck_session_v8_get_abi_version(void);

int ck_session_v8_open(
    const CKSessionConfigV8 *config,
    CKSessionV8 **session_out);

void ck_session_v8_close(CKSessionV8 *session);

int ck_session_v8_get_model_descriptor(
    const CKSessionV8 *session,
    CKModelRuntimeDescriptorV8 *descriptor,
    size_t descriptor_size);

/* Size-query convention: pass output=NULL/capacity=0 to receive the required
 * element or byte count. The returned count excludes a text NUL terminator. */
int ck_session_v8_encode(
    CKSessionV8 *session,
    const char *text,
    int32_t *output,
    int32_t capacity);

int ck_session_v8_decode(
    CKSessionV8 *session,
    const int32_t *tokens,
    int32_t token_count,
    char *output,
    int32_t capacity);

int ck_session_v8_format_chat(
    CKSessionV8 *session,
    const char *system_text,
    const char *user_text,
    char *output,
    int32_t capacity);

int ck_session_v8_generate(
    CKSessionV8 *session,
    const CKSessionGenerateRequestV8 *request,
    ck_session_token_callback_v8 callback,
    void *user_data,
    CKSessionGenerateResultV8 *result);

void ck_session_v8_cancel(CKSessionV8 *session);
int ck_session_v8_reset(CKSessionV8 *session);
const char *ck_session_v8_last_error(const CKSessionV8 *session);

#ifdef __cplusplus
}
#endif

#endif
