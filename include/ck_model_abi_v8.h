#ifndef CK_MODEL_ABI_V8_H
#define CK_MODEL_ABI_V8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CK_MODEL_ABI_V8_VERSION UINT32_C(1)

enum CKModelCapabilityV8 {
    CK_MODEL_CAP_INIT                    = UINT64_C(1) << 0,
    CK_MODEL_CAP_AUTOREGRESSIVE_DECODE   = UINT64_C(1) << 1,
    CK_MODEL_CAP_TEXT_ENCODE             = UINT64_C(1) << 2,
    CK_MODEL_CAP_TOKEN_DECODE            = UINT64_C(1) << 3,
    CK_MODEL_CAP_CHAT_FORMAT             = UINT64_C(1) << 4,
    CK_MODEL_CAP_STOP_TOKENS             = UINT64_C(1) << 5,
    CK_MODEL_CAP_MIXED_EMBEDDING_PREFILL = UINT64_C(1) << 6,
    CK_MODEL_CAP_AUDIO_WAV_ENCODER       = UINT64_C(1) << 7,
    CK_MODEL_CAP_IMAGE_TENSOR_ENCODER    = UINT64_C(1) << 8,
    CK_MODEL_CAP_RAW_IMAGE_ENCODER       = UINT64_C(1) << 9,
    CK_MODEL_CAP_ENCODER_OUTPUT          = UINT64_C(1) << 10,
    CK_MODEL_CAP_ENCODER_MEMORY          = UINT64_C(1) << 11,
    CK_MODEL_CAP_NAMED_ACTIVATIONS       = UINT64_C(1) << 12,
    CK_MODEL_CAP_PROFILE                 = UINT64_C(1) << 13,
    CK_MODEL_CAP_XRAY_KV                 = UINT64_C(1) << 14,
    CK_MODEL_CAP_GENERATION_POLICY       = UINT64_C(1) << 15
};

#define CK_MODEL_CAP_V8_KNOWN_MASK ((UINT64_C(1) << 16) - UINT64_C(1))

enum CKGenerationFlagsV8 {
    CK_GENERATION_FLAG_TIMESTAMPS = 1u << 0
};

enum CKModelArtifactRoleV8 {
    CK_MODEL_ROLE_UNKNOWN = 0,
    CK_MODEL_ROLE_DECODER = 1,
    CK_MODEL_ROLE_ENCODER = 2,
    CK_MODEL_ROLE_COMBINED = 3
};

/* Fixed-layout descriptor. New fields consume reserved slots and require an
 * ABI version bump when their interpretation changes. Hosts must check both
 * struct_size and abi_version before using it. */
typedef struct CKModelRuntimeDescriptorV8 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t capabilities;
    uint32_t artifact_role;
    uint32_t reserved0;
    int32_t context_length;
    int32_t vocab_size;
    int32_t encoder_memory_tokens;
    int32_t encoder_memory_dim;
    int32_t primary_input_tokens;
    int32_t primary_input_dim;
    uint64_t reserved[8];
} CKModelRuntimeDescriptorV8;

typedef uint32_t (*ck_model_get_abi_version_v8_fn)(void);
typedef uint64_t (*ck_model_get_capabilities_v8_fn)(void);
typedef int (*ck_model_get_runtime_descriptor_v8_fn)(
    CKModelRuntimeDescriptorV8 *descriptor,
    size_t descriptor_size);

#ifdef __cplusplus
}
#endif

#endif
