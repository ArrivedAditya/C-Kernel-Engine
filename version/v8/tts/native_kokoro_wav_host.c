#define _POSIX_C_SOURCE 200809L
#ifdef CK_TTS_EXPECTED_LIBRARY_SHA256
#define _GNU_SOURCE
#endif
/* Native Kokoro diagnostic host. The generated model owns every neural
 * operation; this host loads caller-owned storage, optionally fills
 * request-local excitation, and writes only the returned valid WAV samples.
 * CK_TTS_RUNTIME_WAVEFORM selects the bounded runtime-length entry ABI. */
#include "ckernel_tts.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef CK_TTS_RUNTIME_WAVEFORM
#define CK_TTS_GENERATED_ENTRY ck_kokoro_runtime_waveform_bounded
#define CK_TTS_GENERATED_SYMBOL "ck_kokoro_runtime_waveform_bounded"
#else
#define CK_TTS_GENERATED_ENTRY ck_kokoro_waveform_bounded
#define CK_TTS_GENERATED_SYMBOL "ck_kokoro_waveform_bounded"
#endif
#ifdef CK_TTS_EXPECTED_LIBRARY_SHA256
#include <dlfcn.h>
#include <openssl/evp.h>

/* The resolved generated-entry symbol identifies the mapped library. The
 * digest identifies its on-disk artifact at capture time; this is not an
 * embedded build identity or proof against concurrent file replacement. */
static int verify_loaded_library(void) {
    void *symbol = dlsym(RTLD_DEFAULT, CK_TTS_GENERATED_SYMBOL);
    Dl_info info;
    if (!symbol || !dladdr(symbol, &info) || !info.dli_fname) return 0;
    FILE *stream = fopen(info.dli_fname, "rb");
    if (!stream) return 0;
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int count = 0;
    unsigned char buffer[65536];
    int okay = context && EVP_DigestInit_ex(context, EVP_sha256(), NULL) == 1;
    size_t length;
    while (okay && (length = fread(buffer, 1, sizeof(buffer), stream)) != 0)
        okay = EVP_DigestUpdate(context, buffer, length) == 1;
    if (ferror(stream)) okay = 0;
    if (okay) okay = EVP_DigestFinal_ex(context, digest, &count) == 1;
    if (fclose(stream) != 0) okay = 0;
    EVP_MD_CTX_free(context);
    const char *expected = CK_TTS_EXPECTED_LIBRARY_SHA256;
    if (!okay || count != 32 || strlen(expected) != 64) return 0;
    static const char hex[] = "0123456789abcdef";
    for (unsigned int i = 0; i < count; ++i) {
        if (expected[2 * i] != hex[digest[i] >> 4] ||
            expected[2 * i + 1] != hex[digest[i] & 15]) return 0;
    }
    fprintf(stderr, "resolved generated symbol in %s; artifact SHA-256 verified\n",
            info.dli_fname);
    return 1;
}
#endif

/* The host build declares the graph's token capacity. CKTTS003 binds exactly
 * that many IDs; CKTTS004 binds a checked valid length within capacity. */
#ifndef CK_TTS_COMPILED_TOKEN_COUNT
#define CK_TTS_COMPILED_TOKEN_COUNT 0
#endif

#ifdef CK_TTS_RUNTIME_WAVEFORM
extern int CK_TTS_GENERATED_ENTRY(uint8_t *, size_t, int32_t *, int32_t *,
    int32_t *, int32_t *, int32_t *, int32_t *, int32_t *, int32_t *, int32_t *);
#else
extern int CK_TTS_GENERATED_ENTRY(uint8_t *, size_t, int32_t *, int32_t *,
    int32_t *, int32_t *, int32_t *, int32_t *, int32_t *);
#endif

static int read_u32(FILE *stream, uint32_t *value) {
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, stream) != 4) return 0;
    *value = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
        (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
    return 1;
}

static int read_u64(FILE *stream, uint64_t *value) {
    uint32_t low, high;
    if (!read_u32(stream, &low) || !read_u32(stream, &high)) return 0;
    *value = (uint64_t)low | (uint64_t)high << 32;
    return 1;
}

static int bounded_region(uint64_t offset, uint64_t elements,
    uint64_t element_bytes, uint64_t arena_bytes) {
    return element_bytes && offset <= arena_bytes &&
        !(offset % element_bytes) &&
        elements <= (arena_bytes - offset) / element_bytes;
}

static int overlaps(uint64_t a, uint64_t a_bytes,
    uint64_t b, uint64_t b_bytes) {
    return a < b + b_bytes && b < a + a_bytes;
}

static int parse_phoneme_ids(const char *text, int32_t *ids, size_t count) {
    if (count < 3 || count > 40 || !text) return 0;
    const char *cursor = text;
    for (size_t i = 0; i < count; ++i) {
        if (*cursor < '0' || *cursor > '9') return 0;
        char *end = NULL;
        errno = 0;
        const long parsed = strtol(cursor, &end, 10);
        if (errno || end == cursor || parsed < 0 || parsed >= 178) return 0;
        ids[i] = (int32_t)parsed;
        if (i + 1 == count) {
            if (*end) return 0;
        } else {
            if (*end != ',') return 0;
            cursor = end + 1;
        }
    }
    return ids[0] == 0 && ids[count - 1] == 0;
}

static int parse_runtime_phoneme_ids(const char *text, int32_t ids[64],
                                     size_t *valid_count) {
    if (!text || !*text) return 0;
    size_t count = 1;
    for (const char *cursor = text; *cursor; ++cursor)
        if (*cursor == ',' && ++count > 40) return 0;
    if (count < 3 || count > 40 || !parse_phoneme_ids(text, ids, count))
        return 0;
    for (size_t index = count; index < 64; ++index) ids[index] = 178;
    *valid_count = count;
    return 1;
}

/* CKVOICE1: magic, LE row count, LE 256 columns, then row-major FP32.
 * The importer verifies the source artifact; the host validates the native
 * package and reads only the row selected by this request. */
static int read_voice_row(const char *path, size_t row, float values[256]) {
    const uint16_t endian = 1;
    if (sizeof(float) != 4 || FLT_RADIX != 2 ||
        FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 ||
        *(const unsigned char *)&endian != 1) return 0;
    FILE *stream = fopen(path, "rb");
    if (!stream) return 0;
    char magic[8];
    uint32_t rows, columns;
    int okay = fread(magic, 1, 8, stream) == 8 &&
        memcmp(magic, "CKVOICE1", 8) == 0 &&
        read_u32(stream, &rows) && read_u32(stream, &columns) &&
        rows <= 4096 && columns == 256 && row < rows;
    if (okay) {
        const long expected = 16L + (long)rows * 256L * 4L;
        okay = fseek(stream, 0, SEEK_END) == 0 &&
            ftell(stream) == expected &&
            fseek(stream, 16L + (long)row * 256L * 4L, SEEK_SET) == 0 &&
            fread(values, sizeof(float), 256, stream) == 256;
        for (size_t i = 0; okay && i < 256; ++i)
            if (!isfinite(values[i])) okay = 0;
    }
    if (fclose(stream) != 0) okay = 0;
    return okay;
}

static int write_u16(FILE *stream, uint16_t value) {
    return fputc(value & 255, stream) != EOF &&
        fputc((value >> 8) & 255, stream) != EOF;
}

static int write_u32(FILE *stream, uint32_t value) {
    return write_u16(stream, (uint16_t)value) &&
        write_u16(stream, (uint16_t)(value >> 16));
}

static int write_wav(const char *path, const float *samples, size_t count) {
    if (count > (UINT32_MAX - 36u) / 2u) return 0;
    /* O_EXCL also rejects a symlink or hard link to an existing bundle asset.
     * A preflight lstat in main gives an early error; this open closes the race. */
    int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) return 0;
    FILE *stream = fdopen(descriptor, "wb");
    if (!stream) {
        close(descriptor);
        remove(path);
        return 0;
    }
    const uint32_t payload = (uint32_t)(count * 2u);
    int okay = fwrite("RIFF", 1, 4, stream) == 4 &&
        write_u32(stream, payload + 36u) &&
        fwrite("WAVEfmt ", 1, 8, stream) == 8 &&
        write_u32(stream, 16) && write_u16(stream, 1) &&
        write_u16(stream, 1) && write_u32(stream, 24000) &&
        write_u32(stream, 48000) && write_u16(stream, 2) &&
        write_u16(stream, 16) &&
        fwrite("data", 1, 4, stream) == 4 && write_u32(stream, payload);
    for (size_t i = 0; okay && i < count; ++i) {
        if (!isfinite(samples[i])) { okay = 0; break; }
        float value = samples[i];
        if (value > 1.0f) value = 1.0f;
        if (value < -1.0f) value = -1.0f;
        const int32_t scaled = (int32_t)lrintf(value * 32767.0f);
        okay = write_u16(stream, (uint16_t)(int16_t)scaled);
    }
    if (fclose(stream) != 0) okay = 0;
    if (!okay) remove(path);
    return okay;
}

int main(int argc, char **argv) {
#ifdef CK_TTS_EXPECTED_LIBRARY_SHA256
    if (!verify_loaded_library()) {
        fprintf(stderr, "generated library identity mismatch\n");
        return 9;
    }
#endif
    if (argc != 3 && argc != 4 && argc != 6) {
        fprintf(stderr, "usage: %s prepared-arena.cktts output.wav "
            "[seed [phoneme-ids-csv voice-bank.ckvoice]]\n", argv[0]);
        return 2;
    }
    struct stat existing_output;
    if (lstat(argv[2], &existing_output) == 0 || errno != ENOENT) {
        fprintf(stderr, "output path already exists or cannot be inspected\n");
        return 10;
    }
    uint64_t seed = 0;
    if (argc >= 4) {
        char *end = NULL;
        errno = 0;
        const unsigned long long parsed = strtoull(argv[3], &end, 0);
        if (errno || !end || end == argv[3] || *end || argv[3][0] == '-')
            return 2;
        seed = (uint64_t)parsed;
    }
    FILE *input = fopen(argv[1], "rb");
    if (!input) return 3;
    char magic[8];
    uint64_t bytes64, waveform_offset, waveform_capacity;
    uint64_t gaussian_offset = 0, gaussian_capacity = 0;
    uint64_t ids_offset = 0, ids_count = 0;
    uint64_t predictor_offset = 0, decoder_offset = 0;
    uint64_t request_length_offset = 0;
    uint32_t segments;
    if (fread(magic, 1, 8, input) != 8) {
        fclose(input); return 4;
    }
    const int runtime_request = memcmp(magic, "CKTTS004", 8) == 0;
#ifndef CK_TTS_RUNTIME_WAVEFORM
    if (runtime_request) { fclose(input); return 4; }
#endif
    const int request = runtime_request || memcmp(magic, "CKTTS003", 8) == 0;
    const int native_noise = request || memcmp(magic, "CKTTS002", 8) == 0;
    if ((!native_noise && memcmp(magic, "CKTTS001", 8) != 0) ||
        (request ? argc != 6 : native_noise ? argc != 4 : argc != 3) ||
        !read_u64(input, &bytes64) || !read_u64(input, &waveform_offset) ||
        !read_u64(input, &waveform_capacity) ||
        (native_noise && (!read_u64(input, &gaussian_offset) ||
                          !read_u64(input, &gaussian_capacity))) ||
        (request && (!read_u64(input, &ids_offset) ||
                     !read_u64(input, &ids_count) ||
                     !read_u64(input, &predictor_offset) ||
                     !read_u64(input, &decoder_offset))) ||
        (runtime_request && !read_u64(input, &request_length_offset)) ||
        !read_u32(input, &segments) ||
        bytes64 > SIZE_MAX - 63u || bytes64 == 0 ||
        bytes64 > UINT64_C(4) * 1024u * 1024u * 1024u ||
        !waveform_capacity ||
        !bounded_region(waveform_offset, waveform_capacity, 4, bytes64) ||
        (native_noise && (
            !bounded_region(gaussian_offset, gaussian_capacity, 4, bytes64) ||
            gaussian_capacity != 76800u * 9u ||
            overlaps(gaussian_offset, gaussian_capacity * 4,
                waveform_offset, waveform_capacity * 4))) ||
        (request && (CK_TTS_COMPILED_TOKEN_COUNT < 3 ||
            ids_count != CK_TTS_COMPILED_TOKEN_COUNT ||
            (runtime_request ? ids_count != 64 : ids_count > 40) ||
            !bounded_region(ids_offset, ids_count, 4, bytes64) ||
            !bounded_region(predictor_offset, 128, 4, bytes64) ||
            !bounded_region(decoder_offset, 128, 4, bytes64) ||
            overlaps(ids_offset, ids_count * 4, gaussian_offset,
                gaussian_capacity * 4) ||
            overlaps(ids_offset, ids_count * 4, waveform_offset,
                waveform_capacity * 4) ||
            overlaps(predictor_offset, 512, decoder_offset, 512) ||
            overlaps(predictor_offset, 512, gaussian_offset,
                gaussian_capacity * 4) ||
            overlaps(decoder_offset, 512, gaussian_offset,
                gaussian_capacity * 4) ||
            overlaps(predictor_offset, 512, waveform_offset,
                waveform_capacity * 4) ||
            overlaps(decoder_offset, 512, waveform_offset,
                waveform_capacity * 4) ||
            overlaps(ids_offset, ids_count * 4, predictor_offset, 512) ||
            overlaps(ids_offset, ids_count * 4, decoder_offset, 512)))) {
        fclose(input); return 4;
    }
    if (runtime_request && (!bounded_region(request_length_offset, 1, 4, bytes64) ||
            overlaps(request_length_offset, 4, ids_offset, ids_count * 4) ||
            overlaps(request_length_offset, 4, predictor_offset, 512) ||
            overlaps(request_length_offset, 4, decoder_offset, 512) ||
            overlaps(request_length_offset, 4, gaussian_offset,
                     gaussian_capacity * 4) ||
            overlaps(request_length_offset, 4, waveform_offset,
                     waveform_capacity * 4))) {
        fclose(input); return 4;
    }
    int32_t request_ids[64];
    float request_voice[256];
    size_t valid_count = (size_t)ids_count;
    if (request && (!(runtime_request ?
                    parse_runtime_phoneme_ids(argv[4], request_ids, &valid_count) :
                    parse_phoneme_ids(argv[4], request_ids, valid_count)) ||
                    !read_voice_row(argv[5], valid_count - 3u,
                                    request_voice))) {
        fclose(input); return 4;
    }
    const size_t arena_bytes = (size_t)bytes64;
    const size_t padded = (arena_bytes + 63u) & ~(size_t)63u;
    uint8_t *arena = aligned_alloc(64, padded);
    if (!arena) { fclose(input); return 5; }
    memset(arena, 0, padded);
    int okay = segments <= 1000000u;
    for (uint32_t i = 0; okay && i < segments; ++i) {
        uint64_t offset, length;
        okay = read_u64(input, &offset) && read_u64(input, &length) &&
            offset <= bytes64 && length <= bytes64 - offset &&
            fread(arena + (size_t)offset, 1, (size_t)length, input) == length;
    }
    if (fgetc(input) != EOF) okay = 0;
    fclose(input);
    if (!okay) { free(arena); return 6; }
    if (request) {
        memcpy(arena + (size_t)ids_offset, request_ids,
               (size_t)ids_count * 4u);
        if (runtime_request) {
            const int32_t length = (int32_t)valid_count;
            memcpy(arena + (size_t)request_length_offset, &length, 4);
        }
        memcpy(arena + (size_t)decoder_offset, request_voice, 128u * 4u);
        memcpy(arena + (size_t)predictor_offset, request_voice + 128,
               128u * 4u);
    }
    if (native_noise && audio_noise_counter_gaussian_f32(
            (float *)(arena + (size_t)gaussian_offset),
            (size_t)gaussian_capacity, (size_t)gaussian_capacity,
            seed, 0) != CK_AUDIO_EXTENT_OK) {
        free(arena); return 6;
    }
    int32_t lengths[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    const int status = CK_TTS_GENERATED_ENTRY(arena, arena_bytes,
        &lengths[0], &lengths[1], &lengths[2], &lengths[3],
        &lengths[4], &lengths[5], &lengths[6]
#ifdef CK_TTS_RUNTIME_WAVEFORM
        , &lengths[7], &lengths[8]
#endif
        );
#ifdef CK_TTS_RUNTIME_WAVEFORM
    const int sample_index = 5, frame_index = 6;
#else
    const int sample_index = 3, frame_index = 4;
#endif
    if (status || lengths[sample_index] <= 0 ||
        (uint64_t)lengths[sample_index] > waveform_capacity ||
        lengths[frame_index] <= 1 ||
        (int64_t)(lengths[frame_index] - 1) * 5 != lengths[sample_index]) {
        free(arena); return 7;
    }
    const float *waveform = (const float *)(arena + (size_t)waveform_offset);
    okay = write_wav(argv[2], waveform, (size_t)lengths[sample_index]);
    free(arena);
    if (!okay) return 8;
    printf("wrote %d samples at 24000 Hz\n", lengths[sample_index]);
    return 0;
}
