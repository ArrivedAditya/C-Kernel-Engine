/* Fixed-utterance native Kokoro diagnostic host. The generated model owns
 * every neural operation; this host loads caller-owned storage and writes WAV. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int ck_kokoro_waveform_bounded(uint8_t *, size_t, int32_t *, int32_t *,
    int32_t *, int32_t *, int32_t *, int32_t *, int32_t *);

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
    FILE *stream = fopen(path, "wb");
    if (!stream) return 0;
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
    if (argc != 3) {
        fprintf(stderr, "usage: %s prepared-arena.cktts output.wav\n", argv[0]);
        return 2;
    }
    FILE *input = fopen(argv[1], "rb");
    if (!input) return 3;
    char magic[8];
    uint64_t bytes64, waveform_offset, waveform_capacity;
    uint32_t segments;
    if (fread(magic, 1, 8, input) != 8 ||
        memcmp(magic, "CKTTS001", 8) != 0 ||
        !read_u64(input, &bytes64) || !read_u64(input, &waveform_offset) ||
        !read_u64(input, &waveform_capacity) || !read_u32(input, &segments) ||
        bytes64 > SIZE_MAX - 63u || bytes64 == 0 ||
        bytes64 > UINT64_C(4) * 1024u * 1024u * 1024u ||
        waveform_offset > bytes64 ||
        (waveform_offset & (sizeof(float) - 1u)) || !waveform_capacity ||
        waveform_capacity > (bytes64 - waveform_offset) / sizeof(float)) {
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
    int32_t lengths[7] = {-1, -1, -1, -1, -1, -1, -1};
    const int status = ck_kokoro_waveform_bounded(arena, arena_bytes,
        &lengths[0], &lengths[1], &lengths[2], &lengths[3],
        &lengths[4], &lengths[5], &lengths[6]);
    if (status || lengths[3] <= 0 ||
        (uint64_t)lengths[3] > waveform_capacity ||
        lengths[4] <= 1 ||
        (int64_t)(lengths[4] - 1) * 5 != lengths[3]) {
        free(arena); return 7;
    }
    const float *waveform = (const float *)(arena + (size_t)waveform_offset);
    okay = write_wav(argv[2], waveform, (size_t)lengths[3]);
    free(arena);
    if (!okay) return 8;
    printf("wrote %d samples at 24000 Hz\n", lengths[3]);
    return 0;
}
