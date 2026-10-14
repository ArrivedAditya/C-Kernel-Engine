/* Dictionary-only English text preparation for Kokoro's native phoneme
 * dispatcher. This is not the full context-dependent Misaki G2P frontend. */
#define _POSIX_C_SOURCE 200809L
#include "frontend_identity.h"
#include <errno.h>
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int digest_matches(FILE *stream) {
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    unsigned char digest[EVP_MAX_MD_SIZE], bytes[65536];
    unsigned int digest_len = 0;
    int okay = context && EVP_DigestInit_ex(context, EVP_sha256(), NULL) == 1;
    size_t length;
    while (okay && (length = fread(bytes, 1, sizeof(bytes), stream)) != 0)
        okay = EVP_DigestUpdate(context, bytes, length) == 1;
    if (ferror(stream)) okay = 0;
    if (okay) okay = EVP_DigestFinal_ex(context, digest, &digest_len) == 1;
    EVP_MD_CTX_free(context);
    if (fseek(stream, 0, SEEK_SET) != 0) okay = 0;
    if (!okay || digest_len != 32) return 0;
    static const char hex[] = "0123456789abcdef";
    for (unsigned i = 0; i < digest_len; ++i)
        if (CK_TTS_LEXICON_SHA256[2 * i] != hex[digest[i] >> 4] ||
            CK_TTS_LEXICON_SHA256[2 * i + 1] != hex[digest[i] & 15]) return 0;
    return 1;
}

static int append(int32_t ids[40], size_t *count, int32_t id) {
    if (*count >= 39 || id <= 0 || id >= 178) return 0;
    ids[(*count)++] = id;
    return 1;
}

/* The offline exporter caps each record below this fixed line capacity. */
static int lookup(FILE *lexicon, const char *word,
                  int32_t ids[40], size_t *count) {
    char line[512];
    rewind(lexicon);
    if (!fgets(line, sizeof(line), lexicon) || strcmp(line, "CKLEX1\n"))
        return -1;
    while (fgets(line, sizeof(line), lexicon)) {
        if (!strchr(line, '\n')) return -1;
        char *tab = strchr(line, '\t');
        if (!tab) return -1;
        *tab++ = 0;
        if (strcmp(line, word)) continue;
        char *cursor = tab;
        for (;;) {
            char *end = NULL;
            errno = 0;
            const long value = strtol(cursor, &end, 10);
            if (errno || end == cursor || value <= 0 || value >= 178 ||
                !append(ids, count, (int32_t)value)) return -1;
            if (*end == '\n' && end[1] == 0) return 1;
            if (*end != ',') return -1;
            cursor = end + 1;
        }
    }
    return ferror(lexicon) ? -1 : 0;
}

static int punctuation(unsigned char character) {
    switch (character) {
        case '.': return CK_TTS_DOT_ID;
        case ',': return CK_TTS_COMMA_ID;
        case '!': return CK_TTS_EXCLAMATION_ID;
        case '?': return CK_TTS_QUESTION_ID;
        default: return 0;
    }
}

static int encode_text(FILE *lexicon, const char *text,
                       int32_t ids[40], size_t *count) {
    const size_t length = strlen(text);
    if (!length || length > 512) return 0;
    *count = 1;
    ids[0] = 0;
    int have_token = 0, pending_space = 0, word_ended = 0;
    for (size_t i = 0; i < length;) {
        const unsigned char ch = (unsigned char)text[i];
        if (ch == ' ' || ch == '\t' || ch == '\n') {
            if (have_token) pending_space = 1;
            ++i;
            continue;
        }
        if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')) {
            char word[65];
            size_t used = 0;
            while (i < length) {
                unsigned char value = (unsigned char)text[i];
                if (value >= 'A' && value <= 'Z') value += 'a' - 'A';
                if (!((value >= 'a' && value <= 'z') ||
                      (value == '\'' && used && i + 1 < length &&
                       ((text[i + 1] >= 'A' && text[i + 1] <= 'Z') ||
                        (text[i + 1] >= 'a' && text[i + 1] <= 'z'))))) break;
                if (used == sizeof(word) - 1) return 0;
                word[used++] = (char)value;
                ++i;
            }
            word[used] = 0;
            if (have_token && !pending_space) return 0;
            if (pending_space && !append(ids, count, CK_TTS_SPACE_ID)) return 0;
            const int found = lookup(lexicon, word, ids, count);
            if (found != 1) {
                fprintf(stderr, "unsupported Kokoro lexicon word: %s\n", word);
                return 0;
            }
            have_token = 1;
            pending_space = 0;
            word_ended = 1;
            continue;
        }
        const int id = punctuation(ch);
        if (!id || !word_ended || pending_space ||
            !append(ids, count, id)) return 0;
        word_ended = 0;
        have_token = 1;
        ++i;
    }
    if (!have_token || *count >= 40) return 0;
    ids[(*count)++] = 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s bundle-dir lexicon.cklex output.wav seed text\n",
                argv[0]);
        return 2;
    }
    FILE *lexicon = fopen(argv[2], "rb");
    if (!lexicon || !digest_matches(lexicon)) {
        fprintf(stderr, "Kokoro native lexicon identity mismatch\n");
        if (lexicon) fclose(lexicon);
        return 3;
    }
    int32_t ids[40];
    size_t count = 0;
    const int okay = encode_text(lexicon, argv[5], ids, &count);
    if (fclose(lexicon) != 0 || !okay) {
        fprintf(stderr, "unsupported text or phoneme capacity\n");
        return 2;
    }
    char csv[160];
    size_t used = 0;
    for (size_t i = 0; i < count; ++i) {
        int written = snprintf(csv + used, sizeof(csv) - used, "%s%d",
                               i ? "," : "", ids[i]);
        if (written < 0 || (size_t)written >= sizeof(csv) - used) return 2;
        used += (size_t)written;
    }
    char dispatcher[4096];
    int written = snprintf(dispatcher, sizeof(dispatcher),
                           "%s/native-kokoro-dispatch", argv[1]);
    if (written <= 0 || (size_t)written >= sizeof(dispatcher)) return 4;
    char *const child[] = {dispatcher, argv[1], argv[3], argv[4], csv, NULL};
    execv(dispatcher, child);
    perror("Kokoro native dispatch failed");
    return 5;
}
