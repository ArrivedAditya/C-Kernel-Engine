/**
 * @file ck_cli_rwkv7_template.c
 * @brief C-Kernel-Engine v8 CLI Template — RWKV-7
 *
 * RWKV-7 CLI notes:
 *   - Trie tokenizer: vocab comes from rwkv-kernels/tokenizer.json,
 *     65536 entries, loaded by the model library, NOT BPE merges.
 *     Tokenization MUST happen inside the model library (trie walk over
 *     the JSON vocab). This file does no local BPE/merge-table encode.
 *   - Stop token: id 0 (<|endoftext|>) only. No other stop ids.
 *   - g1j prompt wrapper: build prompts as System:/User:/Assistant:
 *     lines, with a bos 0 prefix before the wrapped prompt and an
 *     "Assistant:" generation prefix at the end. Callers should wrap
 *     raw user text this way before encoding.
 *   - No serving-layer extensions here. Prompts and
 *     completions pass through as plain text only.
 *
 * This is a standalone CLI template that demonstrates the proper usage pattern:
 *   1. Allocate memory (via ck_model_init)
 *   2. Load weights (from bump file)
 *   3. Initialize tokenizer (from vocab data in model)
 *   4. Wait for user prompt
 *   5. Tokenize input (via model library)
 *   6. Run prefill (embed + forward)
 *   7. Run decode loop (sample + decode)
 *
 * Build:
 *   gcc -O3 -o ck-cli-rwkv7 ck_cli_rwkv7_template.c -ldl -lm
 *
 * Usage:
 *   ./ck-cli-rwkv7 <libmodel.so> <weights.bump> [-p "prompt"]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <time.h>
#include <math.h>

/* ============================================================================
 * API Function Types (loaded via dlsym)
 * ============================================================================ */

typedef int (*init_fn_t)(const char *weights_path);
typedef void (*free_fn_t)(void);
typedef int (*embed_fn_t)(const int32_t *tokens, int num_tokens);
typedef int (*forward_fn_t)(float *logits_out);
typedef int (*decode_fn_t)(int32_t token, float *logits_out);
typedef int32_t (*sample_fn_t)(void);
typedef float* (*get_logits_fn_t)(void);
typedef int (*get_int_fn_t)(void);
typedef void (*kv_reset_fn_t)(void);
typedef const int32_t* (*get_vocab_offsets_fn_t)(void);
typedef const char* (*get_vocab_strings_fn_t)(void);
typedef const int32_t* (*get_vocab_merges_fn_t)(void);
/* RWKV-7 trie tokenizer provided by the model library (optional symbols). */
typedef int (*tok_encode_fn_t)(const char *text, int text_len, int32_t *ids_out, int max_ids);
typedef const char* (*tok_decode_fn_t)(int32_t id);

/* ============================================================================
 * Model API Structure
 * ============================================================================ */

typedef struct {
    void *handle;

    /* Core API */
    init_fn_t init;
    free_fn_t free_model;
    embed_fn_t embed_tokens;
    forward_fn_t forward;
    decode_fn_t decode;
    sample_fn_t sample_argmax;
    get_logits_fn_t get_logits;
    kv_reset_fn_t kv_reset;

    /* Model info */
    get_int_fn_t get_vocab_size;
    get_int_fn_t get_context_window;
    get_int_fn_t get_active_tokens;

    /* Vocab accessors (for tokenizer) */
    get_int_fn_t get_num_merges;
    get_int_fn_t get_vocab_strings_size;
    get_vocab_offsets_fn_t get_vocab_offsets;
    get_vocab_strings_fn_t get_vocab_strings;
    get_vocab_merges_fn_t get_vocab_merges;

    /* RWKV-7 trie tokenizer (resolved via dlsym when the model provides it) */
    tok_encode_fn_t tokenizer_encode_fn;
    tok_decode_fn_t tokenizer_decode_fn;
} ModelAPI;

/* ============================================================================
 * Tokenizer stub (RWKV-7: no local encode — model library owns the trie)
 * ============================================================================ */

typedef struct {
    int vocab_size;
    const int32_t *vocab_offsets;
    const char *vocab_strings;
    int num_merges;
    const int32_t *vocab_merges;
} SimpleTokenizer;

static SimpleTokenizer* tokenizer_create(void) {
    return (SimpleTokenizer*)calloc(1, sizeof(SimpleTokenizer));
}

static void tokenizer_free(SimpleTokenizer *tok) {
    if (tok) free(tok);
}

static int tokenizer_init(SimpleTokenizer *tok,
                          int vocab_size,
                          const int32_t *vocab_offsets,
                          const char *vocab_strings,
                          int num_merges,
                          const int32_t *vocab_merges) {
    tok->vocab_size = vocab_size;
    tok->vocab_offsets = vocab_offsets;
    tok->vocab_strings = vocab_strings;
    tok->num_merges = num_merges;
    tok->vocab_merges = vocab_merges;
    return 0;
}

/* Refuses to tokenize locally: the model library must provide the RWKV-7
 * trie encode via dlsym (ck_model_tokenizer_encode) or a sidecar loader
 * for tokenizer.json. Returning -1 forces the caller to use the model
 * path instead of falling back to local byte values. */
static int tokenizer_encode(SimpleTokenizer *tok, const char *text, int text_len,
                           int32_t *ids, int max_ids) {
    (void)tok;
    (void)text;
    (void)text_len;
    (void)ids;
    (void)max_ids;
    return -1;
}

/* Decode token ID to string (local vocab view only; prefer model decode). */
static const char* tokenizer_decode(SimpleTokenizer *tok, int32_t id) {
    static char buf[256];
    if (id < 0 || id >= tok->vocab_size || !tok->vocab_offsets || !tok->vocab_strings) {
        buf[0] = '\0';
        return buf;
    }

    int offset = tok->vocab_offsets[id];
    int next_offset = (id + 1 < tok->vocab_size) ? tok->vocab_offsets[id + 1] : offset;
    int len = next_offset - offset;
    if (len < 0 || len > 255) len = 0;

    memcpy(buf, tok->vocab_strings + offset, len);
    buf[len] = '\0';
    return buf;
}

/* Route encode through the model library trie; fail hard when absent.
 * We must NOT silently use byte values as token IDs. */
static int model_tokenizer_encode(ModelAPI *api, const char *text, int text_len,
                                  int32_t *ids, int max_ids) {
    if (api->tokenizer_encode_fn) {
        return api->tokenizer_encode_fn(text, text_len, ids, max_ids);
    }
    fprintf(stderr,
            "Error: model library does not provide ck_model_tokenizer_encode; "
            "load tokenizer.json (65536-entry trie) via the model library or a "
            "sidecar loader. Refusing to fall back to byte values as token IDs.\n");
    exit(1);
}

/* Route decode through the model library when available. */
static const char* model_tokenizer_decode(ModelAPI *api, SimpleTokenizer *tok, int32_t id) {
    if (api->tokenizer_decode_fn) {
        return api->tokenizer_decode_fn(id);
    }
    if (!api->tokenizer_encode_fn) {
        fprintf(stderr,
                "Error: model library does not provide ck_model_tokenizer_decode; "
                "load tokenizer.json (65536-entry trie) via the model library or a "
                "sidecar loader. Refusing to fall back to byte values as token IDs.\n");
        exit(1);
    }
    return tokenizer_decode(tok, id);
}

/* ============================================================================
 * Model Loading
 * ============================================================================ */

static bool load_symbol(void *handle, const char *name, void **out, bool required) {
    *out = dlsym(handle, name);
    if (!*out && required) {
        fprintf(stderr, "Error: missing symbol '%s'\n", name);
        return false;
    }
    return true;
}

static bool load_model(const char *lib_path, ModelAPI *api) {
    memset(api, 0, sizeof(*api));

    api->handle = dlopen(lib_path, RTLD_NOW);
    if (!api->handle) {
        fprintf(stderr, "Error: dlopen failed: %s\n", dlerror());
        return false;
    }

    /* Required functions */
    if (!load_symbol(api->handle, "ck_model_init", (void**)&api->init, true)) return false;
    if (!load_symbol(api->handle, "ck_model_embed_tokens", (void**)&api->embed_tokens, true)) return false;
    if (!load_symbol(api->handle, "ck_model_forward", (void**)&api->forward, true)) return false;
    if (!load_symbol(api->handle, "ck_model_decode", (void**)&api->decode, true)) return false;
    if (!load_symbol(api->handle, "ck_model_sample_argmax", (void**)&api->sample_argmax, true)) return false;

    /* Optional functions */
    load_symbol(api->handle, "ck_model_free", (void**)&api->free_model, false);
    load_symbol(api->handle, "ck_model_get_logits", (void**)&api->get_logits, false);
    /* ck_model_kv_cache_reset resets the recurrent RWKV-7 state
     * (att_shift/ffn_shift/wkv/v_first), not a KV cache. Symbol name is kept
     * for API compatibility with the generic CLI template. */
    load_symbol(api->handle, "ck_model_kv_cache_reset", (void**)&api->kv_reset, false);
    load_symbol(api->handle, "ck_model_get_vocab_size", (void**)&api->get_vocab_size, false);
    load_symbol(api->handle, "ck_model_get_context_window", (void**)&api->get_context_window, false);
    load_symbol(api->handle, "ck_model_get_active_tokens", (void**)&api->get_active_tokens, false);
    load_symbol(api->handle, "ck_model_get_num_merges", (void**)&api->get_num_merges, false);
    load_symbol(api->handle, "ck_model_get_vocab_strings_size", (void**)&api->get_vocab_strings_size, false);
    load_symbol(api->handle, "ck_model_get_vocab_offsets", (void**)&api->get_vocab_offsets, false);
    load_symbol(api->handle, "ck_model_get_vocab_strings", (void**)&api->get_vocab_strings, false);
    load_symbol(api->handle, "ck_model_get_vocab_merges", (void**)&api->get_vocab_merges, false);
    /* RWKV-7 trie tokenizer entry points (optional; required at inference time). */
    load_symbol(api->handle, "ck_model_tokenizer_encode", (void**)&api->tokenizer_encode_fn, false);
    load_symbol(api->handle, "ck_model_tokenizer_decode", (void**)&api->tokenizer_decode_fn, false);

    return true;
}

/* ============================================================================
 * Inference
 * ============================================================================ */

static void run_inference(ModelAPI *api, SimpleTokenizer *tok,
                         const char *prompt, int max_tokens) {
    int vocab_size = api->get_vocab_size ? api->get_vocab_size() : 32000;
    int context_len = api->get_context_window ? api->get_context_window() : 2048;
    (void)vocab_size;

    /* Tokenize input via the model library trie (no local byte fallback).
     * Expected prompt shape: g1j wrapper with System:/User:/Assistant: lines,
     * bos 0 prefix, and trailing Assistant: generation prefix. */
    int32_t *tokens = (int32_t*)malloc(context_len * sizeof(int32_t));
    int num_tokens = model_tokenizer_encode(api, prompt, -1, tokens, context_len);

    printf("Prompt: \"%s\" (%d tokens)\n", prompt, num_tokens);
    printf("Generating...\n\n");

    /* Reset recurrent state: ck_model_kv_cache_reset resets the recurrent
     * RWKV-7 state (att_shift/ffn_shift/wkv/v_first), not a KV cache. */
    if (api->kv_reset) {
        api->kv_reset();
    }

    /* STEP 1: Prefill - embed all prompt tokens and run forward */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    api->embed_tokens(tokens, num_tokens);
    api->forward(NULL);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double prefill_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6;

    /* Sample first token */
    int32_t next_token = api->sample_argmax();

    /* STEP 2: Decode loop - generate tokens one at a time */
    double decode_ms = 0.0;
    int generated = 0;

    printf("Output: ");
    fflush(stdout);

    for (int i = 0; i < max_tokens; i++) {
        /* Output token */
        const char *word = model_tokenizer_decode(api, tok, next_token);
        printf("%s", word);
        fflush(stdout);

        /* RWKV-7 stop condition: token 0 (<|endoftext|>) only. */
        if (next_token == 0) {
            break;
        }

        /* Decode next token */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        api->decode(next_token, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        decode_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 +
                     (t1.tv_nsec - t0.tv_nsec) / 1e6;

        /* Sample next */
        next_token = api->sample_argmax();
        generated++;
    }

    printf("\n\n");

    /* Print timing stats */
    printf("--- Stats ---\n");
    printf("Prefill: %d tokens in %.1f ms (%.1f tok/s)\n",
           num_tokens, prefill_ms, num_tokens / (prefill_ms / 1000.0));
    printf("Decode:  %d tokens in %.1f ms (%.1f tok/s)\n",
           generated, decode_ms, generated > 0 ? generated / (decode_ms / 1000.0) : 0);

    free(tokens);
}

/* ============================================================================
 * REPL (Read-Eval-Print Loop)
 * ============================================================================ */

static void run_repl(ModelAPI *api, SimpleTokenizer *tok) {
    char line[4096];

    printf("\nC-Kernel-Engine v8 CLI\n");
    printf("Type your prompt and press Enter. Type 'quit' to exit.\n\n");

    while (1) {
        printf("You: ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            break;
        }

        /* Remove trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }

        /* Check for quit command */
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) {
            break;
        }

        /* Skip empty lines */
        if (line[0] == '\0') {
            continue;
        }

        /* Reset recurrent state for new conversation: ck_model_kv_cache_reset
         * resets the recurrent RWKV-7 state (att_shift/ffn_shift/wkv/v_first),
         * not a KV cache. */
        if (api->kv_reset) {
            api->kv_reset();
        }

        printf("Assistant: ");
        fflush(stdout);

        /* Run inference */
        run_inference(api, tok, line, 256);
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */

static void print_usage(const char *prog) {
    printf("Usage: %s <libmodel.so> <weights.bump> [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -p, --prompt TEXT    Run single prompt instead of REPL\n");
    printf("  -n, --max-tokens N   Max tokens to generate (default: 256)\n");
    printf("  -h, --help           Show this help\n");
    printf("\nExamples:\n");
    printf("  %s ./model.so ./weights.bump\n", prog);
    printf("  %s ./model.so ./weights.bump -p \"Hello, world!\"\n", prog);
}

int main(int argc, char **argv) {
    const char *lib_path = NULL;
    const char *weights_path = NULL;
    const char *prompt = NULL;
    int max_tokens = 256;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            if (!lib_path) lib_path = argv[i];
            else if (!weights_path) weights_path = argv[i];
        }
    }

    if (!lib_path || !weights_path) {
        print_usage(argv[0]);
        return 1;
    }

    /* Load model library */
    printf("Loading model: %s\n", lib_path);

    ModelAPI api;
    if (!load_model(lib_path, &api)) {
        return 1;
    }

    /* Initialize model (allocate memory, load weights) */
    printf("Initializing model: %s\n", weights_path);

    if (api.init(weights_path) != 0) {
        fprintf(stderr, "Error: ck_model_init failed\n");
        return 1;
    }

    /* Get model info */
    int vocab_size = api.get_vocab_size ? api.get_vocab_size() : 32000;
    int context_len = api.get_context_window ? api.get_context_window() : 2048;

    printf("Model loaded: vocab=%d, context=%d\n", vocab_size, context_len);

    /* Initialize tokenizer: RWKV-7 trie lives in the model library
     * (tokenizer.json, 65536 entries). Require ck_model_tokenizer_encode;
     * never silently fall back to byte values as token IDs. */
    SimpleTokenizer *tok = tokenizer_create();

    if (api.tokenizer_encode_fn) {
        int num_merges = api.get_num_merges ? api.get_num_merges() : 0;
        tokenizer_init(tok, vocab_size,
                       api.get_vocab_offsets ? api.get_vocab_offsets() : NULL,
                       api.get_vocab_strings ? api.get_vocab_strings() : NULL,
                       num_merges,
                       api.get_vocab_merges ? api.get_vocab_merges() : NULL);
        printf("Tokenizer initialized from model\n");
    } else {
        fprintf(stderr,
                "Error: model library does not provide ck_model_tokenizer_encode; "
                "cannot tokenize without the RWKV-7 trie (tokenizer.json). Exiting.\n");
        tokenizer_free(tok);
        if (api.free_model) api.free_model();
        dlclose(api.handle);
        return 1;
    }

    /* Run inference */
    if (prompt) {
        run_inference(&api, tok, prompt, max_tokens);
    } else {
        run_repl(&api, tok);
    }

    /* Cleanup */
    tokenizer_free(tok);
    if (api.free_model) api.free_model();
    dlclose(api.handle);

    printf("Done.\n");
    return 0;
}
