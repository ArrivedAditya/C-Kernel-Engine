Issue #878: externally staged Qwen2 prefill, grouped-row boundary

The selected Q4_K x Q8_K prefill provider declares a four-row reduction
group. An initial 127-token call necessarily stores its last three rows
using the declared tail reduction. Appending token 128 cannot change those
previously stored rows into the reduction used by an uninterrupted 128-row
call. Exact-input replay located the first arithmetic difference at the
layer-2 MLP-down projection; layer-9 K was the first observed KV mismatch.
Changing the tail provider globally would violate its existing numerical
contract and oracle tests.

Generated runtimes now export ck_model_prefill_append_row_group(). For a
provider with group size greater than one, ck_model_embed_tokens returns -3
before native state mutation when an external append starts after a partial
group. An aligned singleton append uses the generated prefill-continuation
path. Full-prompt execution and internal chunking retain their existing
provider arithmetic. Callers with the full prompt can split at a multiple
of the exported row group. A caller that has already committed a partial
group must reset and replay from an aligned boundary before appending.

The adjacent JSON files are artifact-bound local x86 results from the
Qwen2-0.5B-Instruct Q4_K_M fixture, four native threads. Separate
uninterrupted and 127-token-chunk bundles were rebuilt using this change;
they share serialized weights and loaded engine/tokenizer identities.
120+8, 124+4, and 128+1 were exactly equal to uninterrupted execution for
full logits and all 24 KV layers after prefill and three decode steps.
One-call internal 429-token chunking was also exact. The formerly silent
127+1 case returned -3 with unchanged logits, position and all KV layers;
reset and 124+4 retry succeeded. These are local generated-runtime results,
not a claim about every ISA, model, or serving client.

Reproduction, with separate full and chunked bundles built from the same
weights and current source:

  OMP_NUM_THREADS=4 CK_NUM_THREADS=4 python \
    version/v8/scripts/certify_prefill_append_rejection_v8.py \
    --bundle CHUNKED_BUNDLE --tokens TOKENS_128_JSON \
    --split-at 127 --json-out rejected.json

  OMP_NUM_THREADS=4 CK_NUM_THREADS=4 python \
    version/v8/scripts/compare_chunked_prefill_v8.py \
    --full FULL_BUNDLE --chunked CHUNKED_BUNDLE \
    --tokens TOKENS_128_JSON \
    --continuation tests/fixtures/issue833/continuation-3.json \
    --split-at 124 --json-out aligned.json --require-exact

TOKENS_128_JSON is the first 128 integer IDs from
tests/fixtures/issue833/qwen2-natural-429.json. The remaining reports use
the indicated split position or full 429-token fixture. Loaded-library
digests in the reports are verified against their actual backing mappings.
