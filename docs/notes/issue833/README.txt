Issue #833: full versus multi-chunk prefill, 2026-10-21

The adjacent JSON reports are diagnostic evidence from isolated generated
bundles on DGX Spark (aarch64, OMP_NUM_THREADS=4). They bind the weights,
manifest, generated C, model library, engine and tokenizer by SHA-256.
The token arrays are in tests/fixtures/issue833. The comparison tool runs
each bundle in a separate process and checks full logits and every exported
KV layer after prefill and three teacher-forced decode steps.

To replay, build full and chunked bundles from the same source and weights,
then run, for example:

  python version/v8/scripts/compare_chunked_prefill_v8.py \
    --full FULL_BUNDLE --chunked CHUNKED_BUNDLE \
    --tokens tests/fixtures/issue833/gemma-natural-429.json \
    --continuation tests/fixtures/issue833/continuation-3.json \
    --json-out comparison.json --require-exact

Gemma3, chunk 127: exact logits and 18/18 KV layers at all four stages.
Qwen3, chunk 128: exact logits and 28/28 KV layers at all four stages.
Qwen2, chunk 128: nonexact logits and 3/24 exact KV layers; the top token
still agrees at all four stages. The first tiny Qwen2 difference appears in
layer 2's MLP-down projection at prompt length 132: full prefill uses a
multirow Q4_K x Q8_K GEMM, while continuation uses a one-row reduction.
The 5,548-byte Q8_K input row is identical on both sides (SHA-256
acaaa7f59de4dd5a2aa9bcad15cadfb19e3eaa8cb0774c1f9c505bfabb1469c4).
The approximately 2.5e-7 projection difference later crosses a quantized
activation boundary in layer 13. Running the GEMM provider at M=1 produced
the same small difference, so simply swapping the function name does not
recover the multirow reduction order. No tolerance has been promoted from
this diagnostic. Issue #833 should stay open for that numerical contract.

These reports do not certify task quality, model output protocols, or the
actually loaded library mapping used by a serving process.
