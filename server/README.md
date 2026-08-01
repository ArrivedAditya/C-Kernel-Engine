# Experimental Responses Schema Scaffold

This directory is a development scaffold for a subset of the OpenAI Responses
API shape. It validates request, response, conversation, and server-sent event
schemas with deterministic mock data.

It is not an OpenAI-compatible inference server:

- no CKE model is loaded;
- no request invokes CKE inference;
- response text and usage are mocked;
- stores are process-local and non-durable;
- there is no bounded queue, backpressure, authentication, or real
  cancellation.

FastAPI is temporary scaffolding so contributors can iterate on the HTTP
contract quickly. The intended production path is a dedicated C or Rust server
that integrates directly with CKE, loads each model once, owns a bounded
request queue, propagates cancellation into generation, and emits streaming
events from the native token loop.

## Native runtime boundary

The stable host boundary is `include/ck_session_v8.h`, implemented by
`build/libck_session_v8.so`. Build it with:

```bash
make ck-session-v8
```

A Python prototype may load that library with `ctypes` or `cffi`; a Rust server
may bind the same C ABI. The host opens one generated model session, then calls
`ck_session_v8_generate`. CKE performs circuit-derived chat formatting, native
tokenization, model execution, generated stop/timestamp policy, and native
detokenization. The callback receives each token ID and its UTF-8 bytes, which
the HTTP layer can translate into response or SSE events.

Sampling values such as `temperature` and `top_p` belong to each request. They
do not select the tokenizer. The generated model declares tokenizer, chat,
stop-token, and modality capabilities at compile time. Session requests reset
KV/recurrent state by default; callers must explicitly set
`CK_SESSION_REQUEST_CONTINUE_STATE` to continue an existing sequence.

The current session ABI deliberately fails closed when a generated model lacks
the required tokenizer or chat capability. It does not infer a tokenizer or
chat template from the model name. The FastAPI scaffold remains mocked until a
separate PR binds this ABI and adds lifecycle, queue, cancellation, streaming,
and integration tests.

Run the schema tests with:

```bash
python3 -m pip install -r server/requirements.txt
make test-server-schema
```

Do not add server flags to `ck_chat.py` or `ck_run_v8.py` until a real
`cks-v8-run serve` subcommand owns model lifecycle and starts the server.
