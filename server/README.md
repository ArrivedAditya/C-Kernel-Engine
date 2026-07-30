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

Run the schema tests with:

```bash
python3 -m pip install -r server/requirements.txt
make test-server-schema
```

Do not add server flags to `ck_chat.py` or `ck_run_v8.py` until a real
`cks-v8-run serve` subcommand owns model lifecycle and starts the server.
