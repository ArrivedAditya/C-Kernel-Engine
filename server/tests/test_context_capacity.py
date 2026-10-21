"""Context-capacity admission contract for the serving path.

Three cases are kept distinct:
1. The rendered prompt alone exceeds the loaded context capacity -> rejected
   with 400 context_length_exceeded before any model execution, never
   silently truncated.
2. The prompt fits but the requested output reservation exceeds the
   remaining capacity -> rejected under the declared admission policy.
3. An admitted request that actually reaches its output limit -> truthful
   incomplete/length stop (incomplete_details.reason == "max_output_tokens").

Every rejection must leave the session usable for the next request, and the
native side receives CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION so a missed
admission fails instead of truncating.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from server.live import create_app
from server.session_v8 import CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION


class CapacitySession:
    def __init__(self, prompt_token_count: int = 5, stop_reason: int = 1) -> None:
        self.prompt_token_count = prompt_token_count
        self.stop_reason = stop_reason
        self.calls: list[dict] = []

    def count_tokens(self, text: str) -> int:
        return self.prompt_token_count

    def generate(self, system, user, *, max_tokens, temperature, top_p, on_token,
                 flags=0, stop_on_text=(), stop_at_eos=False):
        self.calls.append({"flags": flags, "max_tokens": max_tokens})
        on_token(1, "done")
        return {
            "prompt_tokens": self.prompt_token_count,
            "generated_tokens": 1,
            "stop_reason": self.stop_reason,
        }

    def close(self):
        pass


def _client(session, context_length=64):
    app = create_app(session, model="test", context_length=context_length,
                     allow_untemplated=True)
    return TestClient(app)


def _post(client, text="hello", max_output_tokens=8):
    return client.post(
        "/v1/responses",
        json={"model": "test", "input": text, "max_output_tokens": max_output_tokens},
    )


def test_prompt_alone_over_capacity_rejected_and_session_stays_usable() -> None:
    session = CapacitySession(prompt_token_count=64)
    client = _client(session, context_length=64)
    rejected = _post(client)
    assert rejected.status_code == 400
    error = rejected.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert "before any output tokens are reserved" in error["message"]
    assert session.calls == [], "rejected request must not reach generation"
    session.prompt_token_count = 5
    ok = _post(client, text="hi")
    assert ok.status_code == 200
    assert ok.json()["status"] == "completed"


def test_output_reservation_over_remaining_capacity_rejected_then_usable() -> None:
    session = CapacitySession(prompt_token_count=10)
    client = _client(session, context_length=64)
    rejected = _post(client, max_output_tokens=60)
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "context_length_exceeded"
    assert session.calls == []
    ok = _post(client, max_output_tokens=8)
    assert ok.status_code == 200


def test_output_budget_equal_to_capacity_rejected() -> None:
    session = CapacitySession(prompt_token_count=1)
    client = _client(session, context_length=32)
    rejected = _post(client, max_output_tokens=32)
    assert rejected.status_code == 400
    assert session.calls == []


def test_zero_token_count_cannot_admit_nonempty_prompt() -> None:
    session = CapacitySession(prompt_token_count=0)
    client = _client(session, context_length=64)
    rejected = _post(client, text="long nonempty prompt")
    assert rejected.status_code == 500
    assert rejected.json()["error"]["code"] == "native_tokenization_failed"
    assert session.calls == []
    session.prompt_token_count = 5
    assert _post(client).status_code == 200


def test_oversized_prompt_stream_rejected_before_generation_then_usable() -> None:
    session = CapacitySession(prompt_token_count=65)
    client = _client(session, context_length=64)
    rejected = client.post("/v1/responses", json={
        "model": "test", "input": "too long", "max_output_tokens": 8, "stream": True,
    })
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "context_length_exceeded"
    assert session.calls == []
    session.prompt_token_count = 5
    assert _post(client).status_code == 200


def test_admitted_request_hitting_limit_reports_truthful_length_stop() -> None:
    session = CapacitySession(prompt_token_count=5, stop_reason=2)
    client = _client(session, context_length=64)
    response = _post(client)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "incomplete"
    assert payload["incomplete_details"] == {"reason": "max_output_tokens"}


def test_every_request_carries_native_reject_truncation_backstop() -> None:
    session = CapacitySession(prompt_token_count=5)
    client = _client(session, context_length=64)
    response = _post(client)
    assert response.status_code == 200
    assert session.calls, "admitted request must reach generation"
    for call in session.calls:
        assert call["flags"] & CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION
