from __future__ import annotations

import threading
import asyncio
import json
import socket
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
import uvicorn

from server.batch_decode import BatchTick, CompletionEvent, TokenEvent
from server.batch_http import Batch2HTTPOwner
from server.live import create_app
from server.session_v8 import CK_SESSION_REQUEST_RAW_PROMPT, SessionBusyError


class FakeSession:
    context_length = 128

    def encode_ids(self, prompt):
        return [ord(prompt[0])]

    def count_tokens(self, prompt):
        return len(self.encode_ids(prompt))

    def decode_ids(self, ids):
        return bytes(ids)


class FakeLoop:
    def __init__(self):
        self.guard = threading.Lock()
        self.active = {}
        self.next_ticket = 1
        self.gate = threading.Event()
        self.calls = []
        self.cancelled = set()

    def submit(self, slot, ids, *, max_tokens, **_):
        with self.guard:
            ticket = self.next_ticket
            self.next_ticket += 1
            self.active[ticket] = [slot, ids[0], max_tokens]
            return ticket

    def active_tickets(self):
        with self.guard:
            return tuple(self.active)

    def request_cancel(self, ticket):
        with self.guard:
            if ticket in self.active:
                self.cancelled.add(ticket)

    def advance(self):
        assert self.gate.wait(5)
        tokens, completed = [], []
        with self.guard:
            self.calls.append(tuple(sorted(self.active)))
            mask = sum(1 << state[0] for state in self.active.values())
            for ticket, (slot, token, remaining) in tuple(self.active.items()):
                cancelled = ticket in self.cancelled
                if remaining and not cancelled:
                    tokens.append(TokenEvent(ticket, slot, token))
                    remaining -= 1
                if cancelled or not remaining:
                    completed.append(CompletionEvent(
                        ticket, slot, "cancelled" if cancelled else "token_limit", 1))
                    del self.active[ticket]
                    self.cancelled.discard(ticket)
                else:
                    self.active[ticket][2] = remaining
        return BatchTick(tuple(tokens), tuple(completed), mask)


class ToolLoop(FakeLoop):
    def submit(self, slot, ids, *, max_tokens, **_):
        with self.guard:
            ticket = self.next_ticket
            self.next_ticket += 1
            label = chr(ids[0])
            output = ('<tool_call>{"name":"probe","arguments":{"x":"'
                      + label + '"}}</tool_call>')
            self.active[ticket] = [slot, output, 0]
            return ticket

    def advance(self):
        assert self.gate.wait(5)
        tokens, completed = [], []
        with self.guard:
            self.calls.append(tuple(sorted(self.active)))
            mask = sum(1 << state[0] for state in self.active.values())
            for ticket, (slot, output, index) in tuple(self.active.items()):
                tokens.append(TokenEvent(ticket, slot, ord(output[index])))
                index += 1
                if index == len(output):
                    completed.append(CompletionEvent(ticket, slot, "eos", index))
                    del self.active[ticket]
                else:
                    self.active[ticket][2] = index
        return BatchTick(tuple(tokens), tuple(completed), mask)


class FailOnceLoop(FakeLoop):
    def __init__(self):
        super().__init__()
        self.gate.set()
        self.fail_next = True

    def advance(self):
        if self.fail_next:
            self.fail_next = False
            with self.guard:
                failed = tuple(self.active.items())
                self.active.clear()
            return BatchTick((), tuple(
                CompletionEvent(ticket, state[0], "runtime_error", 0)
                for ticket, state in failed), None)
        return super().advance()


class TextStopLoop(ToolLoop):
    def submit(self, slot, ids, *, max_tokens, **_):
        with self.guard:
            ticket = self.next_ticket
            self.next_ticket += 1
            self.active[ticket] = [slot, "X<STOP>Y", 0]
            return ticket


class ContinuationLoop(ToolLoop):
    def submit(self, slot, ids, *, max_tokens, **_):
        if ids[0] != ord("C"):
            return super().submit(slot, ids, max_tokens=max_tokens)
        with self.guard:
            ticket = self.next_ticket
            self.next_ticket += 1
            self.active[ticket] = [slot, "done", 0]
            return ticket


def test_two_http_clients_have_independent_text_and_output_limits():
    loop = FakeLoop()
    owner = Batch2HTTPOwner(FakeSession(), max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    app = create_app(FakeSession(), model="m", context_length=128,
                     chat_template="{{ messages[0].content }}",
                     batch_http=owner)
    client = TestClient(app)

    def send(letter, limit):
        return client.post("/v1/responses", json={
            "model": "m", "input": letter, "max_output_tokens": limit,
            "temperature": 0,
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(send, "A", 3)
        two = pool.submit(send, "B", 2)
        # Both request IDs own separate slots before generated execution.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with owner.guard:
                if len(owner.slots) == 2 and owner.pending.qsize() and loop.active_tickets():
                    break
            time.sleep(0.001)
        else:
            raise AssertionError("two HTTP requests were not admitted")
        loop.gate.set()
        a, b = one.result(timeout=5), two.result(timeout=5)
    assert a.status_code == b.status_code == 200
    assert a.json()["output_text"] == "AAA"
    assert b.json()["output_text"] == "BB"
    assert a.json()["id"] != b.json()["id"]
    assert a.json()["status"] == b.json()["status"] == "incomplete"
    assert any(len(c) == 2 for c in loop.calls)
    assert owner.shared_steps > 0
    owner.close()


def test_slot_overflow_and_reuse():
    loop = FakeLoop()
    owner = Batch2HTTPOwner(FakeSession(), max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    a, b = owner.reserve(), owner.reserve()
    try:
        try:
            owner.reserve()
        except SessionBusyError:
            pass
        else:
            raise AssertionError("third concurrent request was admitted")
    finally:
        a.release()
        b.release()
    c = owner.reserve()
    c.release()
    owner.close()


def test_http_overflow_is_429_and_native_failure_recovers():
    loop = FailOnceLoop()
    class CountingSession(FakeSession):
        def count_tokens(self, prompt):
            return len(prompt)

    session = CountingSession()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    app = create_app(session, model="m", context_length=128,
                     chat_template="{{ messages[0].content }}", batch_http=owner)
    client = TestClient(app)
    held = (owner.reserve(), owner.reserve())
    overflow = client.post("/v1/responses", json={"model": "m", "input": "C"})
    assert overflow.status_code == 429
    assert overflow.json()["error"]["code"] == "rate_limit_exceeded"
    for lease in held:
        lease.release()
    oversized = client.post("/v1/responses", json={
        "model": "m", "input": "ABCDEFGHI", "max_output_tokens": 1})
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "batch_prompt_limit_exceeded"
    assert not loop.active_tickets()
    failed = client.post("/v1/responses", json={
        "model": "m", "input": "A", "max_output_tokens": 1})
    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    recovered = client.post("/v1/responses", json={
        "model": "m", "input": "B", "max_output_tokens": 1})
    assert recovered.status_code == 200
    assert recovered.json()["output_text"] == "B"
    owner.close()


def test_cancelled_slot_cannot_cancel_reused_request():
    loop = FakeLoop()
    owner = Batch2HTTPOwner(FakeSession(), max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    first = owner.reserve()
    first.start_worker(None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(first.session.generate, None, "A", max_tokens=1,
                             temperature=0, top_p=1, on_token=lambda *_: 0,
                             flags=CK_SESSION_REQUEST_RAW_PROMPT)
        loop.gate.set()
        assert future.result(timeout=5)["generated_tokens"] == 1
    first.release()
    second = owner.reserve()
    first.cancel()
    assert not second.client.cancelled.is_set()
    second.release()
    owner.close()


def test_cancellation_of_one_slot_does_not_stop_the_other():
    loop = FakeLoop()
    owner = Batch2HTTPOwner(FakeSession(), max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    a, b = owner.reserve(), owner.reserve()
    a.start_worker(None)
    b.start_worker(None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(a.session.generate, None, "A", max_tokens=4,
                         temperature=0, top_p=1, on_token=lambda *_: 0,
                         flags=CK_SESSION_REQUEST_RAW_PROMPT)
        fb = pool.submit(b.session.generate, None, "B", max_tokens=2,
                         temperature=0, top_p=1, on_token=lambda *_: 0,
                         flags=CK_SESSION_REQUEST_RAW_PROMPT)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with owner.guard:
                if a.client.ticket is not None and owner.pending.qsize():
                    break
            time.sleep(0.001)
        else:
            raise AssertionError("first ticket and second submission were not ready")
        a.cancel()
        loop.gate.set()
        assert fa.result(timeout=5)["stop_reason"] == 3
        assert fb.result(timeout=5)["generated_tokens"] == 2
    a.release()
    b.release()
    owner.close()


def test_concurrent_tool_calls_keep_arguments_and_call_ids_separate():
    loop = ToolLoop()
    session = FakeSession()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    app = create_app(session, model="m", context_length=128,
                     chat_template="{{ messages[0].content }}|{{ tools | tojson }}",
                     tool_protocol="tagged_json", batch_http=owner)
    client = TestClient(app)

    def send(label):
        return client.post("/v1/responses", json={
            "model": "m", "input": label, "max_output_tokens": 100,
            "tools": [{"type": "function", "name": "probe", "parameters": {
                "type": "object", "properties": {"x": {"type": "string"}}}}],
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(send, "A")
        second = pool.submit(send, "B")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with owner.guard:
                if len(owner.slots) == 2 and owner.pending.qsize() and loop.active_tickets():
                    break
            time.sleep(0.001)
        else:
            raise AssertionError("tool requests were not admitted")
        loop.gate.set()
        results = first.result(timeout=5), second.result(timeout=5)
    assert all(r.status_code == 200 for r in results)
    calls = [next(item for item in r.json()["output"]
                  if item["type"] == "function_call") for r in results]
    assert [call["arguments"] for call in calls] == ['{"x":"A"}', '{"x":"B"}']
    assert calls[0]["call_id"] != calls[1]["call_id"]
    assert any(len(c) == 2 for c in loop.calls)
    owner.close()


def test_declared_text_stop_is_hidden_and_completes_response():
    loop = TextStopLoop()
    loop.gate.set()
    session = FakeSession()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    app = create_app(session, model="m", context_length=128,
                     chat_template="{{ messages[0].content }}", batch_http=owner,
                     stop_on_text=["<STOP>"])
    response = TestClient(app).post("/v1/responses", json={
        "model": "m", "input": "A", "max_output_tokens": 32})
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["output_text"] == "X"
    owner.close()


def test_tool_result_continuation_uses_its_own_response_history():
    loop = ContinuationLoop()
    loop.gate.set()
    session = FakeSession()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    app = create_app(session, model="m", context_length=128,
                     chat_template="{% if messages[-1].role == 'tool' %}C{% else %}A{% endif %}",
                     tool_protocol="tagged_json", batch_http=owner)
    client = TestClient(app)
    tools = [{"type": "function", "name": "probe", "parameters": {}}]
    first = client.post("/v1/responses", json={
        "model": "m", "input": "A", "tools": tools,
        "max_output_tokens": 100}).json()
    call = next(item for item in first["output"]
                if item["type"] == "function_call")
    second = client.post("/v1/responses", json={
        "model": "m", "previous_response_id": first["id"],
        "input": [{"type": "function_call_output", "call_id": call["call_id"],
                   "output": "result A"}],
        "tools": tools, "max_output_tokens": 10}).json()
    assert second["status"] == "completed"
    assert second["output_text"] == "done"
    assert not any(item["type"] == "function_call" for item in second["output"])
    owner.close()


def test_stream_socket_disconnect_releases_only_its_batch_slot():
    loop = FakeLoop()
    session = FakeSession()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    app = create_app(session, model="m", context_length=128,
                     chat_template="{{ messages[0].content }}", batch_http=owner)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]},
                              daemon=True)
    thread.start()
    connection = None
    try:
        deadline = time.monotonic() + 3
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        payload = json.dumps({"model": "m", "input": "A", "stream": True,
                              "max_output_tokens": 4}).encode()
        connection = socket.create_connection(("127.0.0.1", port), timeout=3)
        connection.sendall(
            b"POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nContent-Length: "
            + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        headers = b""
        while b"\r\n\r\n" not in headers:
            headers += connection.recv(4096)
        assert b"200 OK" in headers.split(b"\r\n", 1)[0]
        deadline = time.monotonic() + 3
        while not owner.tickets and time.monotonic() < deadline:
            time.sleep(0.01)
        assert owner.tickets, "native decode did not start before disconnect"
        connection.close()
        connection = None
        # Keep the native step blocked until the disconnect has requested
        # cancellation. The other slot must remain independently usable.
        deadline = time.monotonic() + 3
        while not any(c.cancelled.is_set() for c in owner.slots.values()) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert any(c.cancelled.is_set() for c in owner.slots.values())
        loop.gate.set()
        deadline = time.monotonic() + 3
        while owner.slots and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not owner.slots
        followup = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/responses",
            data=json.dumps({"model": "m", "input": "B",
                             "max_output_tokens": 1}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(followup, timeout=3) as response:
            assert json.load(response)["output_text"] == "B"
    finally:
        loop.gate.set()
        if connection is not None:
            connection.close()
        server.should_exit = True
        thread.join(timeout=3)
        listener.close()
        owner.close()


def test_blocked_stream_send_does_not_hold_completed_native_slot():
    loop = FakeLoop()
    loop.gate.set()
    session = FakeSession()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    app = create_app(session, model="m", context_length=128,
                     chat_template="{{ messages[0].content }}", batch_http=owner)
    payload = json.dumps({"model": "m", "input": "A", "stream": True,
                          "max_output_tokens": 4}).encode()
    scope = {"type": "http", "asgi": {"version": "3.0"},
             "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/responses", "raw_path": b"/v1/responses",
             "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"application/json")],
             "client": ("127.0.0.1", 10000), "server": ("127.0.0.1", 80)}
    blocked = threading.Event()
    unblock = threading.Event()
    errors = []

    def serve():
        async def exchange():
            received = False
            body_sends = 0

            async def receive():
                nonlocal received
                if not received:
                    received = True
                    return {"type": "http.request", "body": payload,
                            "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                nonlocal body_sends
                if message["type"] == "http.response.body":
                    body_sends += 1
                    if body_sends == 3:
                        blocked.set()
                        await asyncio.to_thread(unblock.wait)

            await app(scope, receive, send)

        try:
            asyncio.run(exchange())
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        assert blocked.wait(3), "stream did not reach blocked send"
        deadline = time.monotonic() + 3
        while owner.slots and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not owner.slots, "completed native work retained a slow reader's slot"
        followup = TestClient(app).post("/v1/responses", json={
            "model": "m", "input": "B", "max_output_tokens": 1})
        assert followup.json()["output_text"] == "B"
    finally:
        unblock.set()
        thread.join(timeout=3)
        owner.close()
    assert not thread.is_alive()
    assert not errors


def test_http_cancel_waits_for_batch_step_then_allows_followup():
    loop = FakeLoop()
    session = FakeSession()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    app = create_app(session, model="m", context_length=128,
                     chat_template="{{ messages[0].content }}", batch_http=owner)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]},
                              daemon=True)
    thread.start()
    connection = None
    try:
        deadline = time.monotonic() + 3
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        payload = json.dumps({"model": "m", "input": "A", "stream": True,
                              "max_output_tokens": 4}).encode()
        connection = socket.create_connection(("127.0.0.1", port), timeout=3)
        connection.sendall(
            b"POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nContent-Length: "
            + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        received = b""
        while not re.search(rb"resp_[0-9a-f]{24}", received):
            received += connection.recv(4096)
        response_id = re.search(rb"resp_[0-9a-f]{24}", received).group().decode()
        deadline = time.monotonic() + 3
        while not owner.tickets and time.monotonic() < deadline:
            time.sleep(0.01)
        assert owner.tickets

        def cancel():
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/responses/{response_id}/cancel",
                data=b"", method="POST")
            with urllib.request.urlopen(request, timeout=3) as response:
                return json.load(response)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(cancel)
            deadline = time.monotonic() + 3
            while not any(c.cancelled.is_set() for c in owner.slots.values()) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert any(c.cancelled.is_set() for c in owner.slots.values())
            loop.gate.set()
            assert future.result(timeout=3)["status"] == "cancelled"
        followup = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/responses",
            data=json.dumps({"model": "m", "input": "B",
                             "max_output_tokens": 1}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(followup, timeout=3) as response:
            assert json.load(response)["output_text"] == "B"
    finally:
        loop.gate.set()
        if connection is not None:
            connection.close()
        server.should_exit = True
        thread.join(timeout=3)
        listener.close()
        owner.close()


def test_shutdown_waits_for_inflight_tokenizer_use():
    class BlockingEncode(FakeSession):
        def __init__(self):
            self.entered = threading.Event()
            self.release_encode = threading.Event()

        def encode_ids(self, prompt):
            self.entered.set()
            assert self.release_encode.wait(5)
            return super().encode_ids(prompt)

    session = BlockingEncode()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=FakeLoop())
    lease = owner.reserve()
    assert lease.start_worker(None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        generation = pool.submit(lease.session.generate, None, "A",
                                 max_tokens=1, temperature=0, top_p=1,
                                 on_token=lambda *_: 0,
                                 flags=CK_SESSION_REQUEST_RAW_PROMPT)
        assert session.entered.wait(3)
        shutdown = pool.submit(owner.close)
        deadline = time.monotonic() + 3
        while not owner.shutdown.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert owner.shutdown.is_set()
        assert not lease.client.finished.is_set()
        session.release_encode.set()
        assert generation.result(timeout=3)["stop_reason"] == 3
        lease.release()
        shutdown.result(timeout=3)
