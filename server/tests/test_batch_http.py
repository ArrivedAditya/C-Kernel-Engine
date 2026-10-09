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
from server.session_v8 import CK_SESSION_REQUEST_RAW_PROMPT, SessionBusyError, SessionError


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


class RaisingAdvanceLoop(FakeLoop):
    def advance(self):
        assert self.gate.wait(5)
        raise RuntimeError("injected native owner failure")


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


def test_owner_exception_fails_active_and_pending_consumers_once():
    loop = RaisingAdvanceLoop()
    owner = Batch2HTTPOwner(FakeSession(), max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    a, b = owner.reserve(), owner.reserve()
    assert a.start_worker(None) and b.start_worker(None)
    def generate(lease, text):
        return lease.session.generate(None, text, max_tokens=4,
                                      temperature=0, top_p=1,
                                      on_token=lambda *_: 0,
                                      flags=CK_SESSION_REQUEST_RAW_PROMPT)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(generate, a, "A")
        deadline = time.monotonic() + 3
        while not loop.active_tickets() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert loop.active_tickets()
        fb = pool.submit(generate, b, "B")
        deadline = time.monotonic() + 3
        while owner.pending.qsize() != 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert owner.pending.qsize() == 1
        loop.gate.set()
        for future in (fa, fb):
            try:
                future.result(timeout=3)
            except SessionError:
                pass
            else:
                raise AssertionError("failed owner returned a successful response")
    assert owner.poisoned and not owner.worker.is_alive()
    assert not owner.slots and owner.pending.empty()
    assert a.client.events.empty() and b.client.events.empty()
    try:
        owner.reserve()
    except SessionError:
        pass
    else:
        raise AssertionError("poisoned owner accepted another request")
    a.release()
    b.release()
    owner.close()


def test_owner_exception_rejects_a_request_still_tokenizing():
    class BlockingEncode(FakeSession):
        def __init__(self):
            self.entered = threading.Event()
            self.resume = threading.Event()

        def encode_ids(self, prompt):
            if prompt == "B":
                self.entered.set()
                assert self.resume.wait(5)
            return super().encode_ids(prompt)

    session = BlockingEncode()
    loop = RaisingAdvanceLoop()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop)
    a, b = owner.reserve(), owner.reserve()
    assert a.start_worker(None) and b.start_worker(None)
    def generate(lease, text):
        return lease.session.generate(None, text, max_tokens=4,
                                      temperature=0, top_p=1,
                                      on_token=lambda *_: 0,
                                      flags=CK_SESSION_REQUEST_RAW_PROMPT)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(generate, a, "A")
        deadline = time.monotonic() + 3
        while not loop.active_tickets() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert loop.active_tickets()
        fb = pool.submit(generate, b, "B")
        assert session.entered.wait(3)
        loop.gate.set()
        try:
            fa.result(timeout=3)
        except SessionError:
            pass
        else:
            raise AssertionError("active request did not observe owner failure")
        session.resume.set()
        try:
            fb.result(timeout=3)
        except SessionError:
            pass
        else:
            raise AssertionError("tokenizing request did not observe owner failure")
    assert not owner.slots and owner.pending.empty()
    assert b.client.events.get_nowait() == ("done", "runtime_error")
    assert b.client.events.empty()
    a.release()
    b.release()
    owner.close()


def test_completed_slow_consumers_have_a_separate_admission_cap():
    loop = FakeLoop()
    loop.gate.set()
    owner = Batch2HTTPOwner(FakeSession(), max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop,
                            max_delivery_consumers=2)
    for text in ("A", "B"):
        lease = owner.reserve()
        assert lease.start_worker(None)
        assert lease.session.generate(None, text, max_tokens=1,
                                      temperature=0, top_p=1,
                                      on_token=lambda *_: 0,
                                      flags=CK_SESSION_REQUEST_RAW_PROMPT)["generated_tokens"] == 1
        lease.release()
    assert not owner.slots
    assert len(owner.clients) == 2
    try:
        owner.reserve()
    except SessionBusyError:
        pass
    else:
        raise AssertionError("slow delivery did not bound admission")
    first = next(iter(owner.clients.values()))
    owner.delivery_done(first)
    newer = owner.reserve()
    newer.release()
    owner.close()


def test_decoded_output_byte_cap_fails_only_that_consumer():
    loop = FakeLoop()
    loop.gate.set()
    session = FakeSession()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop,
                            max_output_bytes=2)
    app = create_app(session, model="m", context_length=128,
                     chat_template="{{ messages[0].content }}", batch_http=owner)
    client = TestClient(app)
    failed = client.post("/v1/responses", json={
        "model": "m", "input": "A", "max_output_tokens": 3})
    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    recovered = client.post("/v1/responses", json={
        "model": "m", "input": "B", "max_output_tokens": 2})
    assert recovered.status_code == 200
    assert recovered.json()["output_text"] == "BB"
    owner.close()


def test_output_token_delivery_cap_rejects_before_native_execution():
    loop = FakeLoop()
    loop.gate.set()
    session = FakeSession()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop,
                            max_output_tokens=2)
    app = create_app(session, model="m", context_length=128,
                     chat_template="{{ messages[0].content }}", batch_http=owner)
    client = TestClient(app)
    rejected = client.post("/v1/responses", json={
        "model": "m", "input": "A", "max_output_tokens": 3})
    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "batch_output_limit_exceeded"
    assert not loop.calls
    followup = client.post("/v1/responses", json={
        "model": "m", "input": "B", "max_output_tokens": 1})
    assert followup.json()["output_text"] == "B"
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
        assert len(owner.clients) == 1, "disconnected consumer lost native ownership early"
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
                            max_prompt_tokens=8, loop=loop,
                            max_delivery_consumers=2)
    app = create_app(session, model="m", context_length=128,
                     chat_template="{{ messages[0].content }}", batch_http=owner)
    scope = {"type": "http", "asgi": {"version": "3.0"},
             "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/responses", "raw_path": b"/v1/responses",
             "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"application/json")],
             "client": ("127.0.0.1", 10000), "server": ("127.0.0.1", 80)}
    blocked = [threading.Event(), threading.Event()]
    unblock = [threading.Event(), threading.Event()]
    errors = []

    def serve(index):
        async def exchange():
            received = False
            body_sends = 0
            payload = json.dumps({"model": "m", "input": "AB"[index],
                                  "stream": True, "max_output_tokens": 4}).encode()

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
                        blocked[index].set()
                        await asyncio.to_thread(unblock[index].wait)

            await app(scope, receive, send)

        try:
            asyncio.run(exchange())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=serve, args=(index,), daemon=True)
               for index in range(2)]
    for thread in threads:
        thread.start()
    try:
        assert all(event.wait(3) for event in blocked), "streams did not reach blocked sends"
        deadline = time.monotonic() + 3
        while owner.slots and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not owner.slots, "completed native work retained a slow reader's slot"
        assert len(owner.clients) == 2
        overflow = TestClient(app).post("/v1/responses", json={
            "model": "m", "input": "C", "max_output_tokens": 1})
        assert overflow.status_code == 429
        unblock[0].set()
        threads[0].join(timeout=3)
        assert not threads[0].is_alive()
        followup = TestClient(app).post("/v1/responses", json={
            "model": "m", "input": "C", "max_output_tokens": 1})
        assert followup.json()["output_text"] == "C"
    finally:
        for event in unblock:
            event.set()
        for thread in threads:
            thread.join(timeout=3)
        owner.close()
    assert all(not thread.is_alive() for thread in threads)
    assert not errors


def test_stalled_stream_send_times_out_without_stalling_owner():
    loop = FakeLoop()
    loop.gate.set()
    session = FakeSession()
    owner = Batch2HTTPOwner(session, max_extra_bytes=1,
                            max_prompt_tokens=8, loop=loop,
                            delivery_timeout=0.1)
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
    errors = []
    sent_bodies = []

    def serve():
        async def exchange():
            received = False
            async def receive():
                nonlocal received
                if not received:
                    received = True
                    return {"type": "http.request", "body": payload,
                            "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                if message["type"] == "http.response.body":
                    sent_bodies.append(message.get("body", b""))
                    blocked.set()
                    await asyncio.Event().wait()

            await app(scope, receive, send)
        try:
            asyncio.run(exchange())
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        assert blocked.wait(3)
        thread.join(timeout=3)
        assert not thread.is_alive(), "stalled delivery did not terminate"
        assert not errors
        assert not owner.clients
        response_id = re.search(rb"resp_[0-9a-f]{24}", b"".join(sent_bodies)).group().decode()
        stored = TestClient(app).get(f"/v1/responses/{response_id}")
        assert stored.json()["status"] == "cancelled"
        followup = TestClient(app).post("/v1/responses", json={
            "model": "m", "input": "B", "max_output_tokens": 1})
        assert followup.json()["output_text"] == "B"
    finally:
        owner.close()


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
