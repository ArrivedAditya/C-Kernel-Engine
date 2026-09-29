"""Execution ownership and unsupported-input regressions for every text model."""
import asyncio
import hashlib
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from server import live
from server.schemas.response import CreateResponseRequest
from version.v8.scripts.ck_serve_v8 import create_app

TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}"


class Session:
    def __init__(self):
        self.calls = 0
        self.cancel_calls = 0
        self.release = threading.Event()
        self.finished = threading.Event()

    def generate(self, *args, on_token, **kwargs):
        self.calls += 1
        on_token(0, 'answer')
        if self.calls == 1:
            assert self.release.wait(5)
        self.finished.set()
        return {'prompt_tokens': 1, 'generated_tokens': 1, 'stop_reason': 1}

    def cancel(self):
        self.cancel_calls += 1


def test_native_completion_releases_slot_without_consuming_terminal_event():
    """A paused/disconnected SSE consumer cannot retain the native slot."""
    session = Session()
    app = create_app(session, model='test', chat_template=TEMPLATE)
    def routes(items):
        for route in items:
            if hasattr(route, 'original_router'):
                yield from routes(route.original_router.routes)
            else:
                yield route
    endpoint = next(r.endpoint for r in routes(app.routes) if getattr(r, 'path', '').endswith('/responses') and 'POST' in r.methods)
    request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/responses', 'headers': []})

    async def run():
        response = endpoint(CreateResponseRequest(model='test', input='first', stream=True), request)
        iterator = response.body_iterator
        while 'response.output_text.delta' not in await anext(iterator):
            pass
        assert app.state.flight_lock.locked()
        session.release.set()
        for _ in range(100):
            if not app.state.flight_lock.locked():
                break
            await asyncio.sleep(.01)
        assert not app.state.flight_lock.locked()
        assert not app.state.active_streams
        # The first HTTP iterator is still paused before its terminal event.
        second = endpoint(CreateResponseRequest(model='test', input='second'), request)
        assert second['status'] == 'completed'
        response.lease.disconnect()
        assert session.cancel_calls == 0
        await iterator.aclose()

    asyncio.run(run())


def test_stale_owner_never_cancels_new_request_and_active_owner_retains_slot():
    lock = threading.Lock()
    session = Session()
    lock.acquire()
    old = live._FlightLease(lock, session)
    old.start_worker(threading.Event())
    old.release()
    lock.acquire()
    current = live._FlightLease(lock, session)
    cancelled = threading.Event()
    current.start_worker(cancelled)
    old.disconnect()
    old.cancel()
    assert session.cancel_calls == 0
    current.disconnect()
    assert session.cancel_calls == 1 and cancelled.is_set()
    assert lock.locked()  # Cancellation request is not native completion.
    current.release()
    current.release()
    assert not lock.locked()


def test_disconnect_before_stream_iteration_releases_reserved_slot():
    lock = threading.Lock()
    lock.acquire()
    lease = live._FlightLease(lock, Session())
    lease.disconnect()
    assert not lock.locked()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('route', ['responses', 'chat/completions'])
@pytest.mark.parametrize('media', ['image', 'file', 'tool_image', 'tool_file'])
def test_media_rejected_before_renderer_tokenizer_and_native_execution(monkeypatch, stream, route, media):
    def forbidden(*args, **kwargs):
        pytest.fail('unsupported media reached computation')
    session = Session()
    session.generate = session.count_tokens = forbidden
    monkeypatch.setattr(live, '_render_with_chat_templates', forbidden)
    app = create_app(session, model='test', context_length=4096, chat_template=TEMPLATE)
    client = TestClient(app)
    part = {'type': 'input_image', 'image_url': 'https://example.org/a.png'} if media.endswith('image') else {'type': 'input_file', 'file_id': 'example'}
    if route == 'responses':
        content = [{'type': 'function_call_output', 'call_id': 'call_1', 'output': [part]}] if media.startswith('tool') else [{'type': 'message', 'role': 'user', 'content': [part]}]
        payload = {'model': 'test', 'input': content, 'stream': stream}
    else:
        chat_part = {'type': 'image_url', 'image_url': {'url': 'https://example.org/a.png'}} if media.endswith('image') else {'type': 'file', 'file': {'file_id': 'example'}}
        payload = {'model': 'test', 'messages': [{'role': 'tool' if media.startswith('tool') else 'user', 'tool_call_id': 'call_1', 'content': [chat_part]}], 'stream': stream}
    result = client.post('/v1/' + route, json=payload)
    assert result.status_code in (422, 400)
    assert session.calls == 0


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('user_text', ['literal <think>', 'literal </think>', '/tmp/image.png'])
def test_literal_user_tags_cannot_open_output_reasoning(stream, user_text):
    session = Session()
    session.release.set()
    template = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}assistant\n{% endif %}"
    client = TestClient(create_app(session, model='test', chat_template=template))
    result = client.post('/v1/responses', json={'model': 'test', 'input': user_text, 'stream': stream, 'reasoning': {'effort': 'medium'}})
    assert result.status_code == 200
    if stream:
        assert 'response.reasoning_text.delta' not in result.text
        assert 'response.output_text.delta' in result.text
    else:
        assert result.json()['output_text'] == 'answer'


def test_render_only_order_and_pinned_publisher_fixture():
    root = Path(__file__).parent / 'fixtures'
    provenance = json.loads((root / 'qwen_template_provenance.json').read_text())
    for filename, item in provenance.items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == item['sha256']
    parts = [{'type': 'input_text', 'text': 'A'}, {'type': 'input_image', 'image_url': 'https://example.org/a.png'}, {'type': 'input_text', 'text': 'B'}, {'type': 'input_image', 'image_url': 'https://example.org/b.png'}]
    normalized = live._ordered_content_parts(parts)
    assert [p['type'] for p in normalized] == ['text', 'image', 'text', 'image']
    body = CreateResponseRequest(model='test', input='text')
    template = (root / 'qwen35_08b_publisher.jinja').read_text()
    rendered = live._render_with_chat_templates(template, None, [{'role': 'user', 'content': normalized}], body)
    assert rendered.index('A') < rendered.index('<|image_pad|>') < rendered.index('B') < rendered.rindex('<|image_pad|>')
