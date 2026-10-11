"""Request-owned decode policy over the real session ABI and a fake model."""

import concurrent.futures
import ctypes
import subprocess
import threading
import time

import pytest

from server.batch_decode import Batch2RequestLoop
from server.session_v8 import SESSION_LIB_PATH, SessionError, SessionV8
from tests.test_v8_batch_session_scheduler import MODEL_SOURCE


@pytest.fixture
def fake_bundle(tmp_path):
    source = tmp_path / "fake_model.c"
    source.write_text(MODEL_SOURCE.replace("active->value+i", "active->value-i")
                      .replace("states[i]->value+j", "states[i]->value-j"))
    subprocess.run([
        "cc", "-shared", "-fPIC", "-I", "include", "-I", "version/v8/src",
        str(source), "-o", str(tmp_path / "libmodel.so"),
    ], check=True)
    (tmp_path / "weights.bump").write_bytes(b"")
    (tmp_path / "weights_manifest.map").write_text("")
    assert SESSION_LIB_PATH.is_file(), "build ck-session-v8 before this test"
    return tmp_path


def test_two_requests_cancel_reuse_eos_and_capacity(fake_bundle):
    session = SessionV8.open(fake_bundle, context_length=16, num_threads=1)
    try:
        loop = Batch2RequestLoop(session.enable_batch2(320))
        with pytest.raises(ValueError, match="context"):
            loop.submit(0, [1] * 16, max_tokens=2)
        last_position = loop.submit(0, [1] * 16, max_tokens=1)
        assert loop.advance().completed[0].ticket == last_position
        first = loop.submit(0, [1, 2], max_tokens=4)
        second = loop.submit(1, [3], max_tokens=3)
        with pytest.raises(Exception, match="already owns"):
            loop.submit(0, [4], max_tokens=1)
        tick = loop.advance()
        assert [(x.ticket, x.token_id) for x in tick.tokens] == [(first, 0), (second, 0)]
        assert tick.advanced_mask == 3 and not tick.completed

        model = ctypes.CDLL(str(fake_bundle / "libmodel.so"))
        model.ck_fake_hold_batch.argtypes = [ctypes.c_int]
        model.ck_fake_hold_batch(1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            in_flight = pool.submit(loop.advance)
            try:
                deadline = time.monotonic() + 5
                while not model.ck_fake_batch_entered() and time.monotonic() < deadline:
                    time.sleep(.001)
                assert model.ck_fake_batch_entered()
                assert loop.request_cancel(first)
            finally:
                model.ck_fake_hold_batch(0)
            tick = in_flight.result(timeout=5)
        assert tick.advanced_mask == 3
        assert [(x.ticket, x.reason) for x in tick.completed] == [(first, "cancelled")]
        assert loop.active_tickets() == (second,)
        assert not loop.request_cancel(first)

        reused = loop.submit(0, [7], max_tokens=2)
        assert reused != first
        tick = loop.advance()
        assert tick.advanced_mask == 1
        assert [(x.ticket, x.reason) for x in tick.completed] == [(second, "token_limit")]
        tick = loop.advance()
        assert tick.advanced_mask == 0
        assert [(x.ticket, x.reason) for x in tick.completed] == [(reused, "token_limit")]
        assert loop.active_tickets() == ()

        eos = loop.submit(1, [2], max_tokens=5, stop_ids=[0])
        other = loop.submit(0, [3], max_tokens=2)
        tick = loop.advance()
        assert [(x.ticket, x.token_id) for x in tick.tokens] == [(other, 0)]
        assert tick.advanced_mask == 1
        assert [(x.ticket, x.reason) for x in tick.completed] == [(eos, "eos")]
        assert [(x.ticket, x.reason) for x in loop.advance().completed] == [
            (other, "token_limit")]
    finally:
        session.close()


def test_bad_request_policy_is_rejected_before_prefill(fake_bundle):
    session = SessionV8.open(fake_bundle, context_length=16, num_threads=1)
    try:
        loop = Batch2RequestLoop(session.enable_batch2(320))
        for policy in ({"temperature": float("nan")}, {"top_p": 0.0},
                       {"seed": True}, {"stop_ids": [16]}, {"max_tokens": 0}):
            options = {"max_tokens": 1, **policy}
            with pytest.raises(ValueError):
                loop.submit(0, [1], **options)
        assert loop.active_tickets() == ()
        ticket = loop.submit(0, [1], max_tokens=1)
        assert loop.advance().completed[0].ticket == ticket
    finally:
        session.close()


def test_real_native_staged_prefill_interleaves_and_cancel_reuses(fake_bundle):
    session = SessionV8.open(fake_bundle, context_length=16, num_threads=1)
    try:
        loop = Batch2RequestLoop(session.enable_batch2(320))
        first = loop.submit(0, [1], max_tokens=4)
        second = loop.begin_staged(1, [2, 3, 4], max_tokens=2)
        assert loop.advance_prefill(second, 1) == ()
        assert loop.batch.position(0) == 1
        with pytest.raises(SessionError, match="capability"):
            loop.batch.position(1)
        assert any(token.ticket == first for token in loop.advance().tokens)
        assert loop.advance_prefill(second, 1) == ()
        assert loop.advance_prefill(second, 1) == ()
        assert loop.batch.position(1) == 3
        assert any(token.ticket == second for token in loop.advance().tokens)
        assert loop.request_cancel(second)
        assert any((e.ticket, e.reason) == (second, "cancelled")
                   for e in loop.advance().completed)
        assert not loop.request_cancel(second)
        replacement = loop.begin_staged(1, [7], max_tokens=1)
        loop.advance_prefill(replacement, 1)
        assert any(token.ticket == replacement for token in loop.advance().tokens)
    finally:
        session.close()


class _ControlledBatch:
    """Constant logits make per-request RNG and failure recovery observable."""

    def __init__(self, *, fail_step=False, fail_reset=False):
        self.parent = type("Parent", (), {
            "lib": ctypes.CDLL(str(SESSION_LIB_PATH)), "context_length": 16,
        })()
        self.vocab_size = 4
        self.slots = {}
        self.next_ticket = 1
        self.fail_step = fail_step
        self.fail_reset = fail_reset

    def prefill(self, slot, tokens):
        ticket = self.next_ticket
        self.next_ticket += 1
        logits = (ctypes.c_float * 4)(0, 0, 0, 0)
        self.slots[slot] = (ticket, logits)
        return ticket, logits

    def step(self, tokens):
        if self.fail_step:
            self.fail_step = False
            raise SessionError(-7, "injected native step failure")
        mask = 0
        for slot, (_, logits) in self.slots.items():
            mask |= 1 << slot
            for i in range(4):
                logits[i] = 0
        return mask, tuple(self.slots.get(i, (None, None))[1] for i in (0, 1))

    def reset_slot(self, slot):
        if self.fail_reset:
            self.fail_reset = False
            raise SessionError(-7, "injected native reset failure")
        self.slots.pop(slot, None)

    def request_cancel(self, ticket):
        pass


class _StagedBatch(_ControlledBatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefills = {}
        self.chunks = []

    def begin_prefill(self, slot, total_tokens):
        ticket = self.next_ticket
        self.next_ticket += 1
        self.prefills[slot] = [ticket, total_tokens, 0]
        return ticket

    def append_prefill(self, slot, ticket, tokens):
        state = self.prefills[slot]
        assert state[0] == ticket
        state[2] += len(tokens)
        assert state[2] <= state[1]
        self.chunks.append((ticket, tuple(tokens)))
        if state[2] == state[1]:
            self.slots[slot] = (ticket, (ctypes.c_float * 4)(0, 0, 0, 0))
            del self.prefills[slot]
        return state[2]

    def staged_logits(self, slot):
        return self.slots[slot][1]

    def reset_slot(self, slot):
        super().reset_slot(slot)
        self.prefills.pop(slot, None)


def test_sampling_rng_is_request_local_when_another_request_joins():
    together = Batch2RequestLoop(_ControlledBatch())
    first = together.submit(0, [1], max_tokens=4, temperature=1, seed=123)
    together.submit(1, [2], max_tokens=4, temperature=1, seed=456)
    concurrent_tokens = []
    for _ in range(4):
        concurrent_tokens.extend(e.token_id for e in together.advance().tokens
                                 if e.ticket == first)

    isolated = Batch2RequestLoop(_ControlledBatch())
    alone = isolated.submit(0, [1], max_tokens=4, temperature=1, seed=123)
    isolated_tokens = []
    for _ in range(4):
        isolated_tokens.extend(e.token_id for e in isolated.advance().tokens
                               if e.ticket == alone)
    assert concurrent_tokens == isolated_tokens
    assert len(concurrent_tokens) == 4


def test_staged_prefill_is_bounded_cancelled_and_reusable():
    batch = _StagedBatch()
    loop = Batch2RequestLoop(batch)
    with pytest.raises(ValueError, match="prompt IDs"):
        loop.begin_staged(0, [1, 4], max_tokens=1)
    assert batch.prefills == {}
    first = loop.begin_staged(0, [1, 2, 3, 1, 2], max_tokens=3)
    assert loop.active_tickets() == (first,)
    assert loop.advance_prefill(first, 2) == ()
    assert batch.chunks == [(first, (1, 2))]
    assert loop.advance().tokens == ()
    second = loop.begin_staged(1, [2, 3, 1], max_tokens=2)
    assert loop.request_cancel(second)
    completed = loop.advance_prefill(second, 2)
    assert [(c.ticket, c.reason) for c in completed] == [(second, "cancelled")]
    assert batch.chunks == [(first, (1, 2))]
    assert not loop.request_cancel(second)
    assert loop.advance_prefill(first, 2) == ()
    assert loop.advance_prefill(first, 2) == ()
    assert loop.staged_tickets() == ()
    tick = loop.advance()
    assert tick.tokens[0].ticket == first
    reused = loop.begin_staged(1, [3], max_tokens=1)
    assert reused != second
    loop.advance_prefill(reused, 1)
    tick = loop.advance()
    assert {t.ticket for t in tick.tokens} == {first, reused}
    assert not loop.request_cancel(second)


def test_staged_prefill_native_failure_settles_ready_and_pending():
    batch = _StagedBatch()
    loop = Batch2RequestLoop(batch)
    ready = loop.begin_staged(0, [1], max_tokens=3)
    loop.advance_prefill(ready, 1)
    pending = loop.begin_staged(1, [2, 3], max_tokens=3)
    batch.append_prefill = lambda *_: (_ for _ in ()).throw(SessionError(-7, "injected"))
    settled = loop.advance_prefill(pending, 1)
    assert {(e.ticket, e.reason) for e in settled} == {
        (ready, "runtime_error"), (pending, "runtime_error")}
    assert loop.active_tickets() == ()


def test_staged_cancel_during_append_waits_for_worker_then_reuses_slot():
    class BlockingBatch(_StagedBatch):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def append_prefill(self, slot, ticket, tokens):
            self.entered.set()
            assert self.release.wait(5)
            return super().append_prefill(slot, ticket, tokens)

    batch = BlockingBatch()
    loop = Batch2RequestLoop(batch)
    ticket = loop.begin_staged(0, [1, 2], max_tokens=1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        in_flight = pool.submit(loop.advance_prefill, ticket, 1)
        assert batch.entered.wait(5)
        assert loop.request_cancel(ticket)
        assert loop.active_tickets() == (ticket,)
        batch.release.set()
        result = in_flight.result(timeout=5)
    assert [(e.ticket, e.reason) for e in result] == [(ticket, "cancelled")]
    assert loop.active_tickets() == ()
    fresh = loop.begin_staged(0, [3], max_tokens=1)
    loop.advance_prefill(fresh, 1)
    assert loop.advance().completed[0].ticket == fresh


def test_random_draw_stays_below_one_after_float32_conversion():
    loop = Batch2RequestLoop(_ControlledBatch())
    ticket = loop.submit(0, [1], max_tokens=1, temperature=1, seed=0)
    loop._slots[0].rng = type("NearOne", (), {
        "random": lambda self: 0.9999999999999999,
    })()
    tick = loop.advance()
    assert tick.tokens[0].ticket == ticket
    assert tick.completed[0].reason == "token_limit"


def test_native_failure_reports_unknown_advancement_and_recovers():
    loop = Batch2RequestLoop(_ControlledBatch(fail_step=True))
    first = loop.submit(0, [1], max_tokens=2)
    second = loop.submit(1, [2], max_tokens=2)
    tick = loop.advance()
    assert tick.advanced_mask is None
    assert {c.ticket for c in tick.completed} == {first, second}
    assert {c.reason for c in tick.completed} == {"runtime_error"}
    assert loop.active_tickets() == ()
    recovered = loop.submit(0, [3], max_tokens=1)
    assert loop.advance().completed[0].ticket == recovered


def test_failed_reset_keeps_loop_unavailable_until_explicit_recovery():
    loop = Batch2RequestLoop(_ControlledBatch(fail_step=True, fail_reset=True))
    loop.submit(0, [1], max_tokens=2)
    tick = loop.advance()
    assert tick.advanced_mask is None
    assert tick.completed[0].reason == "runtime_error"
    with pytest.raises(SessionError, match="needs recovery"):
        loop.submit(0, [1], max_tokens=1)
    loop.recover()
    ticket = loop.submit(0, [1], max_tokens=1)
    assert loop.advance().completed[0].ticket == ticket
