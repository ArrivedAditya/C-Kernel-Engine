"""The host batch binding runs the real native session ABI against a fake model."""

import concurrent.futures
import ctypes
import subprocess
import time
from pathlib import Path

import pytest

from server import session_v8
from tests.test_v8_batch_session_scheduler import MODEL_SOURCE
from version.v8.scripts.certify_batch_mixed_one_row_v8 import (
    CAP_BATCH_TWO_ROWS, _compare_prefill_snapshots, _finite_logit_bytes,
    _require_shared_projections, certify,
)


def test_mixed_certificate_rejects_identical_nonfinite_logits_and_missing_shared_work():
    for invalid in (float("nan"), float("inf"), -float("inf")):
        reference = (ctypes.c_float * 3)(1, invalid, 2)
        mixed = (ctypes.c_float * 3)(1, invalid, 2)
        assert bytes(reference) == bytes(mixed)
        with pytest.raises(AssertionError, match="nonfinite logit at index 1"):
            _finite_logit_bytes(reference, "reference")
        with pytest.raises(AssertionError, match="nonfinite logit at index 1"):
            _finite_logit_bytes(mixed, "mixed")
    assert _finite_logit_bytes((ctypes.c_float * 2)(1, 2), "finite")
    for capabilities, groups in ((0, 72), (CAP_BATCH_TWO_ROWS, None),
                                 (CAP_BATCH_TWO_ROWS, 0)):
        with pytest.raises(AssertionError):
            _require_shared_projections(capabilities, groups)
    assert _require_shared_projections(CAP_BATCH_TWO_ROWS, 72) == 72


def test_ordinary_prefill_comparison_keeps_logit_kv_and_position_failures_distinct():
    ordinary = (bytes((ctypes.c_float * 3)(1, 4, 2)), 3, "a" * 64)
    assert _compare_prefill_snapshots(ordinary, ordinary)["status"] == "pass"

    changed_logits = (bytes((ctypes.c_float * 3)(5, 4, 2)), 3, ordinary[2])
    verdict = _compare_prefill_snapshots(ordinary, changed_logits)
    assert verdict["status"] == "fail"
    assert verdict["max_abs_logit_diff"] == 4
    assert not verdict["logits_exact"] and not verdict["top1_equal"]
    assert verdict["kv_exact"] and verdict["position_ordinary"] == verdict["position_staged"]

    changed_kv = _compare_prefill_snapshots(ordinary, (ordinary[0], 3, "b" * 64))
    assert changed_kv["status"] == "fail"
    assert changed_kv["logits_exact"] and not changed_kv["kv_exact"]

    changed_position = _compare_prefill_snapshots(ordinary, (ordinary[0], 2, ordinary[2]))
    assert changed_position["status"] == "fail"
    assert changed_position["logits_exact"] and changed_position["kv_exact"]

    with pytest.raises(AssertionError, match="extents differ"):
        _compare_prefill_snapshots(ordinary, (ordinary[0][:-4], 3, ordinary[2]))
    with pytest.raises(AssertionError, match="nonfinite"):
        _compare_prefill_snapshots(ordinary, (bytes((ctypes.c_float * 3)(1, float("inf"), 2)),
                                              3, ordinary[2]))


def test_ordinary_prefill_comparison_rejects_forced_decode_replay(monkeypatch):
    monkeypatch.setenv("CK_V8_FORCE_DECODE_PREFILL", "1")
    with pytest.raises(ValueError, match="cannot force decode replay"):
        certify(Path("unused"), compare_ordinary=True)


def test_staged_prefill_binding_tracks_progress_and_cancel(tmp_path):
    source = tmp_path / "fake_model.c"
    source.write_text(MODEL_SOURCE)
    subprocess.run([
        "cc", "-shared", "-fPIC", "-I", str(session_v8.PROJECT_ROOT / "include"),
        "-I", str(session_v8.PROJECT_ROOT / "version/v8/src"),
        str(source), "-o", str(tmp_path / "libmodel.so"),
    ], check=True)
    (tmp_path / "weights.bump").write_bytes(b"")
    (tmp_path / "weights_manifest.map").write_text("")
    session = session_v8.SessionV8.open(tmp_path, context_length=16, num_threads=1)
    try:
        batch = session.enable_batch2(320)
        ticket = batch.begin_prefill(0, 3)
        assert batch.append_prefill(0, ticket, [1, 2]) == 2
        with pytest.raises(session_v8.SessionError, match="capability"):
            batch.position(0)
        with pytest.raises(ValueError, match="token IDs"):
            batch.append_prefill(0, ticket, [16])
        assert batch.append_prefill(0, ticket, [3]) == 3
        assert batch.position(0) == 3
        assert batch.staged_logits(0)[0] == 6
        batch.reset_slot(0)

        interrupted = batch.begin_prefill(0, 2)
        assert batch.append_prefill(0, interrupted, [4]) == 1
        batch.request_cancel(interrupted)
        with pytest.raises(session_v8.SessionError, match="cancelled"):
            batch.append_prefill(0, interrupted, [5])
        batch.reset_slot(0)
        fresh = batch.begin_prefill(0, 1)
        batch.request_cancel(interrupted)
        assert batch.append_prefill(0, fresh, [7]) == 1
        assert batch.staged_logits(0)[0] == 7
    finally:
        session.close()


def test_mixed_step_shares_generated_call_and_preserves_both_positions(tmp_path):
    source = tmp_path / "fake_model.c"
    source.write_text(MODEL_SOURCE)
    subprocess.run([
        "cc", "-shared", "-fPIC", "-I", str(session_v8.PROJECT_ROOT / "include"),
        "-I", str(session_v8.PROJECT_ROOT / "version/v8/src"),
        str(source), "-o", str(tmp_path / "libmodel.so"),
    ], check=True)
    (tmp_path / "weights.bump").write_bytes(b"")
    (tmp_path / "weights_manifest.map").write_text("")
    session = session_v8.SessionV8.open(tmp_path, context_length=16, num_threads=1)
    try:
        batch = session.enable_batch2(320)
        model = ctypes.CDLL(str(tmp_path / "libmodel.so"))
        ready, _ = batch.prefill(0, [1, 2])
        staged = batch.begin_prefill(1, 2)
        before = model.ck_fake_batch_calls()
        with pytest.raises(session_v8.SessionError, match="invalid_argument"):
            batch.mixed_step((4, 5), (ready, staged + 1), 1)
        assert model.ck_fake_batch_calls() == before
        with pytest.raises(ValueError, match="token IDs"):
            batch.mixed_step((16, 5), (ready, staged), 1)
        mask, consumed, logits = batch.mixed_step((4, 5), (ready, staged), 1)
        assert (mask, consumed) == (3, 1)
        assert (logits[0][0], logits[1][0]) == (7, 5)
        assert model.ck_fake_batch_calls() == before + 1
        assert batch.position(0) == 3
        with pytest.raises(session_v8.SessionError, match="capability"):
            batch.position(1)
        mask, consumed, logits = batch.mixed_step((1, 6), (ready, staged), 1)
        assert (mask, consumed) == (3, 2)
        assert (logits[0][0], logits[1][0]) == (8, 11)
        assert (batch.position(0), batch.position(1)) == (4, 2)
        mask, logits = batch.step((1, 1))
        assert mask == 3 and (logits[0][0], logits[1][0]) == (9, 12)
        batch.reset_slot(0)
        batch.reset_slot(1)

        ready, _ = batch.prefill(0, [2])
        staged = batch.begin_prefill(1, 2)
        batch.request_cancel(staged)
        before = model.ck_fake_batch_calls()
        with pytest.raises(session_v8.SessionError, match="cancelled"):
            batch.mixed_step((1, 3), (ready, staged), 1)
        assert model.ck_fake_batch_calls() == before
        assert batch.position(0) == 1
        batch.reset_slot(1)
        mask, _ = batch.step((1, 0))
        assert mask == 1 and batch.position(0) == 2
    finally:
        session.close()


def test_batch_host_boundary_budget_tickets_and_recovery(tmp_path, monkeypatch):
    source = tmp_path / "fake_model.c"
    source.write_text(MODEL_SOURCE)
    subprocess.run([
        "cc", "-shared", "-fPIC", "-I", str(session_v8.PROJECT_ROOT / "include"),
        "-I", str(session_v8.PROJECT_ROOT / "version/v8/src"),
        str(source), "-o", str(tmp_path / "libmodel.so"),
    ], check=True)
    (tmp_path / "weights.bump").write_bytes(b"")
    (tmp_path / "weights_manifest.map").write_text("")
    assert session_v8.SESSION_LIB_PATH.is_file(), "build ck-session-v8 before this test"
    session = session_v8.SessionV8.open(tmp_path, context_length=16, num_threads=1)
    try:
        with pytest.raises(session_v8.SessionError, match="text encoding is unavailable"):
            session.encode_ids("fake tokenizer")
        with pytest.raises(session_v8.SessionError, match="requires 320 extra bytes"):
            session.enable_batch2(0)
        batch_type = session_v8.Batch2SessionV8

        def fail_host_allocation(*args):
            raise MemoryError("injected host logits allocation failure")

        monkeypatch.setattr(session_v8, "Batch2SessionV8", fail_host_allocation)
        with pytest.raises(MemoryError, match="host logits"):
            session.enable_batch2(320)
        monkeypatch.setattr(session_v8, "Batch2SessionV8", batch_type)
        batch = session.enable_batch2(320)
        assert session.enable_batch2(320) is batch
        assert batch.required_bytes == 320
        assert batch.native_required_bytes == 192
        assert batch.vocab_size == 16
        for invalid in ([-1, 1], [1, 16], [1, 1 << 32], [True]):
            with pytest.raises(ValueError, match="token IDs"):
                batch.prefill(0, invalid)
        with pytest.raises(ValueError, match="token IDs"):
            batch.step((1 << 32, 1))
        first_ticket, first_logits = batch.prefill(0, [1, 2])
        second_ticket, second_logits = batch.prefill(1, [3])
        assert first_ticket != second_ticket
        assert (first_logits[0], second_logits[0]) == (3, 3)
        assert (batch.position(0), batch.position(1)) == (2, 1)

        mask, logits = batch.step((4, 5))
        assert mask == 3
        assert logits[0] is first_logits and logits[1] is second_logits
        assert (logits[0][0], logits[1][0]) == (7, 8)
        model = ctypes.CDLL(str(tmp_path / "libmodel.so"))
        model.ck_fake_hold_batch.argtypes = [ctypes.c_int]
        model.ck_fake_hold_batch(1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            in_flight = pool.submit(batch.step, (1, 1))
            try:
                deadline = time.monotonic() + 5
                while not model.ck_fake_batch_entered() and time.monotonic() < deadline:
                    time.sleep(.001)
                assert model.ck_fake_batch_entered()
                batch.request_cancel(first_ticket)  # Must not wait on the step lock.
            finally:
                model.ck_fake_hold_batch(0)
            assert in_flight.result(timeout=5)[0] == 3
        mask, logits = batch.step((6, 1))
        assert mask == 2 and logits[1][0] == 10
        assert batch.position(0) == 4
        assert batch.position(1) == 4

        batch.reset_slot(0)
        new_ticket, logits = batch.prefill(0, [7])
        assert new_ticket != first_ticket and logits[0] == 7
        batch.request_cancel(first_ticket)  # Retired ticket cannot cancel reuse.
        mask, logits = batch.step((1, 1))
        assert mask == 3 and (logits[0][0], logits[1][0]) == (8, 11)
        batch.request_cancel(new_ticket)
        assert batch.step((1, 1))[0] == 2
        with pytest.raises(ValueError, match="slot"):
            batch.reset_slot(2)
    finally:
        session.close()
    with pytest.raises(session_v8.SessionError, match="closed"):
        batch.step((1, 1))


def test_missing_optional_batch_abi_fails_capability_check():
    class LegacyLib:
        pass

    with pytest.raises(session_v8.SessionError, match="lacks the batch2 ABI"):
        session_v8._configure_batch_abi(LegacyLib())
    with pytest.raises(session_v8.SessionError, match="lacks mixed batch2 step"):
        session_v8._configure_mixed_step_abi(LegacyLib())
