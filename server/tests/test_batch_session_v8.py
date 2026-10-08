"""The host batch binding runs the real native session ABI against a fake model."""

import concurrent.futures
import ctypes
import subprocess
import time

import pytest

from server import session_v8
from tests.test_v8_batch_session_scheduler import MODEL_SOURCE


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
