"""The host batch binding runs the real native session ABI against a fake model."""

import concurrent.futures
import ctypes
import os
import subprocess
import sys
import time
from array import array
from pathlib import Path

import pytest

from server import session_v8
from tests.test_v8_batch_session_scheduler import MODEL_SOURCE
from version.v8.scripts.certify_batch_mixed_one_row_v8 import (
    CAP_BATCH_TWO_ROWS, _compare_prefill_snapshots, _finite_logit_bytes,
    _logit_distance, _validated_oracle_result, _verify_oracle_checkout,
    _verify_oracle_mappings,
    _require_shared_projections, certify,
)
from version.v8.scripts.certify_batch_slot_isolation_negative_v8 import (
    certify as certify_slot_negative,
)
from version.v8.scripts.xray_batch_prefill_kv_v8 import (
    attention_span, compare_kv_exports, parse_kv_export, retain_prefix_result,
)
from version.v8.scripts.xray_batch_prefill_attention_v8 import (
    compare_dumps, read_dump,
)
from version.v8.scripts.xray_batch_slot_isolation_v8 import (
    assert_isolated_parity, assert_new_logical_rows, assert_parked_unchanged,
    inject_parked_kv_write,
)


def _ckx_fixture(*, position=3, layer=16, heads=2, width=4, dtype=0):
    import struct
    row_bytes = width * (4 if dtype == 0 else 2)
    payload = bytearray(2 * heads * position * row_bytes)
    header = struct.pack("<8I", 0x564B5843, 2, layer, position,
                         heads, 32, width, dtype)
    return header + payload


def test_kv_xray_reports_first_logical_difference_and_rejects_wrong_owner():
    ordinary_bytes = _ckx_fixture()
    staged_bytes = bytearray(ordinary_bytes)
    # V, head 1, logical position 2, channel 3; physical cache padding is absent.
    row = (2 + 1) * 3 + 2
    staged_bytes[32 + row * 16 + 3 * 4] = 1
    ordinary = parse_kv_export(ordinary_bytes, layer=16, position=3)
    staged = parse_kv_export(bytes(staged_bytes), layer=16, position=3)
    assert compare_kv_exports(ordinary, ordinary, ordinary_owner=0,
                              staged_owner=0)["exact"]
    result = compare_kv_exports(ordinary, staged, ordinary_owner=0, staged_owner=0)
    assert {key: result["first_difference"][key] for key in
            ("kind", "head", "position", "channel", "byte_in_element")} == {
        "kind": "V", "head": 1, "position": 2,
        "channel": 3, "byte_in_element": 0,
    }
    assert result["first_difference"]["ordinary"] == {
        "raw_le_hex": "00000000", "value": 0.0, "classification": "finite"}
    assert result["first_difference"]["staged"]["raw_le_hex"] == "01000000"
    assert result["first_difference"]["absolute_difference"] > 0
    with pytest.raises(ValueError, match="owner"):
        compare_kv_exports(ordinary, staged, ordinary_owner=0, staged_owner=1)


def test_kv_xray_rejects_wrong_position_geometry_and_truncated_payload():
    payload = _ckx_fixture(position=3)
    with pytest.raises(ValueError, match="position"):
        parse_kv_export(payload, layer=16, position=2)
    with pytest.raises(ValueError, match="layer"):
        parse_kv_export(payload, layer=15, position=3)
    with pytest.raises(ValueError, match="extent"):
        parse_kv_export(payload[:-1], layer=16, position=3)
    with pytest.raises(ValueError, match="format"):
        parse_kv_export(b"bad!" + payload[4:], layer=16, position=3)
    other_dtype = parse_kv_export(_ckx_fixture(dtype=1), layer=16, position=3)
    with pytest.raises(ValueError, match="geometry"):
        compare_kv_exports(parse_kv_export(payload, layer=16, position=3),
                           other_dtype, ordinary_owner=0, staged_owner=0)


def test_kv_xray_retains_early_failure_preceding_control():
    cases = {}
    previous = None
    for length in range(1, 6):
        previous = retain_prefix_result(
            cases, slot=0, length=length, max_prefix=10,
            by_layer={"16": {"exact": length != 4}}, previous=previous)
    assert list(cases) == ["slot_0/prefix_3", "slot_0/prefix_4"]
    cases = {}
    retain_prefix_result(cases, slot=1, length=1, max_prefix=10,
                         by_layer={"16": {"exact": False}}, previous=None)
    assert list(cases) == ["slot_1/prefix_1"]


@pytest.mark.parametrize("dtype,ordinary,staged,value", [
    (0, b"\x00\x00\x80\x3f", b"\x00\x00\xc0\x7f", 1.0),
    (1, b"\x00\x3c", b"\x00\x7e", 1.0),
    (2, b"\x80\x3f", b"\xc0\x7f", 1.0),
])
def test_kv_xray_decodes_finite_and_nonfinite_first_values(
    dtype, ordinary, staged, value,
):
    left = bytearray(_ckx_fixture(position=1, heads=1, width=1, dtype=dtype))
    right = bytearray(left)
    left[32:32 + len(ordinary)] = ordinary
    right[32:32 + len(staged)] = staged
    result = compare_kv_exports(
        parse_kv_export(bytes(left), layer=16, position=1),
        parse_kv_export(bytes(right), layer=16, position=1),
        ordinary_owner=0, staged_owner=0)["first_difference"]
    assert result["ordinary"] == {
        "raw_le_hex": ordinary.hex(), "value": value, "classification": "finite"}
    assert result["staged"]["raw_le_hex"] == staged.hex()
    assert result["staged"]["classification"] == "nan"
    assert result["staged"]["value"] is None
    assert result["absolute_difference"] is None


def test_kv_xray_validates_declared_causal_and_window_boundaries():
    assert attention_span(position=9, sliding_window=8,
                          observed_first=2, observed_last=9) == {
                              "first": 2, "last": 9, "observed": True}
    assert attention_span(position=9, sliding_window=512) == {
        "first": 0, "last": 9, "observed": False}
    for first, last in ((1, 9), (2, 8), (0, 10)):
        with pytest.raises(ValueError, match="causal/window"):
            attention_span(position=9, sliding_window=8,
                           observed_first=first, observed_last=last)


def test_attention_xray_validates_observed_span_and_logical_v():
    scores = (0.0, 1.0, float("-inf"), float("-inf"))
    ordinary = {
        "kq_scores_h2_q1": scores,
        "kq_soft_h2_q1": (0.4, 0.6, 0.0, 0.0),
        "kqv_out_h2_q1": (1.6,),
        "vcols_h2_q1": (1.0, 2.0, 0.0, 0.0),
    }
    staged = {name.replace("q1", "q0"): values for name, values in ordinary.items()}
    result = compare_dumps(ordinary, staged, ordinary_query=1, staged_query=0,
                           position=1, sliding_window=2)
    assert result["2"]["read_span"] == {"first": 0, "last": 1, "observed": True}
    assert result["2"]["logical_v_exact"]
    staged["kq_scores_h2_q0"] = (float("-inf"), 1.0, float("-inf"), float("-inf"))
    with pytest.raises(ValueError, match="causal/window"):
        compare_dumps(ordinary, staged, ordinary_query=1, staged_query=0,
                      position=1, sliding_window=2)
    staged["kq_scores_h2_q0"] = scores
    staged["vcols_h2_q0"] = (1.0, 3.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="logical attention V"):
        compare_dumps(ordinary, staged, ordinary_query=1, staged_query=0,
                      position=1, sliding_window=2)


def test_attention_xray_rejects_wrong_capture_identity_and_truncation(tmp_path):
    import struct
    header = struct.pack("<8sIi32sII4qIi32s", b"CKDMP\0\0\0", 1, 16,
                         b"kq_scores_h2_q9", 0, 1, 2, 0, 0, 0, 2, 9, b"")
    path = tmp_path / "trace.bin"
    path.write_bytes(header + struct.pack("<2f", 0.0, 1.0))
    assert read_dump(path, layer=16, query=9)["kq_scores_h2_q9"] == (0.0, 1.0)
    with pytest.raises(ValueError, match="invalid or ambiguous"):
        read_dump(path, layer=17, query=9)
    path.write_bytes(header + b"\0")
    with pytest.raises(ValueError, match="truncated attention vector"):
        read_dump(path, layer=16, query=9)


def test_live_batch_slot_xray_rejects_parked_writes_and_wrong_isolated_output():
    before = {"position": 3, "kv": {"0": "a", "16": "b"}}
    assert_parked_unchanged(before, dict(before))
    with pytest.raises(AssertionError, match="non-advancing slot changed"):
        assert_parked_unchanged(before, {"position": 4, "kv": before["kv"]})
    with pytest.raises(AssertionError, match="non-advancing slot changed"):
        assert_parked_unchanged(before, {"position": 3, "kv": {"0": "a", "16": "changed"}})
    shared = {**before, "logits_sha256": "c"}
    assert_isolated_parity(shared, dict(shared))
    with pytest.raises(AssertionError, match="differs from isolated"):
        assert_isolated_parity(shared, {**shared, "logits_sha256": "different"})


def test_parked_write_injection_changes_real_pointer_bytes():
    import hashlib

    arena = (ctypes.c_ubyte * 4)(42, 0, 0, 0)

    class Batch:
        def position(self, slot):
            assert slot == 1
            return 2

    class Model:
        def ck_model_get_named_activation_ptr(self, name):
            assert name == b"kv_cache"
            return ctypes.addressof(arena)

    def snapshot():
        return {"position": 2, "kv": {"0": hashlib.sha256(bytes(arena)).hexdigest()}}

    before = snapshot()
    inject_parked_kv_write(Batch(), Model(), 1)
    assert arena[0] == 43
    with pytest.raises(AssertionError, match="non-advancing slot changed"):
        assert_parked_unchanged(before, snapshot())


def test_slot_negative_driver_runs_control_and_both_subprocess_directions(tmp_path, monkeypatch):
    probe = tmp_path / "fake_probe.py"
    probe.write_text('''
import argparse
import hashlib
import json
import os
import sys
p = argparse.ArgumentParser()
p.add_argument("bundle")
p.add_argument("--layers")
p.add_argument("--threads")
p.add_argument("--json-out")
p.add_argument("--inject-parked-write", type=int)
a = p.parse_args()
mode = os.environ.get("FAKE_SLOT_PROBE", "valid")
if a.inject_parked_write is None:
    if mode == "control_failure":
        print("RuntimeError: model load failed", file=sys.stderr)
        raise SystemExit(1)
    row = {"schema": "cke.v8.batch-slot-isolation-xray.v1", "status": "pass",
           "cases": {"0": {}, "1": {}}}
    for name, filename in (("loaded_model_sha256", "libmodel.so"),
                           ("loaded_engine_sha256", "libckernel_engine.so"),
                           ("weights_sha256", "weights.bump")):
        row[name] = hashlib.sha256(open(os.path.join(a.bundle, filename), "rb").read()).hexdigest()
    for name in ("loaded_session_sha256", "generated_source_sha256",
                 "diagnostic_source_sha256"):
        row[name] = "a" * 64
    open(a.json_out, "w").write(json.dumps(row))
else:
    if mode == "model_load_failure":
        print("RuntimeError: model load failed", file=sys.stderr)
    elif mode == "import_failure":
        print("ModuleNotFoundError: fake dependency", file=sys.stderr)
    elif mode == "unrelated_assertion":
        print("AssertionError: wrong slot advanced", file=sys.stderr)
    elif mode == "wrong_direction":
        print("InjectedParkedWriteDetected: non-advancing slot 9 logical KV changed",
              file=sys.stderr)
    else:
        parked = a.inject_parked_write ^ 1
        print(f"InjectedParkedWriteDetected: non-advancing slot {parked} logical KV changed",
              file=sys.stderr)
    raise SystemExit(1)
''')
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for filename in ("libmodel.so", "libckernel_engine.so", "weights.bump"):
        (bundle / filename).write_bytes(filename.encode())
    result = certify_slot_negative(bundle, probe=probe)
    assert result["status"] == "pass"
    assert set(result["negative_cases"]) == {
        "advancing_0/parked_1", "advancing_1/parked_0"}
    for mode, error in (("control_failure", "control run did not complete"),
                        ("model_load_failure", "did not reach its logical-KV guard"),
                        ("import_failure", "did not reach its logical-KV guard"),
                        ("unrelated_assertion", "did not reach its logical-KV guard"),
                        ("wrong_direction", "did not reach its logical-KV guard")):
        monkeypatch.setenv("FAKE_SLOT_PROBE", mode)
        with pytest.raises(AssertionError, match=error):
            certify_slot_negative(bundle, probe=probe)


def test_live_batch_slot_xray_rejects_header_only_progress():
    before = {"position": 2, "kv": {"16": {"ckx_sha256": "old",
                                              "last_row_nonzero": True}}}
    after = {"position": 3, "kv": {"16": {"ckx_sha256": "new",
                                             "last_row_nonzero": False}}}
    with pytest.raises(AssertionError, match="new logical KV row was not written"):
        assert_new_logical_rows(before, after)
    after["kv"]["16"]["last_row_nonzero"] = True
    assert_new_logical_rows(before, after)


def test_live_batch_slot_xray_direct_file_cli_without_pythonpath():
    script = Path(__file__).resolve().parents[2] / "version/v8/scripts/xray_batch_slot_isolation_v8.py"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=script.parents[3],
                            env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--json-out" in result.stdout


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


def test_llama_logit_diagnostic_keeps_full_vector_error_visible():
    reference = bytes((ctypes.c_float * 3)(1, 4, 2))
    result = _logit_distance(reference, bytes((ctypes.c_float * 3)(5, 4, 2)))
    assert result["max_abs_logit_diff"] == 4
    assert result["logit_rmse"] > 2
    assert result["top1_subject"] == 1 and result["top1_oracle"] == 0
    assert _logit_distance(reference, reference)["max_abs_logit_diff"] == 0
    with pytest.raises(AssertionError, match="extents differ"):
        _logit_distance(reference, reference[:-4])
    with pytest.raises(AssertionError, match="nonfinite"):
        _logit_distance(reference, bytes((ctypes.c_float * 3)(1, float("nan"), 2)))
    with pytest.raises(ValueError, match="requires the ordinary"):
        certify(Path("unused"), oracle_gguf=Path("unused"))


def test_oracle_mapping_attestation_rejects_alternate_and_replaced_files(tmp_path):
    names = ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so")
    mappings = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(name.encode())
        info = path.stat()
        mappings[name] = {
            "path": str(path),
            "device": f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}",
            "inode": info.st_ino,
        }
    meta = {"loaded_libraries": mappings}
    assert set(_verify_oracle_mappings(meta, tmp_path)) == set(names)
    old_inode = mappings["libllama.so"]["inode"]
    (tmp_path / "replacement").write_bytes(b"other")
    (tmp_path / "replacement").replace(tmp_path / "libllama.so")
    assert (tmp_path / "libllama.so").stat().st_ino != old_inode
    with pytest.raises(AssertionError, match="stale loaded"):
        _verify_oracle_mappings(meta, tmp_path)
    mappings["libllama.so"]["inode"] = (tmp_path / "libllama.so").stat().st_ino
    alternate = tmp_path / "alternate" / "libllama.so"
    alternate.parent.mkdir()
    alternate.write_bytes(b"other")
    mappings["libllama.so"]["path"] = str(alternate)
    with pytest.raises(AssertionError, match="alternate"):
        _verify_oracle_mappings(meta, tmp_path)


def test_oracle_response_rejects_malformed_and_nonfinite_rows(tmp_path):
    for value in ({}, {"meta": {}}, {"meta": {"n_vocab": 1}, "logits": array("f", [1])}):
        with pytest.raises(AssertionError):
            _validated_oracle_result(value, [100], "batched", tmp_path)
    meta = {"n_vocab": 1, "token_count": 1, "tokens": [100],
            "decode_mode": "batched", "effective_settings": {"context": 256}}
    with pytest.raises(AssertionError, match="nonfinite"):
        _validated_oracle_result({"meta": meta, "logits": array("f", [float("nan")])},
                                 [100], "batched", tmp_path)


def test_oracle_checkout_rejects_wrong_pin_and_dirty_tracked_source(monkeypatch, tmp_path):
    values = iter(["160000 commit aaaaaaaa\tllama.cpp\n", "bbbbbbbb\n"])
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: next(values))
    with pytest.raises(AssertionError, match="repository pin"):
        _verify_oracle_checkout(tmp_path, tmp_path / "llama.cpp")
    values = iter(["160000 commit aaaaaaaa\tllama.cpp\n", "aaaaaaaa\n",
                   " M src/llama.cpp\n"])
    with pytest.raises(AssertionError, match="tracked source changes"):
        _verify_oracle_checkout(tmp_path, tmp_path / "llama.cpp")


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
        assert batch.position(0) == 2
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
        assert batch.position(1) == 1
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
