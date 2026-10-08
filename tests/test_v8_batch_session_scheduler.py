"""Native two-slot scheduling boundary with a deterministic generated-model stand-in."""

import ctypes
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "build/libck_session_v8.so"


MODEL_SOURCE = r'''
#include "ck_model_abi_v8.h"
#include <stdatomic.h>
#include <string.h>

typedef struct { int position; int value; } State;
static State *states[2], *active;
static int create_calls, fail_create_call, batch_calls, single_calls;
static atomic_int hold_batch, batch_entered;
static const uint64_t caps = CK_MODEL_CAP_INIT |
    CK_MODEL_CAP_AUTOREGRESSIVE_DECODE | CK_MODEL_CAP_TEXT_ENCODE |
    CK_MODEL_CAP_TOKEN_DECODE | CK_MODEL_CAP_SEQUENCE_STATE_SWITCH |
    CK_MODEL_CAP_BATCH_DECODE_TWO_ROWS;
uint32_t ck_model_get_abi_version(void) { return CK_MODEL_ABI_V8_VERSION; }
uint64_t ck_model_get_capabilities(void) { return caps; }
int ck_model_get_runtime_descriptor(CKModelRuntimeDescriptorV8 *out, size_t size) {
    if (!out || size < sizeof(*out)) return -1;
    *out = (CKModelRuntimeDescriptorV8){.struct_size=sizeof(*out),
        .abi_version=CK_MODEL_ABI_V8_VERSION, .capabilities=caps,
        .artifact_role=CK_MODEL_ROLE_DECODER, .context_length=16,
        .vocab_size=16};
    return 0;
}
int ck_model_init(const char *weights) { (void)weights; return 0; }
void ck_model_free(void) { states[0]=states[1]=active=0; }
int ck_model_encode_text(const char *text, int length) { (void)text; (void)length; return 0; }
int ck_model_decode_tokens(const int32_t *ids, int count, char *text, int capacity) {
    (void)ids; (void)count; (void)text; (void)capacity; return 0;
}
int ck_model_get_context_window(void) { return 16; }
int ck_model_get_vocab_size(void) { return 16; }
int ck_model_get_active_tokens(void) { return active ? active->position : 0; }
int ck_model_kv_cache_enable(int count) { (void)count; return 0; }
void ck_model_kv_cache_reset(void) { if (active) memset(active, 0, sizeof(*active)); }
int ck_model_sequence_state_requirements(size_t *bytes, size_t *alignment) {
    *bytes=64; *alignment=64; return 0;
}
int ck_model_sequence_state_create(void *arena, size_t bytes, uint64_t *handle) {
    create_calls++;
    if (!arena || bytes < 64 || create_calls == fail_create_call) return -1;
    for (int i=0; i<2; ++i) if (!states[i]) {
        states[i]=(State *)arena; memset(arena, 0, bytes);
        *handle=(uint64_t)(i+1); return 0;
    }
    return -1;
}
int ck_model_sequence_state_activate(uint64_t handle) {
    if (handle < 1 || handle > 2 || !states[handle-1]) return -1;
    active=states[handle-1]; return 0;
}
int ck_model_sequence_state_destroy(uint64_t handle) {
    if (handle < 1 || handle > 2 || !states[handle-1]) return -1;
    if (active == states[handle-1]) active=0;
    states[handle-1]=0; return 0;
}
int ck_model_embed_tokens(const int32_t *tokens, int count) {
    if (!active || !tokens || count < 1) return -1;
    for (int i=0; i<count; ++i) active->value += tokens[i];
    active->position += count; return 0;
}
int ck_model_forward(float *logits) {
    if (!active || !logits) return -1;
    for (int i=0; i<16; ++i) logits[i]=(float)(active->value+i);
    return 0;
}
int ck_model_decode(int32_t token, float *logits) {
    if (!active || !logits || token == 15) return -1;
    single_calls++; active->value += token; active->position++;
    for (int i=0; i<16; ++i) logits[i]=(float)(active->value+i);
    return 0;
}
int ck_model_batch_decode_workspace(size_t *bytes, size_t *alignment) {
    *bytes=64; *alignment=64; return 0;
}
int ck_model_decode_batch2(const CKModelBatchDecodeRowV8 *rows,
                            size_t count, void *work, size_t bytes) {
    if (!rows || count != 2 || !work || bytes < 64) return -1;
    for (int i=0; i<2; ++i)
        if (rows[i].sequence_handle != (uint64_t)(i+1) ||
            rows[i].position != states[i]->position || rows[i].token == 15)
            return -1;
    atomic_store(&batch_entered, 1);
    while (atomic_load(&hold_batch)) { }
    batch_calls++;
    for (int i=0; i<2; ++i) {
        states[i]->value += rows[i].token; states[i]->position++;
        for (int j=0; j<16; ++j)
            rows[i].logits[j]=(float)(states[i]->value+j);
    }
    return 0;
}
int ck_fake_batch_calls(void) { return batch_calls; }
int ck_fake_single_calls(void) { return single_calls; }
void ck_fake_hold_batch(int value) {
    atomic_store(&batch_entered, 0);
    atomic_store(&hold_batch, value);
}
int ck_fake_batch_entered(void) { return atomic_load(&batch_entered); }
void ck_fake_fail_create_on_call(int call) { fail_create_call=call; }
'''


class Config(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("model_library_path", ctypes.c_char_p), ("weights_path", ctypes.c_char_p),
        ("manifest_path", ctypes.c_char_p), ("context_length", ctypes.c_int32),
        ("num_threads", ctypes.c_int32), ("required_capabilities", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class GenerateRequest(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("system_text", ctypes.c_char_p), ("user_text", ctypes.c_char_p),
        ("max_tokens", ctypes.c_int32), ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float), ("flags", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32), ("reserved", ctypes.c_uint64 * 8),
    ]


class GenerateResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("prompt_tokens", ctypes.c_int32), ("generated_tokens", ctypes.c_int32),
        ("stop_reason", ctypes.c_int32), ("reserved0", ctypes.c_int32),
        ("prefill_time_ms", ctypes.c_double), ("decode_time_ms", ctypes.c_double),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class BatchSessionSchedulerTests(unittest.TestCase):
    def test_two_slots_cancel_reuse_and_failure_recovery(self):
        self.assertTrue(SESSION.is_file(), "build ck-session-v8 before this test")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "fake_model.c"
            model_path = Path(temporary) / "libmodel.so"
            source.write_text(MODEL_SOURCE)
            subprocess.run(["cc", "-shared", "-fPIC", "-I", str(ROOT / "include"),
                            str(source), "-o", str(model_path)], check=True)
            model = ctypes.CDLL(str(model_path))
            session = ctypes.CDLL(str(SESSION))
            session.ck_session_v8_open.argtypes = [ctypes.POINTER(Config), ctypes.POINTER(ctypes.c_void_p)]
            session.ck_session_v8_open.restype = ctypes.c_int
            session.ck_session_v8_close.argtypes = [ctypes.c_void_p]
            session.ck_session_v8_batch2_enable.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
            session.ck_session_v8_batch2_prefill.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint64)]
            session.ck_session_v8_batch2_step.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                ctypes.POINTER(ctypes.c_uint32)]
            session.ck_session_v8_batch2_reset_slot.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            session.ck_session_v8_batch2_position.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
            session.ck_session_v8_batch2_request_cancel.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            session.ck_session_v8_generate.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(GenerateRequest), ctypes.c_void_p,
                ctypes.c_void_p, ctypes.POINTER(GenerateResult)]
            session.ck_session_v8_reset.argtypes = [ctypes.c_void_p]
            config = Config(ctypes.sizeof(Config), 1, str(model_path).encode(),
                            b"unused", None, 16, 1, 0, (ctypes.c_uint64 * 8)())
            handle = ctypes.c_void_p()
            self.assertEqual(session.ck_session_v8_open(ctypes.byref(config), ctypes.byref(handle)), 0)
            try:
                needed = ctypes.c_size_t()
                self.assertEqual(session.ck_session_v8_batch2_enable(
                    handle, 0, ctypes.byref(needed)), -8)
                self.assertEqual(needed.value, 192)
                model.ck_fake_fail_create_on_call(2)
                self.assertEqual(session.ck_session_v8_batch2_enable(
                    handle, needed.value, ctypes.byref(needed)), -7)
                self.assertEqual(session.ck_session_v8_batch2_enable(
                    handle, needed.value, ctypes.byref(needed)), 0)
                self.assertEqual(session.ck_session_v8_batch2_enable(
                    handle, 0, ctypes.byref(needed)), -8)
                request = GenerateRequest()
                request.struct_size = ctypes.sizeof(request)
                request.abi_version = 1
                request.user_text = b"hello"
                result = GenerateResult()
                result.struct_size = ctypes.sizeof(result)
                result.abi_version = 1
                self.assertEqual(session.ck_session_v8_generate(
                    handle, ctypes.byref(request), None, None,
                    ctypes.byref(result)), -6)
                self.assertEqual(session.ck_session_v8_reset(handle), -6)
                self.assertEqual(session.ck_session_v8_batch2_enable(
                    handle, needed.value, ctypes.byref(needed)), 0)
                outputs = [(ctypes.c_float * 16)() for _ in range(2)]
                output_ptrs = (ctypes.POINTER(ctypes.c_float) * 2)(*outputs)

                def prefill(slot, tokens):
                    values = (ctypes.c_int32 * len(tokens))(*tokens)
                    ticket = ctypes.c_uint64()
                    rc = session.ck_session_v8_batch2_prefill(
                        handle, slot, values, len(tokens), outputs[slot], ctypes.byref(ticket))
                    self.assertEqual(rc, 0)
                    return ticket.value

                def step(a, b):
                    advanced = ctypes.c_uint32(99)
                    rc = session.ck_session_v8_batch2_step(
                        handle, (ctypes.c_int32 * 2)(a, b), output_ptrs,
                        ctypes.byref(advanced))
                    return rc, advanced.value

                ticket_a = prefill(0, [1, 2])
                ticket_b = prefill(1, [3])
                self.assertEqual(step(4, 5), (0, 3))
                self.assertEqual([outputs[0][0], outputs[1][0]], [7, 8])
                self.assertEqual(model.ck_fake_batch_calls(), 1)
                position = ctypes.c_int32()
                self.assertEqual(session.ck_session_v8_batch2_position(
                    handle, 0, ctypes.byref(position)), 0)
                self.assertEqual(position.value, 3)
                self.assertEqual(session.ck_session_v8_batch2_position(
                    handle, 1, ctypes.byref(position)), 0)
                self.assertEqual(position.value, 2)

                # A cancel arriving inside native work cannot undo that step;
                # it excludes only the next scheduling boundary.
                model.ck_fake_hold_batch(1)
                in_flight = []
                worker = threading.Thread(target=lambda: in_flight.append(step(1, 1)))
                worker.start()
                try:
                    deadline = time.monotonic() + 2
                    while not model.ck_fake_batch_entered() and time.monotonic() < deadline:
                        time.sleep(0.001)
                    self.assertTrue(model.ck_fake_batch_entered())
                    session.ck_session_v8_batch2_request_cancel(handle, ticket_a)
                finally:
                    model.ck_fake_hold_batch(0)
                    worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(in_flight, [(0, 3)])
                self.assertEqual(step(6, 1), (0, 2))
                self.assertEqual(outputs[1][0], 10)
                self.assertEqual(model.ck_fake_single_calls(), 1)

                self.assertEqual(session.ck_session_v8_batch2_reset_slot(handle, 0), 0)
                new_ticket = prefill(0, [7])
                self.assertNotEqual(new_ticket, ticket_a)
                session.ck_session_v8_batch2_request_cancel(handle, ticket_a)
                self.assertEqual(step(1, 1), (0, 3))
                self.assertEqual(outputs[0][0], 8)

                self.assertEqual(step(15, 1), (-7, 0))
                self.assertEqual(step(1, 1), (-6, 0))
                self.assertEqual(session.ck_session_v8_batch2_reset_slot(handle, 0), 0)
                self.assertEqual(step(1, 1), (-6, 0))
                self.assertEqual(session.ck_session_v8_batch2_reset_slot(handle, 1), 0)
                self.assertNotEqual(prefill(0, [2]), new_ticket)
                prefill(1, [4])
                self.assertEqual(step(1, 1), (0, 3))
                self.assertEqual(step(16, 1), (-1, 0))
                self.assertEqual(step(1, 1), (0, 3))

                # A full context rejects the entire two-row step; the other
                # slot can advance after the full slot is cancelled.
                self.assertEqual(session.ck_session_v8_batch2_reset_slot(handle, 0), 0)
                full_ticket = prefill(0, [1] * 16)
                previous_b = outputs[1][0]
                self.assertEqual(step(1, 1), (-1, 0))
                self.assertEqual(outputs[1][0], previous_b)
                session.ck_session_v8_batch2_request_cancel(handle, full_ticket)
                self.assertEqual(step(1, 1), (0, 2))
            finally:
                session.ck_session_v8_close(handle)


if __name__ == "__main__":
    unittest.main()
