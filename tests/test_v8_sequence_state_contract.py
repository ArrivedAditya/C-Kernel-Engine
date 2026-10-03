"""Small compiled proof for serialized sequence state switching."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from sequence_state_codegen_v8 import emit_sequence_state_api  # noqa: E402
from sequence_state_contract_v8 import resolve_sequence_state_contract  # noqa: E402
import codegen_core_v8  # noqa: E402
import codegen_v8  # noqa: E402


def _layout(*, extra: list[dict] | None = None) -> dict:
    return {
        "memory": {
            "arena": {"total_size": 256},
            "activations": {
                "buffers": [
                    {"name": "kv_cache", "abs_offset": 64, "size": 32, "lifetime": "sequence", "mutable": True},
                    {"name": "logits", "abs_offset": 128, "size": 32, "lifetime": "call", "mutable": True},
                    *(extra or []),
                ]
            },
        }
    }


class SequenceStateContractTests(unittest.TestCase):
    def test_generated_capability_matches_generated_api(self) -> None:
        layout = _layout()
        config = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 4}
        dense = codegen_core_v8.emit_model_and_api(layout=layout, config=config)
        declared = codegen_v8._emit_runtime_capability_api(
            {"config": config, "operations": []}, layout, None
        )
        self.assertIn("ck_model_sequence_state_create", dense)
        self.assertIn("CK_MODEL_CAP_SEQUENCE_STATE_SWITCH", declared)

        hybrid_layout = _layout(
            extra=[{"name": "arbitrary_buffer", "abs_offset": 160, "size": 16, "lifetime": "sequence", "mutable": True}]
        )
        hybrid = codegen_core_v8.emit_model_and_api(layout=hybrid_layout, config=config)
        hybrid_declared = codegen_v8._emit_runtime_capability_api(
            {"config": config, "operations": []}, hybrid_layout, None
        )
        self.assertNotIn("ck_model_sequence_state_create", hybrid)
        self.assertNotIn("CK_MODEL_CAP_SEQUENCE_STATE_SWITCH", hybrid_declared)

    def test_named_kv_lookup_uses_active_handle(self) -> None:
        source = codegen_v8._inject_activation_lookup_api(
            "", _layout(), sequence_switch=True
        )
        self.assertIn('strcmp(name, "kv_cache") == 0', source)
        self.assertIn("return (uintptr_t)g_model->kv_cache", source)
        self.assertIn("g_model->kv_cache != (float *)(g_model->bump + A_KV_CACHE)", source)
        legacy = codegen_v8._inject_activation_lookup_api(
            "", _layout(), sequence_switch=False
        )
        self.assertNotIn("return (uintptr_t)g_model->kv_cache", legacy)

    def test_dense_kv_contract_comes_from_layout(self) -> None:
        self.assertEqual(resolve_sequence_state_contract(_layout(), {}), {
        "kv_offset": 64,
        "kv_bytes": 32,
        "kv_alignment": 64,
        })
        self.assertIsNotNone(resolve_sequence_state_contract(_layout(
            extra=[{"name": "shared_table", "abs_offset": 160, "size": 16,
                    "lifetime": "model", "mutable": False}]), {}))
        self.assertIsNone(resolve_sequence_state_contract(_layout(), {"uses_cross_attention": True}))
        self.assertIsNone(resolve_sequence_state_contract(
            _layout(extra=[{"name": "any_name", "abs_offset": 160, "size": 16, "lifetime": "sequence", "mutable": True}]), {}
        ))
        self.assertIsNone(resolve_sequence_state_contract(
            _layout(extra=[{"name": "anything", "abs_offset": 160, "size": 16}]), {}
        ))
        self.assertIsNone(resolve_sequence_state_contract(
            _layout(extra=[{"name": "state", "abs_offset": 160, "size": 16, "lifetime": "model", "mutable": True}]), {}
        ))

    def test_invalid_kv_regions_do_not_receive_switching(self) -> None:
        for kv in (
            {"name": "kv_cache", "abs_offset": 250, "size": 32, "lifetime": "sequence", "mutable": True},
            {"name": "kv_cache", "abs_offset": -1, "size": 32, "lifetime": "sequence", "mutable": True},
            {"name": "kv_cache", "abs_offset": 65, "size": 32, "lifetime": "sequence", "mutable": True},
            {"name": "kv_cache", "abs_offset": 64, "size": 0, "lifetime": "sequence", "mutable": True},
        ):
            layout = _layout()
            layout["memory"]["activations"]["buffers"][0] = kv
            with self.assertRaises(ValueError):
                resolve_sequence_state_contract(layout, {})

    def test_overlapping_activation_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlaps"):
            resolve_sequence_state_contract(
                _layout(extra=[{"name": "scratch", "abs_offset": 80, "size": 16, "lifetime": "call", "mutable": True}]), {}
            )

    def test_emitted_switching_keeps_two_sequences_independent(self) -> None:
        source = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define A_KV_CACHE 0
#define KV_CACHE_SIZE 16
typedef struct {
    uint8_t *bump;
    float *kv_cache;
    uint16_t *kv_cache_f16;
    int pos;
    int rope_pos;
} CKModel;
static CKModel *g_model;
''' + emit_sequence_state_api() + r'''
static void advance(uint8_t token) {
    assert(g_model->pos < KV_CACHE_SIZE);
    ((uint8_t *)g_model->kv_cache)[g_model->pos++] = token;
    g_model->rope_pos++;
}
int main(void) {
    _Alignas(64) uint8_t bump[64] = {0};
    _Alignas(64) uint8_t arena_a[64] = {0};
    _Alignas(64) uint8_t arena_b[64] = {0};
    _Alignas(64) uint8_t arena_c[64] = {0};
    CKModel model = {.bump = bump, .kv_cache = (float *)bump,
                     .kv_cache_f16 = (uint16_t *)bump};
    g_model = &model;
    size_t bytes = 0, alignment = 0;
    assert(ck_model_sequence_state_requirements(&bytes, &alignment) == 0);
    assert(bytes == 16 && alignment == 64);
    uint64_t original = ck_model_sequence_state_default();
    uint64_t a = 0, b = 0, rejected = 123;
    assert(original != 0);
    assert(ck_model_sequence_state_create(NULL, bytes, &rejected) == -2 && rejected == 0);
    assert(ck_model_sequence_state_create(arena_a + 1, bytes, &rejected) == -2);
    assert(ck_model_sequence_state_create(arena_a, bytes - 1, &rejected) == -2);
    assert(ck_model_sequence_state_create(arena_a, bytes, &a) == 0);
    assert(ck_model_sequence_state_create(arena_a, bytes, &rejected) == -2);
    assert(ck_model_sequence_state_create(arena_b, bytes, &b) == 0);
    assert(ck_model_sequence_state_create(arena_c, bytes, &rejected) == -3 && rejected == 0);
    assert(ck_model_sequence_state_create(bump, bytes, &rejected) == -2);
    assert(ck_model_sequence_state_create(arena_b, bytes, &rejected) == -2);
    assert(ck_model_sequence_state_activate(a) == 0);
    advance(11); advance(12);
    assert(ck_model_sequence_state_activate(b) == 0);
    advance(21);
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(model.pos == 2 && model.rope_pos == 2);
    assert(((uint8_t *)model.kv_cache)[0] == 11);
    assert(((uint8_t *)model.kv_cache)[1] == 12);
    advance(13);
    assert(ck_model_sequence_state_activate(b) == 0);
    assert(model.pos == 1 && model.rope_pos == 1);
    assert(((uint8_t *)model.kv_cache)[0] == 21);
    assert(((uint8_t *)model.kv_cache)[1] == 0);
    memset(model.kv_cache, 0, KV_CACHE_SIZE);
    model.pos = 0;
    model.rope_pos = 0;
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(model.pos == 3 && ((uint8_t *)model.kv_cache)[0] == 11);
    assert(ck_model_sequence_state_activate(b) == 0);
    assert(model.pos == 0 && ((uint8_t *)model.kv_cache)[0] == 0);
    assert(ck_model_sequence_state_activate(original) == 0);
    assert(model.pos == 0 && model.kv_cache == (float *)bump);
    assert(ck_model_sequence_state_destroy(a) == 0);
    assert(ck_model_sequence_state_activate(a) == -1);
    uint64_t old_a = a;
    assert(ck_model_sequence_state_create(arena_a, bytes, &a) == 0);
    assert(a != old_a && ck_model_sequence_state_activate(old_a) == -1);
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(ck_model_sequence_state_destroy(a) == 0);
    assert(model.kv_cache == (float *)bump);
    assert(ck_model_sequence_state_destroy(a) == -1);
    assert(ck_model_sequence_state_destroy(b) == 0);
    assert(ck_model_sequence_state_destroy(original) == -1);
    ck_sequence_release_all();
    assert(ck_model_sequence_state_activate(original) == -1);
    uint64_t new_default = ck_model_sequence_state_default();
    assert(new_default != original);
    assert(ck_model_sequence_state_activate(original) == -1);
    assert(ck_model_sequence_state_activate(new_default) == 0);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sequence.c").write_text(source)
            subprocess.run(
                ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "sequence.c", "-o", "sequence"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(root / "sequence")], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
