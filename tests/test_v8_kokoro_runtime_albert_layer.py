"""One generated ALBERT layer executes bounded, changing valid token counts."""
import ctypes
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import export_kokoro_bump as exporter
from build_kokoro_runtime_albert_circuit import build_circuit
from build_kokoro_albert_circuit import build_circuit as fixed_circuit
from tests.v8_kokoro_fixture_support import load_first_layer_fixtures
from tests.v8_checked_graph_test_support import compile_native_graph
sys.path.insert(0, str(ROOT / 'version/v8/scripts'))
import build_ir_v8


class KokoroRuntimeAlbertLayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tensors = load_first_layer_fixtures(cls)
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        origins = {name: {'source_name': name, 'transform': 'identity'}
                   for name in tensors}
        bundle = exporter.write_bundle(root, tensors, origins, {
            'n_token': 178, 'hidden_dim': 512,
            'plbert': {'intermediate_size': 2048, 'max_position_embeddings': 512,
                       'num_attention_heads': 12}},
            {'source': 'pinned independent first-layer PyTorch captures'})
        exporter.verify_bundle(root)
        cls.bump = (root / 'weights.bump').read_bytes()
        cls.entries = {entry['name']: entry for entry in bundle['entries']}
        graph = build_circuit()
        circuit = root / 'circuit.json'
        circuit.write_text(json.dumps(graph))
        source = {'config': {
            'model': graph['name'], 'arch': graph['name'], 'num_layers': 1,
            'embed_dim': 128, 'num_heads': 1, 'num_kv_heads': 1,
            'head_dim': 128, 'intermediate_size': 256,
            'context_length': 36, 'max_seq_len': 36, 'vocab_size': 178,
            'T': 36, 'C': 128, 'epsilon': 1e-12,
            'activation_buffer_dtypes': {
                'word_ids': 'i32', 'type_ids': 'i32',
                'request_length': 'i32', 'valid_length': 'i32',
                'valid_residual_elements': 'i32'}},
            'entries': bundle['entries'], 'quant_summary': {},
            'template': graph}
        cls.source = source
        cls.circuit = circuit
        cls.layout, cls.call_ir, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(root, source, circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
        cls.activations = {item['name']: item for item in
                           cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
                       cls.layout['memory']['weights']['entries']}
        cls.arena_size = cls.layout['memory']['arena']['total_size']
        fixed = fixed_circuit(8)
        fixed_path = root / 'fixed8.json'
        fixed_path.write_text(json.dumps(fixed))
        fixed_source = json.loads(json.dumps(source))
        fixed_source['template'] = fixed
        fixed_source['config']['context_length'] = 8
        fixed_source['config']['max_seq_len'] = 8
        fixed_source['config']['T'] = 8
        (root / 'fixed8').mkdir()
        cls.fixed_layout, cls.fixed_ir, _, cls.fixed_loaded, cls.fixed_fn = (
            compile_native_graph(root / 'fixed8', fixed_source, fixed_path))
        cls.fixed_activations = {item['name']: item for item in
                                 cls.fixed_layout['memory']['activations']['buffers']}
        cls.fixed_weights = {item['name']: item for item in
                             cls.fixed_layout['memory']['weights']['entries']}
        cls.fixed_arena_size = cls.fixed_layout['memory']['arena']['total_size']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, shape, dtype=np.float32):
        return np.ndarray(shape, dtype=dtype, buffer=arena,
                          offset=self.activations[name]['abs_offset'])

    def arena(self):
        raw = (ctypes.c_uint8 * (self.arena_size + 63))()
        arena = (ctypes.c_uint8 * self.arena_size).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        for name, planned in self.weights.items():
            entry = self.entries[name]
            payload = self.bump[entry['file_offset']:
                                entry['file_offset'] + entry['size']]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry['sha256'])
            arena[planned['abs_offset']:planned['abs_offset'] + len(payload)] = payload
        self.view(arena, 'word_ids', (36,), np.int32)[:] = self.embed['ids']
        self.view(arena, 'type_ids', (36,), np.int32)[:] = self.embed['type_ids']
        return arena

    def fixed_arena(self):
        raw = (ctypes.c_uint8 * (self.fixed_arena_size + 63))()
        arena = (ctypes.c_uint8 * self.fixed_arena_size).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        for name, planned in self.fixed_weights.items():
            entry = self.entries[name]
            payload = self.bump[entry['file_offset']:
                                entry['file_offset'] + entry['size']]
            arena[planned['abs_offset']:planned['abs_offset'] + len(payload)] = payload
        for name in ('word_ids', 'type_ids'):
            offset = self.fixed_activations[name]['abs_offset']
            values = self.embed['ids' if name == 'word_ids' else 'type_ids'][:8]
            np.ndarray((8,), dtype=np.int32, buffer=arena, offset=offset)[:] = values
        return arena

    def test_checked_runtime_layer(self):
        self.assertEqual(self.call_ir['errors'], [])
        self.assertEqual(len(self.call_ir['operations']), 16)
        self.assertEqual(sum(op['function'] == 'audio_scaled_residual_add_f32'
                             for op in self.call_ir['operations']), 2)
        for op in self.call_ir['operations']:
            for checkpoint in op.get('semantic_checkpoints', []):
                self.assertNotEqual(checkpoint['resolved_contract_id'],
                                    'unresolved')
        selected = [op['call_abi']['kernel_id'] for op in
                    self.call_ir['operations'] if
                    op['function'] == 'audio_scaled_residual_add_f32']
        self.assertEqual(selected, ['audio_scaled_residual_add_runtime_f32'] * 2)
        arena = self.arena()
        request = self.view(arena, 'request_length', (1,), np.int32)
        output = self.view(arena, 'albert_layer_output', (36, 768))
        attention_residual = self.view(arena, 'attention_residual_output', (36, 768))
        layer_residual = self.view(arena, 'layer_residual_output', (36, 768))
        out_tokens = ctypes.c_int32(-1)
        residual_elements = ctypes.c_int32(-1)
        fixed_arena = self.fixed_arena()
        self.assertEqual(self.fixed_fn(fixed_arena, len(fixed_arena)), 0)
        fixed_output = np.ndarray((8, 768), dtype=np.float32,
                                  buffer=fixed_arena,
                                  offset=self.fixed_activations[
                                      'albert_layer_output']['abs_offset']).copy()
        for tokens in (2, 8, 36, 8):
            request[0] = tokens
            output[:] = -777
            attention_residual[:] = -777
            layer_residual[:] = -777
            out_tokens.value = -1
            residual_elements.value = -1
            self.assertEqual(self.fn(arena, len(arena), ctypes.byref(out_tokens),
                                     ctypes.byref(residual_elements)), 0)
            self.assertEqual(out_tokens.value, tokens)
            self.assertEqual(residual_elements.value, tokens * 768)
            self.assertTrue(np.isfinite(output[:tokens]).all())
            self.assertTrue(np.all(output[tokens:] == -777))
            self.assertTrue(np.all(attention_residual[tokens:] == -777))
            self.assertTrue(np.all(layer_residual[tokens:] == -777))
            if tokens == 8:
                np.testing.assert_allclose(output[:8], fixed_output, atol=0, rtol=0)
            if tokens == 36:
                error = np.abs(output - self.layer['layer_output_expected'])
                self.assertLessEqual(float(error.max()), 1e-5)
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': 'kokoro.runtime-first-albert-36.pytorch28',
                    'name': 'Runtime-bound first ALBERT layer at full capacity',
                    'provider': 'audio_scaled_residual_add_runtime_f32',
                    'dtype': 'fp32', 'direction': 'inference',
                    'oracle': 'pinned PyTorch 2.8 full first-layer capture',
                    'status': 'pass', 'gate': 'composition', 'blocking': True,
                    'max_diff': float(error.max()), 'tolerance': 1e-5,
                    'metric': 'maximum absolute final-layer error',
                    'configuration': 'capacity=36; valid tokens=36',
                    'reproduction_command': 'python3 -m unittest '
                        'tests.test_v8_kokoro_runtime_albert_layer',
                }, sort_keys=True))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.runtime-first-albert-8.fixed8-control',
            'name': 'Runtime-bound first ALBERT layer matches fixed8 graph',
            'provider': 'audio_scaled_residual_add_runtime_f32',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'separately compiled CKE fixed8 graph; not independent PyTorch',
            'status': 'pass', 'gate': 'composition', 'blocking': True,
            'max_diff': 0.0, 'tolerance': 0.0,
            'metric': 'exact eight-token final-layer equality',
            'configuration': 'one capacity-36 entry, valid=8; fixed8 control',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_runtime_albert_layer',
        }, sort_keys=True))
        for invalid in (-1, 0, 37, 2147483647):
            request[0] = invalid
            output[:] = -777
            out_tokens.value = -1
            residual_elements.value = -1
            self.assertNotEqual(self.fn(arena, len(arena),
                                        ctypes.byref(out_tokens),
                                        ctypes.byref(residual_elements)), 0)
            self.assertEqual(out_tokens.value, -1)
            self.assertEqual(residual_elements.value, -1)
            self.assertTrue(np.all(output == -777))
        request[0] = 8
        ids = self.view(arena, 'word_ids', (36,), np.int32)
        original = int(ids[0])
        ids[0] = 178
        output[:] = -777
        out_tokens.value = residual_elements.value = -1
        self.assertNotEqual(self.fn(arena, len(arena), ctypes.byref(out_tokens),
                                    ctypes.byref(residual_elements)), 0)
        self.assertEqual(out_tokens.value, -1)
        self.assertEqual(residual_elements.value, -1)
        self.assertTrue(np.all(output == -777))
        ids[0] = original
        out_tokens.value = residual_elements.value = -1
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(out_tokens),
                                 ctypes.byref(residual_elements)), 0)
        np.testing.assert_array_equal(output[:8], fixed_output)
        self.assertTrue(np.all(output[8:] == -777))

    def test_residual_capacity_metadata_must_match_plan(self):
        for mutation in ('shape', 'element_count', 'binding_override'):
            source = json.loads(json.dumps(self.source))
            graph = source['template']
            residual = next(op for op in graph['block_types']['phoneme_encoder']
                            ['body']['ops'] if op['op'] == 'audio_scaled_residual_add')
            if mutation == 'shape':
                graph['activation_buffers']['attention_residual_output'][
                    'shape'][1] = 767
            elif mutation == 'element_count':
                residual['params']['N'] = 27647
            else:
                residual['params']['call_constants'] = {
                    'residual_elements': 27648}
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    RuntimeError, 'HARD (CALL CONSTANT|RUNTIME SCALAR) FAULT'):
                with contextlib.redirect_stdout(io.StringIO()):
                    registry = build_ir_v8.load_kernel_registry()
                    ir1 = build_ir_v8.build_ir1_direct(
                        source, self.circuit, mode='prefill')
                    low1 = build_ir_v8.generate_ir_lower_1(
                        ir1, registry, source, 'prefill')
                    layout = build_ir_v8.generate_memory_layout(
                        low1, source, registry, mode='prefill', context_len=36)
                    low2 = build_ir_v8.generate_ir_lower_2(
                        low1, layout, source, registry, mode='prefill')
                    build_ir_v8.generate_ir_lower_3(low2, mode='prefill')


if __name__ == '__main__':
    unittest.main()
