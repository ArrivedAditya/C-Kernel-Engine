"""Bounded complete phoneme encoder via one ordinary generated native entry."""
import ctypes
import hashlib
import json
from pathlib import Path
import unittest
import sys
import shutil
import subprocess
import tempfile
import os

import numpy as np
from tests import test_v8_kokoro_generated_albert_layer as first_layer
from tests import test_v8_linear_rows_oracle as linear_oracle
from tests.v8_checked_graph_test_support import compile_native_graph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_encoder_circuit as circuit_author


class KokoroGeneratedEncoderTest(unittest.TestCase):
    vector = first_layer.KokoroGeneratedAlbertLayerTest.vector
    arena = first_layer.KokoroGeneratedAlbertLayerTest.arena

    @classmethod
    def setUpClass(cls):
        # Reuse certified import fixtures, not the first layer's execution as
        # model input. The complete entry generates every intermediate itself.
        previous = os.environ.get('CKE_KOKORO_LAYER_EVIDENCE_DIR')
        evidence = os.environ.get('CKE_KOKORO_ENCODER_EVIDENCE_DIR')
        if evidence:
            os.environ['CKE_KOKORO_LAYER_EVIDENCE_DIR'] = str(Path(evidence).resolve() / 'prefix')
        else:
            os.environ.pop('CKE_KOKORO_LAYER_EVIDENCE_DIR', None)
        try:
            first_layer.KokoroGeneratedAlbertLayerTest.setUpClass.__func__(cls)
        finally:
            if previous is None:
                os.environ.pop('CKE_KOKORO_LAYER_EVIDENCE_DIR', None)
            else:
                os.environ['CKE_KOKORO_LAYER_EVIDENCE_DIR'] = previous
        root = Path(cls.temp.name) / 'encoder'
        root.mkdir(exist_ok=True)
        fixture = ROOT / 'tests/fixtures/tts/kokoro_encoder_pinned.npz'
        cls.encoder = dict(np.load(fixture))
        cls.encoder_meta = json.loads(fixture.with_suffix('.json').read_text())
        if hashlib.sha256(fixture.read_bytes()).hexdigest() != cls.encoder_meta['fixture_sha256']:
            raise RuntimeError('encoder fixture hash mismatch')
        tensors = {name: np.frombuffer(cls.bump, dtype=np.float32,
                                      count=entry['size']//4, offset=entry['file_offset']).copy().reshape(entry['shape'])
                   for name, entry in cls.entries.items()}
        for kind in ('weight', 'bias'):
            tensors[f'phoneme_projection.{kind}'] = cls.encoder[f'weight__phoneme_projection__{kind}']
        origins = {name: {'source_name': name, 'transform': 'identity'} for name in tensors}
        bundle = first_layer.exporter.write_bundle(root, tensors, origins,
            {'n_token': 178, 'hidden_dim': 512, 'plbert': {'intermediate_size': 2048,
             'max_position_embeddings': 512, 'num_attention_heads': 12}},
            {'source': 'pinned encoder and prior certified effective-weight fixtures'})
        first_layer.exporter.verify_bundle(root)
        cls.bump = (root / 'weights.bump').read_bytes()
        cls.entries = {entry['name']: entry for entry in bundle['entries']}
        circuit = ROOT / 'version/v8/circuits/kokoro_phoneme_encoder_bounded.json'
        template = json.loads(circuit.read_text())
        source = {'config': {
            'model': template['name'], 'arch': template['name'], 'num_layers': 1,
            'embed_dim': 128, 'num_heads': 1, 'num_kv_heads': 1, 'head_dim': 128,
            'intermediate_size': 256, 'context_length': 36, 'max_seq_len': 36,
            'vocab_size': 178, 'T': 36, 'C': 128, 'epsilon': 1e-12,
            'activation_buffer_dtypes': {'word_ids': 'i32', 'type_ids': 'i32'}},
            'entries': bundle['entries'], 'quant_summary': {}, 'template': template}
        cls.layout, cls.call_ir, cls.library, cls.loaded, cls.fn = compile_native_graph(root, source, circuit)
        cls.weights = {entry['name']: entry for entry in cls.layout['memory']['weights']['entries']}
        cls.activations = {entry['name']: entry for entry in cls.layout['memory']['activations']['buffers']}
        cls.arena_size = cls.layout['memory']['arena']['total_size']
        cls.root = root
        cls.source = source

    def specialized_graph(self, tokens):
        import copy
        root = self.root / f'tokens-{tokens}'
        root.mkdir(exist_ok=True)
        template = circuit_author.build_circuit(tokens)
        circuit = root / 'circuit.json'
        circuit.write_text(json.dumps(template))
        source = copy.deepcopy(self.source)
        source['template'] = template
        source['config'].update(T=tokens, context_length=tokens, max_seq_len=tokens)
        return compile_native_graph(root, source, circuit)

    def arena_for(self, layout, ids):
        bytes_required = layout['memory']['arena']['total_size']
        raw = (ctypes.c_uint8 * (bytes_required + 63))()
        arena = (ctypes.c_uint8 * bytes_required).from_buffer(raw, (-ctypes.addressof(raw)) & 63)
        for entry in layout['memory']['weights']['entries']:
            stored = self.entries[entry['name']]
            payload = self.bump[stored['file_offset']:stored['file_offset']+stored['size']]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), stored['sha256'])
            arena[entry['abs_offset']:entry['abs_offset']+stored['size']] = payload
        buffers = {x['name']: x for x in layout['memory']['activations']['buffers']}
        for name, entry in buffers.items():
            dtype = np.int32 if name in ('word_ids', 'type_ids') else np.float32
            vector = np.ndarray((entry['size']//4,), dtype, buffer=arena, offset=entry['abs_offset'])
            vector[:] = ids if name == 'word_ids' else 0 if name == 'type_ids' else -777.
        return arena, buffers

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_every_encoder_boundary(self):
        self.assertEqual(len(self.call_ir['operations']), 147)
        self.assertEqual(len(self.weights), 25)
        self.assertEqual(self.call_ir['errors'], [])
        arena = self.arena()
        self.assertEqual(self.fn(arena, len(arena)), 0)
        errors = {}
        propagation = self.verify_affine_arithmetic(arena, self.layout, self.call_ir, self.encoder, 36)
        report = {'torch': {'tensors': {}}, 'comparisons': {}}
        thresholds = {}
        for name, expected in self.encoder.items():
            if name.startswith('weight__') or name == 'word_ids':
                continue
            actual = self.vector(arena, name, np.float32, expected.size).reshape(expected.shape)
            self.assertTrue(np.isfinite(actual).all(), name)
            difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            limit = self.composition_limit(name, expected, propagation.get(name))
            self.assertTrue(np.all(difference <= limit), name)
            self.assertLessEqual(float(np.sqrt(np.mean(difference*difference))), 3e-5, name)
            errors[name] = {'max_abs': float(difference.max()),
                            'rmse': float(np.sqrt(np.mean(difference*difference))),
                            'worst_sample': list(map(int, np.unravel_index(difference.argmax(), difference.shape))),
                            'oracle_at_worst': float(expected[np.unravel_index(difference.argmax(), difference.shape)])}
            thresholds[name] = float(np.max(limit))
            candidate_path = self.root / f'{name}-native.f32'
            oracle_path = self.root / f'{name}-pytorch.f32'
            actual.tofile(candidate_path)
            expected.tofile(oracle_path)
            point = next(point for op in self.call_ir['operations'] for point in op.get('semantic_checkpoints', [])
                         if point['tensor'] == name)
            selector = name if point['layer'] < 0 else f'{name}@{point["layer"]}'
            report['torch']['tensors'][selector] = {'path': str(oracle_path), 'shape': list(expected.shape)}
            report['comparisons'][selector] = {'ck_path': str(candidate_path), 'shape': list(expected.shape),
                'capacity_shape': list(expected.shape), 'valid_shape': list(expected.shape),
                'physical_strides': [expected.shape[1], 1]}
            self.emit_comparison(name, errors[name]['max_abs'], thresholds[name], difference, limit, 36, point)
        (self.root / 'encoder-errors.json').write_text(json.dumps(errors, indent=2))
        self.verify_xray(report, thresholds)

    @staticmethod
    def composition_limit(name, reference, propagation=None):
        # A new connected-encoder contract, NOT a changed primitive bound.
        # Normalized states/final features retain a strict 3e-5 absolute bound.
        # Other stages use the prior 5e-5 affine absolute allowance plus 2e-6
        # relative allowance for twelve connected calls. Independent oracle-fed
        # FFN checks below prove the FP64 arithmetic separately.
        if name.endswith(('attention_norm_output', 'albert_layer_output', 'phoneme_features')):
            return np.full(reference.shape, 3e-5, np.float64)
        allowance = 5e-5 + 2e-6 * np.abs(reference.astype(np.float64))
        # Affine cancellation depends on input perturbation, not output scale.
        # |W dx| <= |W| |dx| is calculated BEFORE looking at candidate outputs.
        return allowance if propagation is None else allowance + propagation

    def verify_affine_arithmetic(self, arena, layout, calls, reference, tokens):
        buffers = {x['name']: x for x in layout['memory']['activations']['buffers']}
        weights = {x['name']: x for x in layout['memory']['weights']['entries']}
        def view(entry, shape):
            return np.ndarray(shape, np.float32, buffer=arena, offset=entry['abs_offset'])
        propagated = {}
        summaries = []
        residual_count = 0
        for op in calls['operations']:
            if op['function'] == 'audio_scaled_residual_add_f32':
                arguments = {a['name']: a for a in op['args']}
                left_name = arguments['residual']['buffer_ref']
                right_name = arguments['branch']['buffer_ref']
                output_name = arguments['output']['buffer_ref']
                shape = reference[output_name].shape
                left = view(buffers[left_name], shape)
                right = view(buffers[right_name], shape)
                actual = view(buffers[output_name], shape)
                scale = np.float32(float(arguments['scale']['expr'].rstrip('f')))
                expected = left + scale * right
                np.testing.assert_array_equal(actual, expected, err_msg=output_name)
                propagated[output_name] = (np.abs(left.astype(np.float64)-reference[left_name].astype(np.float64))
                    + abs(float(scale))*np.abs(right.astype(np.float64)-reference[right_name].astype(np.float64)))
                residual_count += 1
                continue
            if op['function'] != 'linear_rows_checked_f32':
                continue
            arguments = {a['name']: a for a in op['args']}
            m, k, n = (int(arguments[name]['expr']) for name in ('rows', 'input_channels', 'output_channels'))
            input_name = arguments['input']['buffer_ref']
            output_name = arguments['output']['buffer_ref']
            x = view(buffers[input_name], (m, k)).astype(np.float64)
            w = view(weights[arguments['weight']['weight_ref']], (n, k)).astype(np.float64)
            b = view(weights[arguments['bias']['weight_ref']], (n,)).astype(np.float64)
            actual = view(buffers[output_name], (m, n))
            # Independent NumPy elementwise operations implement the declared
            # bias-first, ascending FP64 sum, with separate multiply and add.
            expected = np.broadcast_to(b, (m, n)).copy()
            for channel in range(k):
                expected += x[:, channel, None] * w[None, :, channel]
            expected = expected.astype(np.float32)
            np.testing.assert_array_equal(actual, expected, err_msg=output_name)
            delta = np.abs(x-reference[input_name].astype(np.float64))
            propagated[output_name] = delta @ np.abs(w).T
            summaries.append({'operation_index': op['idx'], 'output': output_name,
                              'native_vs_fp64_ascending_max_abs': float(np.max(np.abs(actual-expected))),
                              'propagated_input_bound_max': float(propagated[output_name].max())})
        self.assertEqual(len(summaries), 74)
        self.assertEqual(residual_count, 24)
        folder = self.root if tokens == 36 else self.root / f'tokens-{tokens}'
        (folder / 'affine-arithmetic.json').write_text(json.dumps(summaries, indent=2))
        return propagated

    def emit_comparison(self, name, maximum, bound, diff, limit, tokens, point):
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': f'kokoro.encoder.{tokens}.{name}.connected-vs-pytorch28',
            'name': name, 'provider': point['kernel_id'], 'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pytorch', 'backend_version': self.encoder_meta['dependencies']['torch'],
            'configuration': f'tokens={tokens}; twelve shared layers; connected graph',
            'evidence_kind': 'numerical', 'max_diff': maximum, 'tolerance': bound,
            'max_elementwise_scaled_error': float(np.max(diff/limit)),
            'elementwise_rule': '3e-5 normalized/final; otherwise 5e-5 + 2e-6*abs(reference) + propagated affine/residual input bound',
            'status': 'pass' if np.all(diff <= limit) else 'fail',
            'reproduction_command': 'python3 -m unittest ' + self.id()}, sort_keys=True))

    def verify_xray(self, tensor_report, thresholds):
        builder, xray = first_layer.xray_builder, first_layer.xray
        identity = builder.capture_runtime_library_identity(self.loaded, 'ck_kokoro_phoneme_encoder')
        candidate = builder.build_manifest(backend='ck', call_ir=self.call_ir, tensor_report=tensor_report,
            model='kokoro_phoneme_encoder_bounded', source='generated_native', phase='prefill',
            loaded_library=self.library, runtime_library=identity)
        oracle = builder.build_manifest(backend='pytorch', call_ir=self.call_ir, tensor_report=tensor_report,
            model='kokoro_phoneme_encoder_bounded', source='pinned_capture', phase='prefill')
        self.assertEqual(len(candidate['checkpoints']), 147)
        self.assertEqual(len(oracle['checkpoints']), 147)
        reports = []
        for op in self.call_ir['operations']:
            for point in op.get('semantic_checkpoints', []):
                self.assertNotEqual(point['resolved_contract_id'], 'unresolved')
                profile = {'schema': 'cke.parity_profile', 'schema_version': 1, 'name': 'kokoro_encoder_connected',
                    'backend': 'pytorch', 'contract_schema_version': 1,
                    'required_match_fields': ['checkpoint_id', 'producer', 'logical_layout', 'axis_names',
                                             'resolved_contract_id', 'kernel_id', 'function'],
                    'observed_storage': {'default': 'fp32', 'checkpoints': {}},
                    'dtype_thresholds': {'fp32': {'cosine_min': 0.99999, 'rmse_max': 3e-5,
                        'relative_rmse_max': 3e-5, 'max_abs_max': thresholds[point['tensor']], 'finite_required': True}},
                    'checkpoint_order': [point['id']], 'interval_expansions': {}, 'backend_mappings': {}}
                # Keep full captures on disk, but avoid repeatedly validating
                # 147 irrelevant records for each one-checkpoint profile.
                subject = {**candidate, 'checkpoints': [x for x in candidate['checkpoints']
                                                       if x['checkpoint_id'] == point['id']]}
                expected = {**oracle, 'checkpoints': [x for x in oracle['checkpoints']
                                                     if x['checkpoint_id'] == point['id']]}
                result = xray.compare_manifests(subject, expected, profile)
                self.assertEqual(result['status'], 'pass', point['id'])
                reports.append(result)
        for filename, contents in [('native-checkpoints.json', candidate), ('pytorch-checkpoints.json', oracle),
                                   ('xray-reports.json', reports)]:
            (self.root / filename).write_text(json.dumps(contents, indent=2))

    def test_local_ffn_affine_against_independent_fp64(self):
        # Same oracle-fed input distinguishes local arithmetic differences from
        # errors carried through previous connected layer invocations.
        oracle = linear_oracle.LinearRowsOracleTest
        oracle.setUpClass()
        instance = oracle(methodName='test_pinned_kokoro_projection_against_pytorch')
        try:
            name = 'phoneme_encoder.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.'
            entries = self.entries
            def weight(kind):
                entry = entries[name + kind]
                return np.frombuffer(self.bump, np.float32, count=entry['size']//4,
                                     offset=entry['file_offset']).reshape(entry['shape'])
            w, b = weight('weight'), weight('bias')
            report = {}
            for invocation in range(12):
                prefix = f'l{invocation:02d}_'
                x = self.encoder[prefix + 'ffn_gelu_output']
                expected = self.encoder[prefix + 'ffn_projection_output']
                actual = np.full(expected.shape, -777., np.float32)
                self.assertEqual(instance.call(x, w, b, actual, 36, 2048, 768), 0)
                independent = (x.astype(np.float64) @ w.astype(np.float64).T + b.astype(np.float64)).astype(np.float32)
                self.assertTrue(np.isfinite(actual).all())
                local_diff = np.abs(actual.astype(np.float64) - independent.astype(np.float64))
                torch_diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                report[prefix] = {'native_vs_fp64_max_abs': float(local_diff.max()),
                                  'native_vs_pytorch_max_abs': float(torch_diff.max())}
                self.assertLessEqual(float(local_diff.max()), 1e-7)
            (self.root / 'encoder-affine-oracle.json').write_text(json.dumps(report, indent=2))
            print('ENCODER_AFFINE_LOCAL ' + json.dumps(report, sort_keys=True))
        finally:
            oracle.tearDownClass()

    def test_repeat_all_checkpoints_bitwise(self):
        arena = self.arena()
        names = [name for name in self.encoder if not name.startswith('weight__') and name != 'word_ids']
        self.assertEqual(self.fn(arena, len(arena)), 0)
        first = {name: self.vector(arena, name, np.float32, self.encoder[name].size).copy()
                 for name in names}
        for _ in range(3):
            self.assertEqual(self.fn(arena, len(arena)), 0)
            for name in names:
                np.testing.assert_array_equal(self.vector(arena, name, np.float32, first[name].size), first[name])

    def test_undersized_arena_and_invalid_ids_stop_before_outputs(self):
        arena = self.arena()
        original = bytes(arena)
        self.assertEqual(self.fn(arena, len(arena)-1), -2)
        self.assertEqual(bytes(arena), original)
        for invalid in (-1, 178):
            self.vector(arena, 'word_ids', np.int32, 36)[0] = invalid
            original = bytes(arena)
            self.assertNotEqual(self.fn(arena, len(arena)), 0)
            self.assertEqual(bytes(arena), original)

    def test_failed_shared_normalization_and_final_projection_leave_consumers_untouched(self):
        for weight, first_unwritten, expected_status in (
            ('phoneme_encoder.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight',
             'l00_attention_norm_output', -3),
            ('phoneme_projection.weight', 'phoneme_features', -1)):
            arena = self.arena()
            planned = self.weights[weight]
            np.ndarray((planned['size']//4,), np.float32, buffer=arena,
                       offset=planned['abs_offset'])[0] = np.nan
            self.assertEqual(self.fn(arena, len(arena)), expected_status)
            self.assertTrue(np.all(self.vector(arena, first_unwritten, np.float32,
                                              self.encoder[first_unwritten].size) == -777.))
            self.assertTrue(np.all(self.vector(arena, 'phoneme_features', np.float32, 36*512) == -777.))

    def test_standalone_encoder_replay_without_python_or_checkout(self):
        arena = self.arena()
        output = self.activations['phoneme_features']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'arena.bin').write_bytes(bytes(arena))
            shutil.copyfile(self.library, root / 'generated.so')
            source = root / 'host.c'
            source.write_text(f'''#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
extern int ck_kokoro_phoneme_encoder(uint8_t *, size_t);
int main(int argc, char **argv) {{
    if (argc != 3) return 10;
    const size_t bytes = {self.arena_size}u;
    uint8_t *arena = aligned_alloc(64, (bytes+63u)&~(size_t)63u);
    if (!arena) return 11;
    FILE *in = fopen(argv[1], "rb");
    if (!in) {{ free(arena); return 12; }}
    size_t received = fread(arena, 1, bytes, in);
    int extra = fgetc(in); fclose(in);
    if (received != bytes || extra != EOF) {{ free(arena); return 13; }}
    int status = ck_kokoro_phoneme_encoder(arena, bytes);
    if (status) {{ free(arena); return 14; }}
    FILE *out = fopen(argv[2], "wb");
    if (!out) {{ free(arena); return 15; }}
    size_t written = fwrite(arena+{output['abs_offset']}u, 1, {36*512*4}u, out);
    int closed = fclose(out); free(arena);
    return written == {36*512*4}u && closed == 0 ? 0 : 16;
}}
''')
            binary = root / 'native-encoder'
            subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', str(source),
                '-L', str(root), '-l:generated.so', '-Wl,-rpath,$ORIGIN', '-o', str(binary)], check=True)
            subprocess.run([str(binary), 'arena.bin', 'features.f32'], cwd=root, check=True)
            standalone = np.fromfile(root / 'features.f32', np.float32)
            self.assertTrue(np.isfinite(standalone).all())
            self.assertEqual(self.fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(standalone, self.vector(arena, 'phoneme_features', np.float32, 36*512))
            np.testing.assert_allclose(standalone, self.encoder['phoneme_features'].ravel(), atol=3e-5, rtol=0)
            replay = self.root / 'standalone'
            replay.mkdir(exist_ok=True)
            for filename in ('host.c', 'native-encoder', 'generated.so', 'arena.bin', 'features.f32'):
                shutil.copy2(root / filename, replay / filename)
            (root / 'arena.bin').write_bytes(b'invalid')
            result = subprocess.run([str(binary), 'arena.bin', 'rejected.f32'], cwd=root)
            self.assertEqual(result.returncode, 13)
            self.assertFalse((root / 'rejected.f32').exists())

    def test_short_and_long_specializations(self):
        results = {}
        for tokens in (2, 8, 64):
            with self.subTest(tokens=tokens):
                path = ROOT / f'tests/fixtures/tts/kokoro_encoder_{tokens}_pinned.npz'
                meta = json.loads(path.with_suffix('.json').read_text())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), meta['fixture_sha256'])
                self.assertEqual(meta['dependencies'], self.encoder_meta['dependencies'])
                self.assertEqual(meta['model_pin'], self.encoder_meta['model_pin'])
                data = dict(np.load(path))
                layout, calls, library, loaded, fn = self.specialized_graph(tokens)
                arena, buffers = self.arena_for(layout, data['word_ids'])
                self.assertEqual(len(calls['operations']), 147)
                self.assertEqual(len(layout['memory']['weights']['entries']), 25)
                self.assertEqual(fn(arena, len(arena)), 0)
                propagation = self.verify_affine_arithmetic(arena, layout, calls, data, tokens)
                errors = {}
                for name, expected in data.items():
                    if name.startswith('weight__') or name == 'word_ids':
                        continue
                    actual = np.ndarray(expected.shape, np.float32, buffer=arena,
                                        offset=buffers[name]['abs_offset'])
                    self.assertTrue(np.isfinite(actual).all(), name)
                    diff = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
                    limit = self.composition_limit(name, expected, propagation.get(name))
                    self.assertTrue(np.all(diff <= limit), (tokens, name))
                    self.assertLessEqual(float(np.sqrt(np.mean(diff*diff))), 3e-5)
                    errors[name] = {'max_abs': float(diff.max()),
                                    'max_scaled': float(np.max(diff/limit))}
                    point = next(point for op in calls['operations'] for point in op.get('semantic_checkpoints', [])
                                 if point['tensor'] == name)
                    self.emit_comparison(name, errors[name]['max_abs'], float(np.max(limit)), diff, limit, tokens, point)
                results[str(tokens)] = errors
        (self.root / 'specialization-errors.json').write_text(json.dumps(results, indent=2))

    def test_full_pinned_bump_replay_when_available(self):
        location = os.environ.get('CKE_KOKORO_BUMP_DIR')
        if not location:
            self.skipTest('full pinned BUMP unavailable; fixture subset is independently tested')
        directory = Path(location)
        bundle = first_layer.exporter.verify_bundle(directory)
        self.assertEqual(bundle['pin']['model'], self.encoder_meta['model_pin'])
        self.assertEqual(bundle['provenance']['source_asset_sha256'], self.encoder_meta['asset_hashes'])
        entries = {x['name']: x for x in bundle['entries']}
        arena = self.arena()
        with (directory / 'weights.bump').open('rb') as stream:
            for name, planned in self.weights.items():
                entry = entries[name]
                self.assertEqual(entry['sha256'], self.entries[name]['sha256'])
                self.assertEqual(entry['size'], planned['size'])
                stream.seek(entry['file_offset'])
                payload = stream.read(entry['size'])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry['sha256'])
                arena[planned['abs_offset']:planned['abs_offset']+len(payload)] = payload
        self.assertEqual(self.fn(arena, len(arena)), 0)
        actual = self.vector(arena, 'phoneme_features', np.float32, 36*512)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, self.encoder['phoneme_features'].ravel(), atol=3e-5, rtol=0)

    def test_unsupported_mask_port_is_rejected_by_normal_lowering(self):
        import copy
        source = copy.deepcopy(self.source)
        source['template']['activation_buffers']['attention_mask'] = {'shape': [36]}
        source['template']['activation_bindings']['attention_mask'] = 'attention_mask'
        op = next(op for op in source['template']['block_types']['phoneme_encoder']['body']['ops']
                  if op['op'] == 'attention_full_token_major_checked')
        op['graph_slots']['inputs']['mask'] = 'external:attention_mask'
        source['config']['activation_buffer_dtypes']['attention_mask'] = 'i32'
        folder = self.root / 'unsupported-mask'
        folder.mkdir(exist_ok=True)
        circuit = folder / 'circuit.json'
        circuit.write_text(json.dumps(source['template']))
        with self.assertRaisesRegex(RuntimeError, 'HARD CIRCUIT INTERFACE FAULT: circuit declares unknown input ports'):
            compile_native_graph(folder, source, circuit)
        self.assertFalse((folder / 'generated.c').exists())


class KokoroEncoderCircuitAuthoringTest(unittest.TestCase):
    def test_canonical_circuit_and_shared_edges(self):
        circuit = circuit_author.build_circuit()
        self.assertEqual(circuit, json.loads(circuit_author.OUTPUT.read_text()))
        block = circuit['block_types']['phoneme_encoder']
        self.assertEqual(len(block['header']), 2)
        self.assertEqual(len(block['body']['ops']), 144)
        self.assertEqual(len(block['footer']), 1)
        exports = circuit['semantic_checkpoints']['exports']
        self.assertEqual(len(exports), 147)
        identities = [point['id'] for export in exports.values() for point in export['checkpoints']]
        self.assertEqual(len(set(identities)), 147)
        for i in range(12):
            op = block['body']['ops'][12*i]
            self.assertEqual(op['weight_refs']['weight'],
                'phoneme_encoder.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight')
            self.assertEqual(op['graph_slots']['inputs']['input'],
                'projection_output' if i == 0 else f'l{i-1:02d}_albert_layer_output')

    def test_specializations_and_rejected_geometry(self):
        for tokens in (2, 8, 36, 64, 512):
            graph = circuit_author.build_circuit(tokens)
            self.assertEqual(graph['activation_buffers']['phoneme_features']['shape'], [tokens, 512])
            self.assertEqual(graph['runtime_constants']['tokens'], tokens)
        for tokens in (0, 1, 513, -1, True, 2.0):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                circuit_author.build_circuit(tokens)


if __name__ == '__main__':
    unittest.main()
