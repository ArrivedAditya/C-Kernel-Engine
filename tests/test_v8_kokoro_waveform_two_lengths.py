"""Two explicit Kokoro phoneme-count buckets through the generated WAV graph."""

import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import wave

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_duration_fixture_support import populated_duration_arena

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_waveform_circuit as author
import export_kokoro_voice_bank as voice_export

FIXTURE = ROOT / 'tests/fixtures/tts/kokoro_waveform_hello_8_pinned.npz'


class KokoroWaveformTwoLengthsTest(unittest.TestCase):
    def test_exact_bucket_authoring_preserves_default(self):
        self.assertEqual(json.loads(author.OUTPUT.read_text()), author.build_circuit())
        for tokens in (3, 8, 40):
            with self.subTest(tokens=tokens):
                graph = author.build_circuit(tokens)
                self.assertEqual(graph['runtime_constants']['phoneme_count'], tokens)
                self.assertEqual(graph['activation_buffers']['word_ids']['shape'], [tokens])
                self.assertEqual(graph['activation_buffers']['text_encoder_features']['shape'], [tokens, 512])
                self.assertEqual(graph['activation_buffers']['generator_waveform']['shape'], [76800])
                text = graph['block_types']['text_encoder']['body']['ops']
                self.assertEqual(next(op for op in text if op['id'] == 'text_lstm')['params']['T'], tokens)
        for tokens in (0, 1, 2, 41, 512, True):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                author.build_circuit(tokens)

    @unittest.skipUnless(os.environ.get('CKE_KOKORO_MODEL_DIR'),
                         'pinned voice bank unavailable; set CKE_KOKORO_MODEL_DIR')
    def test_selected_voice_styles_match_pinned_bank(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable for pinned voice-bank comparison')

        model_dir = Path(os.environ['CKE_KOKORO_MODEL_DIR'])
        voice_path = model_dir / 'voices/af_heart.pt'
        short_meta = json.loads(FIXTURE.with_suffix('.json').read_text())
        long_meta = json.loads((ROOT / 'version/v8/tts/reference/fixture_manifest.json').read_text())
        expected_hash = short_meta['asset_sha256']['voices/af_heart.pt']
        self.assertEqual(expected_hash, long_meta['assets']['voices/af_heart.pt'])
        self.assertEqual(hashlib.sha256(voice_path.read_bytes()).hexdigest(), expected_hash)
        bank = torch.load(voice_path, map_location='cpu', weights_only=True).detach().numpy()
        self.assertEqual(bank.shape[-1], 256)
        bank = bank.reshape(bank.shape[0], 256)
        with tempfile.TemporaryDirectory() as directory:
            native_path = Path(directory) / 'af_heart.ckvoice'
            export_meta = voice_export.export_voice_bank(voice_path, native_path)
            self.assertEqual(export_meta['source_sha256'], expected_hash)
            self.assertEqual(export_meta['native_sha256'],
                '50a2b7d543cc5a9bdefb3217e3bf1a8d87e947dd5b4e8babf685c0bc4b3b4177')
            packed = native_path.read_bytes()
            self.assertEqual(packed[:8], b'CKVOICE1')
            self.assertEqual(struct.unpack_from('<II', packed, 8), (510, 256))
            np.testing.assert_array_equal(np.frombuffer(packed, dtype='<f4',
                offset=16).reshape(510, 256), bank)
        for tokens in (3, 8, 36, 40):
            self.assertLess(tokens - 3, bank.shape[0])
        with np.load(FIXTURE, allow_pickle=False) as short, \
                np.load(ROOT / 'tests/fixtures/tts/kokoro_generator_stage0_pinned.npz',
                        allow_pickle=False) as long:
            for tokens, metadata, fixture in ((8, short_meta, short), (36, long_meta, long)):
                with self.subTest(tokens=tokens):
                    row = tokens - 3
                    self.assertEqual(metadata['voice_row_index'], row)
                    np.testing.assert_array_equal(fixture['decoder_style'].ravel(), bank[row, :128])
                    np.testing.assert_array_equal(fixture['predictor_style'].ravel(), bank[row, 128:])
        self.assertFalse(np.array_equal(bank[5], bank[33]))

    @unittest.skipUnless(os.environ.get('CKE_TTS_RUN_TWO_LENGTH_WAVEFORM'),
                         'asset-backed two-length waveform replay not requested')
    def test_loaded_exact_buckets_short_long_short(self):
        from tests.test_v8_kokoro_generated_waveform import KokoroGeneratedWaveformTest
        meta = json.loads(FIXTURE.with_suffix('.json').read_text())
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                         meta['fixture_sha256'])
        self.assertEqual(meta['input_text'], 'Hello.')
        self.assertEqual(meta['voice_row_index'], len(meta['input_ids']) - 3)
        long_reference = json.loads((ROOT / 'version/v8/tts/reference/fixture_manifest.json').read_text())
        self.assertEqual(long_reference['voice_row_index'], len(long_reference['input_ids']) - 3)
        self.assertEqual(len(long_reference['input_ids']), 36)
        with np.load(FIXTURE, allow_pickle=False) as archive:
            short = {name: archive[name].copy() for name in archive.files}
        self.assertEqual(short['word_ids'].tolist(), meta['input_ids'])
        baseline = KokoroGeneratedWaveformTest
        baseline.setUpClass()
        try:
            self.assertEqual(meta['model_pin']['revision'],
                             baseline.spectral_meta['model_pin'])
            self.assertEqual(meta['code_pin']['kokoro'],
                             baseline.spectral_meta['code_pin'])
            self.assertEqual(meta['asset_sha256'],
                             baseline.spectral_meta['asset_sha256'])
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                circuit = root / 'kokoro-waveform-8.json'
                circuit.write_text(json.dumps(author.build_circuit(8)))
                source = json.loads(json.dumps(baseline.waveform_source))
                source['template'] = author.build_circuit(8)
                source['config'].update(model=source['template']['name'],
                    arch=source['template']['name'], context_length=8,
                    max_seq_len=8, T=8)
                layout, calls, library, loaded, short_fn = compile_native_graph(
                    root, source, circuit)
                self.assertEqual(calls['errors'], [])
                self.assertEqual(len(calls['operations']), len(baseline.calls['operations']))
                short_fn.argtypes = [ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_size_t] + [ctypes.POINTER(ctypes.c_int32)] * 7
                short_arena = populated_duration_arena(layout,
                    baseline.entries, baseline.bump, baseline.encoder,
                    baseline.duration, word_ids=short['word_ids'])
                view = baseline.view
                view(layout, short_arena, 'predictor_style')[:] = \
                    short['predictor_style'].ravel()
                view(layout, short_arena, 'decoder_style')[:] = \
                    short['decoder_style'].ravel()
                gaussian = view(layout, short_arena, 'source_gaussian').reshape(-1, 9)
                gaussian[:32400] = short['gaussian'].reshape(32400, 9)
                view(layout, short_arena, 'source_stft_window')[:] = short['window']
                taps = np.arange(20, dtype=np.float64)
                angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
                view(layout, short_arena, 'source_stft_cos').reshape(11, 20)[:] = np.cos(angles)
                view(layout, short_arena, 'source_stft_sin').reshape(11, 20)[:] = np.sin(angles)
                long_arena = baseline.arena()
                short_output = view(layout, short_arena, 'generator_waveform')
                short_output.fill(-91.)
                rejected_lengths = [ctypes.c_int32(-1) for _ in range(7)]
                self.assertNotEqual(short_fn(short_arena, len(short_arena) - 1,
                    *(ctypes.byref(value) for value in rejected_lengths)), 0)
                self.assertEqual([value.value for value in rejected_lengths], [-1] * 7)
                self.assertTrue(np.all(short_output == -91.))
                outputs = []
                report = []
                for label, fn, arena, selected_layout, expected in (
                    ('short', short_fn, short_arena, layout, short['waveform']),
                    ('long', baseline.fn, long_arena, baseline.layout,
                     baseline.spectral['waveform'][0, 0]),
                    ('short', short_fn, short_arena, layout, short['waveform'])):
                    output = view(selected_layout, arena, 'generator_waveform')
                    output.fill(-91.)
                    lengths = [ctypes.c_int32(-1) for _ in range(7)]
                    started = time.perf_counter()
                    self.assertEqual(fn(arena, len(arena), *(
                        ctypes.byref(value) for value in lengths)), 0)
                    elapsed = time.perf_counter() - started
                    count = lengths[3].value
                    self.assertEqual(count, len(expected))
                    self.assertTrue(np.isfinite(output[:count]).all())
                    self.assertTrue(np.all(output[count:] == -91.))
                    actual = output[:count].copy()
                    error = np.abs(actual.astype(np.float64) -
                                   expected.astype(np.float64))
                    report.append({'bucket': label, 'tokens': 8 if label == 'short' else 36,
                        'sample_count': count, 'seconds': elapsed,
                        'oracle_max_abs': float(error.max()),
                        'oracle_rmse': float(np.sqrt(np.mean(error * error))),
                        'oracle_numerical_status': 'pass' if error.max() <= 1e-5 else 'fail'})
                    if label == 'short' and not outputs:
                        durations = view(layout, short_arena, 'runtime_values', np.int32)[:8]
                        np.testing.assert_array_equal(durations, short['durations'])
                        f0 = view(layout, short_arena, 'f0_output')[:short['f0'].size]
                        report[-1]['f0_max_abs'] = float(np.max(np.abs(
                            f0.astype(np.float64) - short['f0'].ravel().astype(np.float64))))
                    outputs.append(actual)
                np.testing.assert_array_equal(outputs[0], outputs[2])
                self.assertNotEqual(hashlib.sha256(outputs[0].tobytes()).hexdigest(),
                                    hashlib.sha256(outputs[1].tobytes()).hexdigest())
                host_root = root / 'standalone-short'
                host_root.mkdir()
                for source_path, name in (
                    (library, 'generated.so'),
                    (ROOT / 'version/v8/tts/native_kokoro_wav_host.c', 'host.c'),
                    (ROOT / 'src/kernels/audio_noise_counter.c', 'audio_noise_counter.c'),
                    (ROOT / 'src/kernels/checked_float_buffer.h', 'checked_float_buffer.h'),
                    (ROOT / 'include/ckernel_tts.h', 'ckernel_tts.h')):
                    shutil.copy2(source_path, host_root / name)
                binary = host_root / 'native-kokoro-wav'
                subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra',
                    '-Werror', '-pedantic', '-DCK_TTS_COMPILED_TOKEN_COUNT=8',
                    '-I', str(host_root),
                    str(host_root / 'host.c'), str(host_root / 'audio_noise_counter.c'),
                    '-L', str(host_root), '-l:generated.so', '-Wl,-rpath,$ORIGIN',
                    '-lm', '-o', str(binary)], check=True, capture_output=True)
                buffers = {item['name']: item for item in
                    layout['memory']['activations']['buffers']}
                arena_bytes = layout['memory']['arena']['total_size']
                package = host_root / 'prepared-hello.cktts'
                with mmap.mmap(-1, arena_bytes) as packed:
                    for item in layout['memory']['weights']['entries']:
                        entry = baseline.entries[item['name']]
                        start = entry['file_offset']
                        packed[item['abs_offset']:item['abs_offset'] + entry['size']] = \
                            baseline.bump[start:start + entry['size']]
                    def fill(name, values, dtype=np.float32):
                        item = buffers[name]
                        array = np.asarray(values, dtype=dtype).ravel()
                        np.ndarray((array.size,), dtype, buffer=packed,
                            offset=item['abs_offset'])[:] = array
                    fill('word_ids', short['word_ids'], np.int32)
                    fill('predictor_style', short['predictor_style'])
                    fill('decoder_style', short['decoder_style'])
                    fill('source_gaussian', short['gaussian'])
                    fill('source_stft_window', short['window'])
                    fill('source_stft_cos', np.cos(angles))
                    fill('source_stft_sin', np.sin(angles))
                    output_item = buffers['generator_waveform']
                    segments = 0
                    with package.open('wb') as stream:
                        stream.write(b'CKTTS001')
                        stream.write(struct.pack('<QQQI', arena_bytes,
                            output_item['abs_offset'], output_item['size'] // 4, 0))
                        for offset in range(0, arena_bytes, 65536):
                            chunk = packed[offset:offset + 65536]
                            if any(chunk):
                                stream.write(struct.pack('<QQ', offset, len(chunk)))
                                stream.write(chunk)
                                segments += 1
                        stream.seek(32)
                        stream.write(struct.pack('<I', segments))
                wav_path = host_root / 'hello-generated.wav'
                native = subprocess.run([str(binary), str(package), str(wav_path)],
                    cwd=host_root, check=True, capture_output=True, text=True)
                self.assertIn('wrote 32400 samples at 24000 Hz', native.stdout)
                with wave.open(str(wav_path), 'rb') as audio:
                    self.assertEqual((audio.getnchannels(), audio.getsampwidth(),
                                      audio.getframerate(), audio.getnframes()),
                                     (1, 2, 24000, 32400))
                    pcm = audio.readframes(32400)
                expected_pcm = np.rint(np.clip(outputs[0], -1., 1.) *
                    np.float32(32767.)).astype('<i2')
                np.testing.assert_array_equal(np.frombuffer(pcm, dtype='<i2'),
                                              expected_pcm)
                # Keep the captured-noise oracle lane above intact. Independently
                # exercise the shorter graph with CKE's request-owned noise and
                # compare its generated output with the standalone native host.
                noise_library = host_root / 'audio_noise_counter.so'
                subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra',
                    '-Werror', '-pedantic', '-shared', '-fPIC', '-I', str(host_root),
                    str(host_root / 'audio_noise_counter.c'), '-lm',
                    '-o', str(noise_library)], check=True, capture_output=True)
                noise = ctypes.CDLL(str(noise_library)).audio_noise_counter_gaussian_f32
                noise.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                                  ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64]
                noise.restype = ctypes.c_int
                short_gaussian = view(layout, short_arena, 'source_gaussian')
                self.assertEqual(noise(short_gaussian.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)), short_gaussian.size,
                    short_gaussian.size, 7, 0), 0)
                short_output.fill(-91.)
                seeded_lengths = [ctypes.c_int32(-1) for _ in range(7)]
                self.assertEqual(short_fn(short_arena, len(short_arena), *(
                    ctypes.byref(value) for value in seeded_lengths)), 0)
                self.assertEqual(seeded_lengths[3].value, 32400)
                self.assertTrue(np.isfinite(short_output[:32400]).all())
                self.assertTrue(np.all(short_output[32400:] == -91.))
                seeded_pcm = np.rint(np.clip(short_output[:32400], -1., 1.) *
                    np.float32(32767.)).astype('<i2')
                self.assertFalse(np.array_equal(seeded_pcm, expected_pcm))
                gaussian_item = buffers['source_gaussian']
                self.assertEqual(gaussian_item['size'] // 4, 76800 * 9)
                native_package = host_root / 'prepared-hello-native-noise.cktts'
                with package.open('rb') as captured, native_package.open('wb') as stream:
                    stream.write(b'CKTTS002')
                    stream.write(struct.pack('<QQQQQI', arena_bytes,
                        output_item['abs_offset'], output_item['size'] // 4,
                        gaussian_item['abs_offset'], gaussian_item['size'] // 4,
                        segments))
                    captured.seek(36)
                    shutil.copyfileobj(captured, stream, length=65536)
                seeded_wav = host_root / 'hello-native-seed-7.wav'
                subprocess.run([str(binary), str(native_package),
                    str(seeded_wav), '7'], cwd=host_root, check=True,
                    capture_output=True, text=True)
                with wave.open(str(seeded_wav), 'rb') as audio:
                    self.assertEqual((audio.getnchannels(), audio.getsampwidth(),
                                      audio.getframerate(), audio.getnframes()),
                                     (1, 2, 24000, 32400))
                    actual_seeded_pcm = audio.readframes(32400)
                np.testing.assert_array_equal(
                    np.frombuffer(actual_seeded_pcm, dtype='<i2'), seeded_pcm)
                # CKTTS003 binds request IDs and the selected voice-bank row
                # at the native host boundary. Use the complete pinned bank
                # when supplied; otherwise package the verified fixture row.
                voice_bank = host_root / 'af_heart-fixture-row.ckvoice'
                full_bank = os.environ.get('CKE_TTS_NATIVE_VOICE_BANK')
                if full_bank:
                    voice_bank_bytes = Path(full_bank).read_bytes()
                    self.assertEqual(hashlib.sha256(voice_bank_bytes).hexdigest(),
                        '50a2b7d543cc5a9bdefb3217e3bf1a8d87e947dd5b4e8babf685c0bc4b3b4177')
                    rows_count, columns = struct.unpack_from('<II',
                        voice_bank_bytes, 8)
                    self.assertEqual((voice_bank_bytes[:8], rows_count, columns),
                                     (b'CKVOICE1', 510, 256))
                    rows = np.frombuffer(voice_bank_bytes, dtype='<f4',
                        offset=16).reshape(rows_count, columns)
                    voice_source = 'complete pinned af_heart bank'
                    voice_bank.write_bytes(voice_bank_bytes)
                else:
                    rows = np.zeros((6, 256), dtype='<f4')
                    rows[5, :128] = short['decoder_style'].ravel()
                    rows[5, 128:] = short['predictor_style'].ravel()
                    voice_source = 'pinned fixture row only'
                    voice_bank.write_bytes(b'CKVOICE1' +
                        struct.pack('<II', 6, 256) + rows.tobytes())
                np.testing.assert_array_equal(rows[5, :128],
                                              short['decoder_style'].ravel())
                np.testing.assert_array_equal(rows[5, 128:],
                                              short['predictor_style'].ravel())
                request_package = host_root / 'prepared-hello-request.cktts'
                with package.open('rb') as captured, request_package.open('wb') as stream:
                    stream.write(b'CKTTS003')
                    stream.write(struct.pack('<QQQQQQQQQI', arena_bytes,
                        output_item['abs_offset'], output_item['size'] // 4,
                        gaussian_item['abs_offset'], gaussian_item['size'] // 4,
                        buffers['word_ids']['abs_offset'], 8,
                        buffers['predictor_style']['abs_offset'],
                        buffers['decoder_style']['abs_offset'], segments))
                    captured.seek(36)
                    shutil.copyfileobj(captured, stream, length=65536)
                request_wav = host_root / 'hello-native-request.wav'
                ids_csv = ','.join(str(int(value)) for value in short['word_ids'])
                subprocess.run([str(binary), str(request_package),
                    str(request_wav), '7', ids_csv, str(voice_bank)],
                    cwd=host_root, check=True, capture_output=True, text=True)
                self.assertEqual(request_wav.read_bytes(), seeded_wav.read_bytes())
                evidence = {
                    'schema': 'cke.kokoro.two_length_waveform_diagnostic.v1',
                    'execution_status': 'pass',
                    'numerical_status': 'fail' if any(
                        item['oracle_numerical_status'] == 'fail' for item in report)
                        else 'pass',
                    'scope': 'two precompiled exact-length graphs loaded in one process; '
                             'captured Gaussian oracle replay plus seeded short '
                             'standalone replay; no text frontend or bucket padding',
                    'model_revision': meta['model_pin']['revision'],
                    'reference_revision': meta['code_pin']['kokoro'],
                    'fixture_sha256': meta['fixture_sha256'],
                    'voice_row_index': {'short': meta['voice_row_index'],
                                        'long': long_reference['voice_row_index']},
                    'capacity_rejection_and_recovery': 'pass',
                    'generated_library_sha256': {
                        'short': hashlib.sha256(library.read_bytes()).hexdigest(),
                        'long': hashlib.sha256((baseline.native_dir / 'generated.so').read_bytes()).hexdigest()},
                    'library_artifact_paths': {'short': str(library),
                                               'long': str(baseline.native_dir / 'generated.so')},
                    'runtime_build_identity': 'NOT_VERIFIED',
                    'standalone_short_wav_sha256': hashlib.sha256(
                        wav_path.read_bytes()).hexdigest(),
                    'native_noise_short': {
                        'seed': 7,
                        'status': 'pass',
                        'wav_sha256': hashlib.sha256(
                            seeded_wav.read_bytes()).hexdigest(),
                        'pcm_matches_generated_seeded_output': True,
                        'pytorch_noise_parity': 'NOT_TESTED',
                    },
                    'native_request_short': {
                        'status': 'pass',
                        'phoneme_count': 8,
                        'voice_row': 5,
                        'voice_source': voice_source,
                        'voice_bank_sha256': hashlib.sha256(
                            voice_bank.read_bytes()).hexdigest(),
                        'seed': 7,
                        'wav_sha256': hashlib.sha256(
                            request_wav.read_bytes()).hexdigest(),
                        'pcm_matches_seeded_host': True,
                    },
                    'host': platform.platform(),
                    'reproduction_command': 'CKE_TTS_RUN_TWO_LENGTH_WAVEFORM=1 '
                        'python3 -m unittest '
                        'tests.test_v8_kokoro_waveform_two_lengths.'
                        'KokoroWaveformTwoLengthsTest.'
                        'test_loaded_exact_buckets_short_long_short',
                    'requests': report,
                }
                retain_dir = os.environ.get('CKE_TTS_RETAIN_DIR')
                if retain_dir:
                    retained = Path(retain_dir) / 'kokoro-two-length-short'
                    retained.mkdir(parents=True, exist_ok=True)
                    retained_wavs = {}
                    for label, source in (('captured-noise', wav_path),
                                          ('seeded-noise', seeded_wav),
                                          ('native-request', request_wav)):
                        target = retained / f'hello-{label}.wav'
                        shutil.copy2(source, target)
                        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),
                                         hashlib.sha256(source.read_bytes()).hexdigest())
                        retained_wavs[label] = str(target)
                    evidence['retained_wavs'] = retained_wavs
                    (retained / 'reproduction.json').write_text(
                        json.dumps(evidence, indent=2) + '\n')
                output_path = os.environ.get('CKE_TTS_TWO_LENGTH_REPORT')
                if output_path:
                    Path(output_path).write_text(json.dumps(evidence, indent=2) + '\n')
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': 'kokoro.waveform.exact-bucket-native-request-v1',
                    'name': 'Kokoro 8-ID native request and voice-row binding',
                    'provider': 'generated_kokoro_waveform+native_request_host',
                    'dtype': 'fp32', 'direction': 'inference',
                    'oracle': 'same-seed generated-entry PCM',
                    'status': 'pass', 'gate': 'session', 'blocking': True,
                    'max_diff': 0, 'tolerance': 0,
                    'configuration': '8 supplied IDs, fixture voice row 5, '
                        'seed 7, 32400 samples; request WAV equals seeded host WAV',
                    'reproduction_command': evidence['reproduction_command'],
                }, sort_keys=True))
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': 'kokoro.waveform.exact-bucket-short-native-noise-v1',
                    'name': 'Kokoro 8-ID seeded native host PCM replay',
                    'provider': 'audio_noise_counter_gaussian_f32+generated_kokoro_waveform',
                    'dtype': 'fp32', 'direction': 'inference',
                    'oracle': 'same-seed generated-entry PCM',
                    'status': 'pass', 'gate': 'session', 'blocking': True,
                    'max_diff': 0, 'tolerance': 0,
                    'configuration': '8 IDs, af_heart, seed 7, 32400 samples; '
                        'native host PCM equals in-process seeded generated output',
                    'pytorch_noise_parity': 'NOT_TESTED',
                    'reproduction_command': evidence['reproduction_command'],
                }, sort_keys=True))
                for ordinal, item in enumerate(report[:2]):
                    label = 'short' if ordinal == 0 else 'long'
                    print('CKE_NUMERICAL_CASE ' + json.dumps({
                        'case_id': f'kokoro.waveform.exact-bucket-{label}-diagnostic-v1',
                        'name': f'Kokoro {item["tokens"]}-ID connected waveform',
                        'provider': 'generated_kokoro_waveform', 'dtype': 'fp32',
                        'direction': 'inference',
                        'oracle': 'pinned-PyTorch-2.8-full-model-waveform',
                        'status': item['oracle_numerical_status'],
                        'gate': 'diagnostic', 'blocking': False,
                        'max_diff': item['oracle_max_abs'], 'tolerance': 1e-5,
                        'rmse': item['oracle_rmse'],
                        'configuration': f'{item["tokens"]} IDs, af_heart, '
                            f'captured Gaussian, {item["sample_count"]} samples',
                        'reproduction_command': evidence['reproduction_command'],
                    }, sort_keys=True))
                print('CKE_TTS_TWO_LENGTH_DIAGNOSTIC ' + json.dumps(evidence, sort_keys=True))
        finally:
            baseline.tearDownClass()


if __name__ == '__main__':
    unittest.main()
