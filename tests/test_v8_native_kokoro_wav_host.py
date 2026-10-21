"""Cheap native host contract for seeded excitation packages."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_COUNT = 76800 * 9
GAUSSIAN_BYTES = GAUSSIAN_COUNT * 4
WAVEFORM_OFFSET = GAUSSIAN_BYTES
IDS_OFFSET = WAVEFORM_OFFSET + 128
PREDICTOR_OFFSET = IDS_OFFSET + 64
DECODER_OFFSET = PREDICTOR_OFFSET + 512
ARENA_BYTES = DECODER_OFFSET + 512


class NativeKokoroWavHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        stub = cls.root / 'stub.c'
        stub.write_text('''#include <stddef.h>
#include <stdint.h>
int ck_kokoro_waveform_bounded(uint8_t *arena, size_t capacity,
    int32_t *a, int32_t *b, int32_t *c, int32_t *d,
    int32_t *e, int32_t *f, int32_t *g) {
    if (!arena || capacity < ''' + str(ARENA_BYTES) + ''') return -1;
    const float *noise = (const float *)arena;
    float *audio = (float *)(arena + ''' + str(GAUSSIAN_BYTES) + ''');
    for (int i = 0; i < 20; ++i) audio[i] = .01f * noise[i];
    const int32_t *ids = (const int32_t *)(arena + ''' + str(IDS_OFFSET) + ''');
    const float *predictor = (const float *)(arena + ''' + str(PREDICTOR_OFFSET) + ''');
    const float *decoder = (const float *)(arena + ''' + str(DECODER_OFFSET) + ''');
    for (int i = 0; i < 20; ++i)
        audio[i] += .0001f * (float)ids[1] +
            .1f * predictor[0] + .1f * decoder[0];
    *a = 1; *b = 1; *c = 1; *d = 20; *e = 5; *f = 1; *g = 1;
    return 0;
}\n''')
        cls.binary = cls.root / 'wav-host'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-pedantic', '-DCK_TTS_COMPILED_TOKEN_COUNT=8',
            '-I', str(ROOT / 'include'), '-I',
            str(ROOT / 'src/kernels'),
            str(ROOT / 'version/v8/tts/native_kokoro_wav_host.c'),
            str(ROOT / 'src/kernels/audio_noise_counter.c'), str(stub),
            '-lm', '-o', str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_loaded_generated_library_identity_is_checked_before_package(self):
        library = self.root / 'stub-generated.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra',
            '-Werror', '-shared', '-fPIC', str(self.root / 'stub.c'),
            '-o', str(library)], check=True, capture_output=True)
        actual = hashlib.sha256(library.read_bytes()).hexdigest()
        for label, expected, status in (
                ('matching', actual, 3), ('mismatch', '0' * 64, 9)):
            with self.subTest(label=label):
                binary = self.root / f'identity-{label}'
                subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra',
                    '-Werror', '-pedantic', '-DCK_TTS_COMPILED_TOKEN_COUNT=8',
                    f'-DCK_TTS_EXPECTED_LIBRARY_SHA256="{expected}"',
                    '-I', str(ROOT / 'include'), '-I',
                    str(ROOT / 'src/kernels'),
                    str(ROOT / 'version/v8/tts/native_kokoro_wav_host.c'),
                    str(ROOT / 'src/kernels/audio_noise_counter.c'),
                    '-L', str(self.root), '-l:stub-generated.so',
                    '-Wl,-rpath,$ORIGIN', '-lm', '-ldl', '-lcrypto',
                    '-o', str(binary)], check=True, capture_output=True,
                    text=True)
                run = subprocess.run([str(binary),
                    str(self.root / 'missing.cktts'),
                    str(self.root / 'never.wav'), '7',
                    '0,4,4,4,4,4,4,0', str(self.root / 'missing.ckvoice')],
                    capture_output=True, text=True)
                self.assertEqual(run.returncode, status, run.stderr)
                self.assertIn('resolved generated symbol in' if label ==
                    'matching' else 'generated library identity mismatch',
                    run.stderr)
                self.assertFalse((self.root / 'never.wav').exists())

    def package(self, name, gaussian_offset=0, gaussian_count=GAUSSIAN_COUNT,
                waveform_offset=WAVEFORM_OFFSET):
        path = self.root / name
        path.write_bytes(struct.pack('<8sQQQQQI', b'CKTTS002', ARENA_BYTES,
            waveform_offset, 20, gaussian_offset, gaussian_count, 0))
        return path

    def test_runtime_entry_uses_valid_sample_count(self):
        stub = self.root / 'runtime-stub.c'
        stub.write_text('''#include <stddef.h>
#include <stdint.h>
int ck_kokoro_runtime_waveform_bounded(uint8_t *arena, size_t capacity,
    int32_t *a, int32_t *b, int32_t *c, int32_t *d, int32_t *e,
    int32_t *f, int32_t *g, int32_t *h, int32_t *i) {
    if (!arena || capacity < 256) return -1;
    float *audio = (float *)arena;
    for (int j = 0; j < 20; ++j) audio[j] = .25f;
    *a = 8; *b = 6144; *c = 1; *d = 2; *e = 4;
    *f = 20; *g = 5; *h = 4; *i = 5;
    return 0;
}
''')
        binary = self.root / 'runtime-wav-host'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-pedantic', '-DCK_TTS_RUNTIME_WAVEFORM',
            '-I', str(ROOT / 'include'), '-I', str(ROOT / 'src/kernels'),
            str(ROOT / 'version/v8/tts/native_kokoro_wav_host.c'),
            str(ROOT / 'src/kernels/audio_noise_counter.c'), str(stub),
            '-lm', '-o', str(binary)], check=True, capture_output=True)
        package = self.root / 'runtime.cktts'
        package.write_bytes(struct.pack('<8sQQQI', b'CKTTS001',
            256, 0, 64, 0))
        wav = self.root / 'runtime.wav'
        run = subprocess.run([str(binary), str(package), str(wav)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        with wave.open(str(wav), 'rb') as audio:
            self.assertEqual((audio.getnframes(), audio.getframerate()),
                             (20, 24000))
            self.assertTrue(any(audio.readframes(20)))

    def test_runtime_request_binds_length_padding_and_voice_row(self):
        ids_offset = GAUSSIAN_BYTES + 128
        predictor_offset = ids_offset + 64 * 4
        decoder_offset = predictor_offset + 512
        length_offset = decoder_offset + 512
        arena_bytes = length_offset + 4
        stub = self.root / 'runtime-request-stub.c'
        stub.write_text('''#include <stddef.h>
#include <stdint.h>
int ck_kokoro_runtime_waveform_bounded(uint8_t *arena, size_t capacity,
    int32_t *a, int32_t *b, int32_t *c, int32_t *d, int32_t *e,
    int32_t *f, int32_t *g, int32_t *h, int32_t *i) {
    if (!arena || capacity < ''' + str(arena_bytes) + ''') return -1;
    const int32_t n = *(const int32_t *)(arena + ''' + str(length_offset) + ''');
    const int32_t *ids = (const int32_t *)(arena + ''' + str(ids_offset) + ''');
    const float *style = (const float *)(arena + ''' + str(predictor_offset) + ''');
    if (n != 8 && n != 13 && n != 36) return -2;
    for (int32_t j = n; j < 64; ++j) if (ids[j] != 178) return -3;
    if (style[0] != (float)(n - 3) / 100.f) return -4;
    const int32_t samples = n == 8 ? 20 : n == 13 ? 25 : 30;
    float *audio = (float *)(arena + ''' + str(WAVEFORM_OFFSET) + ''');
    for (int32_t j = 0; j < samples; ++j)
        audio[j] = .1f + .001f * n + .0001f * ids[1];
    *a = n; *b = n * 768; *c = 1; *d = 2; *e = 4;
    *f = samples; *g = samples / 5 + 1; *h = samples / 5; *i = *g;
    return 0;
}
''')
        binary = self.root / 'runtime-request-host'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-pedantic', '-DCK_TTS_COMPILED_TOKEN_COUNT=64',
            '-DCK_TTS_RUNTIME_WAVEFORM', '-I', str(ROOT / 'include'),
            '-I', str(ROOT / 'src/kernels'),
            str(ROOT / 'version/v8/tts/native_kokoro_wav_host.c'),
            str(ROOT / 'src/kernels/audio_noise_counter.c'), str(stub),
            '-lm', '-o', str(binary)], check=True, capture_output=True,
            text=True)
        bank = self.root / 'runtime-voice.ckvoice'
        voices = [0.] * (34 * 256)
        for row in range(34):
            voices[row * 256 + 128] = row / 100.
        bank.write_bytes(b'CKVOICE1' + struct.pack('<II', 34, 256) +
                         struct.pack('<' + 'f' * len(voices), *voices))

        def package(name, *, request_offset=length_offset, count=64):
            path = self.root / name
            path.write_bytes(struct.pack('<8s10QI', b'CKTTS004', arena_bytes,
                WAVEFORM_OFFSET, 30, 0, GAUSSIAN_COUNT, ids_offset, count,
                predictor_offset, decoder_offset, request_offset, 0))
            return path

        packed = package('runtime-request.cktts')
        first = None
        for count in (8, 13, 36, 8):
            ids = ','.join(map(str, [0] + [50] * (count - 2) + [0]))
            output = self.root / f'runtime-request-{count}.wav'
            output.unlink(missing_ok=True)
            run = subprocess.run([str(binary), str(packed), str(output), '7',
                                  ids, str(bank)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            with wave.open(str(output), 'rb') as audio:
                self.assertEqual(audio.getnframes(),
                                 {8: 20, 13: 25, 36: 30}[count])
            if count == 8:
                digest = hashlib.sha256(output.read_bytes()).hexdigest()
                if first is None:
                    first = digest
                else:
                    self.assertEqual(digest, first)
        for bad in (package('bad-request-offset.cktts',
                            request_offset=ids_offset),
                    package('bad-id-capacity.cktts', count=40)):
            output = self.root / 'rejected-runtime.wav'
            run = subprocess.run([str(binary), str(bad), str(output), '7',
                '0,50,50,50,50,50,50,0', str(bank)], capture_output=True)
            self.assertEqual(run.returncode, 4)
            self.assertFalse(output.exists())
        for ids in ('0,50,0', '0,50,178,0', '1,50,0',
                    ','.join(map(str, [0] + [50] * 39 + [0]))):
            output = self.root / 'rejected-runtime.wav'
            run = subprocess.run([str(binary), str(packed), str(output), '7',
                                  ids, str(bank)], capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertFalse(output.exists())

        # The worker must retain its loaded package across requests. Removing
        # the package after READY makes an accidental per-chunk reload fail.
        worker = subprocess.Popen([str(binary), str(packed), '--session',
                                   str(bank)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.assertEqual(worker.stdout.readline().strip(), 'READY\tloads=1')
        packed.unlink()
        requests = []
        for index, count in enumerate((8, 13, 36, 8)):
            ids = ','.join(map(str, [0] + [50] * (count - 2) + [0]))
            output = self.root / f'persistent-{index}.wav'
            requests.append((output, count))
            worker.stdin.write(f'RUN\t{output}\t7\t{ids}\n')
            worker.stdin.flush()
            sample_count = {8: 20, 13: 25, 36: 30}[count]
            self.assertEqual(worker.stdout.readline().strip(),
                             f'WAV_READY\t{sample_count}\t{output}')
        invalid = self.root / 'persistent-invalid.wav'
        ids = ','.join(map(str, [0] + [50] * 14 + [0]))
        worker.stdin.write(f'RUN\t{invalid}\t7\t{ids}\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'ERROR\t7')
        self.assertFalse(invalid.exists())
        worker.stdin.write(f'RUN\t{invalid}\tbad\t{ids}\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'ERROR\t2')
        self.assertFalse(invalid.exists())
        worker.stdin.write(f'RUN\t{requests[0][0]}\t7\t{ids}\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'ERROR\t10')
        recovered = self.root / 'persistent-recovered.wav'
        ids = ','.join(map(str, [0] + [50] * 6 + [0]))
        worker.stdin.write(f'RUN\t{recovered}\t7\t{ids}\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(),
                         f'WAV_READY\t20\t{recovered}')
        aggregate = self.root / 'aggregate.wav'
        worker.stdin.write(f'BEGIN\t{aggregate}\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'BEGUN')
        for count, expected in ((8, 20), (13, 25), (36, 30)):
            ids = ','.join(map(str, [0] + [50] * (count - 2) + [0]))
            worker.stdin.write(f'APPEND\t7\t{ids}\n')
            worker.stdin.flush()
            response = worker.stdout.readline().strip().split('\t')
            self.assertEqual(response[:2], ['CHUNK_READY', str(expected)])
        self.assertFalse(aggregate.exists())
        worker.stdin.write('END\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(),
                         f'WAV_READY\t75\t{aggregate}')
        with wave.open(str(aggregate), 'rb') as audio:
            self.assertEqual(audio.getnframes(), 75)
            joined_pcm = audio.readframes(75)
        isolated_pcm = b''.join(output.read_bytes()[44:] for output, _ in requests[:3])
        self.assertEqual(joined_pcm, isolated_pcm)

        abandoned = self.root / 'abandoned.wav'
        worker.stdin.write(f'BEGIN\t{abandoned}\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'BEGUN')
        worker.stdin.write(f'APPEND\t7\t{ids}\n')
        worker.stdin.flush()
        self.assertTrue(worker.stdout.readline().startswith('CHUNK_READY\t'))
        worker.stdin.write('APPEND\tbad\t0,50,50,50,50,50,50,0\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'ERROR\t2')
        self.assertFalse(abandoned.exists())
        self.assertFalse(list(self.root.glob('abandoned.wav.partial.*')))
        worker.stdin.write(f'BEGIN\t{abandoned}\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'BEGUN')
        worker.stdin.write('APPEND\t7\t0,50,50,50,50,50,50,0\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'CHUNK_READY\t20\t20')
        invalid_ids = ','.join(map(str, [0] + [50] * 14 + [0]))
        worker.stdin.write(f'APPEND\t7\t{invalid_ids}\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'ERROR\t7')
        self.assertFalse(abandoned.exists())
        self.assertFalse(list(self.root.glob('abandoned.wav.partial.*')))
        worker.stdin.write(f'BEGIN\t{abandoned}\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'BEGUN')
        worker.stdin.write('APPEND\t7\t0,50,50,50,50,50,50,0\nEND\nQUIT\n')
        worker.stdin.flush()
        self.assertEqual(worker.stdout.readline().strip(), 'CHUNK_READY\t20\t20')
        self.assertEqual(worker.stdout.readline().strip(),
                         f'WAV_READY\t20\t{abandoned}')
        worker.stdin.close()
        self.assertEqual(worker.wait(timeout=10), 0, worker.stderr.read())
        worker.stdout.close()
        worker.stderr.close()
        for output, count in requests:
            with wave.open(str(output), 'rb') as audio:
                self.assertEqual(audio.getnframes(), {8: 20, 13: 25, 36: 30}[count])
        self.assertEqual(requests[0][0].read_bytes(), requests[3][0].read_bytes())
        self.assertEqual(requests[0][0].read_bytes(), recovered.read_bytes())
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.native-wav-host.persistent-requests-v1',
            'name': 'native host retains package across 8/13/36/8 requests',
            'provider': 'native_kokoro_wav_host', 'dtype': 'pcm16',
            'direction': 'inference',
            'oracle': 'isolated WAVs and exact repeated-request identity',
            'status': 'pass', 'gate': 'session', 'blocking': True,
            'max_diff': 0, 'tolerance': 0,
            'configuration': 'synthetic generated entry; package removed after READY; invalid request then recovery',
            'reproduction_command': 'python3 -m unittest tests.test_v8_native_kokoro_wav_host',
        }))

    def request_package(self, name, ids_offset=IDS_OFFSET, ids_count=8,
                        predictor_offset=PREDICTOR_OFFSET,
                        decoder_offset=DECODER_OFFSET):
        path = self.root / name
        path.write_bytes(struct.pack('<8sQQQQQQQQQI', b'CKTTS003', ARENA_BYTES,
            WAVEFORM_OFFSET, 20, 0, GAUSSIAN_COUNT, ids_offset, ids_count,
            predictor_offset, decoder_offset, 0))
        return path

    def voice_bank(self, name, selected_decoder=.25,
                   selected_predictor=.5):
        path = self.root / name
        values = [0.] * (6 * 256)
        values[5 * 256] = selected_decoder
        values[5 * 256 + 128] = selected_predictor
        path.write_bytes(b'CKVOICE1' + struct.pack('<II', 6, 256) +
                         struct.pack('<' + 'f' * len(values), *values))
        return path

    def test_seeded_replay_distinct_seed_and_bounds(self):
        package = self.package('native.cktts')
        hashes = []
        for index, seed in enumerate(('7', '7', '8')):
            wav = self.root / f'noise-{index}.wav'
            subprocess.run([str(self.binary), str(package), str(wav), seed],
                           check=True, capture_output=True)
            with wave.open(str(wav), 'rb') as stream:
                self.assertEqual((stream.getnchannels(),
                    stream.getsampwidth(), stream.getframerate(),
                    stream.getnframes()), (1, 2, 24000, 20))
                self.assertTrue(any(stream.readframes(20)))
            hashes.append(hashlib.sha256(wav.read_bytes()).hexdigest())
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.native-wav-host.seeded-noise-v1',
            'name': 'native host seeded excitation package and WAV identity',
            'provider': 'audio_noise_counter_gaussian_f32',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'deterministic-replay-and-different-seed-control',
            'status': 'pass', 'gate': 'session', 'blocking': True,
            'max_diff': 0, 'tolerance': 0,
            'same_seed_wav_sha256': hashes[0],
            'different_seed_wav_sha256': hashes[2],
            'configuration': 'synthetic generated-entry stub; 20 PCM16 samples at 24 kHz',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_native_kokoro_wav_host'}))
        for seed in ('-1', '18446744073709551616', 'bad'):
            wav = self.root / 'rejected.wav'
            rejected = subprocess.run([str(self.binary), str(package),
                                       str(wav), seed], capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(wav.exists())
        bad = self.package('overrun.cktts', gaussian_offset=ARENA_BYTES - 4)
        wav = self.root / 'overrun.wav'
        rejected = subprocess.run([str(self.binary), str(bad), str(wav), '7'],
                                  capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(wav.exists())
        bad = self.package('wrong-size.cktts', gaussian_count=GAUSSIAN_COUNT - 1)
        rejected = subprocess.run([str(self.binary), str(bad), str(wav), '7'],
                                  capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(wav.exists())
        bad = self.package('overlap.cktts', waveform_offset=0)
        rejected = subprocess.run([str(self.binary), str(bad), str(wav), '7'],
                                  capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(wav.exists())

    def test_existing_output_is_rejected_before_package_execution(self):
        package = self.package('output-guard.cktts')
        output = self.root / 'existing-output.wav'
        output.write_bytes(b'keep this output')
        original_package = package.read_bytes()
        for label, path in (('ordinary', output), ('package', package)):
            with self.subTest(label=label):
                run = subprocess.run([str(self.binary), str(package),
                    str(path), '7'], capture_output=True, text=True)
                self.assertEqual(run.returncode, 10, run.stderr)
        self.assertEqual(output.read_bytes(), b'keep this output')
        self.assertEqual(package.read_bytes(), original_package)
        for kind in ('symlink', 'hardlink'):
            alias = self.root / f'host-{kind}-to-package.wav'
            if kind == 'symlink':
                alias.symlink_to(package)
            else:
                os.link(package, alias)
            with self.subTest(alias=kind):
                run = subprocess.run([str(self.binary), str(package),
                    str(alias), '7'], capture_output=True, text=True)
                self.assertEqual(run.returncode, 10, run.stderr)
                self.assertEqual(package.read_bytes(), original_package)
        fresh = self.root / 'fresh-after-output-guard.wav'
        run = subprocess.run([str(self.binary), str(package), str(fresh),
            '7'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        with wave.open(str(fresh), 'rb') as stream:
            self.assertEqual(stream.getnframes(), 20)

    def test_native_request_ids_and_selected_voice_row(self):
        package = self.request_package('request.cktts')
        bank = self.voice_bank('voice.ckvoice')
        ids = '0,50,83,54,156,31,4,0'
        wav = self.root / 'request.wav'
        command = [str(self.binary), str(package), str(wav), '7', ids,
                   str(bank)]
        subprocess.run(command, check=True, capture_output=True)
        with wave.open(str(wav), 'rb') as stream:
            self.assertEqual((stream.getframerate(), stream.getnframes()),
                             (24000, 20))
        expected = hashlib.sha256(wav.read_bytes()).hexdigest()
        wav.unlink()
        subprocess.run(command, check=True, capture_output=True)
        self.assertEqual(hashlib.sha256(wav.read_bytes()).hexdigest(), expected)
        wav.unlink()
        altered = self.voice_bank('other.ckvoice', selected_decoder=.125)
        subprocess.run(command[:-1] + [str(altered)], check=True,
                       capture_output=True)
        self.assertNotEqual(hashlib.sha256(wav.read_bytes()).hexdigest(),
                            expected)
        wav.unlink()
        changed_ids = '0,51,83,54,156,31,4,0'
        subprocess.run(command[:4] + [changed_ids, str(bank)], check=True,
                       capture_output=True)
        self.assertNotEqual(hashlib.sha256(wav.read_bytes()).hexdigest(),
                            expected)
        wav.unlink()
        for bad_ids in ('0,50,83,54,156,31,0',
                        '0,50,83,54,178,31,4,0',
                        '1,50,83,54,156,31,4,0',
                        '0,50,83,54,156,31,4,0,'):
            with self.subTest(ids=bad_ids):
                rejected = subprocess.run(command[:4] + [bad_ids, str(bank)],
                                          capture_output=True)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertFalse(wav.exists())
        for bad_package in (
                self.request_package('wrong-count.cktts', ids_count=9),
                self.request_package('overlap-ids.cktts', ids_offset=0),
                self.request_package('overlap-styles.cktts',
                                     predictor_offset=DECODER_OFFSET)):
            with self.subTest(package=bad_package.name):
                rejected = subprocess.run([str(self.binary), str(bad_package),
                    str(wav), '7', ids, str(bank)], capture_output=True)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertFalse(wav.exists())
        malformed = self.root / 'truncated.ckvoice'
        malformed.write_bytes(bank.read_bytes()[:-4])
        rejected = subprocess.run(command[:-1] + [str(malformed)],
                                  capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(wav.exists())
        missing_row = self.root / 'missing-row.ckvoice'
        raw = bank.read_bytes()
        missing_row.write_bytes(raw[:8] + struct.pack('<II', 5, 256) +
                                raw[16:16 + 5 * 256 * 4])
        rejected = subprocess.run(command[:-1] + [str(missing_row)],
                                  capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(wav.exists())
        nonfinite = self.root / 'nan-row.ckvoice'
        raw = bytearray(raw)
        struct.pack_into('<f', raw, 16 + 5 * 256 * 4, float('nan'))
        nonfinite.write_bytes(raw)
        rejected = subprocess.run(command[:-1] + [str(nonfinite)],
                                  capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(wav.exists())
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.native-wav-host.request-binding-v1',
            'name': 'native host exact-length phoneme and voice-row binding',
            'provider': 'native_kokoro_wav_host',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'synthetic-entry changed-input-and-voice controls',
            'status': 'pass', 'gate': 'session', 'blocking': True,
            'max_diff': 0, 'tolerance': 0,
            'same_request_wav_sha256': expected,
            'configuration': 'CKTTS003, 8 IDs, CKVOICE1 row 5; invalid '
                'count, layout, ID and bank inputs rejected before WAV output',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_native_kokoro_wav_host',
        }))


if __name__ == '__main__':
    unittest.main()
