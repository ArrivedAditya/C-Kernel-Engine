"""Optional pinned-model diagnostic for source/STFT impact on waveform.

This feeds alternative first source-convolution outputs to the Python reference
model only. It does not certify generated C waveform execution.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from kokoro import KModel

from tests.test_v8_kokoro_generated_source_stft_conv import (
    KokoroGeneratedSourceStftConvTest as Fixture,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'version/v8/tts/reference/fixture_manifest.json').read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(reference, actual):
    error = np.abs(reference - actual)
    index = int(np.argmax(error))
    return {'max_abs': float(error[index]), 'rmse': float(np.sqrt(np.mean(
        (reference.astype(np.float64) - actual.astype(np.float64)) ** 2))),
        'worst_index': index, 'reference': float(reference[index]),
        'actual': float(actual[index])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--capture-dir', required=True, type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    capture_root = args.capture_dir.resolve()
    capture_manifest = json.loads((capture_root / 'manifest.json').read_text())
    if (capture_manifest['pin']['model'] != MANIFEST['pin']['model'] or
        capture_manifest['pin']['reference_code'] != MANIFEST['pin']['reference_code'] or
        capture_manifest['assets'] != MANIFEST['assets'] or
        capture_manifest['phonemes'] != MANIFEST['phonemes']):
        raise RuntimeError('capture identity differs from pinned fixture')
    for package in ('torch', 'kokoro'):
        expected = MANIFEST['environment']['packages'][package]
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f'{package} {actual} != pinned {expected}')
    for name, expected in MANIFEST['assets'].items():
        path = model_dir / name
        if digest(path) != expected:
            raise RuntimeError(f'asset hash mismatch: {path}')
    for name in ('waveform_f32', 'generator_initial_phase_random',
                 'generator_harmonic_gaussian', 'generator_source_gaussian'):
        path = capture_root / capture_manifest['tensors'][name]['file']
        if digest(path) != capture_manifest['tensors'][name]['sha256']:
            raise RuntimeError(f'capture hash mismatch: {path}')
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    Fixture.setUpClass()
    try:
        case = Fixture()
        arena = case.arena()
        if case.execute(arena) != (0, (103, 206, 61800, 12361, 2060)):
            raise RuntimeError('generated prefix failed')
        source_generated = case.view(arena, 'source_waveform')[:61800].copy()
        source_reference = case.oracle['source'][0].copy()
        window = torch.from_numpy(case.oracle['window'].copy())

        def torch_channels(samples):
            spectrum = torch.stft(torch.from_numpy(samples.copy()), 20, 5, 20,
                                  window=window, return_complex=True).numpy()
            return np.ascontiguousarray(np.concatenate((
                np.abs(spectrum), np.angle(spectrum)), axis=0))

        alternatives = {
            'pinned_source_cke_stft': case.native_source_convolution(
                case.native_stft(source_reference)),
            'generated_source_torch_stft': case.native_source_convolution(
                torch_channels(source_generated)),
            'generated_source_cke_stft': case.native_source_convolution(
                case.native_stft(source_generated)),
        }
        baseline_conv = case.oracle['first_source_conv'][0]
        for label, value in alternatives.items():
            if value.shape != baseline_conv.shape or not np.isfinite(value).all():
                raise RuntimeError(f'invalid candidate: {label}')

        config = model_dir / 'config.json'
        model = KModel(repo_id=MANIFEST['pin']['model']['repository'],
                       config=str(config),
                       model=str(model_dir / 'kokoro-v1_0.pth')).eval()
        voice_pack = torch.load(model_dir / 'voices/af_heart.pt', map_location='cpu',
                                weights_only=True)
        style = voice_pack[MANIFEST['voice_row_index']]
        if style.ndim == 1:
            style = style.unsqueeze(0)
        phonemes = MANIFEST['phonemes']
        module = dict(model.named_modules())['decoder.generator.noise_convs.0']
        random_values = {
            'phase': torch.from_numpy(np.load(capture_root / 'generator_initial_phase_random.npy')),
            'harmonic': torch.from_numpy(np.load(capture_root / 'generator_harmonic_gaussian.npy')),
            'source': torch.from_numpy(np.load(capture_root / 'generator_source_gaussian.npy')),
        }

        def run(replacement=None):
            hook = None
            original_rand, original_randn_like = torch.rand, torch.randn_like
            def fixed_rand(*shape, **kwargs):
                if tuple(shape) == (1, 9):
                    return random_values['phase'].clone()
                return original_rand(*shape, **kwargs)
            def fixed_randn_like(value, *args, **kwargs):
                if tuple(value.shape) == (1, 61800, 9):
                    return random_values['harmonic'].clone()
                if tuple(value.shape) == (1, 61800, 1):
                    return random_values['source'].clone()
                return original_randn_like(value, *args, **kwargs)
            if replacement is not None:
                tensor = torch.from_numpy(replacement).unsqueeze(0)
                hook = module.register_forward_hook(lambda _m, _i, output: (
                    tensor.to(dtype=output.dtype, device=output.device)))
            torch.manual_seed(MANIFEST['pin']['fixture']['torch_seed'])
            torch.rand, torch.randn_like = fixed_rand, fixed_randn_like
            try:
                output = model(phonemes, style, speed=1.0, return_output=True)
                return output.audio.detach().cpu().numpy().copy()
            finally:
                torch.rand, torch.randn_like = original_rand, original_randn_like
                if hook is not None:
                    hook.remove()

        baseline = run()
        fixture_wav = np.load(capture_root / capture_manifest['tensors']['waveform_f32']['file'])
        np.testing.assert_array_equal(fixture_wav.reshape(-1), baseline.reshape(-1))
        print(json.dumps({'baseline_vs_pinned_waveform': metrics(
            fixture_wav.reshape(-1), baseline.reshape(-1)),
            'sample_count': int(baseline.size)}))
        candidate_outputs = {}
        for label, replacement in alternatives.items():
            candidate = run(replacement)
            candidate_outputs[label] = candidate
            print(json.dumps({'case': label,
                'source_conv_vs_reference': metrics(
                    baseline_conv.reshape(-1), replacement.reshape(-1)),
                'waveform_vs_baseline': metrics(
                    baseline.reshape(-1), candidate.reshape(-1))}))
        print(json.dumps({'case': 'generated_source_stft_incremental',
            'waveform_cke_stft_vs_torch_stft': metrics(
                candidate_outputs['generated_source_torch_stft'].reshape(-1),
                candidate_outputs['generated_source_cke_stft'].reshape(-1))}))
    finally:
        Fixture.tearDownClass()


if __name__ == '__main__':
    main()
