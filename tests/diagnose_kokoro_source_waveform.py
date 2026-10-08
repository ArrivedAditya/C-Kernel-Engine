"""Optional pinned-model diagnostic for source/STFT impact on waveform.

This feeds alternative F0 curves or first source-convolution outputs to the
Python reference model only. It does not certify generated C waveform execution.
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
from tests.test_v8_kokoro_generated_prosody_complete import complete_prosody_weights


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'version/v8/tts/reference/fixture_manifest.json').read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(reference, actual):
    if reference.shape != actual.shape:
        raise RuntimeError(f'comparison shape mismatch: {reference.shape} != {actual.shape}')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise RuntimeError('comparison contains nonfinite values')
    difference = reference.astype(np.float64) - actual.astype(np.float64)
    error = np.abs(difference)
    index = int(np.argmax(error))
    return {'max_abs': float(error[index]), 'rmse': float(np.sqrt(np.mean(
        difference ** 2))),
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
        f0_generated = case.view(arena, 'f0_output')[:206].reshape(1, 206).copy()
        f0_reference = case.oracle['f0'].copy()
        if f0_generated.shape != f0_reference.shape:
            raise RuntimeError('generated and reference F0 shapes differ')
        weights, direct, _ = complete_prosody_weights()
        block_generated = case.view(arena, 'f0_block2_output').reshape(256, 256)[:, :206]
        block_reference = direct['predictor_F0_2']
        projection_weight = weights['duration_prosody.F0_proj.weight'].reshape(256)
        projection_bias = weights['duration_prosody.F0_proj.bias'].reshape(-1)[0]
        projected_generated = (projection_weight.astype(np.float64)
                               @ block_generated.astype(np.float64)
                               + float(projection_bias))
        projected_reference = (projection_weight.astype(np.float64)
                               @ block_reference.astype(np.float64)
                               + float(projection_bias))
        print(json.dumps({'case': 'generated_f0_error_decomposition',
            'kind': 'diagnostic_comparison',
            'third_block_vs_reference': metrics(
                block_reference.reshape(-1), block_generated.reshape(-1)),
            'upstream_projection_effect': metrics(
                projected_reference, projected_generated),
            'native_projection_vs_fp64_same_input': metrics(
                projected_generated, f0_generated.reshape(-1)),
            'pinned_projection_vs_fp64_same_input': metrics(
                projected_reference, f0_reference.reshape(-1))}))
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
        modules = dict(model.named_modules())
        module = modules['decoder.generator.noise_convs.0']
        generator = modules['decoder.generator']
        source_module = modules['decoder.generator.m_source']
        random_values = {
            'phase': torch.from_numpy(np.load(capture_root / 'generator_initial_phase_random.npy')),
            'harmonic': torch.from_numpy(np.load(capture_root / 'generator_harmonic_gaussian.npy')),
            'source': torch.from_numpy(np.load(capture_root / 'generator_source_gaussian.npy')),
        }

        def run(replacement=None, f0_override=None):
            hooks = []
            captured = {}
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
            def bind_f0(_module, inputs):
                actual = inputs[2]
                captured['f0_input'] = actual.detach().cpu().numpy().copy()
                if f0_override is None:
                    return None
                if actual.shape != f0_override.shape:
                    raise RuntimeError('F0 override shape differs from generator input')
                supplied = torch.from_numpy(f0_override).to(
                    dtype=actual.dtype, device=actual.device)
                return (*inputs[:2], supplied)

            def capture_source(_module, _inputs, output):
                captured['source'] = output[0].detach().cpu().numpy().copy()

            def capture_conv(_module, _inputs, output):
                captured['first_source_conv'] = output.detach().cpu().numpy().copy()

            hooks.append(generator.register_forward_pre_hook(bind_f0))
            hooks.append(source_module.register_forward_hook(capture_source))
            hooks.append(module.register_forward_hook(capture_conv))
            if replacement is not None:
                tensor = torch.from_numpy(replacement).unsqueeze(0)
                hooks.append(module.register_forward_hook(lambda _m, _i, output: (
                    tensor.to(dtype=output.dtype, device=output.device))))
            torch.manual_seed(MANIFEST['pin']['fixture']['torch_seed'])
            torch.rand, torch.randn_like = fixed_rand, fixed_randn_like
            try:
                output = model(phonemes, style, speed=1.0, return_output=True)
                captured['audio'] = output.audio.detach().cpu().numpy().copy()
                return captured
            finally:
                torch.rand, torch.randn_like = original_rand, original_randn_like
                for hook in hooks:
                    hook.remove()

        baseline = run()
        fixture_wav = np.load(capture_root / capture_manifest['tensors']['waveform_f32']['file'])
        np.testing.assert_array_equal(fixture_wav.reshape(-1), baseline['audio'].reshape(-1))
        np.testing.assert_array_equal(f0_reference, baseline['f0_input'])
        np.testing.assert_array_equal(source_reference, baseline['source'].reshape(-1))
        np.testing.assert_array_equal(baseline_conv, baseline['first_source_conv'][0])
        print(json.dumps({'baseline_vs_pinned_waveform': metrics(
            fixture_wav.reshape(-1), baseline['audio'].reshape(-1)),
            'sample_count': int(baseline['audio'].size)}))
        f0_substitution = run(f0_override=f0_generated)
        print(json.dumps({'case': 'generated_f0_into_pinned_full_model',
            'kind': 'diagnostic_substitution',
            'generated_f0_vs_reference': metrics(
                f0_reference.reshape(-1), f0_generated.reshape(-1)),
            'source_vs_reference': metrics(
                baseline['source'].reshape(-1),
                f0_substitution['source'].reshape(-1)),
            'first_source_conv_vs_reference': metrics(
                baseline['first_source_conv'].reshape(-1),
                f0_substitution['first_source_conv'].reshape(-1)),
            'waveform_vs_reference': metrics(
                baseline['audio'].reshape(-1),
                f0_substitution['audio'].reshape(-1))}))
        print(json.dumps({'case': 'cke_source_arithmetic_with_identical_generated_f0',
            'kind': 'diagnostic_comparison',
            'source_vs_pinned_model': metrics(
                f0_substitution['source'].reshape(-1),
                source_generated.reshape(-1))}))
        source_with_generated_f0 = f0_substitution['source'].reshape(-1)
        reference_channels = torch_channels(source_reference)
        changed_channels = torch_channels(source_with_generated_f0)
        raw_phase_delta = np.abs(reference_channels[11:] - changed_channels[11:])
        branch_flips = np.argwhere(raw_phase_delta > np.pi)
        generated_channels = torch_channels(source_generated)
        arithmetic_phase_delta = np.abs(
            changed_channels[11:] - generated_channels[11:])
        arithmetic_branch_flips = np.argwhere(arithmetic_phase_delta > np.pi)
        print(json.dumps({'case': 'generated_f0_phase_sensitivity',
            'kind': 'diagnostic_substitution',
            'raw_phase_branch_cut_mismatch_count': int(len(branch_flips)),
            'first_raw_phase_branch_cut_mismatches': branch_flips[:8].tolist(),
            'max_raw_phase_delta': float(raw_phase_delta.max()),
            'near_identical_source_branch_cut_mismatch_count': int(
                len(arithmetic_branch_flips)),
            'first_near_identical_source_branch_cut_mismatches':
                arithmetic_branch_flips[:8].tolist(),
            'source_conv_native_vs_pinned_model_with_same_f0': metrics(
                f0_substitution['first_source_conv'].reshape(-1),
                alternatives['generated_source_torch_stft'].reshape(-1))}))
        candidate_outputs = {}
        for label, replacement in alternatives.items():
            candidate = run(replacement)
            candidate_outputs[label] = candidate
            print(json.dumps({'case': label,
                'source_conv_vs_reference': metrics(
                    baseline_conv.reshape(-1), replacement.reshape(-1)),
                'waveform_vs_baseline': metrics(
                    baseline['audio'].reshape(-1), candidate['audio'].reshape(-1))}))
        print(json.dumps({'case': 'generated_source_stft_incremental',
            'waveform_cke_stft_vs_torch_stft': metrics(
                candidate_outputs['generated_source_torch_stft']['audio'].reshape(-1),
                candidate_outputs['generated_source_cke_stft']['audio'].reshape(-1))}))
    finally:
        Fixture.tearDownClass()


if __name__ == '__main__':
    main()
