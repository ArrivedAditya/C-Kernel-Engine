"""Assemble the pinned effective weights used by connected decoder tests."""

from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture


def decoder_weights_and_references():
    fixtures = {}
    for key in ('prosody_first_block', 'prosody_norm', 'prosody_shared',
                'text_encoder', 'text_embedding',
                'prosody_second_block_weights'):
        fixtures[key], _ = verified_fixture(key)
    direct, direct_meta = verified_fixture('decoder_ingress')
    encode, encode_meta = verified_fixture('decoder_encode')
    decode = {}
    decode_meta = {}
    for index in range(4):
        decode[index], decode_meta[index] = verified_fixture(
            f'decoder_decode{index}')
    weights = {'acoustic_text_encoder.embedding.weight':
               fixtures['text_embedding']['table'].copy()}
    text = fixtures['text_encoder']
    for index in range(3):
        for kind in ('weight', 'bias'):
            weights[f'acoustic_text_encoder.conv{index}.{kind}'] = \
                text[f'conv{index}_{kind}']
        weights[f'acoustic_text_encoder.norm{index}.weight'] = \
            text[f'norm{index}_gamma']
        weights[f'acoustic_text_encoder.norm{index}.bias'] = \
            text[f'norm{index}_beta']
    for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
        weights[f'acoustic_text_encoder.lstm.{kind}'] = text[f'lstm_{kind}']
        weights[f'duration_prosody.shared_scan.{kind}'] = \
            fixtures['prosody_shared'][kind]
    for branch in ('F0', 'N'):
        prefix = f'duration_prosody.{branch}.0'
        for norm_name in ('norm1', 'norm2'):
            for kind in ('fc_weight', 'fc_bias', 'norm_weight', 'norm_bias'):
                canonical = f'{prefix}.{norm_name}.{kind.replace("_", ".")}'
                source = ('prosody_norm' if norm_name == 'norm1' else
                          'prosody_first_block')
                source_name = (f'{branch}_{kind}' if norm_name == 'norm1' else
                               f'{branch}_norm2_{kind}')
                weights[canonical] = fixtures[source][source_name]
        for index in (1, 2):
            for kind in ('weight', 'bias'):
                weights[f'{prefix}.conv{index}.{kind}'] = \
                    fixtures['prosody_first_block'][f'{branch}_conv{index}_{kind}']
    weights.update(fixtures['prosody_second_block_weights'])
    for collection in (direct, encode, *decode.values()):
        for name, value in collection.items():
            if name.startswith('waveform_decoder.'):
                weights[name] = value
    return weights, direct, direct_meta, encode, encode_meta, decode, decode_meta
