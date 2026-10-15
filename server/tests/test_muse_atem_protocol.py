"""Recipient-scoped Muse output, using publisher ATEM examples as fixtures."""

from __future__ import annotations

import pytest

from server.muse_atem_protocol import MuseSyntaxError, parse_muse_turn


def _tool(name: str, key: str, value: str, ending: str = "<|eot|>") -> str:
    return (
        f'<|start|>assistant to={name}<|message|><atem:function_calls>\n'
        f'<atem:invoke name="{name}">\n'
        f'<atem:parameter name="{key}">{value}</atem:parameter>\n'
        f'</atem:invoke>\n</atem:function_calls>{ending}'
    )


def test_reasoning_and_two_tool_channels_continue_across_eom() -> None:
    raw = (
        ' to=self<|message|>Need both files.<|eom|>'
        + _tool('read_file', 'path', '/tmp/a', '<|eom|>')
        + _tool('read_file', 'path', '/tmp/b')
    )
    turn = parse_muse_turn(raw)
    assert turn.reasoning == 'Need both files.'
    assert turn.final_text == ''
    assert [call['arguments']['path'] for call in turn.tool_calls] == ['/tmp/a', '/tmp/b']


def test_final_channel_does_not_execute_quoted_atem() -> None:
    raw = (' to=self<|message|>Think.<|eom|>'
           '<|start|>assistant to=user<|message|>'
           'Example: <atem:invoke name="read_file"></atem:invoke><|eot|>')
    turn = parse_muse_turn(raw)
    assert turn.reasoning == 'Think.'
    assert 'Example:' in turn.final_text
    assert turn.tool_calls == []


def test_tool_arguments_preserve_raw_strings_and_decode_json() -> None:
    raw = _tool('edit', 'content', '  <svg>"A" & B</svg>  ', '<|eom|>') + _tool(
        'edit', 'options', '{"count":3,"list":[true,null]}')
    turn = parse_muse_turn(raw)
    assert turn.tool_calls[0]['arguments']['content'] == '  <svg>"A" & B</svg>  '
    assert turn.tool_calls[1]['arguments']['options'] == {'count': 3, 'list': [True, None]}


@pytest.mark.parametrize('raw', [
    ' to=self<|message|>reasoning only<|eom|>',
    ' to=self<|message|>reasoning only<|eot|>',
    _tool('read_file', 'path', '/tmp/a').replace('</atem:function_calls>', ''),
    _tool('read_file', 'path', '/tmp/a').replace('to=read_file', 'to=other'),
    _tool('read_file', 'n', '1e999'),
    _tool('read_file', 'n', '{"value":1e999}'),
    _tool('read_file', 'n', '9' * 1100),
    _tool('read_file', 'n', '{"value":' + '9' * 1100 + '}'),
    _tool('read_file', 'n', '{"a":1,"a":2}'),
    _tool('read_file', 'path', '/tmp/a') + '<|start|>assistant to=user<|message|>late',
    ' to=user<|message|>done<|eom|>',
    '<|start|>assistant to=read_file<|message|><atem:function_calls></atem:function_calls><|eot|>',
])
def test_malformed_or_unsafe_turn_is_rejected(raw: str) -> None:
    with pytest.raises(MuseSyntaxError):
        parse_muse_turn(raw)


def test_plain_text_and_stop_stripped_final() -> None:
    assert parse_muse_turn('Hello.').final_text == 'Hello.'
    assert parse_muse_turn(' to=user<|message|>Hello.').final_text == 'Hello.'
