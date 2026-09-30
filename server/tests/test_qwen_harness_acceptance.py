"""The local Qwen task must agree across tool events, artifact and recording."""

from pathlib import Path

from server.qwen_harness_acceptance import EXPECTED_HTML, evaluate


SESSION = '7e218e94-5a31-4ccb-a284-130c935bdd79'
MODEL = 'qwen-test'
SUMMARY = 'Changed the heading and ran python3 test_html.py; PASS: expected heading present.'


def _evidence():
    events = [
        {'type': 'system', 'subtype': 'init', 'session_id': SESSION, 'model': MODEL},
    ]
    recording = [{'type': 'user', 'sessionId': SESSION}]
    for number, name in enumerate(('read_file', 'edit', 'run_shell_command'), 1):
        call_id = f'call_{number}'
        events.append({'type': 'assistant', 'session_id': SESSION, 'message': {
            'content': [{'type': 'tool_use', 'id': call_id, 'name': name}],
        }})
        events.append({'type': 'user', 'session_id': SESSION, 'message': {
            'content': [{'type': 'tool_result', 'tool_use_id': call_id,
                         'is_error': False,
                         'content': 'PASS: expected heading present' if number == 3 else 'ok'}],
        }})
        recording.append({'type': 'assistant', 'provenance': 'assistant_output',
                          'sessionId': SESSION, 'message': {'parts': [
                              {'functionCall': {'id': call_id, 'name': name}}]}})
        recording.append({'type': 'tool_result', 'sessionId': SESSION,
                          'toolCallResult': {'callId': call_id, 'status': 'success'}})
    recording.append({'type': 'assistant', 'provenance': 'assistant_output',
                      'sessionId': SESSION, 'message': {'parts': [{'text': SUMMARY}]}})
    events.append({'type': 'result', 'session_id': SESSION, 'subtype': 'success',
                   'is_error': False, 'result': SUMMARY})
    return events, recording


def _check(events, recording, *, html=EXPECTED_HTML, exit_code=0):
    return evaluate(events, recording, session_id=SESSION, html=html,
                    exit_code=exit_code, expected_model=MODEL)


def test_complete_tool_task_passes():
    events, recording = _evidence()
    result = _check(events, recording)
    assert result['status'] == 'pass'
    assert result['tool_sequence'] == ['read_file', 'edit', 'run_shell_command']


def test_missing_stream_final_fails_even_if_recording_and_artifact_pass():
    events, recording = _evidence()
    result = _check(events[:-1], recording)
    assert result['status'] == 'fail'
    assert 'stream_final_missing_or_failed' in result['errors']
    assert result['recorded_summary_present']


def test_wrong_call_id_fails_before_final_summary_can_certify():
    events, recording = _evidence()
    events[2]['message']['content'][0]['tool_use_id'] = 'call_from_other_response'
    result = _check(events, recording)
    assert result['status'] == 'fail'
    assert 'missing_or_unmatched_tool_result' in result['errors']


def test_failed_test_or_unrelated_edit_fails():
    events, recording = _evidence()
    events[-2]['message']['content'][0]['content'] = 'FAIL: missing heading'
    result = _check(events, recording, html=EXPECTED_HTML + '<!-- unrelated edit -->')
    assert result['status'] == 'fail'
    assert 'validator_result_missing' in result['errors']
    assert 'artifact_mismatch' in result['errors']


def test_recording_mismatch_fails_even_when_stream_claims_success():
    events, recording = _evidence()
    recording[-1]['message']['parts'][0]['text'] = 'A different result.'
    result = _check(events, recording)
    assert result['status'] == 'fail'
    assert 'stream_recording_summary_mismatch' in result['errors']


def test_optional_read_of_validator_is_allowed_but_other_paths_are_rejected():
    events, recording = _evidence()
    events[1]['message']['content'][0]['input'] = {'file_path': 'index.html'}
    events[3]['message']['content'][0]['input'] = {'file_path': 'index.html'}
    events[5]['message']['content'][0]['input'] = {'command': 'python3 test_html.py'}
    events.insert(3, {'type': 'assistant', 'session_id': SESSION, 'message': {
        'content': [{'type': 'tool_use', 'id': 'call_extra', 'name': 'read_file',
                     'input': {'file_path': 'test_html.py'}}],
    }})
    events.insert(4, {'type': 'user', 'session_id': SESSION, 'message': {
        'content': [{'type': 'tool_result', 'tool_use_id': 'call_extra',
                     'is_error': False, 'content': 'validator source'}],
    }})
    recording.insert(3, {'type': 'assistant', 'provenance': 'assistant_output',
                         'sessionId': SESSION, 'message': {'parts': [
                             {'functionCall': {'id': 'call_extra', 'name': 'read_file'}}]}})
    recording.insert(4, {'type': 'tool_result', 'sessionId': SESSION,
                         'toolCallResult': {'callId': 'call_extra', 'status': 'success'}})
    assert _check(events, recording)['status'] == 'pass'
    assert evaluate(events, recording, session_id=SESSION, html=EXPECTED_HTML,
                    exit_code=0, expected_model=MODEL,
                    fixture_dir=Path('/tmp/fixture'))['status'] == 'pass'
    events[3]['message']['content'][0]['input']['file_path'] = '../private.txt'
    result = evaluate(events, recording, session_id=SESSION, html=EXPECTED_HTML,
                      exit_code=0, expected_model=MODEL,
                      fixture_dir=Path('/tmp/fixture'))
    assert 'unexpected_tool_path' in result['errors']


def test_shell_command_must_be_exact_validator():
    events, recording = _evidence()
    events[1]['message']['content'][0]['input'] = {'file_path': 'index.html'}
    events[3]['message']['content'][0]['input'] = {'file_path': 'index.html'}
    events[5]['message']['content'][0]['input'] = {
        'command': 'python3 test_html.py; echo misleading PASS'}
    result = evaluate(events, recording, session_id=SESSION, html=EXPECTED_HTML,
                      exit_code=0, expected_model=MODEL,
                      fixture_dir=Path('/tmp/fixture'))
    assert 'unexpected_validation_command' in result['errors']


def test_stale_session_or_missing_recorded_tool_result_fails():
    events, recording = _evidence()
    events[1]['session_id'] = 'older-session'
    recording[2]['toolCallResult']['callId'] = 'another-call'
    result = _check(events, recording)
    assert result['status'] == 'fail'
    assert 'mixed_stream_sessions' in result['errors']
    assert 'stream_recording_tool_results_mismatch' in result['errors']


def test_nonzero_harness_exit_cannot_pass_with_good_artifact():
    events, recording = _evidence()
    result = _check(events, recording, exit_code=124)
    assert result['status'] == 'fail'
    assert 'qwen_nonzero_exit' in result['errors']


def test_tool_result_must_follow_its_own_call_before_next_step():
    events, recording = _evidence()
    events[2], events[4] = events[4], events[2]
    result = _check(events, recording)
    assert result['status'] == 'fail'
    assert 'tool_result_order_mismatch' in result['errors']


def test_parallel_reads_finish_before_edit():
    events, recording = _evidence()
    events[1]['message']['content'].append(
        {'type': 'tool_use', 'id': 'call_extra', 'name': 'read_file'})
    events.insert(3, {'type': 'user', 'session_id': SESSION, 'message': {
        'content': [{'type': 'tool_result', 'tool_use_id': 'call_extra',
                     'is_error': False, 'content': 'validator source'}]}})
    recording[1]['message']['parts'].append(
        {'functionCall': {'id': 'call_extra', 'name': 'read_file'}})
    recording.insert(3, {'type': 'tool_result', 'sessionId': SESSION,
                         'toolCallResult': {'callId': 'call_extra', 'status': 'success'}})
    assert _check(events, recording)['status'] == 'pass'

    early_edit = list(events)
    early_edit[3], early_edit[4] = early_edit[4], early_edit[3]
    result = _check(early_edit, recording)
    assert 'tool_result_order_mismatch' in result['errors']
