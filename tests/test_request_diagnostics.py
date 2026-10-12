import json

import numpy as np

from ruri.common.zmq import unpack
from ruri.server.diagnostics import RequestRecorder


def test_capture_keeps_inputs_before_mutation_and_rtc_boundary(tmp_path):
    recorder = RequestRecorder(tmp_path / 'capture', {'checkpoint': 'test'})
    state = np.arange(7, dtype=np.float32)
    actions = np.tile(state, (30, 1))
    actions[4:, 0] += 8
    request = {
        'observation.state': state.copy(),
        'observation.images.top': np.zeros((12, 16, 3), dtype=np.uint8),
        'context.rtc.prev_chunk_left_over': np.tile(state, (22, 1)),
        'context.rtc.consumed_steps': 8,
    }
    response = {'action_chunk': actions.copy(), 'rtc.execute_from': 4}
    try:
        capture = recorder.begin(request)
        request['observation.state'][:] = 999
        recorder.finish(capture, response)
        response['action_chunk'][:] = 999
        recorder.pending.join()
        saved = unpack((tmp_path / 'capture/00000001.msgpack').read_bytes())
        np.testing.assert_array_equal(saved['request']['observation.state'], state)
        np.testing.assert_array_equal(saved['response']['action_chunk'], actions)
        summary = json.loads((tmp_path / 'capture/summary.jsonl').read_text())
        assert summary['prefix_max_abs_error'] == 0
        assert summary['prefix_boundary_delta'] == [8, 0, 0, 0, 0, 0, 0]
        assert summary['first_executable_minus_state'] == [8, 0, 0, 0, 0, 0, 0]
    finally:
        recorder.close()


def test_budget_keeps_summaries_and_error_responses(tmp_path):
    recorder = RequestRecorder(tmp_path / 'limited', {}, max_bytes=1)
    try:
        recorder.finish(recorder.begin({'type': 'infer'}), {'error': 'example'})
        recorder.pending.join()
        summary = json.loads((tmp_path / 'limited/summary.jsonl').read_text())
        assert summary['raw_file'] is None
        assert summary['response_info']['error'] == 'example'
        assert not list((tmp_path / 'limited').glob('000*.msgpack'))
    finally:
        recorder.close()
