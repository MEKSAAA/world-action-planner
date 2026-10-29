import hashlib
import json
import sys

import numpy as np
import pytest

from robowits.multigoal_episode_report import review_indices, terminal_outcome
from robowits import multigoal_rescore


def test_partial_and_error_are_not_native_failures():
    for status in ('running', 'error', 'diagnostic_complete'):
        assert terminal_outcome(dict(status=status, success=False, steps=1000), 1000) is None
    assert terminal_outcome(dict(status='complete', success=False, steps=1000), 1000) is False
    assert terminal_outcome(dict(status='complete', success=True, steps=514, terminated=True), 1000) is True
    with pytest.raises(ValueError, match='endpoint'):
        terminal_outcome(dict(status='complete', success=False, steps=50), 1000)
    assert review_indices(100) == [0, 25, 50, 75, 99]
    assert review_indices(51) == [0, 25, 50]
    assert review_indices(0) == []


@pytest.fixture
def recorded_bank(tmp_path, monkeypatch):
    run = tmp_path/'run'
    videos = run/'decision-0000'
    videos.mkdir(parents=True)
    current = np.zeros((2, 2, 3), dtype=np.uint8)
    for i in range(3):
        folder = videos/f'candidate-{i}'
        folder.mkdir()
        np.save(folder/'prediction.npy', np.zeros((20, 2, 2, 3), dtype=np.uint8))
        np.savez(folder/'input.npz', history=current[None])
    def scores(values):
        return [dict(id=i, valid=True, score=v, reason='evidence') for i, v in enumerate(values)]
    decision = dict(selected=1, no_wm={'qwen': scores([.3, .2, .1])},
        wm={'qwen': scores([.1, .7, .2])},
        candidate_map=[dict(goal={'language': f'goal {i}'}) for i in range(3)],
        observation_sha256=hashlib.sha256(current.tobytes()).hexdigest())
    protocol = dict(task_id='robowits/03-gap-retrieve-v0', primary_judge='qwen',
                    judges=['qwen'], value_candidate_text=False)
    (run/'protocol.json').write_text(json.dumps(protocol))
    (run/'decisions.json').write_text(json.dumps([decision]))
    config = tmp_path/'judges.json'
    config.write_text(json.dumps(dict(judges=['qwen', 'gemini'])))
    out = tmp_path/'out'
    monkeypatch.setattr(sys, 'argv', ['rescore', '--run', str(run), '--videos', str(videos),
        '--out', str(out), '--judge-config', str(config), '--kind', 'wm', '--reuse-recorded-primary'])
    calls = []
    monkeypatch.setattr(multigoal_rescore, 'backend', lambda config, name: name)
    class Judge:
        def __init__(self, backend, *, candidate_text):
            assert backend == 'gemini', 'Online primary must not be queried again'
            assert candidate_text is False
        def without_wm(self, current, task, languages, out):
            calls.append('control')
            return scores([.3, .2, .1])
        def judge(self, current, predictions, task, languages, out, *, valid_future_frames):
            assert valid_future_frames == 6 and all(len(p) == 6 for p in predictions)
            calls.append('wm')
            return scores([.1, .2, .8])
    monkeypatch.setattr(multigoal_rescore, 'TaskValue', Judge)
    return run, videos, out, calls


def test_cross_score_keeps_actual_online_primary(recorded_bank):
    run, videos, out, calls = recorded_bank
    multigoal_rescore.main()
    results = json.loads((out/'scores.json').read_text())
    assert results['qwen']['selected'] == 1
    assert results['qwen']['score_source'] == 'recorded_online'
    assert results['gemini']['selected'] == 2
    assert results['gemini']['score_source'] == 'offline_cross_score'
    assert calls == ['control', 'wm']
    protocol = json.loads((out/'protocol.json').read_text())
    assert len(protocol['candidate_bank_sha256']) == 3


def test_cross_score_rejects_different_observation(recorded_bank):
    run, videos, out, calls = recorded_bank
    for i in range(3):
        np.savez(videos/f'candidate-{i}'/'input.npz', history=np.ones((1, 2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='observation'):
        multigoal_rescore.main()
    assert calls == []
