import hashlib
import json
import sys

import numpy as np
import pytest

from robowits.multigoal_episode_report import inference_usage, review_indices, terminal_outcome
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


def test_usage_counts_retries_once_and_excludes_reused_proposals(tmp_path):
    def save(relative, value):
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    base = 'decision-0000'
    save(f'{base}/candidate-bank-reuse.json', {'source': 'another-run'})
    save(f'{base}/proposals/attempt-0/request.json', {'backend': 'gemini'})
    save(f'{base}/proposals/attempt-0/provider-response.json',
         {'usageMetadata': {'totalTokenCount': 10000}})
    attempt = f'{base}/wm-gemini/candidate-0/vlm/attempt-0'
    save(f'{attempt}/request.json', {'backend': 'gemini'})
    # Provider returned a billed response, but validation failed before a
    # normalized response was saved. It must still count.
    save(f'{attempt}/provider-response.json',
         {'usageMetadata': {'totalTokenCount': 100, 'thoughtsTokenCount': 70}})
    retry = f'{base}/wm-gemini/candidate-0/vlm/attempt-1'
    save(f'{retry}/request.json', {'backend': 'gemini'})
    save(f'{retry}/provider-response.json', {'usageMetadata': {'totalTokenCount': 50}})
    save(f'{retry}/response.json', {'usage': {'totalTokenCount': 50}, 'seconds': 3})
    save('decision-0001/proposals/attempt-0/request.json', {'backend': 'gemini'})
    local = 'decision-0001/wm-qwen/candidate-0/vlm/attempt-0'
    save(f'{local}/request.json', {'prompt': 'score'})
    save(f'{local}/response.json', {'output_tokens': 12, 'seconds': 2})
    summary = inference_usage(tmp_path)
    gemini = summary['providers']['gemini']
    assert gemini['requests'] == 3
    assert gemini['responses_with_usage'] == 2
    assert gemini['requests_without_usage'] == 1
    assert gemini['token_counts'] == {'totalTokenCount': 150, 'thoughtsTokenCount': 70}
    assert gemini['recorded_response_seconds'] == 3
    assert summary['providers']['qwen']['token_counts'] == {'localOutputTokens': 12}
    assert summary['copied_proposal_requests_excluded'] == 1


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
