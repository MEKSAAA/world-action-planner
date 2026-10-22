import numpy as np

from robowits.closed_loop_score import choose, visual_score


def video(x=100):
    frames = np.zeros((20, 448, 448, 3), np.uint8)
    frames[:, 90:110, 170:190] = (0, 200, 0)
    frames[:, 96:104, x:x+8] = (240, 170, 0)
    return frames


def test_missing_object_cannot_win_by_disappearing():
    good = visual_score(video())
    missing = video()
    missing[:, 96:104, 100:108] = 0
    bad = visual_score(missing)
    assert good['valid'] and not bad['valid']
    assert choose([good, bad]) == 0
    assert choose([bad, good]) == 0  # Conservative: no comparable baseline evidence.


def test_duplicate_object_is_invalid():
    frames = video()
    frames[:, 130:140, 130:140] = (240, 170, 0)
    assert not visual_score(frames)['valid']


def test_fixed_margin_and_ties_preserve_baseline():
    far, close = visual_score(video(100)), visual_score(video(150))
    assert choose([far, close]) == 1
    assert choose([close, far]) == 0
    assert choose([far, far]) == 0
    assert choose([far, visual_score(video(101))]) == 0


def test_only_predicted_ego_is_scored():
    frames = video()
    score = visual_score(frames)
    frames[:, 224:] = np.random.default_rng(0).integers(0, 256, frames[:, 224:].shape, dtype=np.uint8)
    assert visual_score(frames) == score


def test_initial_orientation_and_velocity_must_match():
    from copy import deepcopy
    from robowits.closed_loop import same_state
    a = {'robot': [0]*16, 'qpos': [0]*18, 'velocities': [0]*18,
         'orientations': {'lemon': [1, 0, 0, 0]}, 'objects': {'lemon': [0, 0, .8]}}
    b = deepcopy(a)
    assert same_state(a, b)
    b['orientations']['lemon'] = [0, 0, 0, 1]
    assert not same_state(a, b)
    b = deepcopy(a)
    b['velocities'][0] = .01
    assert not same_state(a, b)


def test_resume_rejects_changed_weights_or_scenes():
    import pytest
    from robowits.closed_loop import validate_resume_protocol
    original = {'cases': [10, 11], 'policy_weights_sha256': 'original', 'max_steps': 1000}
    validate_resume_protocol(original, dict(original, wap_commit='resume-only-fix'))
    for key, value in [('cases', [11, 12]), ('policy_weights_sha256', 'changed'), ('max_steps', 120)]:
        with pytest.raises(ValueError, match='protocol'):
            validate_resume_protocol(original, dict(original, **{key: value}))


def test_resume_preserves_completed_and_archives_interrupted(tmp_path):
    import json
    from robowits.closed_loop import completed_or_archive
    folder = tmp_path/'wm-10'
    folder.mkdir()
    record = {'status': 'complete', 'success': False}
    (folder/'result.json').write_text(json.dumps(record))
    assert completed_or_archive(tmp_path, 'wm', 10) == record
    (folder/'result.json').write_text('{"status":"running"}')
    (folder/'evidence.txt').write_text('preserve me')
    assert completed_or_archive(tmp_path, 'wm', 10) is None
    assert not folder.exists()
    archived = next((tmp_path/'interrupted').iterdir())
    assert (archived/'evidence.txt').read_text() == 'preserve me'
