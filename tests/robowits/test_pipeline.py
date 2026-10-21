import numpy as np
import pandas as pd
import pytest
from robowits.data import Episodes, VideoWriter, decode_frames, sample_indices
from robowits.geometry import draw_skeleton, look_at, pack
from robowits.windows import boundaries
from robowits.compare import compare


def test_resampling_does_not_stretch_endpoint():
    indices = sample_indices(30, 30, 20)
    assert len(indices) == 20
    assert indices[0] == 0 and indices[-1] == 28
    assert np.max(np.abs(indices/30 - np.arange(20)/20)) <= 1/60 + 1e-10
    with pytest.raises(ValueError):
        sample_indices(10, 20, 30)


def test_shared_video_offset_and_eof(tmp_path):
    path = tmp_path/'video.mp4'
    writer = VideoWriter(path, fps=20, crf=0)
    for value in range(10):
        writer.write(np.full((448, 448, 3), value*20, dtype=np.uint8))
    writer.close()
    frames = list(decode_frames(path, np.array([5, 7, 9])/20, 20))
    assert np.allclose([f.mean() for f in frames], [100, 140, 180], atol=2)
    with pytest.raises(ValueError, match='ended'):
        list(decode_frames(path, [1.], 20))


def test_split_precedes_subset_selection():
    source = object.__new__(Episodes)
    source.tasks = {'a': 0, 'b': 1}
    source.meta = pd.DataFrame([{'episode_index': i, 'tasks': ['a' if i < 8 else 'b']} for i in range(16)])
    rows = source.records(3, 2)
    assert {r['episode_index'] for r in rows if r['split']=='holdout'} == {0, 1, 2, 8, 9, 10}
    assert rows[5]['split'] == 'training'
    assert rows[3]['split'] == 'validation'


def test_projection_clips_out_of_frame_and_rejects_behind_camera():
    pose = look_at([0, 0, 0], [0, 0, -1], [0, 1, 0])
    colors = np.array([[255, 0, 0]])
    clipped = draw_skeleton([([-10, 0, -1], [10, 0, -1])], colors, pose, 65)
    behind = draw_skeleton([([-1, 0, 1], [1, 0, 1])], colors, pose, 65)
    assert np.count_nonzero(clipped) >= 224
    assert not behind.any()


def test_panel_contract_preserves_both_wrists():
    frames = {c: np.full((4, 4, 3), i, dtype=np.uint8) for i, c in enumerate(('ego','wrist_right','wrist_left'))}
    panel = pack(frames)
    assert panel.shape == (448, 448, 3)
    assert panel[0, 0, 0] == panel[0, 300, 0] == 0
    assert panel[300, 0, 0] == 1 and panel[300, 300, 0] == 2


def test_probe_windows_never_score_padding():
    points = boundaries(200, 60, 80, 40, 3)
    assert points.tolist() == [80, 120, 160]
    with pytest.raises(ValueError):
        boundaries(100, 60, 80, 40, 3)


def test_compare_rejects_mismatched_gt_or_random_seed():
    a = [{'window':'w', 'window_sha256':'abc', 'seed':0, 'episode_index':0}]
    for field, replacement in [('seed', 1), ('window_sha256', 'def'), ('episode_index', 2)]:
        with pytest.raises(ValueError, match='mismatch'):
            compare(a, [{**a[0], field:replacement}])
    with pytest.raises(ValueError):
        compare(a, [])


def test_region_scores_count_unique_views_and_ignore_codec_noise():
    from robowits.probe import score
    rgb = np.zeros((1, 448, 448, 3), dtype=np.uint8)
    cond = np.full_like(rgb, 3)  # near-black codec noise is not a robot mask
    scores = score(rgb, rgb, cond, rgb[0])
    assert set(scores) == {'ego', 'wrist_right', 'wrist_left'}
    assert scores['ego']['mse'] == 0
    assert scores['ego']['robot_region_mse'] is None
    assert scores['ego']['changed_region_mse'] is None


def test_hydra_paths_with_equals_are_quoted():
    parser = pytest.importorskip('hydra.core.override_parser.overrides_parser').OverridesParser.create()
    import json
    path = '/tmp/run with spaces/checkpoints/step=2000.ckpt'
    override = parser.parse_override('load=' + json.dumps(path))
    assert override.value() == path


def test_permanent_checkpoint_uses_owning_run_config(tmp_path):
    from robowits.probe import checkpoint_config
    folder = tmp_path/'run/.hydra'
    folder.mkdir(parents=True)
    (folder/'config.yaml').write_text('{}')
    (folder/'hydra.yaml').write_text('{}')
    ckpt = tmp_path/'run/checkpoints/keep/step=2000.ckpt'
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    assert checkpoint_config(ckpt) == folder/'config.yaml'
    with pytest.raises(FileNotFoundError):
        checkpoint_config(tmp_path/'missing.ckpt')
