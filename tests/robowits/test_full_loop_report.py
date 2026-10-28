import json

import imageio.v2 as imageio
import numpy as np
import pytest

from robowits.full_loop_report import export


def save(path, data):
    path.write_text(json.dumps(data))


def fixture_pair(root):
    """Synthetic one-step runs; these are report tests, not robot experiments."""
    frame=np.zeros((448,448,3),np.uint8)
    initial=dict(robot=[0.]*16,qpos=[0.],velocities=[0.],objects={},orientations={})
    for name in ('baseline','selection'):
        folder=root/name;folder.mkdir()
        save(folder/'result.json',dict(status='complete',success=False,steps=1))
        save(folder/'initial.json',initial)
        np.savez(folder/'trajectory.npz',actions=np.zeros((1,20)),joints=np.zeros((2,16)))
        with imageio.get_writer(folder/'execution.mp4',fps=30,codec='libx264') as writer:
            writer.append_data(frame);writer.append_data(frame)
    folder=root/'selection'
    save(folder/'protocol.json',dict(judged_future_frames=1))
    save(folder/'decisions.json',[dict(step=0,selected=0,
        scores=[dict(id=i,score=0.,valid=True,reason='Synthetic static fixture') for i in range(2)],
        policy_seconds=1.,wm_seconds=2.,proxy_seconds=3.,vlm_seconds=4.)])
    save(folder/'contact-replay.json',dict(summary=dict(success=False,steps=1,max_joint_error=0.,
        maximum_lift_m=0.,finger_lemon_contact_frames=0,terminated=False,truncated=True,failure_stage='synthetic')))
    for i in range(2):
        candidate=folder/f'decision-0000/candidate-{i}';candidate.mkdir(parents=True)
        np.save(candidate/'prediction.npy',frame[None])
        np.savez(candidate/'input.npz',history=frame[None])


def test_full_report_requires_actual_completion_and_consistent_native_result(tmp_path):
    pytest.importorskip('imageio_ffmpeg')
    fixture_pair(tmp_path)
    result=tmp_path/'selection/result.json'
    save(result,dict(status='paused_for_review',success=None,steps=1))
    with pytest.raises(ValueError,match='completed episodes'):export(tmp_path)
    assert not (tmp_path/'summary.json').exists()
    save(result,dict(status='complete',success=True,steps=1))
    with pytest.raises(ValueError,match='Replay result disagrees'):export(tmp_path)
    save(result,dict(status='complete',success=False,steps=1))
    summary=export(tmp_path)
    assert summary['selection']['success'] is False
    assert summary['changed_choices']==0
    assert summary['actions_equal_on_common_prefix'] is True
    assert summary['inference_seconds']['wm_seconds']==2.
    with imageio.get_reader(tmp_path/'execution-comparison.mp4') as reader:
        assert reader.count_frames()==2
