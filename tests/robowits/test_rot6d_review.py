import json
import numpy as np
import pytest
from PIL import Image

from robowits.illustrated_value import view_strips
from robowits.rot6d_loop import paired_candidates


def test_view_strips_keep_cameras_separate_and_hide_unused_future(tmp_path):
    current=np.zeros((448,448,3),np.uint8)
    current[:224,:224]=[10,0,0]
    current[224:,:224]=[0,20,0]
    current[224:,224:]=[0,0,30]
    future=np.repeat(current[None],24,axis=0)
    future[20:]=255
    paths=view_strips(current,future,tmp_path,count=20)
    for path,expected in zip(paths,[[10,0,0],[0,20,0],[0,0,30]]):
        a=np.array(Image.open(path))[24:]
        np.testing.assert_array_equal(a[100,100],expected)
        assert a.max()<255
    with pytest.raises(ValueError):view_strips(current,future,tmp_path,count=25)


def test_rot6d_pairing_requires_full_input_task_and_seed_match(tmp_path):
    source=tmp_path/'source';current=tmp_path/'current'
    source.mkdir();current.mkdir()
    for folder in [source,current]:
        np.savez(folder/'policy-input.npz',state=np.zeros(20),ego=np.zeros((4,4,3),np.uint8))
        (folder/'policy-request.json').write_text(json.dumps({'task':'lemon','seeds':[1,2]}))
    saved=np.zeros((2,50,20));saved[1]=2
    np.save(source/'candidates.npy',saved)
    fresh=np.ones((2,50,20))
    a=paired_candidates(source,current,fresh)
    assert not a[0].any() and a[1].all()
    np.testing.assert_array_equal(paired_candidates(source,current,fresh,all_candidates=True),saved)
    (current/'policy-request.json').write_text(json.dumps({'task':'different task','seeds':[1,2]}))
    with pytest.raises(RuntimeError,match='task or seeds'):paired_candidates(source,current,fresh)
    np.savez(current/'policy-input.npz',state=np.ones(20),ego=np.zeros((4,4,3),np.uint8))
    with pytest.raises(RuntimeError,match='inputs differ'):paired_candidates(source,current,fresh)


def test_illustrated_eef_reward_excludes_padded_images_and_motion_measurements(tmp_path, monkeypatch):
    import robowits.illustrated_value as module
    current=np.zeros((448,448,3),np.uint8)
    future=np.repeat(current[None],20,axis=0)
    future[6:]=255
    manifest=tmp_path/'examples.json'
    manifest.write_text(json.dumps({'examples':[]}))
    def generate(socket,prompt,images,out,*,examples):
        for path in images:
            assert not np.asarray(Image.open(path))[24:].any()
        measurements=json.loads(prompt.split('QUERY image measurements (computed from supplied images only): ')[1].split('\n')[0])
        assert all(x['mean_absolute_rgb_change']==0 and x['changed_pixel_percent']==0 for x in measurements.values())
        return json.dumps({'candidates':[dict(id=0,valid=True,score=0.,reason='Static query')]})
    monkeypatch.setattr(module,'generate',generate)
    result=module.IllustratedValueReward('unused',manifest).judge(current,[future],'task','subgoal',tmp_path/'reward',valid_future_frames=6)
    assert result[0]['score']==0
