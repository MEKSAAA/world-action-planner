import json
from pathlib import Path

import numpy as np
import pytest

from robowits.multigoal_backend import Backend, strict_json, validated_complete
from robowits.multigoal_proposals import parse_goals, Proposer
from robowits.multigoal_value import TaskValue, best, select_outcome


def test_json_rejects_duplicate_and_nonfinite():
    for raw in ('{"goals":[],"goals":[]}', '{"x":NaN}', '{"x":Infinity}'):
        with pytest.raises(ValueError):
            strict_json(raw)


def test_goals_reject_bad_grounding_and_duplicates():
    pytest.importorskip('hier_policy')
    rows = [{'language': f'Pick up object {i}', 'arm': 'right', 'anchor': [.5,.5],
             'offset': [0,0,.02]} for i in range(4)]
    assert len(parse_goals(json.dumps({'goals':rows}),4)) == 4
    rows[0]['anchor'][0] = True
    with pytest.raises(ValueError):
        parse_goals(json.dumps({'goals':rows}),4)
    rows[0]['anchor'][0] = .5
    rows[1]['language'] = rows[0]['language']
    with pytest.raises(ValueError):
        parse_goals(json.dumps({'goals':rows}),4)


def test_manual_draft_cannot_execute(tmp_path):
    path = tmp_path/'draft.json'
    path.write_text(json.dumps({'review_status':'draft'}))
    with pytest.raises(ValueError, match='review'):
        Proposer(None,4,path)


class Recorder:
    def __init__(self):
        self.calls=[]
    def complete(self,prompt,images,out):
        self.calls.append((prompt,images))
        n = 2 if 'Estimate appropriateness' in prompt else 1
        return json.dumps({'candidates':[{'id':i,'score':.1+i*.5,'valid':True,
                          'reason':'Visible object and scene evidence'} for i in range(n)]})


def test_no_wm_control_shares_scores_across_draws(tmp_path):
    recorder=Recorder()
    scores=TaskValue(recorder).without_wm(np.zeros((448,448,3),np.uint8),
        'Place the lemon', ['Pick lemon','Pick lemon','Move block','Move block'],tmp_path)
    assert [s['score'] for s in scores] == [.1,.1,.6,.6]
    assert best(scores) == 2
    assert len(recorder.calls[0][1]) == 1


def test_scoring_never_exposes_padded_future(tmp_path):
    from PIL import Image
    recorder=Recorder()
    current=np.zeros((448,448,3),np.uint8)
    pred=np.zeros((20,448,448,3),np.uint8)
    pred[6:]=255
    scores=TaskValue(recorder).judge(current,[pred],'Task','Pick lemon'.splitlines(),tmp_path)
    assert len(scores)==1
    for p in recorder.calls[0][1]:
        # Ignore label band: all queried picture pixels must remain black.
        assert np.asarray(Image.open(p))[24:].max()==0
    assert 'OVERALL task' in recorder.calls[0][0]


def test_all_invalid_stops_selection():
    with pytest.raises(RuntimeError):
        best([{'id':0,'valid':False,'score':0}])


def test_backend_reads_only_named_credential(tmp_path,monkeypatch):
    monkeypatch.delenv('TEST_GEMINI_KEY',raising=False)
    path=tmp_path/'.env'
    path.write_text('UNRELATED=do_not_read\nTEST_GEMINI_KEY="local-test-value"\n')
    backend=Backend('gemini',credential_file=path,key_env='TEST_GEMINI_KEY')
    assert backend.key=='local-test-value'
    assert 'UNRELATED' not in vars(backend)


def test_invalid_json_requeries_without_repairing_scores(tmp_path):
    class InvalidThenValid:
        def __init__(self):self.calls=0
        def complete(self,prompt,images,out):
            self.calls+=1
            if self.calls==2:
                assert 'Validation error to correct: Extra data' in prompt
            return '{"a":0} trailing' if self.calls==1 else '{"a":1}'
    backend=InvalidThenValid()
    assert validated_complete(backend,'prompt',[],tmp_path,strict_json)=={'a':1}
    assert backend.calls==2
    assert (tmp_path/'attempt-0/validation-error.json').exists()


def test_wm_tie_and_invalid_are_explicit_control_fallbacks():
    control=[{'id':0,'valid':True,'score':.1},{'id':1,'valid':True,'score':.8}]
    flat=[{'id':0,'valid':True,'score':0.},{'id':1,'valid':True,'score':0.}]
    assert select_outcome(flat,control)==(1,'no_wm_choice_also_maximum_wm_score')
    rejected=[dict(r,valid=False) for r in flat]
    assert select_outcome(rejected,control)==(1,'all_wm_invalid_fallback_to_no_wm')
    flat[0]['score']=.3
    assert select_outcome(flat,control)==(0,'higher_valid_wm_score')


def test_paired_bank_rejects_changed_inputs_and_seeds(tmp_path,monkeypatch):
    pytest.importorskip('hier_policy')
    from hier_policy import rollout
    from robowits.eef_actions import chunk_targets
    from robowits.multigoal_pairing import reuse_bank
    monkeypatch.setattr(rollout,'policy_observation',lambda obs,goal,point,valid:
                        {'state':obs['state'],'prompt':goal.language})
    source=tmp_path/'source/decision-0000'
    (source/'proposals').mkdir(parents=True)
    (source/'goal-0').mkdir()
    goal={'language':'Pick up lemon','arm':'right','anchor':[.5,.5],'offset':[0,0,.02]}
    (source/'proposals/goals.json').write_text(json.dumps({'goals':[
        dict(goal,point_3d=[.5,0,-.3],valid=True)]}))
    metadata=[{'id':i,'goal_id':0,'draw':i,'seed':i+2,'goal':goal} for i in range(2)]
    (source/'candidate-map.json').write_text(json.dumps(metadata))
    obs={'state':np.array([1,2]),'agent_pos_ee':np.zeros(14,np.float32)}
    np.savez(source/'goal-0/policy-input.npz',state=obs['state'])
    bank=np.zeros((2,10,14),np.float32)
    np.save(source/'candidates.npy',bank)
    np.save(source/'targets.npy',np.stack([chunk_targets(obs['agent_pos_ee'],c) for c in bank]))
    folder=tmp_path/'accepted';folder.mkdir()
    reused,_,_=reuse_bank(source.parent,0,obs,[2,3],folder)
    assert np.array_equal(bank,reused)
    with pytest.raises(ValueError,match='inputs'):
        reuse_bank(source.parent,0,dict(obs,state=np.array([1,3])),[2,3],tmp_path/'bad')
    with pytest.raises(ValueError,match='seeds'):
        reuse_bank(source.parent,0,obs,[3,4],tmp_path/'bad')
