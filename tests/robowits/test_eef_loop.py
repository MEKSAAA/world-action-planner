import json
from types import SimpleNamespace

import numpy as np
import pytest

from robowits.vlm_reward import parse_rewards
from robowits.eef_loop import candidate_seeds, history_inputs


def test_reward_schema_reorders_and_rejects_missing_duplicate_nan():
    good = {'candidates': [{'id': i, 'valid': True, 'score': i/2, 'reason': 'visible progress'} for i in (1, 0)]}
    assert [r['id'] for r in parse_rewards(json.dumps(good), 2)] == [0, 1]
    for rows in [good['candidates'][:1], good['candidates']*2,
                 [dict(good['candidates'][0], score=float('nan')), good['candidates'][1]],
                 [dict(good['candidates'][0], valid='yes'), good['candidates'][1]]]:
        with pytest.raises(ValueError):
            parse_rewards(json.dumps({'candidates': rows}), 2)
    with pytest.raises(ValueError):
        parse_rewards('{"candidates":[],"candidates":[]}', 2)


def test_seed_count_independence_and_history_padding():
    assert candidate_seeds(7, 3, 2) == candidate_seeds(7, 3, 4)[:2]
    assert candidate_seeds(7, 3, 2) != candidate_seeds(7, 4, 2)
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb, cond = history_inputs([(0., a, a+1)])
    assert rgb.shape[0] == cond.shape[0] == 80
    assert np.all(rgb == 0) and np.all(cond == 1)


def test_reward_never_sees_terminal_padding(tmp_path, monkeypatch):
    import robowits.vlm_reward as module
    from PIL import Image
    seen = []
    def generate(socket, prompt, images, out):
        seen.extend(images)
        return json.dumps({'candidates': [{'id': 0, 'valid': True, 'score': .5, 'reason': 'visible'}]})
    monkeypatch.setattr(module, 'generate', generate)
    future = np.zeros((20, 448, 448, 3), dtype=np.uint8)
    future[6:] = (255, 0, 0)
    module.VLMReward('unused').judge(future[0], [future], 'task', 'goal', tmp_path)
    sheet = np.array(Image.open(seen[1]))[24:]
    assert not sheet.any()  # Red padded frames cannot influence VLM judgment.


def test_eef_target_composition_matches_native_and_holds_other_arm():
    native = pytest.importorskip('hier_policy.eef')
    from robowits.eef_actions import chunk_targets, control_action
    start = np.zeros(14, np.float32)
    start[12:] = .08
    deltas = np.zeros((10, 14), np.float32)
    deltas[:, 0] = .001
    deltas[:, 6] = .08
    deltas[:, 13] = .08
    targets = chunk_targets(start, deltas)
    expected = start.copy()
    for row, target in zip(deltas, targets):
        expected = native.apply_eef_delta(expected, row)
        np.testing.assert_array_equal(target, expected)
    assert targets[-1, 0] == pytest.approx(.01)
    np.testing.assert_array_equal(control_action(start, targets[0]),
                                  native.to_env_ee_delta(native.eef_between(start, targets[0])))


@pytest.mark.parametrize('all_invalid,mode,stop_after', [(False,'selection',None),(True,'selection',None),(False,'baseline',None),(False,'selection',2)])
def test_real_loop_orchestration_executes_selected_prefix_and_reobserves(tmp_path, monkeypatch, all_invalid, mode, stop_after):
    pytest.importorskip('hier_policy.rollout')
    import robowits.eef_loop as module
    import imageio.v2
    class Env:
        dt = 1/30
        task_description = 'Place the lemon on the green target area.'
        _robot = object()
        _step_count = 0
        def set_eval_dataset_idx(self, case): pass
        def obs(self):
            return {'pixels': self._step_count, 'agent_pos_ee': np.zeros(14),
                    'agent_pos_joint': np.zeros(16)}
        def reset(self, seed): return self.obs(), {}
        def step(self, action):
            executed.append(float(action[0]))
            self._step_count += 1
            return self.obs(), 0., False, self._step_count >= 3, {'is_success': False}
    class Policy:
        def sample(self, inp, seeds, out):
            assert len(seeds) == (1 if mode == 'baseline' else 2)
            seen.append(inp['observation/images/ego'])
            return np.stack([np.zeros((10,14)), np.ones((10,14))]), .1
    class Proxy:
        last_states = np.zeros((30,16))
        def predict_targets(self, env, targets): return targets
    class WM:
        def predict(self, out, rgb, past, future, seed):
            imagined.append(float(future[0, 0]))
            return np.zeros((20,448,448,3),np.uint8), .2
    class Judge:
        def judge(self, current, predictions, task, goal, out, **kwargs):
            assert len(predictions) == 2
            return [{'valid': not all_invalid, 'score': .2}, {'valid': not all_invalid, 'score': .8}]
    seen, executed, imagined = [], [], []
    monkeypatch.setattr(module, 'snapshot', lambda env: {})
    monkeypatch.setattr(module, 'state16', lambda robot: np.zeros(16))
    monkeypatch.setattr(module, 'pack', lambda x: np.full((448,448,3),x,np.uint8))
    monkeypatch.setattr(module, 'current_subgoal', lambda *a: (SimpleNamespace(language='Move obstacle',arm='left'), np.zeros(3), True))
    # Keep the policy-observation helper native; pixels use a simple mapping.
    original_obs = Env.obs
    Env.obs = lambda self: dict(original_obs(self), pixels={k:self._step_count for k in ('ego','wrist_right','wrist_left')})
    monkeypatch.setattr(module, 'pack', lambda x: np.full((448,448,3),x['ego'] if isinstance(x,dict) else x,np.uint8))
    monkeypatch.setattr(module, 'chunk_targets', lambda start, delta: delta)
    monkeypatch.setattr(module, 'control_action', lambda current, target: target)
    class Writer:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def append_data(self, frame): pass
    monkeypatch.setattr(imageio.v2, 'get_writer', lambda *a, **kw: Writer())
    args = SimpleNamespace(out=tmp_path, case=0, seed=2, num_candidates=2, execute_steps=2,
                           subgoal_steps=50, max_steps=3, vlm_socket='unused', force_candidate_zero=False, mode=mode,
                           stop_after_steps=stop_after)
    fk = SimpleNamespace(skeleton=lambda state: 0)
    if all_invalid:
        with pytest.raises(RuntimeError, match='rejected every'):
            module.run_episode(Env(), Policy(), WM(), Judge(), Proxy(), fk, args)
        assert executed == []
        assert json.loads((tmp_path/'result.json').read_text())['success'] is None
    else:
        result = module.run_episode(Env(), Policy(), WM(), Judge(), Proxy(), fk, args)
        count = stop_after or 3
        assert executed == ([0.] if mode == 'baseline' else [1.])*count
        assert seen == ([0] if stop_after else [0, 2])
        assert imagined == ([] if mode == 'baseline' else [0., 1.]*(1 if stop_after else 2))
        assert result['steps'] == count
        assert result['status'] == ('paused_for_review' if stop_after else 'complete')
        if stop_after:
            assert result['success'] is None and result['success_at_stop'] is False


def test_independent_judge_hides_original_candidate_index(tmp_path, monkeypatch):
    import robowits.vlm_reward as module
    seen = []
    def judge(self, current, predictions, task, subgoal, out, **kwargs):
        assert len(predictions) == 1
        seen.append(predictions[0])
        return [{'id':0,'valid':True,'score':.4,'reason':'motion'}]
    monkeypatch.setattr(module.VLMReward, 'judge', judge)
    rows = module.IndependentVLMReward('unused').judge(None, [17,23], 'task','goal', tmp_path)
    assert seen == [17,23] and [r['id'] for r in rows] == [0,1]


def test_reward_placeholder_is_not_evidence():
    with pytest.raises(ValueError, match='reason'):
        parse_rewards(json.dumps({'candidates':[{'id':0,'valid':True,'score':.5,
                                                  'reason':'brief visual evidence'}]}),1)


def test_selection_margin_retains_zero_for_observed_judge_noise():
    from robowits.closed_loop_score import choose
    rows = [{'valid':True,'score':.0}, {'valid':True,'score':.05}]
    assert choose(rows, margin=.1, require_baseline=False) == 0
    rows[1]['score'] = .2
    assert choose(rows, margin=.1, require_baseline=False) == 1
    rows[0]['score'], rows[1]['score'] = .35, .45
    assert choose(rows, margin=.1, require_baseline=False) == 0
    rows[1]['score'] = .450001
    assert choose(rows, margin=.1, require_baseline=False) == 1


def test_imagination_reuse_requires_all_inputs_identical(tmp_path):
    from robowits.eef_loop import reuse_prediction
    source, destination = tmp_path/'source', tmp_path/'destination'
    source.mkdir(); destination.mkdir()
    a = np.zeros((2,2,3),np.uint8)
    np.savez(source/'input.npz', history=a, past=a+1, future=a+2)
    np.save(source/'prediction.npy', a+3)
    assert reuse_prediction(source,destination,a,a+1,a+4) is None
    assert not (destination/'prediction.npy').exists()
    np.testing.assert_array_equal(reuse_prediction(source,destination,a,a+1,a+2),a+3)
    assert json.loads((destination/'reuse.json').read_text())['inputs_exactly_equal']


def test_paired_policy_draw_reuse_rejects_input_prompt_seed_changes(tmp_path):
    from robowits.eef_loop import anchor_candidate_zero
    reference, destination = tmp_path/'reference',tmp_path/'destination'
    source = reference/'decision-0000'
    source.mkdir(parents=True); destination.mkdir(); (reference/'policy-queue').mkdir()
    obs = {'state':np.zeros(14), 'prompt':'pick up lemon'}
    np.savez(source/'policy-input.npz',state=obs['state'])
    np.save(source/'candidates.npy',np.zeros((1,10,14)))
    (reference/'policy-queue/request-000001.json').write_text(json.dumps({'prompt':obs['prompt'],'seeds':[5]}))
    for changed,seeds in [(dict(obs,state=np.ones(14)),[5]),(dict(obs,prompt='wrong'),[5]),(obs,[6])]:
        with pytest.raises(RuntimeError,match='Cannot reuse'):
            anchor_candidate_zero(reference,0,changed,seeds,np.ones((2,10,14)),destination)
    actual = anchor_candidate_zero(reference,0,obs,[5,6],np.ones((2,10,14)),destination)
    assert not actual[0].any() and actual[1].all()
    assert np.load(destination/'sampled-candidates.npy').all()
    bank = np.stack([np.zeros((10,14)),np.full((10,14),2.)])
    np.save(source/'candidates.npy',bank)
    (reference/'policy-queue/request-000001.json').write_text(json.dumps({'prompt':obs['prompt'],'seeds':[5,6]}))
    with pytest.raises(RuntimeError,match='seeds differ'):
        anchor_candidate_zero(reference,0,obs,[5,7],actual,destination,all_candidates=True)
    np.testing.assert_array_equal(anchor_candidate_zero(reference,0,obs,[5,6],actual,destination,all_candidates=True),bank)
    assert np.load(destination/'sampled-candidates.npy').all()  # Preserve fresh inference evidence.


def test_reward_selects_best_of_many_and_excludes_invalid():
    from robowits.closed_loop_score import choose
    scores = [{'valid': v, 'score': s} for v,s in [(False, .99),(True,.3),(True,.9),(True,.7)]]
    assert choose(scores, margin=0, require_baseline=False) == 2
    scores[2]['score'] = float('nan')
    assert choose(scores, margin=0, require_baseline=False) == 3
