"""Exact shared-prefix candidate reuse for paired multi-goal experiments."""
import json
from pathlib import Path
import shutil

import numpy as np

from .closed_loop import write_json


def validate_reference(protocol, reference):
    reference = Path(reference)
    result = json.loads((reference/'result.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('Full paired comparison requires a completed reference episode')
    previous = json.loads((reference/'protocol.json').read_text())
    if protocol.get('all_invalid_action', 'error') != previous.get('all_invalid_action', 'error'):
        raise ValueError('Paired reference differs: all_invalid_action')
    for key in ('task_id','seed','case','eval_json','policy_checkpoint','openpi_runtime',
                'num_subgoals','samples_per_subgoal','proposer','judges','primary_judge',
                'gemini_model','gemini_thinking_level','qwen_thinking','value_candidate_text',
                'manual_sha256','max_steps','sim_commit'):
        if protocol.get(key) != previous.get(key):
            raise ValueError(f'Paired reference differs: {key}')
    for name in ('multigoal_loop.py','multigoal_proposals.py','multigoal_backend.py','multigoal_value.py',
                 'eef_actions.py','openpi_worker.py'):
        if protocol['source_sha256'][name] != previous['source_sha256'][name]:
            raise ValueError(f'Paired source differs: {name}; use the same frozen code')
    if previous['mode'] != 'without_wm':
        raise ValueError('Paired reference must be the no-WM control')


def reuse_bank(reference, index, obs, expected_seeds, folder):
    """Verify complete policy input, prompt/goal, seeds and targets before reuse."""
    from hier_policy.planner import Subgoal
    from hier_policy.rollout import policy_observation
    from .eef_actions import chunk_targets
    source = Path(reference)/f'decision-{index:04d}'
    proposal = json.loads((source/'proposals/goals.json').read_text())
    metadata = json.loads((source/'candidate-map.json').read_text())
    if [m['seed'] for m in metadata] != expected_seeds:
        raise ValueError('Paired candidate seeds differ')
    for g, row in enumerate(proposal['goals']):
        goal = Subgoal(row['language'],row['arm'],tuple(row['anchor']),tuple(row['offset']))
        inp = policy_observation(obs,goal,np.asarray(row['point_3d'],np.float32),row['valid'])
        arrays = {k:v for k,v in inp.items() if k != 'prompt'}
        with np.load(source/f'goal-{g}/policy-input.npz') as saved:
            if set(arrays) != set(saved.files) or any(not np.array_equal(saved[k],v) for k,v in arrays.items()):
                raise ValueError('Paired complete policy inputs differ')
        matching = [m for m in metadata if m['goal_id'] == g]
        if not matching or any(m['goal'] != dict(language=goal.language,arm=goal.arm,
                anchor=list(goal.anchor),offset=list(goal.offset)) for m in matching):
            raise ValueError('Paired candidate goal/prompt differs')
    candidates = np.load(source/'candidates.npy')
    if candidates.shape != (len(metadata),10,14) or not np.isfinite(candidates).all():
        raise ValueError('Invalid paired action bank')
    targets = np.asarray([chunk_targets(obs['agent_pos_ee'], c) for c in candidates])
    if not np.array_equal(targets,np.load(source/'targets.npy')):
        raise ValueError('Paired measured-pose targets differ')
    shutil.copytree(source/'proposals',folder/'proposals')
    for g in range(len(proposal['goals'])):
        shutil.copytree(source/f'goal-{g}',folder/f'goal-{g}')
    write_json(folder/'candidate-bank-reuse.json',{'source':str(source),
        'complete_inputs_goals_seeds_and_targets_equal':True})
    return candidates, [m['goal']['language'] for m in metadata], metadata
