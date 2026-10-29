"""Frozen Amber policy + multi-subgoal providers + WM + task-level VLM value.

JSON configuration is explicit and saved with source hashes. Use --config FILE.
No policy training, checkpoint mutation, or task-object rollout in selection.
"""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

from .closed_loop import Worker, file_sha256, snapshot, same_state, state16, write_json
from .eef_actions import EEFProxy, chunk_targets, control_action
from .eef_loop import candidate_seeds, history_inputs
from .geometry import Marvin, pack
from .multigoal_backend import Backend
from .multigoal_proposals import Proposer
from .multigoal_value import TaskValue, best, select_outcome
from .multigoal_pairing import reuse_bank, validate_reference
from .openpi_client import OpenPIClient


def backend(config, name):
    return Backend(name, socket=config.get('vlm_socket'),
        model=config.get('gemini_model', 'gemini-3-flash-preview'),
        credential_file=config.get('credential_file'),
        key_env=config.get('key_env', 'RPAD_GEMINI_API_KEY'),
        thinking_level=config.get('gemini_thinking_level', 'low'),
        qwen_thinking=config.get('qwen_thinking',False))


def execution_targets(current, targets, chosen):
    """Explicit abstention holds the measured pose; never relabel it a policy draw."""
    if chosen is None:
        return np.repeat(np.asarray(current, dtype=np.float32)[None], 10, axis=0)
    return targets[chosen]


def run(env, policy, wm, proxy, fk, config, out):
    import imageio.v2 as imageio
    from hier_policy.rollout import policy_observation
    env.set_eval_dataset_idx(config['case'])
    obs, _ = env.reset(seed=config['seed'])
    write_json(out/'initial.json', snapshot(env))
    reference = Path(config['reference_run']) if config.get('reference_run') else None
    paired = reference is not None
    reference_decisions = reference_trajectory = None
    if paired:
        if not same_state(snapshot(env), json.loads((reference/'initial.json').read_text())):
            raise ValueError('Paired initial states differ')
        reference_decisions = json.loads((reference/'decisions.json').read_text())
        reference_trajectory = np.load(reference/'trajectory.npz')
    proposer = Proposer(backend(config, config['proposer']), config['num_subgoals'], config.get('manual'))
    judges = {name: TaskValue(backend(config, name),candidate_text=config.get('value_candidate_text',True))
              for name in config['judges']}
    history = deque(maxlen=180)
    actions, joints, decisions = [], [], []
    started = time.monotonic()
    mode = config['mode']
    result = {'status': 'running', 'success': None, 'steps': 0, 'mode': mode}
    terminated = truncated = False
    with imageio.get_writer(out/'execution.mp4', fps=1/env.dt, codec='libx264') as video:
        def observe():
            q = state16(env._robot)
            rgb = pack(obs['pixels'])
            history.append((env._step_count*env.dt, rgb, pack(fk.skeleton(q))))
            joints.append(q)
            video.append_data(rgb)
        observe()
        try:
            if config.get('prefix_run'):
                source = Path(config['prefix_run'])
                start_step = config['prefix_steps']
                if not same_state(snapshot(env), json.loads((source/'initial.json').read_text())):
                    raise ValueError('Diagnostic prefix initial state differs')
                with np.load(source/'trajectory.npz') as recorded:
                    if not 0 < start_step < min(len(recorded['actions']), config['max_steps']):
                        raise ValueError('Prefix must end before the episode endpoint')
                    for step, action in enumerate(recorded['actions'][:start_step], 1):
                        obs, _, terminated, truncated, info = env.step(action)
                        if terminated or truncated or not np.allclose(
                                state16(env._robot), recorded['joints'][step], atol=1e-5, rtol=0):
                            raise ValueError('Diagnostic prefix replay diverged')
                        actions.append(action)
                        observe()
                result.update(steps=start_step, replayed_prefix_steps=start_step)
                write_json(out/'prefix.json', {'source': str(source), 'steps': start_step,
                    'trajectory_sha256': file_sha256(source/'trajectory.npz'),
                    'initial_and_each_joint_state_verified': True,
                    'scope': 'diagnostic state reached using recorded past actions'})
            while env._step_count < config['max_steps']:
                index = len(decisions)
                folder = out/f'decision-{index:04d}'
                folder.mkdir()
                if paired:
                    if index >= len(reference_decisions):
                        raise ValueError('Paired reference ended before unchanged prefix')
                    previous = reference_decisions[index]
                    seeds = candidate_seeds(config['seed'],index,len(previous['candidate_map']))
                    if previous['step'] != env._step_count or previous['observation_sha256'] != hashlib.sha256(pack(obs['pixels']).tobytes()).hexdigest():
                        raise ValueError('Paired step or RGB diverged before alternative selection')
                    candidates, languages, metadata = reuse_bank(reference,index,obs,seeds,folder)
                    proposal_seconds = policy_seconds = 0.
                else:
                    t0 = time.monotonic()
                    goals = proposer.propose(env, folder/'proposals', config['seed']+index)
                    seeds = candidate_seeds(config['seed'],index,len(goals)*config['samples_per_subgoal'])
                    proposal_seconds = time.monotonic()-t0
                    candidates, languages, metadata = [], [], []
                    policy_seconds = 0.
                    for g, (goal, point, valid) in enumerate(goals):
                        inp = policy_observation(obs, goal, point, valid)
                        goal_dir = folder/f'goal-{g}'
                        goal_dir.mkdir()
                        offset = g*config['samples_per_subgoal']
                        goal_seeds = seeds[offset:offset+config['samples_per_subgoal']]
                        sampled, seconds = policy.sample(inp, goal_seeds, goal_dir)
                        policy_seconds += seconds
                        for draw, action in enumerate(sampled):
                            metadata.append({'id': len(candidates), 'goal_id': g, 'draw': draw,
                                             'seed': goal_seeds[draw], 'goal': asdict(goal)})
                            candidates.append(action)
                            languages.append(goal.language)
                    candidates = np.asarray(candidates)
                np.save(folder/'candidates.npy', candidates)
                write_json(folder/'candidate-map.json', metadata)
                targets = np.asarray([chunk_targets(obs['agent_pos_ee'], c) for c in candidates])
                np.save(folder/'targets.npy', targets)
                # Report physical position differences separately from orientation
                # and gripper width; do not mix their units into one norm.
                position_spread = np.linalg.norm(targets[:,None,:,:6]-targets[None,:,:,:6],axis=-1).mean(axis=-1)
                np.save(folder/'pairwise-position-distance-m.npy', position_spread)
                current = pack(obs['pixels'])
                if paired:
                    no_wm = previous['no_wm']
                    write_json(folder/'no-wm-reuse.json',{'source':str(reference/f'decision-{index:04d}'),
                               'same_current_observation_and_candidates':True,'scores':no_wm})
                else:
                    no_wm = {name: judge.without_wm(current, env.task_description, languages,
                        folder/f'no-wm-{name}') for name, judge in judges.items()}
                scores, predictions = {}, []
                wm_seconds = proxy_seconds = 0.
                if mode == 'selection':
                    history_rgb, past = history_inputs(history)
                    for i, target in enumerate(targets):
                        candidate_dir = folder/f'candidate-{i}'
                        candidate_dir.mkdir()
                        t0 = time.monotonic()
                        future = proxy.predict_targets(env, target)
                        proxy_seconds += time.monotonic()-t0
                        np.save(candidate_dir/'robot_only_states.npy', proxy.last_states)
                        np.save(candidate_dir/'targets.npy', target)
                        prediction, seconds = wm.predict(candidate_dir, history_rgb, past, future, config['seed']+index)
                        wm_seconds += seconds
                        predictions.append(prediction)
                    for name, judge in judges.items():
                        scores[name] = judge.judge(current, predictions, env.task_description,
                            languages, folder/f'wm-{name}', valid_future_frames=6)
                primary = config['primary_judge']
                allow_abstention = config.get('all_invalid_action', 'error') == 'hold'
                chosen, choice_reason = (select_outcome(scores[primary], no_wm[primary],
                    allow_abstention=allow_abstention) if mode == 'selection'
                    else (best(no_wm[primary], allow_abstention=allow_abstention), 'no_wm_control'))
                if chosen is None and mode == 'without_wm':
                    choice_reason = 'all_no_wm_invalid_hold'
                executed_targets = execution_targets(obs['agent_pos_ee'], targets, chosen)
                np.save(folder/'executed-targets.npy', executed_targets)
                decision = {'step': env._step_count, 'selected': chosen, 'candidate_map': metadata,
                    'abstained': chosen is None,
                    'choice_reason': choice_reason, 'paired_candidate_bank': paired,
                    'no_wm': no_wm, 'wm': scores, 'proposal_seconds': proposal_seconds,
                    'policy_seconds': policy_seconds, 'proxy_seconds': proxy_seconds, 'wm_seconds': wm_seconds,
                    'observation_sha256': hashlib.sha256(current.tobytes()).hexdigest()}
                decisions.append(decision)
                write_json(out/'decisions.json', decisions)
                if paired and chosen != previous['selected']:
                    paired = False
                for target in executed_targets:
                    action = control_action(obs['agent_pos_ee'], target)
                    obs, _, terminated, truncated, info = env.step(action)
                    actions.append(action)
                    observe()
                    if paired:
                        step = len(actions)
                        if not np.array_equal(action, reference_trajectory['actions'][step-1]) or not np.allclose(
                                joints[-1],reference_trajectory['joints'][step],atol=1e-5,rtol=0):
                            raise ValueError('Paired executed state diverged without an alternative selection')
                    result.update(steps=env._step_count, success=bool(info['is_success']))
                    if terminated or truncated or env._step_count >= config['max_steps']:
                        break
                write_json(out/'status.json', dict(result, pid=os.getpid(), decisions=len(decisions)))
                np.savez_compressed(out/'trajectory.npz', actions=np.asarray(actions), joints=np.asarray(joints))
                print(json.dumps({'step': env._step_count, 'selected': chosen,
                                  'goal': languages[chosen] if chosen is not None else 'HOLD (abstention)',
                                  'success': result['success']}), flush=True)
                if terminated or truncated:
                    break
                if config.get('diagnostic_decisions') and len(decisions) >= config['diagnostic_decisions']:
                    break
            result.update(status='complete' if terminated or truncated or env._step_count >= config['max_steps']
                          else 'diagnostic_complete', terminated=bool(terminated), truncated=bool(truncated))
            if result['status'] == 'diagnostic_complete':
                result.update(success_at_stop=result['success'], success=None)
        except BaseException:
            result.update(status='error', success=None, error=traceback.format_exc())
            raise
        finally:
            result.update(wall_seconds=time.monotonic()-started, decisions=len(decisions))
            write_json(out/'result.json', result)
            np.savez_compressed(out/'trajectory.npz', actions=np.asarray(actions), joints=np.asarray(joints))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['mode'] not in ('selection', 'without_wm'):
        raise ValueError('Invalid selection mode')
    if config.get('all_invalid_action', 'error') not in ('error', 'hold'):
        raise ValueError('all_invalid_action must be error or hold')
    if not 3 <= config['num_subgoals'] <= 5 or not 1 <= config['samples_per_subgoal'] <= 4:
        raise ValueError('Invalid candidate counts')
    if config['primary_judge'] not in config['judges'] or len(set(config['judges'])) != len(config['judges']):
        raise ValueError('Invalid judges')
    if config['max_steps'] < 1 or config.get('diagnostic_decisions', 1) < 1:
        raise ValueError('Positive step and diagnostic budgets required')
    if config.get('prefix_run') and not config.get('diagnostic_decisions'):
        raise ValueError('Recorded prefix warm starts are restricted to diagnostic runs')
    if config.get('reference_run') and (config.get('prefix_run') or config['mode'] != 'selection'):
        raise ValueError('Full paired references require selection mode and no warm-start prefix')
    # Resolve relative experiment paths before workers change working directories.
    for key in ('out','sim_root','policy_python','openpi_runtime','policy_checkpoint',
                'wm_python','wm_checkpoint','ckpts','wan','urdf','eval_json','manual','reference_run'):
        if config.get(key) is not None:
            # Preserve virtualenv executable symlinks: resolving bin/python to
            # the system binary would silently drop the environment's packages.
            config[key] = str(Path(config[key]).absolute())
    out = Path(config['out'])
    out.mkdir(parents=True, exist_ok=False)
    sim_root = Path(config['sim_root'])
    sys.path[:0] = [str(sim_root), str(sim_root/'third_party/lerobot/src')]
    import gs_gym
    env = policy = wm = None
    try:
        # Catch credentials/catalogue/schema errors before loading GPU models.
        Proposer(backend(config, config['proposer']), config['num_subgoals'], config.get('manual'))
        for name in config['judges']:
            backend(config, name)
        protocol = dict(config, action_mode='EE_DELTA', observation_mode='JOINT', execute_steps=10,
            wm_future_frames=20, judged_future_frames=6, terminal_pose_padding_steps=20,
            wm_choice_rule='prefer no-WM choice on a maximum-score tie; all-invalid fallback to no-WM choice',
            all_invalid_action=config.get('all_invalid_action', 'error'),
            policy_training='none; checkpoint and normalization read-only',
            sim_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=sim_root,text=True).strip(),
            source_sha256={n: file_sha256(Path(__file__).with_name(n)) for n in (
                'multigoal_loop.py','multigoal_backend.py','multigoal_proposals.py','multigoal_value.py',
                'eef_actions.py','openpi_worker.py','closed_loop_worker.py','multigoal_pairing.py')},
            wm_sha256=file_sha256(Path(config['wm_checkpoint'])),
            manual_sha256=file_sha256(Path(config['manual'])) if config.get('manual') else None)
        write_json(out/'protocol.json', protocol)
        if config.get('reference_run'):
            validate_reference(protocol,config['reference_run'])
        write_json(out/'status.json', {'status':'initializing','pid':os.getpid()})
        env = gs_gym.get_task_class(config['task_id'])(
            control_mode='EE_DELTA', observation_mode='JOINT', max_episode_steps=config['max_steps'],
            genesis_backend='cpu', device='cpu', show_viewer=False, camera_type='rasterizer',
            eval_dataset_json=config.get('eval_json'))
        if not np.isclose(env.dt, 1/30, atol=5e-5, rtol=0):
            raise ValueError('Expected native 30 Hz control')
        env.set_eval_dataset_idx(config['case'])
        env.reset(seed=config['seed'])
        fk = Marvin(Path(config['urdf']))
        proxy = EEFProxy(env, fk) if config['mode'] == 'selection' else None
        policy = OpenPIClient(Path(config['policy_python']), Path(config['openpi_runtime']),
            Path(config['policy_checkpoint']), out/'policy-queue', config['policy_gpu'])
        if config['mode'] == 'selection':
            worker_config = dict(config, out=out, resume=False)
            wm = Worker(SimpleNamespace(**worker_config))
        result = run(env, policy, wm, proxy, fk, config, out)
        write_json(out/'status.json', result)
    except BaseException:
        write_json(out/'status.json', {'status':'error','error':traceback.format_exc()})
        raise
    finally:
        if wm is not None:
            wm.close()
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
