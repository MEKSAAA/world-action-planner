"""Amber lemon VLA -> sampled actions -> WAP imagination -> VLM reward -> execution.

Run in the Genesis Python; policy inference runs in a separate OpenPI process,
WM inference in the WAP Python, and VLM inference via an explicit local socket.
"""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

from .closed_loop import Worker, snapshot, same_state, state16, write_json, file_sha256
from .closed_loop_score import choose
from .geometry import Marvin, pack
from .openpi_client import OpenPIClient
from .eef_actions import EEFProxy, chunk_targets, control_action
from .vlm_reward import VLMReward, IndependentVLMReward, generate


def candidate_seeds(seed, decision, count):
    return [int.from_bytes(hashlib.sha256(f'{seed}:{decision}:{i}'.encode()).digest()[:4], 'little')
            for i in range(count)]


def history_inputs(history):
    times = np.asarray([h[0] for h in history])
    target = times[-1]+np.arange(-79, 1)/20
    indices = np.abs(times[:, None]-target).argmin(axis=0)
    return (np.stack([history[i][1] for i in indices]), np.stack([history[i][2] for i in indices]))


def reuse_prediction(source, destination, history, past, future):
    """Reuse only byte-identical WM inputs; callers also verify model/seed provenance."""
    if not (source/'prediction.npy').exists() or not (source/'input.npz').exists():
        return None
    with np.load(source/'input.npz') as saved:
        if any(not np.array_equal(saved[k], v) for k,v in
               [('history',history),('past',past),('future',future)]):
            return None
    prediction = np.load(source/'prediction.npy')
    for filename in ('input.npz','prediction.npy'):
        shutil.copyfile(source/filename, destination/filename)
    write_json(destination/'reuse.json', {'source':str(source), 'inputs_exactly_equal':True})
    return prediction


def anchor_candidate_zero(reference, index, observation, seeds, candidates, directory, *, all_candidates=False):
    """Pair on a saved policy draw only after proving the complete input is identical."""
    source = reference/f'decision-{index:04d}'
    with np.load(source/'policy-input.npz') as saved:
        inputs = {k:v for k,v in observation.items() if k != 'prompt'}
        if set(saved.files) != set(inputs) or any(not np.array_equal(saved[k],v) for k,v in inputs.items()):
            raise RuntimeError('Cannot reuse baseline draw: policy inputs differ')
    request = json.loads((reference/f'policy-queue/request-{index+1:06d}.json').read_text())
    if request['prompt'] != observation['prompt'] or request['seeds'][0] != seeds[0]:
        raise RuntimeError('Cannot reuse baseline draw: prompt or seed differs')
    if all_candidates and request['seeds'] != seeds:
        raise RuntimeError('Cannot reuse candidate set: seeds differ')
    original = np.load(source/'candidates.npy')
    if all_candidates and original.shape != candidates.shape:
        raise RuntimeError('Cannot reuse candidate set: shape differs')
    if not (directory/'sampled-candidates.npy').exists():
        np.save(directory/'sampled-candidates.npy', candidates)
    name = 'candidate-set-reuse.json' if all_candidates else 'candidate-zero-reuse.json'
    write_json(directory/name, dict(source=str(source),
               inputs_prompt_seed_exactly_equal=True,
               fresh_draw_max_abs_diff=float(np.max(np.abs(candidates-original if all_candidates else candidates[0]-original[0]))),
               reason='Pair on the same saved policy draw; independent processes need not be bitwise deterministic'))
    candidates = candidates.copy()
    if all_candidates:
        candidates[:] = original
    else:
        candidates[0] = original[0]
    np.save(directory/'candidates.npy', candidates)
    return candidates


def current_subgoal(env, socket, directory):
    from hier_policy.planner import make_prompt, parse_subgoal, compose_point
    from hier_policy.readers.robowits_replay import _read_ego_rgbd
    rgb, depth, camera = _read_ego_rgbd(env)
    directory.mkdir()
    image = directory/'ego.png'
    Image.fromarray(rgb).save(image)
    raw = generate(socket, make_prompt(env.task_description), [image], directory/'vlm')
    goal = parse_subgoal(raw)
    valid = goal.arm != 'none'
    point = compose_point(goal.anchor, goal.offset, depth, camera) if valid else np.zeros(3, np.float32)
    write_json(directory/'subgoal.json', dict(asdict(goal), point_3d=point.tolist(), valid=valid,
                                             source='current RGB + depth; not object-state oracle'))
    return goal, point, valid


def run_episode(env, policy, wm, judge, proxy, fk, args):
    from hier_policy.rollout import policy_observation
    import imageio.v2 as imageio
    env.set_eval_dataset_idx(args.case)
    obs, info = env.reset(seed=args.seed)
    write_json(args.out/'initial.json', snapshot(env))
    mode = getattr(args, 'mode', 'selection')
    reference = getattr(args, 'reference_run', None)
    reference_decisions = reference_trajectory = None
    imagination_reference = getattr(args, 'reuse_imagination_from', None)
    imagined_decisions = (json.loads((imagination_reference/'decisions.json').read_text())
                         if imagination_reference is not None else [])
    if reference is not None:
        if not same_state(snapshot(env), json.loads((reference/'initial.json').read_text())):
            raise RuntimeError('Paired initial states differ')
        reference_decisions = json.loads((reference/'decisions.json').read_text())
        reference_trajectory = np.load(reference/'trajectory.npz')
    diverged_by_selection = False
    history = deque(maxlen=180)
    all_actions, all_joints, decisions = [], [], []
    result = {'status': 'running', 'success': None, 'steps': 0}
    started = time.monotonic()
    with imageio.get_writer(args.out/'execution.mp4', fps=1/env.dt, codec='libx264') as writer:
        def observe():
            joints = state16(env._robot)
            rgb = pack(obs['pixels'])
            history.append((env._step_count*env.dt, rgb, pack(fk.skeleton(joints))))
            all_joints.append(joints)
            writer.append_data(rgb)
        observe()
        try:
            goal = point = None
            valid = False
            last_goal_step = -args.subgoal_steps
            while env._step_count < args.max_steps:
                directory = args.out/f'decision-{len(decisions):04d}'
                directory.mkdir()
                reference_decision = None
                if reference is not None and not diverged_by_selection:
                    if len(decisions) >= len(reference_decisions):
                        raise RuntimeError('Paired baseline ended before the unchanged prefix')
                    reference_decision = reference_decisions[len(decisions)]
                    current_hash = hashlib.sha256(pack(obs['pixels']).tobytes()).hexdigest()
                    if current_hash != reference_decision['observation_sha256']:
                        raise RuntimeError('Paired observation diverged before any selected alternative')
                if goal is None or env._step_count-last_goal_step >= args.subgoal_steps:
                    if reference_decision is not None:
                        from hier_policy.planner import Subgoal
                        payload = json.loads((reference/f'decision-{len(decisions):04d}/subgoal/subgoal.json').read_text())
                        goal = Subgoal(payload['language'],payload['arm'],tuple(payload['anchor']),tuple(payload['offset']))
                        point,valid = np.asarray(payload['point_3d'],np.float32),payload['valid']
                        (directory/'subgoal').mkdir()
                        write_json(directory/'subgoal/subgoal.json',dict(payload,reused_from=str(reference)))
                    else:
                        goal, point, valid = current_subgoal(env, args.vlm_socket, directory/'subgoal')
                    last_goal_step = env._step_count
                inp = policy_observation(obs, goal, point, valid)
                seeds = candidate_seeds(args.seed, len(decisions), 1 if mode == 'baseline' else args.num_candidates)
                candidates, policy_time = policy.sample(inp, seeds, directory)
                if reference_decision is not None:
                    if getattr(args, 'reuse_baseline_candidate_zero', False):
                        candidates = anchor_candidate_zero(reference,len(decisions),inp,seeds,candidates,directory)
                    if imagination_reference is not None:
                        candidates = anchor_candidate_zero(imagination_reference,len(decisions),inp,seeds,candidates,directory,
                                                           all_candidates=True)
                    first = np.load(reference/f'decision-{len(decisions):04d}/candidates.npy')[0]
                    if not np.array_equal(first,candidates[0]):
                        raise RuntimeError('Paired candidate zero differs before any alternative was executed')
                targets = [chunk_targets(obs['agent_pos_ee'], action) for action in candidates]
                rgb, past = history_inputs(history)
                if mode == 'baseline':
                    selected = 0
                    decision = {'step': env._step_count, 'seeds': seeds, 'subgoal': goal.language,
                                'scores': [], 'recommended': 0, 'selected': 0,
                                'policy_seconds': policy_time, 'wm_seconds': 0.,
                                'proxy_seconds': 0., 'vlm_seconds': 0.}
                else:
                    predictions = []
                    wm_time = proxy_time = 0.
                    reused = 0
                    for i, target in enumerate(targets):
                        candidate_dir = directory/f'candidate-{i}'
                        candidate_dir.mkdir()
                        np.save(candidate_dir/'targets.npy', target)
                        before = time.monotonic()
                        future = proxy.predict_targets(env, target)
                        proxy_time += time.monotonic()-before
                        np.save(candidate_dir/'robot_only_states.npy', proxy.last_states)
                        prediction = None
                        if imagination_reference is not None:
                            source = imagination_reference/f'decision-{len(decisions):04d}/candidate-{i}'
                            prediction = reuse_prediction(source, candidate_dir, rgb, past, future)
                        if prediction is None:
                            prediction, seconds = wm.predict(candidate_dir, rgb, past, future,
                                                             args.seed+len(decisions))
                        else:
                            seconds = 0.
                            reused += 1
                        wm_time += seconds
                        predictions.append(prediction)
                    before = time.monotonic()
                    cached_decision = imagined_decisions[len(decisions)] if len(decisions) < len(imagined_decisions) else None
                    if (reused == len(predictions) and cached_decision is not None and
                            cached_decision['subgoal'] == goal.language and
                            cached_decision['observation_sha256'] == hashlib.sha256(pack(obs['pixels']).tobytes()).hexdigest()):
                        judgments = cached_decision['scores']
                        write_json(directory/'reward-reuse.json', {'source':str(imagination_reference/f'decision-{len(decisions):04d}/reward'),
                                   'scores':judgments, 'same_observation_subgoal_predictions':True})
                        vlm_time = 0.
                    else:
                        judgments = judge.judge(pack(obs['pixels']), predictions, env.task_description,
                                                goal.language, directory/'reward', valid_future_frames=6)
                        vlm_time = time.monotonic()-before
                    if not any(s['valid'] for s in judgments):
                        write_json(directory/'judgments.json', judgments)
                        raise RuntimeError('VLM rejected every imagined candidate; no action executed')
                    recommended = choose(judgments, margin=getattr(args, 'selection_margin', 0.), require_baseline=False)
                    selected = 0 if args.force_candidate_zero else recommended
                    decision = {'step': env._step_count, 'seeds': seeds, 'subgoal': goal.language,
                                'scores': judgments, 'recommended': recommended, 'selected': selected,
                                'policy_seconds': policy_time, 'wm_seconds': wm_time,
                                'proxy_seconds': proxy_time, 'vlm_seconds': vlm_time,
                                'reused_wm_predictions':reused,
                                'reference_wm_seconds':cached_decision['wm_seconds'] if reused == len(predictions) and cached_decision else 0.,
                                'reference_vlm_seconds':cached_decision['vlm_seconds'] if vlm_time == 0. and cached_decision else 0.}
                decision['mode'] = mode
                decision['observation_sha256'] = hashlib.sha256(pack(obs['pixels']).tobytes()).hexdigest()
                diverged_by_selection |= selected != 0
                decisions.append(decision)
                write_json(args.out/'decisions.json', decisions)
                # Compose at query time, then correct relative to the measured pose
                # each real step, exactly as Amber's RecordedScene.step does.
                for target in targets[selected][:args.execute_steps]:
                    action = control_action(obs['agent_pos_ee'], target)
                    obs, reward, terminated, truncated, info = env.step(action)
                    all_actions.append(action)
                    observe()
                    if reference_trajectory is not None and not diverged_by_selection:
                        step = len(all_actions)
                        if not np.array_equal(action,reference_trajectory['actions'][step-1]) or not np.allclose(
                                all_joints[-1],reference_trajectory['joints'][step],atol=1e-5,rtol=0):
                            raise RuntimeError('Paired executed trajectory diverged without an alternative selection')
                    result.update(steps=env._step_count, success=bool(info['is_success']))
                    if terminated or truncated or env._step_count >= args.max_steps:
                        break
                write_json(args.out/'status.json', {'status': 'running', 'steps': env._step_count,
                                                   'decisions': len(decisions), 'pid': os.getpid()})
                print(json.dumps(decision), flush=True)
                if terminated or truncated:
                    break
            result['status'] = 'complete'
        except BaseException:
            result.update(status='error', success=None, error=traceback.format_exc())
            raise
        finally:
            result.update(wall_seconds=time.monotonic()-started, decisions=len(decisions),
                          changed_choices=sum(d['selected'] != 0 for d in decisions))
            write_json(args.out/'result.json', result)
            np.savez_compressed(args.out/'trajectory.npz', actions=np.asarray(all_actions), joints=np.asarray(all_joints))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sim-root', 'policy-python', 'openpi-runtime', 'policy-checkpoint', 'wm-python',
                 'wm-checkpoint', 'ckpts', 'wan', 'urdf', 'eval-json', 'out', 'vlm-socket'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--policy-gpu', default='1')
    p.add_argument('--wm-gpu', default='0')
    p.add_argument('--case', type=int, default=10)
    p.add_argument('--seed', type=int, default=1010)
    p.add_argument('--num-candidates', type=int, default=4)
    p.add_argument('--execute-steps', type=int, default=5)
    p.add_argument('--subgoal-steps', type=int, default=50)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--reference-run', type=Path, help='Completed baseline; reuse matched subgoals and audit the unchanged prefix')
    p.add_argument('--reuse-baseline-candidate-zero', action='store_true', help='Pair on baseline draw after exact input/prompt/seed checks; retain fresh draws for audit')
    p.add_argument('--reuse-imagination-from', type=Path, help='Completed control trial; reuse exact matching WM inputs and judgments with verified provenance')
    p.add_argument('--mode', choices=['baseline','selection'], default='selection')
    p.add_argument('--reward-mode', choices=['joint','independent'], default='joint')
    p.add_argument('--selection-margin', type=float, default=0., help='Require this score improvement over valid candidate 0')
    p.add_argument('--force-candidate-zero', action='store_true')
    args = p.parse_args()
    if args.num_candidates < 2 or not 1 <= args.execute_steps <= 10 or min(args.subgoal_steps, args.max_steps) < 1 or not np.isfinite(args.selection_margin) or args.selection_margin < 0:
        p.error('Require >=2 candidates, 1–10 execution steps, and positive budgets')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    args.resume = False
    sys.path[:0] = [str(args.sim_root), str(args.sim_root/'third_party/lerobot/src')]
    import gs_gym
    env = policy = wm = None
    try:
        protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        protocol.update(sim_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.sim_root,text=True).strip(),
                        action_mode='EE_DELTA', policy_action_horizon=10, wm_future_frames=20,
                        judged_future_frames=6, terminal_pose_padding_steps=20,
                        policy_training='none; checkpoint and normalization read-only',
                        source_sha256={name:file_sha256(Path(__file__).parent/name) for name in
                                       ('eef_loop.py','eef_actions.py','vlm_reward.py','openpi_worker.py','closed_loop_worker.py')},
                        wm_sha256=file_sha256(args.wm_checkpoint))
        write_json(args.out/'protocol.json', protocol)
        if args.reference_run is not None:
            reference_protocol = json.loads((args.reference_run/'protocol.json').read_text())
            for key in ('seed','case','eval_json','policy_checkpoint','openpi_runtime','sim_commit',
                        'execute_steps','subgoal_steps','max_steps'):
                if protocol[key] != reference_protocol[key]:
                    raise ValueError(f'Paired baseline protocol differs: {key}')
        if args.reuse_imagination_from is not None:
            source_protocol = json.loads((args.reuse_imagination_from/'protocol.json').read_text())
            for key in ('wm_sha256','seed','case','eval_json','ckpts','wan','source_sha256','reward_mode','vlm_socket','sim_commit'):
                if protocol[key] != source_protocol[key]:
                    raise ValueError(f'Imagination reuse provenance differs: {key}')
            if json.loads((args.reuse_imagination_from/'status.json').read_text())['status'] != 'complete':
                raise ValueError('Imagination source must be complete')
        write_json(args.out/'status.json', {'status': 'initializing', 'pid': os.getpid()})
        env = gs_gym.get_task_class('robowits/03-gap-retrieve-v0')(
            control_mode='EE_DELTA', observation_mode='JOINT', max_episode_steps=args.max_steps,
            genesis_backend='cpu', device='cpu', show_viewer=False, camera_type='rasterizer',
            eval_dataset_json=str(args.eval_json))
        # Native YAML stores 0.0333, a rounded 30 Hz timestep.
        if not np.isclose(env.dt, 1/30, atol=5e-5, rtol=0):
            raise ValueError('EEF/WM timing adapter requires native 30 Hz control')
        protocol['native_dt_seconds'] = float(env.dt)
        write_json(args.out/'protocol.json', protocol)
        fk = Marvin(args.urdf)
        # Simulator must be reset/built before constructing the proxy scene.
        env.set_eval_dataset_idx(args.case)
        env.reset(seed=args.seed)
        proxy = EEFProxy(env, fk) if args.mode != 'baseline' else None
        policy = OpenPIClient(args.policy_python, args.openpi_runtime, args.policy_checkpoint,
                              args.out/'policy-queue', args.policy_gpu)
        wm = Worker(args) if args.mode != 'baseline' else None
        judge = (IndependentVLMReward if args.reward_mode == 'independent' else VLMReward)(args.vlm_socket)
        result = run_episode(env, policy, wm, judge, proxy, fk, args)
        write_json(args.out/'status.json', result)
    except BaseException:
        write_json(args.out/'status.json', {'status': 'error', 'error': traceback.format_exc()})
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
