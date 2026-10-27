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
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

from .closed_loop import Worker, snapshot, state16, write_json, file_sha256
from .closed_loop_score import choose
from .geometry import Marvin, pack
from .openpi_client import OpenPIClient
from .eef_actions import EEFProxy, chunk_targets, control_action
from .vlm_reward import VLMReward, generate


def candidate_seeds(seed, decision, count):
    return [int.from_bytes(hashlib.sha256(f'{seed}:{decision}:{i}'.encode()).digest()[:4], 'little')
            for i in range(count)]


def history_inputs(history):
    times = np.asarray([h[0] for h in history])
    target = times[-1]+np.arange(-79, 1)/20
    indices = np.abs(times[:, None]-target).argmin(axis=0)
    return (np.stack([history[i][1] for i in indices]), np.stack([history[i][2] for i in indices]))


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
                if goal is None or env._step_count-last_goal_step >= args.subgoal_steps:
                    goal, point, valid = current_subgoal(env, args.vlm_socket, directory/'subgoal')
                    last_goal_step = env._step_count
                inp = policy_observation(obs, goal, point, valid)
                seeds = candidate_seeds(args.seed, len(decisions), args.num_candidates)
                candidates, policy_time = policy.sample(inp, seeds, directory)
                targets = [chunk_targets(obs['agent_pos_ee'], action) for action in candidates]
                rgb, past = history_inputs(history)
                predictions = []
                wm_time = proxy_time = 0.
                for i, target in enumerate(targets):
                    candidate_dir = directory/f'candidate-{i}'
                    candidate_dir.mkdir()
                    np.save(candidate_dir/'targets.npy', target)
                    before = time.monotonic()
                    future = proxy.predict_targets(env, target)
                    proxy_time += time.monotonic()-before
                    np.save(candidate_dir/'robot_only_states.npy', proxy.last_states)
                    prediction, seconds = wm.predict(candidate_dir, rgb, past, future,
                                                     args.seed+len(decisions))
                    wm_time += seconds
                    predictions.append(prediction)
                before = time.monotonic()
                judgments = judge.judge(pack(obs['pixels']), predictions, env.task_description,
                                        goal.language, directory/'reward', valid_future_frames=6)
                vlm_time = time.monotonic()-before
                if not any(s['valid'] for s in judgments):
                    write_json(directory/'judgments.json', judgments)
                    raise RuntimeError('VLM rejected every imagined candidate; no action executed')
                recommended = choose(judgments, margin=0., require_baseline=False)
                selected = 0 if args.force_candidate_zero else recommended
                decision = {'step': env._step_count, 'seeds': seeds, 'subgoal': goal.language,
                            'scores': judgments, 'recommended': recommended, 'selected': selected,
                            'policy_seconds': policy_time, 'wm_seconds': wm_time,
                            'proxy_seconds': proxy_time, 'vlm_seconds': vlm_time}
                decisions.append(decision)
                write_json(args.out/'decisions.json', decisions)
                # Compose at query time, then correct relative to the measured pose
                # each real step, exactly as Amber's RecordedScene.step does.
                for target in targets[selected][:args.execute_steps]:
                    action = control_action(obs['agent_pos_ee'], target)
                    obs, reward, terminated, truncated, info = env.step(action)
                    all_actions.append(action)
                    observe()
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
    p.add_argument('--force-candidate-zero', action='store_true')
    args = p.parse_args()
    if args.num_candidates < 2 or not 1 <= args.execute_steps <= 10 or min(args.subgoal_steps, args.max_steps) < 1:
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
                        wm_sha256=file_sha256(args.wm_checkpoint))
        write_json(args.out/'protocol.json', protocol)
        write_json(args.out/'status.json', {'status': 'initializing', 'pid': os.getpid()})
        env = gs_gym.get_task_class('robowits/03-gap-retrieve-v0')(
            control_mode='EE_DELTA', observation_mode='JOINT', max_episode_steps=args.max_steps,
            genesis_backend='cpu', device='cpu', show_viewer=False, camera_type='rasterizer',
            eval_dataset_json=str(args.eval_json))
        if not np.isclose(env.dt, 1/30):
            raise ValueError('EEF/WM timing adapter requires native 30 Hz control')
        fk = Marvin(args.urdf)
        # Simulator must be reset/built before constructing the proxy scene.
        env.set_eval_dataset_idx(args.case)
        env.reset(seed=args.seed)
        proxy = EEFProxy(env, fk)
        policy = OpenPIClient(args.policy_python, args.openpi_runtime, args.policy_checkpoint,
                              args.out/'policy-queue', args.policy_gpu)
        wm = Worker(args)
        result = run_episode(env, policy, wm, VLMReward(args.vlm_socket), proxy, fk, args)
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
