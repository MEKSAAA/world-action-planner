"""Offline physical branches for saved multi-goal candidates, never online value inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np

from .closed_loop import snapshot, same_state, state16, write_json
from .eef_actions import control_action
from .geometry import pack
from .replay_contacts import contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--decision', type=int, default=0)
    args = parser.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    decisions = json.loads((args.run/'decisions.json').read_text())
    decision = decisions[args.decision]
    sim = Path(protocol['sim_root'])
    sys.path[:0] = [str(sim), str(sim/'third_party/lerobot/src')]
    import gs_gym
    env = gs_gym.get_task_class(protocol['task_id'])(
        control_mode='EE_DELTA', observation_mode='JOINT', max_episode_steps=protocol['max_steps'],
        genesis_backend='cpu', device='cpu', show_viewer=False, camera_type='rasterizer',
        eval_dataset_json=protocol.get('eval_json'))
    args.out.mkdir(parents=True, exist_ok=False)
    trajectory = np.load(args.run/'trajectory.npz')
    targets = np.load(args.run/f'decision-{args.decision:04d}/targets.npy')
    rows = []
    try:
        for i, chunk in enumerate(targets):
            env.set_eval_dataset_idx(protocol['case'])
            obs, _ = env.reset(seed=protocol['seed'])
            if not same_state(snapshot(env), json.loads((args.run/'initial.json').read_text())):
                raise ValueError('Audit initial state differs')
            prefix_error = 0.
            for step, action in enumerate(trajectory['actions'][:decision['step']], 1):
                obs, _, terminated, truncated, _ = env.step(action)
                prefix_error = max(prefix_error, float(np.max(np.abs(state16(env._robot)-trajectory['joints'][step]))))
                if terminated or truncated or prefix_error > 1e-5:
                    raise ValueError('Audit prefix diverged')
            current = pack(obs['pixels'])
            if hashlib.sha256(current.tobytes()).hexdigest() != decision['observation_sha256']:
                raise ValueError('Audit RGB differs before branch')
            folder = args.out/f'candidate-{i}'
            folder.mkdir()
            frames, states, events = [current], [], []
            initial = snapshot(env)
            for target in chunk:
                obs, _, terminated, truncated, info = env.step(control_action(obs['agent_pos_ee'], target))
                frames.append(pack(obs['pixels']))
                states.append(state16(env._robot))
                events.append({'step': env._step_count, 'success': bool(info['is_success']),
                    'state': snapshot(env), 'contacts': contacts(env)})
                if terminated or truncated:
                    break
            with imageio.get_writer(folder/'actual.mp4', fps=1/env.dt, codec='libx264') as writer:
                for frame in frames:
                    writer.append_data(frame)
            indices = np.abs(np.arange(1,len(states)+1)[:,None]*env.dt-np.arange(1,7)[None,:]/20).argmin(axis=0)
            if len(states) >= 9:
                np.save(folder/'truth.npy', np.stack(frames)[indices+1])
            np.save(folder/'native-states.npy', states)
            imageio.imwrite(folder/'current.png', current)
            summary = {'candidate': i, 'selected': i == decision['selected'],
                'language': decision['candidate_map'][i]['goal']['language'],
                'prefix_joint_error': prefix_error, 'steps': len(states),
                'native_success': bool(info['is_success']),
                'object_displacement_m': {k: float(np.linalg.norm(np.asarray(v)-initial['objects'][k]))
                    for k,v in events[-1]['state']['objects'].items()},
                'native_frame_indices': (indices+1).tolist()}
            write_json(folder/'physics.json', {'summary': summary, 'initial': initial, 'events': events})
            rows.append(summary)
            write_json(args.out/'summary.json', rows)
            print(json.dumps(summary), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
