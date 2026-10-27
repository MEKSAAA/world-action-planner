"""Post-hoc contact diagnosis for EEF runs; never used for action ranking."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from .closed_loop import snapshot, same_state, state16, write_json
from .replay_contacts import contacts


def audit(env, directory, source_name, result_override=None):
    result = result_override or json.loads((directory/'result.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('Only completed trajectories can be diagnosed')
    initial = json.loads((directory/'initial.json').read_text())
    if not same_state(snapshot(env), initial):
        raise ValueError('Replay initial state differs')
    source = np.load(directory/source_name)
    events, heights = [], []
    finger_frames = wall_frames = 0
    max_error = 0.
    for step, action in enumerate(source['actions'], 1):
        _, _, terminated, truncated, info = env.step(action)
        max_error = max(max_error, float(np.max(np.abs(state16(env._robot)-source['joints'][step]))))
        if max_error > 1e-5:
            raise RuntimeError(f'Replay diverged at step {step}: {max_error}')
        current = contacts(env)
        if current:
            events.append(dict(step=step, contacts=current))
        finger_frames += bool({'Link8_R','Link9_R','Link8_L','Link9_L'}.intersection(
            current.get('lemon', {}).get('robot_links', [])))
        wall_frames += any('boundary_block' in name for name in current)
        heights.append(float(env._entities['lemon']['entity'].get_pos(relative=False).detach().cpu().numpy().reshape(3)[2]))
        if (terminated or truncated) and step != len(source['actions']):
            raise RuntimeError('Replay termination differs')
    if bool(info['is_success']) != result['success']:
        raise RuntimeError('Replay success differs')
    lift = max(heights)-initial['objects']['lemon'][2]
    if result['success']:
        diagnosis = 'success'
    elif min(heights) < .70:
        diagnosis = 'lemon_below_table'
    elif lift < .025:
        diagnosis = 'finger_contact_without_lift' if finger_frames else 'no_finger_contact_or_lift'
    else:
        diagnosis = 'lift_without_successful_placement_review_video'
    summary = dict(success=result['success'], steps=len(heights), max_joint_error=max_error,
                   finger_lemon_contact_frames=finger_frames, robot_wall_contact_frames=wall_frames,
                   maximum_lift_m=lift, minimum_lemon_z_m=min(heights), failure_stage=diagnosis,
                   contact_force_threshold_N=.01, lift_threshold_m=.025)
    write_json(directory/'contact-replay.json', dict(summary=summary, events=events, heights_m=heights))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--sim-root', type=Path, required=True)
    p.add_argument('--native', action='store_true')
    p.add_argument('--dataset', type=Path)
    p.add_argument('--annotations', type=Path)
    args = p.parse_args()
    sys.path[:0] = [str(args.sim_root), str(args.sim_root/'third_party/lerobot/src')]
    if args.native:
        import pyarrow.parquet as pq
        spec = importlib.util.spec_from_file_location('amber_eval', args.sim_root/'scripts/robowits/eval/eval_pi05_lemon.py')
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        episodes = json.loads((args.run/'results.json').read_text())
        rows = {int(r['episode_index']):r for r in pq.read_table(args.dataset/'meta/episodes/chunk-000/file-000.parquet').to_pylist()}
        catalog = {'dataset':args.dataset, 'by_id':{}}
        for result in episodes:
            ep = result['episode']
            rec = json.loads((args.annotations/f'reconstruction/{ep}/reconstruction.json').read_text())
            catalog['by_id'][ep] = dict(index=ep, length=int(rows[ep]['length']), recording_path=rec['recording_path'],
                file_index=int(rows[ep]['data/file_index']), instruction='Place the lemon on the green target area.')
        env = native._make_env('cpu', max(r['length'] for r in catalog['by_id'].values()), 'eef')
        scene = native.RecordedScene(env, catalog, args.run/'contact-audit-scenes', 'eef')
        summaries = []
        try:
            for result in episodes:
                ep = result['episode']
                scene.prepare(ep); scene.reset()
                directory = args.run/f'episode-{ep}'
                # Native evaluator predates the common status field.
                wrapped = dict(result, status='complete' if result['error'] is None else 'error')
                # Pass a result override without altering the saved native result.
                summary = audit_native(env, directory, wrapped)
                summaries.append(dict(episode=ep, **summary))
                print(json.dumps(summaries[-1]), flush=True)
        finally:
            env.close()
        write_json(args.run/'contact-summary.json', summaries)
    else:
        import gs_gym
        protocol = json.loads((args.run/'protocol.json').read_text())
        env = gs_gym.get_task_class('robowits/03-gap-retrieve-v0')(
            control_mode='EE_DELTA', observation_mode='JOINT', max_episode_steps=protocol['max_steps'],
            genesis_backend='cpu', device='cpu', show_viewer=False, camera_type='rasterizer',
            eval_dataset_json=protocol['eval_json'])
        try:
            env.set_eval_dataset_idx(protocol['case']); env.reset(seed=protocol['seed'])
            print(json.dumps(audit(env, args.run, 'trajectory.npz')), flush=True)
        finally:
            env.close()


def audit_native(env, directory, result):
    return audit(env, directory, 'native.npz', result_override=result)


if __name__ == '__main__':
    main()
