"""Post-hoc native replay for lemon and cube episodes; no policy or WM queries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .closed_loop import file_sha256, same_state, snapshot, state16, write_json
from .geometry import pack
from .multigoal_episode_report import terminal_outcome
from .replay_contacts import contacts


OBJECTS = {'robowits/03-gap-retrieve-v0': 'lemon', 'robowits/02-retrieve-cube-v0': 'cube'}
FINGERS = {'Link8_R', 'Link9_R', 'Link8_L', 'Link9_L'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-diagnostic', action='store_true',
                        help='Verify a completed short diagnostic, without counting it as task success/failure')
    args = parser.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    result = json.loads((args.run/'result.json').read_text())
    outcome = terminal_outcome(result, protocol['max_steps'])
    diagnostic = result['status'] == 'diagnostic_complete' and args.allow_diagnostic
    if outcome is None and not diagnostic:
        raise ValueError('Only terminal episodes or explicitly allowed completed diagnostics can be replayed')
    target = OBJECTS[protocol['task_id']]
    sim = Path(protocol['sim_root'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=sim, text=True).strip()
    if commit != protocol['sim_commit']:
        raise ValueError('Simulator revision differs from the recorded run')
    decisions = json.loads((args.run/'decisions.json').read_text())
    hashes = {d['step']: d['observation_sha256'] for d in decisions}
    with np.load(args.run/'trajectory.npz') as data:
        actions, joints = data['actions'].copy(), data['joints'].copy()
    if not len(actions) or len(actions) != result['steps'] or len(joints) != len(actions)+1:
        raise ValueError('Recorded result and trajectory lengths disagree')
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'protocol.json', dict(source=str(args.run.resolve()), task_id=protocol['task_id'],
        source_sha256={name: file_sha256(args.run/name) for name in
                       ('protocol.json', 'result.json', 'trajectory.npz', 'decisions.json')},
        sim_commit=commit, audit_sha256=file_sha256(Path(__file__)),
        scope='post-hoc replay only; no simulator future or object coordinates enter online ranking'))
    sys.path[:0] = [str(sim), str(sim/'third_party/lerobot/src')]
    import gs_gym
    env = None
    try:
        env = gs_gym.get_task_class(protocol['task_id'])(control_mode='EE_DELTA', observation_mode='JOINT',
            max_episode_steps=protocol['max_steps'], genesis_backend='cpu', device='cpu',
            show_viewer=False, camera_type='rasterizer', eval_dataset_json=protocol.get('eval_json'))
        env.set_eval_dataset_idx(protocol['case'])
        obs, _ = env.reset(seed=protocol['seed'])
        initial = snapshot(env)
        if not same_state(initial, json.loads((args.run/'initial.json').read_text())):
            raise ValueError('Replay initial state differs')
        object_names = list(initial['objects'])
        states, events = [initial], []
        max_error = 0.
        checked_images = 0
        for step, action in enumerate(actions):
            if step in hashes:
                if hashlib.sha256(pack(obs['pixels']).tobytes()).hexdigest() != hashes[step]:
                    raise ValueError(f'Replay decision RGB differs at step {step}')
                checked_images += 1
            obs, _, terminated, truncated, info = env.step(action)
            max_error = max(max_error, float(np.max(np.abs(state16(env._robot)-joints[step+1]))))
            if max_error > 1e-5:
                raise ValueError(f'Replay joint trajectory differs at step {step+1}')
            states.append(snapshot(env))
            events.append({'step': step+1, 'contacts': contacts(env), 'success': bool(info['is_success'])})
            if (terminated or truncated) and step+1 != len(actions):
                raise ValueError('Replay reached an earlier native endpoint')
        expected_success = result['success_at_stop'] if diagnostic else outcome
        if bool(info['is_success']) != expected_success:
            raise ValueError('Replay native success differs')
        if bool(terminated) != result['terminated'] or bool(truncated) != result['truncated']:
            raise ValueError('Replay endpoint flags differ')
        positions = np.array([[s['objects'][name] for name in object_names] for s in states])
        orientations = np.array([[s['orientations'][name] for name in object_names] for s in states])
        item = positions[:, object_names.index(target)]
        goal = positions[:, object_names.index('target_area')]
        distances = np.linalg.norm((item-goal)[:, :2], axis=1)
        contact_counts = {name: sum(name in e['contacts'] for e in events) for name in object_names}
        finger_frames = sum(bool(FINGERS.intersection(e['contacts'].get(target, {}).get('robot_links', [])))
                            for e in events)
        summary = dict(status='diagnostic_verified' if diagnostic else 'complete_verified',
            native_success=outcome, success_at_stop=bool(info['is_success']), steps=len(actions),
            task_object=target, terminated=bool(terminated), truncated=bool(truncated),
            max_joint_error=max_error, decision_images_verified=checked_images,
            finger_object_contact_frames=finger_frames, robot_object_contact_frames=contact_counts,
            maximum_lift_m=float(item[:, 2].max()-item[0, 2]), minimum_object_z_m=float(item[:, 2].min()),
            initial_target_xy_distance_m=float(distances[0]), final_target_xy_distance_m=float(distances[-1]),
            minimum_target_xy_distance_m=float(distances.min()), contact_force_threshold_N=.01)
        np.savez_compressed(args.out/'objects.npz', names=np.array(object_names), positions=positions,
                            orientations=orientations, target_xy_distance=distances)
        write_json(args.out/'contacts.json', events)
        write_json(args.out/'final-state.json', states[-1])
        write_json(args.out/'summary.json', summary)
        lines = ['# 物理重放核查 / Native replay audit', '',
            f'任务：`{protocol["task_id"]}`；物体：{target}；已执行 {len(actions)} 步。',
            f'完整回合原生成功：{outcome}；状态：`{summary["status"]}`。', '',
            f'最大关节误差 {max_error:.6g}；逐决策真实画面哈希通过 {checked_images} 次。',
            f'物体最大抬升 {summary["maximum_lift_m"]*1000:.1f} mm；指尖接触 {finger_frames} 帧。',
            f'到目标中心平面距离：{distances[0]*1000:.1f} → {distances[-1]*1000:.1f} mm。', '',
            '接触、抬升、距离都不是任务成功判据，也不能单独证明稳定抓握或方块已脱离容器。'
            '成功由原生模拟器判定；动作不足完整回合的诊断保留 native_success=null。', '',
            '[统计](summary.json) · [逐帧接触](contacts.json) · [逐帧物体轨迹](objects.npz) · [来源](protocol.json)']
        (args.out/'review.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(summary), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
