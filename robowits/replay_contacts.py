"""Replay recorded actions for native contact evidence, verifying trajectory agreement."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .closed_loop import snapshot, same_state, state16, write_json


def contacts(env):
    robot = env._robot.robot_entity
    data = robot.get_contacts(exclude_self_contact=True)
    ga, gb = (data[k].detach().cpu().numpy().reshape(-1) for k in ('geom_a', 'geom_b'))
    la, lb = (data[k].detach().cpu().numpy().reshape(-1) for k in ('link_a', 'link_b'))
    forces = np.linalg.norm(data['force_a'].detach().cpu().numpy().reshape(-1, 3), axis=1)
    mask = np.ones(len(ga), bool) if 'valid_mask' not in data else data['valid_mask'].detach().cpu().numpy().reshape(-1)
    links = {link.idx: link.name for link in robot.links}
    result = {}
    for i in np.flatnonzero(mask & (forces > .01)):
        a_is_robot = robot.geom_start <= ga[i] < robot.geom_end
        other = gb[i] if a_is_robot else ga[i]
        link = links[int(la[i] if a_is_robot else lb[i])]
        for name, obj in env._entities.items():
            entity = obj['entity']
            if entity.geom_start <= other < entity.geom_end:
                row = result.setdefault(name, {'points': 0, 'force_norm_sum_N': 0., 'robot_links': []})
                row['points'] += 1
                row['force_norm_sum_N'] += float(forces[i])
                if link not in row['robot_links']:
                    row['robot_links'].append(link)
                break
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--modes', nargs='+', choices=['baseline', 'wm'], default=['baseline'])
    args = p.parse_args()
    root = args.run.resolve()
    protocol = json.loads((root/'protocol.json').read_text())
    sim = Path(protocol['sim_root'])
    sys.path.insert(0, str(sim))
    import gs_gym
    import torch
    env = gs_gym.get_task_class('robowits/03-gap-retrieve-v0')(
        device='cpu', genesis_backend='cpu', show_viewer=False, max_episode_steps=protocol['max_steps'],
        control_mode='EE_ABS', observation_mode='EE', eval_dataset_json=protocol['eval_json'])
    summaries = []
    try:
        for mode in args.modes:
            for case in protocol['cases']:
                directory = root/f'{mode}-{case:02d}'
                result = json.loads((directory/'result.json').read_text())
                if result['status'] != 'complete':
                    raise ValueError('Contact audit requires complete episodes')
                torch.manual_seed(1000+case)
                np.random.seed(1000+case)
                env.set_eval_dataset_idx(case)
                env.reset(seed=1000+case)
                if not same_state(snapshot(env), json.loads((directory/'initial.json').read_text())):
                    raise ValueError('Replay initial state does not match original rollout')
                source = np.load(directory/'trajectory.npz')
                events = []
                maximum_error = 0.
                heights = []
                finger_frames = wall_frames = 0
                for step, action in enumerate(source['actions'], 1):
                    _, _, terminated, truncated, info = env.step(action)
                    maximum_error = max(maximum_error, float(np.max(np.abs(state16(env._robot)-source['joints'][step]))))
                    if maximum_error > 1e-5:
                        raise RuntimeError(f'Replay diverged at {mode}/{case}/{step}: {maximum_error}')
                    current = contacts(env)
                    if current:
                        events.append({'step': step, 'contacts': current})
                    fingers = {'Link8_R', 'Link9_R', 'Link8_L', 'Link9_L'}
                    finger_frames += bool(fingers.intersection(current.get('lemon', {}).get('robot_links', [])))
                    wall_frames += any('boundary_block' in key for key in current)
                    heights.append(float(env._entities['lemon']['entity'].get_pos(relative=False).detach().cpu().numpy().reshape(3)[2]))
                    if (terminated or truncated) and step != len(source['actions']):
                        raise RuntimeError('Replay termination differs from original')
                if bool(info['is_success']) != result['success']:
                    raise RuntimeError('Replay task success differs from original')
                start_z = json.loads((directory/'initial.json').read_text())['objects']['lemon'][2]
                lifted = max(heights)-start_z
                if result['success']:
                    diagnosis = '成功'
                elif min(heights) < .70:
                    diagnosis = 'lemon 落到桌面以下'
                elif lifted < .025:
                    diagnosis = '指尖接触过 lemon，但未有效抬起' if finger_frames else '未记录到有效指尖接触，未建立抓取'
                else:
                    diagnosis = 'lemon 曾抬升，但未完成放置；掉落/误放需结合录像'
                summary = {'mode': mode, 'case': case, 'success': result['success'], 'steps': len(heights),
                           'max_joint_error': maximum_error, 'finger_lemon_contact_frames': finger_frames,
                           'robot_wall_contact_frames': wall_frames, 'maximum_lift_m': lifted,
                           'failure_stage': diagnosis, 'contact_force_threshold_N': .01}
                write_json(directory/'contact-replay.json', {'summary': summary, 'events': events})
                summaries.append(summary)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        env.close()
    lines = ['# 原生接触信息复核', '',
             '重放每回合已经执行的动作，并逐步验证测量关节轨迹误差不超过 0.00001。这里的接触数据仅用于事后诊断，不参与世界模型评分。', '',
             '| 模式 | 场景 | 指尖接触 lemon 的帧数 | 与挡板接触帧数 | 最高抬升 / mm | 失败阶段 |',
             '|---|---|---:|---:|---:|---|']
    for r in summaries:
        lines.append(f"| {r['mode']} | {r['case']} | {r['finger_lemon_contact_frames']} | {r['robot_wall_contact_frames']} | {r['maximum_lift_m']*1000:.2f} | {r['failure_stage']} |")
    lines += ['', '接触帧数要求原生接触点力大于 0.01 N；连续接触会跨多帧，因此不等于独立碰撞次数。推挡板属于任务的一部分，不能把所有挡板接触都标成碰撞失败。每回合的接触对象、机器人连杆和力记录见 contact-replay.json。']
    (root/'contact-review.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
