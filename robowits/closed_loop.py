"""Paired Genesis131 lemon policy / image-only WM action-selection pilot.

The native simulator is advanced only for executed actions. Future robot
conditions come from a separate robot-only scene, never task-object rollout.
"""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from .geometry import Marvin, pack
from .closed_loop_score import visual_score, choose

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def state16(robot):
    q = robot.joint_positions.detach().cpu().numpy().reshape(-1)
    if q.shape != (18,):
        raise ValueError(f'Expected single robot, got {q.shape}')
    return np.r_[q[:7], q[7:9].sum(), q[9:16], q[16:18].sum()]


def snapshot(env):
    return {'robot': state16(env._robot).tolist(),
            'qpos': env._robot.robot_entity.get_qpos().detach().cpu().numpy().reshape(-1).tolist(),
            'velocities': env._robot.robot_entity.get_dofs_velocity().detach().cpu().numpy().reshape(-1).tolist(),
            'orientations': {name: data['entity'].get_quat(relative=False).detach().cpu().numpy().reshape(-1).tolist()
                             for name, data in env._entities.items()},
            'objects': {name: data['entity'].get_pos(relative=False).detach().cpu().numpy().reshape(-1).tolist()
                        for name, data in env._entities.items()}}


def same_state(a, b):
    return (np.allclose(a['robot'], b['robot'], atol=1e-6, rtol=0)
            and np.allclose(a['qpos'], b['qpos'], atol=1e-6, rtol=0)
            and np.allclose(a['velocities'], b['velocities'], atol=1e-6, rtol=0)
            and a['objects'].keys() == b['objects'].keys()
            and all(np.allclose(a['orientations'][n], b['orientations'][n], atol=1e-6, rtol=0) for n in a['objects'])
            and all(np.allclose(a['objects'][n], b['objects'][n], atol=1e-6, rtol=0) for n in a['objects']))


class RobotProxy:
    def __init__(self, env, fk):
        from gs_gym.robots.bimanual_marvin import BimanualMarvinRobot
        from gs_gym.scenes.flat_scene import FlatScene
        self.scene = FlatScene(1, deepcopy(env._args.scene_args), enable_collision=True, add_ground_plane=False)
        self.robot = BimanualMarvinRobot(1, self.scene.scene, deepcopy(env._args.robot_args), env.device,
                                       control_mode=env.control_mode, urdf_path=env._robot._urdf_path)
        self.scene.build()
        self.robot.init_dof_indices()
        self.fk = fk

    def predict(self, env, actions):
        import torch
        before = snapshot(env)
        step = env._step_count
        self.scene.scene.reset()
        source = env._robot.robot_entity
        target = self.robot.robot_entity
        target.set_qpos(source.get_qpos().clone(), zero_velocity=False)
        target.set_dofs_velocity(source.get_dofs_velocity().clone())
        self.robot._ik_solver = None
        states = []
        for action in actions:
            self.robot.apply_action(torch.as_tensor(action, device=self.robot.device, dtype=torch.float32)[None])
            self.scene.scene.step()
            states.append(state16(self.robot))
        if step != env._step_count or not same_state(before, snapshot(env)):
            raise RuntimeError('Robot-only projection mutated the evaluated scene')
        times = np.arange(1, 21)/20
        source_times = np.arange(1, len(states)+1)*env.dt
        indices = np.abs(source_times[:, None]-times).argmin(axis=0)
        if np.max(np.abs(source_times[indices]-times)) > env.dt/2:
            raise ValueError('Candidate does not cover the 1-second WM horizon')
        self.last_states = np.asarray(states)
        return np.stack([pack(self.fk.skeleton(states[i])) for i in indices])


class Worker:
    def __init__(self, args):
        suffix = f'-resume-{time.time_ns()}' if args.resume else ''
        self.queue = args.out/f'queue{suffix}'
        self.queue.mkdir()
        cmd = [str(args.wm_python), '-u', '-m', 'robowits.closed_loop_worker', '--queue', str(self.queue),
               '--checkpoint', str(args.wm_checkpoint), '--ckpts', str(args.ckpts), '--wan', str(args.wan)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.wm_gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        # Keep simulator-specific packages out of the model environment.
        env.pop('PYTHONPATH', None)
        with (args.out/f'wm-worker{suffix}.log').open('w') as log:
            self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            self.wait(self.queue/'ready', 600)
        except BaseException:
            self.proc.terminate()
            self.proc.wait()
            raise
        self.calls = 0

    def wait(self, file, timeout):
        start = time.monotonic()
        while not file.exists():
            if self.proc.poll() is not None:
                raise RuntimeError('World-model worker exited; see wm-worker.log')
            if time.monotonic()-start > timeout:
                raise TimeoutError(f'Waiting for {file}')
            time.sleep(.2)

    def predict(self, directory, history, past, future, seed):
        self.calls += 1
        inp, out = directory/'input.npz', directory/'prediction.npy'
        np.savez(inp, history=history, past=past, future=future)
        name = f'{self.calls:05d}'
        write_json(self.queue/f'request-{name}.json', {'input': str(inp), 'output': str(out), 'seed': seed})
        response = self.queue/f'response-{name}.json'
        self.wait(response, 600)
        info = json.loads(response.read_text())
        if info['status'] != 'complete':
            raise RuntimeError(info['traceback'])
        return np.load(out), info['seconds']

    def close(self):
        (self.queue/'stop').touch()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=10)


def diagnose(samples, success):
    if success:
        return '成功（原生环境判据）'
    if any(s['lemon'][2] < .70 for s in samples):
        return 'lemon 掉落桌下'
    held = [i for i, s in enumerate(samples) if s['held']]
    if held:
        if not samples[-1]['held'] and np.linalg.norm(np.array(samples[-1]['lemon'])[:2]-samples[-1]['target'][:2]) > .10:
            return '曾持物，结束时未持物且偏离目标（疑似掉落/误放，需看视频）'
        return '曾持物，但未满足放置成功条件'
    return '未检测到闭爪接近且抬起物体（抓空/受阻原因需看视频）'


def rollout(env, policy, pre, post, fk, proxy, worker, args, case, mode, baseline_initial=None):
    import torch
    import imageio.v2 as imageio
    from lerobot.envs.utils import preprocess_observation
    from gs_gym.envs.robowits.utils import check_being_held
    directory = args.out/f'{mode}-{case:02d}'
    directory.mkdir()
    start = time.monotonic()
    result = {'case': case, 'mode': mode, 'seed': 1000+case, 'status': 'running', 'success': None}
    write_json(directory/'result.json', result)
    writer = None
    actions, diagnostics, decisions, scores, joint_states = [], [], [], [], []
    try:
        torch.manual_seed(1000+case)
        np.random.seed(1000+case)
        env.set_eval_dataset_idx(case)
        obs, info = env.reset(seed=1000+case)
        from .prepare_policy_data import ee14
        expected_ee = ee14(fk, state16(env._robot)[None])[0]
        observed_ee = np.asarray(obs['agent_pos'])
        if not np.allclose(expected_ee, observed_ee, atol=2e-5, rtol=0):
            raise RuntimeError('Policy-data EE conversion differs from native environment observation')
        write_json(directory/'interface-check.json', {'frame': 'robot-base-relative',
                   'max_error': float(np.max(np.abs(expected_ee-observed_ee))),
                   'converted': expected_ee.tolist(), 'native': observed_ee.tolist()})
        initial = snapshot(env)
        write_json(directory/'initial.json', initial)
        if baseline_initial is not None and not same_state(initial, baseline_initial):
            raise RuntimeError('Paired initial robot/object positions differ')
        policy.reset()
        history = deque(maxlen=180)
        def observe():
            joints = state16(env._robot)
            rgb = pack(obs['pixels'])
            history.append((env._step_count*env.dt, rgb, pack(fk.skeleton(joints))))
            joint_states.append(joints)
            writer.append_data(rgb)
            if env._step_count % 5 == 0:
                objects = env.collect_objs_info()
                bounds = np.asarray(objects['lemon']['bounds'])
                closed_near = False
                for pose, opening in ((env._robot.right_ee_pose, joints[7]), (env._robot.left_ee_pose, joints[15])):
                    xyz = pose[0, :3].detach().cpu().numpy()
                    closed_near |= bool(opening < .07 and np.all(xyz >= bounds[0]-.03) and np.all(xyz <= bounds[1]+.03))
                lifted = float(objects['lemon']['pos'][2]) > initial['objects']['lemon'][2]+.025
                diagnostics.append({'step': env._step_count,
                                    'lemon': np.asarray(objects['lemon']['pos']).tolist(),
                                    'target': np.asarray(objects['target_area']['pos']).tolist(),
                                    'near_gripper_native': bool(check_being_held(objects['lemon'], env._robot)),
                                    'held': bool(closed_near and lifted), 'lifted': lifted})
        writer = imageio.get_writer(directory/'execution.mp4', fps=1/env.dt, codec='libx264')
        observe()
        terminated = truncated = False
        chunk = 0
        policy_seconds = wm_seconds = proxy_seconds = 0.
        while not (terminated or truncated):
            batch = preprocess_observation(obs)
            batch['task'] = [env.task_description]
            batch = pre(batch)
            use_wm = mode == 'wm' and chunk > 0 and chunk % 3 == 0 and chunk <= 30
            candidates = []
            started = time.monotonic()
            for k in range(2 if use_wm else 1):
                # Independent, paired per-decision seeds; WM RNG cannot alter policy RNG.
                seed = 100000+case*10000+chunk*2+k
                generator = torch.Generator(device=policy.config.device).manual_seed(seed)
                noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim),
                                    device=policy.config.device, generator=generator)
                with torch.inference_mode():
                    pred = policy.predict_action_chunk(batch, noise=noise)
                    action = post(pred)[0, :30].detach().cpu().numpy()
                if action.shape != (30, 14) or not np.isfinite(action).all():
                    raise ValueError('Expected finite native EE_ABS action chunk (30,14)')
                candidates.append(action)
            policy_seconds += time.monotonic()-started
            selected = 0
            decision = {'step': env._step_count, 'chunk': chunk, 'wm_used': use_wm, 'selected': 0}
            if use_wm:
                real_times = np.array([h[0] for h in history])
                times = real_times[-1]+np.arange(-79, 1)/20
                indices = np.abs(real_times[:, None]-times).argmin(axis=0)
                rgb = np.stack([history[i][1] for i in indices])
                past = np.stack([history[i][2] for i in indices])
                evaluations = []
                for k, candidate in enumerate(candidates):
                    target = directory/f'decision-{chunk:03d}-candidate-{k}'
                    target.mkdir()
                    np.save(target/'actions.npy', candidate)
                    started = time.monotonic()
                    future = proxy.predict(env, candidate)
                    np.save(target/'robot_only_states.npy', proxy.last_states)
                    proxy_seconds += time.monotonic()-started
                    predicted, seconds = worker.predict(target, rgb, past, future, 200000+case*100+chunk)
                    wm_seconds += seconds
                    metric = visual_score(predicted)
                    write_json(target/'score.json', metric)
                    evaluations.append(metric)
                selected = choose(evaluations)
                decision.update(selected=selected, scores=evaluations)
            decisions.append(decision)
            write_json(directory/'decisions.json', decisions)
            for action in candidates[selected]:
                obs, reward, terminated, truncated, info = env.step(action)
                actions.append(action)
                scores.append(float(info.get('score', reward)))
                observe()
                if terminated or truncated:
                    break
            print(json.dumps({'mode': mode, 'case': case, 'step': env._step_count,
                              'selected': selected, 'score': scores[-1], 'success': bool(info['is_success'])}), flush=True)
            chunk += 1
        success = bool(info['is_success'])
        result.update(status='complete', success=success, steps=env._step_count,
                      max_score=max(scores), failure_type=diagnose(diagnostics, success),
                      policy_seconds=policy_seconds, wm_seconds=wm_seconds, proxy_seconds=proxy_seconds,
                      wm_decisions=sum(d['wm_used'] for d in decisions),
                      changed_choices=sum(d['selected'] != 0 for d in decisions))
    except BaseException:
        result.update(status='error', error=traceback.format_exc())
        raise
    finally:
        if writer is not None:
            writer.close()
        result['wall_seconds'] = time.monotonic()-start
        write_json(directory/'diagnostics.json', diagnostics)
        write_json(directory/'result.json', result)
        np.savez_compressed(directory/'trajectory.npz', actions=np.asarray(actions),
                            scores=np.asarray(scores), joints=np.asarray(joint_states))
    return result


def report(out, results):
    lines = ['# Lemon 策略与世界模型筛选：配对闭环实验', '',
             '使用 Genesis131 原生成功判据；这是小样本探索实验，不是 WAP 论文规划器复现。', '',
             '| 场景 | 模式 | 成功 | 步数 | 最高进度 | 耗时（秒） | WM 改选次数 | 结果诊断 |',
             '|---|---|---|---:|---:|---:|---:|---|']
    for r in results:
        lines.append(f"| {r['case']} | {r['mode']} | {r['success']} | {r['steps']} | {r['max_score']:.3f} | {r['wall_seconds']:.1f} | {r['changed_choices']} | {r['failure_type']} |")
    for mode in ('baseline', 'wm'):
        rows = [r for r in results if r['mode'] == mode]
        if rows:
            lines += ['', f"{mode}：{sum(r['success'] for r in rows)}/{len(rows)} 成功；平均 {np.mean([r['wall_seconds'] for r in rows]):.1f} 秒/回合。"]
    lines += ['', '两组均每 30 个原生控制步重规划，使用相同的策略候选 0 随机种子。WM 在第 90、180、…、900 步（回合尚未结束时）比较两个策略采样；其他时间沿用候选 0。',
              '', '评分仅取预测最后五帧的主视角黄色 lemon 与绿色目标区域中心距离；物体缺失/重复或目标不可见时判无效，双方均有效且候选 1 改善超过 0.02 图像宽度才改选。该颜色启发式不等于成功判据，也不充分评价移开障碍物的长期价值。',
              '', '失败类型每五步采样：夹爪宽度小于 0.07 m、末端位于 lemon 包围盒外扩 0.03 m 内、lemon 高于初始中心 0.025 m，三者同时成立才标为疑似持物。这不是接触力证明；具体抓空、碰撞、掉落原因仍需录像核对。',
              '', '当前状态与预测机器人的状态用于绘制骨架；未来骨架由无任务物体的独立机器人场景产生。世界模型及评分器不接收真实未来 RGB、任务物体状态或奖励。',
              '', '详见 protocol.json、每回合 execution.mp4、initial.json、decisions.json 与 diagnostics.json。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def validate_resume_protocol(previous, current):
    keys = ('sim_commit', 'eval_sha256', 'policy_weights_sha256', 'wm_checkpoint_sha256',
            'score_sha256', 'cases', 'max_steps', 'baseline_only', 'control_mode',
            'observation_mode', 'policy_chunk_executed', 'wm_future_frames',
            'wm_history_frames', 'wm_fps', 'urdf', 'wan', 'ckpts')
    changed = [k for k in keys if previous.get(k) != current.get(k)]
    if changed:
        raise ValueError(f'Resume would change experiment protocol: {changed}')


def completed_or_archive(out, mode, case):
    directory = out/f'{mode}-{case:02d}'
    result = directory/'result.json'
    if result.exists():
        r = json.loads(result.read_text())
        if r['status'] == 'complete':
            return r
    if directory.exists():
        archive = out/'interrupted'
        archive.mkdir(exist_ok=True)
        directory.rename(archive/f'{mode}-{case:02d}-{time.time_ns()}')
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-root', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--eval-json', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--wm-python', type=Path, required=True)
    p.add_argument('--wm-checkpoint', type=Path, required=True)
    p.add_argument('--ckpts', type=Path, required=True)
    p.add_argument('--wan', type=Path, required=True)
    p.add_argument('--wm-gpu', default='0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cases', type=int, nargs='+', default=[10, 11, 12, 13, 14])
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--baseline-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='Reuse completed episodes after validating protocol; archive and restart interrupted episodes')
    args = p.parse_args()
    args.out = args.out.resolve()
    args.wm_checkpoint = args.wm_checkpoint.resolve()
    inventory = json.loads(args.eval_json.read_text())
    if len(set(args.cases)) != len(args.cases) or any(c < 0 or c >= len(inventory) for c in args.cases):
        p.error('Require distinct valid evaluation scene indices')
    if args.resume and not (args.out/'protocol.json').exists():
        p.error('Resume requires an existing protocol.json')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    run_lock = (args.out/'.run.lock').open('w')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(args.out/'status.json', {'status': 'initializing', 'pid': os.getpid()})
    sys.path[:0] = [str(args.sim_root), str(args.sim_root/'third_party/lerobot/src')]
    import torch
    import gs_gym
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    torch.set_num_threads(4)
    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.device = 'cuda'
    cfg.pretrained_path = str(args.policy)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(sim_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.sim_root, text=True).strip(),
                    wap_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    eval_sha256=hashlib.sha256(args.eval_json.read_bytes()).hexdigest(),
                    wm_checkpoint_sha256=file_sha256(args.wm_checkpoint),
                    policy_weights_sha256=file_sha256(args.policy/'model.safetensors'),
                    control_mode='EE_ABS', observation_mode='EE', policy_chunk_executed=30,
                    wm_future_frames=20, wm_history_frames=80, wm_fps=20,
                    score_rule='closed_loop_score.py; yellow HSV 12:40, green HSV 40:90, margin .02',
                    score_sha256=hashlib.sha256((ROOT/'robowits/closed_loop_score.py').read_bytes()).hexdigest())
    if args.resume:
        validate_resume_protocol(json.loads((args.out/'protocol.json').read_text()), protocol)
        write_json(args.out/f'resume-protocol-{time.time_ns()}.json', protocol)
    else:
        write_json(args.out/'protocol.json', protocol)
    policy = PI05Policy.from_pretrained(args.policy, config=cfg).to('cuda').eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(args.policy),
                                        preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    env = gs_gym.get_task_class('robowits/03-gap-retrieve-v0')(
        device='cpu', genesis_backend='cpu', show_viewer=False, max_episode_steps=args.max_steps,
        control_mode='EE_ABS', observation_mode='EE', eval_dataset_json=str(args.eval_json))
    fk = Marvin(args.urdf)
    worker = proxy = None
    results = []
    try:
        # Run every baseline first; do not choose test cases based on outcomes.
        for case in args.cases:
            write_json(args.out/'status.json', {'status': 'running', 'mode': 'baseline', 'case': case, 'pid': os.getpid()})
            previous = completed_or_archive(args.out, 'baseline', case) if args.resume else None
            results.append(previous or rollout(env, policy, pre, post, fk, None, None, args, case, 'baseline'))
            report(args.out, results)
        if not args.baseline_only:
            proxy = RobotProxy(env, fk)
            worker = Worker(args)
            for case in args.cases:
                write_json(args.out/'status.json', {'status': 'running', 'mode': 'wm', 'case': case, 'pid': os.getpid()})
                initial = json.loads((args.out/f'baseline-{case:02d}/initial.json').read_text())
                previous = completed_or_archive(args.out, 'wm', case) if args.resume else None
                results.append(previous or rollout(env, policy, pre, post, fk, proxy, worker, args, case, 'wm', initial))
                report(args.out, results)
        write_json(args.out/'status.json', {'status': 'complete', 'results': results})
    except BaseException:
        write_json(args.out/'status.json', {'status': 'error', 'error': traceback.format_exc(), 'completed': results})
        raise
    finally:
        if worker is not None:
            worker.close()
        if proxy is not None:
            proxy.scene.scene.destroy()
        env.close()


if __name__ == '__main__':
    main()
