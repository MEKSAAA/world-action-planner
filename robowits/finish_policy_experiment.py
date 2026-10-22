"""Finish a running lemon training job, then execute a predefined paired pilot."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from .closed_loop import same_state, write_json

ROOT = Path(__file__).resolve().parents[1]


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def render(root, pilot, trained, cases, training_log, status):
    lines = ['# Lemon：真实任务基线与世界模型动作筛选', '',
             f"流程状态：**{status['stage']}**。更新时间：{time.strftime('%Y-%m-%d %H:%M:%S %Z')}。", '',
             '成功仅依据 Genesis131 原生环境判据。进度分、视频预测质量和训练 loss 均不等于成功率。', '',
             '## 原有策略检查', '',
             '现有 PI05 正式基线固定场景 10–14：0/5 成功。ACT 的独立开发场景 0–4：0/5 成功。', '',
             f'[PI05 配对实验](paired-five/report.md) · [原策略失败复核](paired-five/failure-review.md) · [ACT 结果](act-development-check/eval_info.json)', '',
             '## Lemon 单任务微调', '',
             '从现有 PI05 开始，只更新动作模块；45 条训练轨迹（105–149），2 条验证轨迹（103/104）；100–102 不参与。批量 8，固定 2,000 步。EE 位置以机器人基座为原点，并与实际环境观测和原生 IK 验证。']
    if training_log.exists():
        matches = re.findall(r'step:\d+[^\n]*', training_log.read_text(errors='replace'))
        if matches:
            lines += ['', f'最近训练记录：`{matches[-1]}`']
    lines += ['', '先前世界坐标版本的数据与训练试跑已标为无效，不用于评估；GPU 显存不足的试跑也不算完成训练。有效训练路径：`pi05-lemon-base-expert-2k/`。', '',
              '[留出轨迹动作预测检查](policy-validation.json)（完成训练后本地运行，不作为任务成功率）。', '',
              '## 固定场景的策略比较', '',
              f'微调后的场景在结果出来前固定为 {cases}，每回合最多 1,000 步；仍用相同候选生成、评分和执行规则。未完成回合显示“待完成”，不会计为失败。', '',
              '| 场景 | 原策略 | 原策略 + WM | 微调策略 | 微调策略 + WM |',
              '|---|---|---|---|---|']
    for case in cases:
        values = []
        for directory, mode in ((pilot, 'baseline'), (pilot, 'wm'), (trained, 'baseline'), (trained, 'wm')):
            record = read(directory/f'{mode}-{case:02d}/result.json', {})
            values.append('成功' if record.get('success') else '失败' if record.get('status') == 'complete' else '运行错误' if record.get('status') == 'error' else '待完成')
        lines.append(f"| {case} | {' | '.join(values)} |")
    for label, directory, mode in (('原策略', pilot, 'baseline'), ('原策略 + WM', pilot, 'wm'), ('微调策略', trained, 'baseline'), ('微调策略 + WM', trained, 'wm')):
        records = [read(directory/f'{mode}-{case:02d}/result.json', {}) for case in cases]
        completed = [r for r in records if r.get('status') == 'complete']
        if completed:
            lines += ['', f"{label}：已完成 {len(completed)}/{len(cases)}，其中 {sum(r['success'] for r in completed)}/{len(completed)} 成功。"]
    lines += ['', f'[微调策略配对报告]({trained.name}/report.md) · [逐回合证据与额外耗时]({trained.name}/audit.md)', '',
              '## 耗时和结论边界', '',
              '机器同时存在其他计算任务，墙钟耗时不是独占 GPU 吞吐量。[共享机器说明](paired-five/runtime-notes.md)。世界模型评分仅看一秒后图像中的 lemon—目标距离，不能充分评价移障的长期价值；必须同时查看可评分次数与实际改选次数。', '',
              '三个固定场景只支持小样本探索，不能宣称稳定成功或总体成功率提高。若两组都失败，结论就是本次设置没有验证出收益。', '',
              '[流程状态](policy-experiment-status.json) · [训练启动配置](pi05-lemon-base-expert-2k-job.json) · [预先固定的评估配置](policy-followup-protocol.json)']
    if status.get('error'):
        lines += ['', '流程错误：', '', '```text', status['error'], '```']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cases', type=int, nargs='+', default=[10, 11, 12])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    pilot = root/'paired-five'
    train = root/'pi05-lemon-base-expert-2k'
    trained = root/'finetuned-paired-three'
    log = root/'pi05-lemon-base-expert-2k.log'
    job = read(root/'pi05-lemon-base-expert-2k-job.json')
    protocol = read(pilot/'protocol.json')
    if not set(args.cases).issubset(protocol['cases']) or len(set(args.cases)) != len(args.cases):
        p.error('Follow-up cases must be distinct members of the original pilot')
    prior = read(root/'policy-followup-protocol.json')
    previous_status = read(root/'policy-experiment-status.json', {})
    if args.resume:
        if not prior or prior['cases'] != args.cases:
            p.error('Resume must preserve the recorded case list')
        if alive(previous_status.get('pid', -1)):
            p.error('An existing supervisor is still running')
    elif trained.exists() or prior:
        p.error('Follow-up already exists; inspect it instead of starting a duplicate')
    if not args.resume:
        write_json(root/'policy-followup-protocol.json', {'cases': args.cases, 'steps': 2000,
               'selected_before_training_evaluation': True, 'evaluation_checkpoint': 'final step 2000, not best test result',
               'max_steps': protocol['max_steps'], 'training_pid': job['pid'], 'started_at': time.strftime('%Y-%m-%d %H:%M:%S %Z')})
    status = {'stage': '等待训练与原策略配对实验完成', 'pid': os.getpid()}
    def update(stage):
        status['stage'] = stage
        write_json(root/'policy-experiment-status.json', status)
        render(root, pilot, trained, args.cases, log, status)
    try:
        while alive(job['pid']):
            update('等待单任务训练完成')
            time.sleep(15)
        checkpoint = train/'checkpoints/last'
        step = read(checkpoint/'training_state/training_step.json', {}).get('step')
        if step != 2000 or 'End of training' not in log.read_text(errors='replace'):
            raise RuntimeError(f'Training did not finish cleanly at step 2000 (checkpoint step={step}); see {log}')
        sim_python = str(Path(protocol['sim_root'])/'.venv/bin/python')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYOPENGL_PLATFORM='egl',
                   GS_GYM_ASSET_PATHS='/home/junwenmiao/code/RoboWits/assets',
                   HF_HOME='/home/junwenmiao/code/RoboWits/.cache/huggingface',
                   XDG_CACHE_HOME=str(root/'cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        env.pop('PYTHONPATH', None)
        update('训练完成，本地检查留出轨迹动作预测')
        validation = [sim_python, '-m', 'robowits.validate_policy_local', '--sim-root', protocol['sim_root'],
                      '--checkpoint', str((checkpoint/'pretrained_model').resolve()),
                      '--data', str(root/'policy-ee-base-data'), '--out', str(root/'policy-validation.json')]
        with (root/'policy-validation.log').open('w') as stream:
            subprocess.run(validation, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        while read(pilot/'status.json', {}).get('status') != 'complete':
            current = read(pilot/'status.json', {})
            if current.get('status') == 'error' or not alive(current.get('pid', -1)):
                raise RuntimeError('Original paired pilot is not running or failed; inspect its status before continuing')
            update('训练完成，等待原策略配对实验释放资源')
            time.sleep(15)
        cmd = [sim_python, '-u', '-m', 'robowits.closed_loop']
        for key in ('sim_root', 'eval_json', 'urdf', 'wm_python', 'wm_checkpoint', 'ckpts', 'wan', 'wm_gpu'):
            cmd += ['--'+key.replace('_', '-'), protocol[key]]
        cmd += ['--policy', str((checkpoint/'pretrained_model').resolve()), '--out', str(trained),
                '--max-steps', str(protocol['max_steps']), '--cases', *map(str, args.cases)]
        if args.resume and trained.exists():
            cmd.append('--resume')
        write_json(root/'finetuned-paired-command.json', cmd)
        with (root/'finetuned-paired-three.log').open('w') as stream:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status['evaluation_pid'] = proc.pid
        while proc.poll() is None:
            update('运行微调策略的配对闭环实验')
            time.sleep(15)
        if proc.returncode:
            raise RuntimeError('Finetuned paired evaluation failed; see finetuned-paired-three.log')
        for case in args.cases:
            a = read(pilot/f'baseline-{case:02d}/initial.json')
            b = read(trained/f'baseline-{case:02d}/initial.json')
            if not same_state(a, b):
                raise RuntimeError(f'Original and finetuned initial states differ for case {case}')
        for directory in (pilot, trained):
            subprocess.run([sim_python, '-m', 'robowits.summarize_closed_loop', str(directory)], cwd=ROOT, env=env, check=True)
        update('全部完成')
    except BaseException:
        status['error'] = traceback.format_exc()
        update('流程出错，未完成结果不计入成功率')
        raise


if __name__ == '__main__':
    main()
