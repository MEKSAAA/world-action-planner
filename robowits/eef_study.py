"""Run predeclared paired trials sequentially, preserving every trial's evidence.

The JSON specification contains ``python``, ``environment``, ``common`` (eef_loop
arguments), and ``trials`` (case/seed dictionaries). Completed trials are skipped;
partial/error trials are never overwritten. Run baselines first, then controls.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .closed_loop import write_json


def summarize(root):
    rows = []
    lines = ['# Lemon 配对闭环实验', '',
             '只有 status=complete 的完整回合才能计入任务成功率；运行错误不计为任务失败。', '',
             '| 布局 | 条件 | 状态 | 成功 | 步数 | 更换候选次数 | 墙钟秒 |',
             '|---|---|---|---|---:|---:|---:|']
    for path in sorted(root.glob('case-*/*/status.json')):
        status = json.loads(path.read_text())
        row = dict(case=int(path.parent.parent.name.split('-')[-1]), condition=path.parent.name, **status)
        rows.append(row)
        wall = f"{row['wall_seconds']:.1f}" if 'wall_seconds' in row else '—'
        lines.append(f"| {row['case']} | {row['condition']} | {row['status']} | {row.get('success','—')} | "
                     f"{row.get('steps','—')} | {row.get('changed_choices','—')} | {wall} |")
    lines += ['', 'baseline：原策略；forced_zero：运行 WM/VLM 但固定执行候选 0；selection：按 VLM 分数选择。', '',
              '每轮候选 0、初始状态和未分叉前的观测/执行轨迹都有一致性检查。只在 WM 输入完全一致且模型/代码/种子一致时复用对照组预测；同一观测、子目标和预测下复用评分。', '',
              '墙钟时间包含已复用计算的节省；公平比较推理成本时须另外加上 decisions.json 的 reference_wm_seconds 与 reference_vlm_seconds。结果只是三个固定布局的可行性试验。', '',
              '每个回合的 execution.mp4、decisions.json、protocol.json 与日志均保存在对应 case 目录。']
    write_json(root/'results.json', rows)
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--conditions', nargs='+', choices=['baseline', 'forced_zero', 'selection'],
                   default=['baseline', 'forced_zero', 'selection'])
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out/'spec.json'
    if frozen.exists() and json.loads(frozen.read_text()) != spec:
        raise ValueError('Study specification changed; choose a new output directory')
    write_json(frozen, spec)
    env = dict(os.environ, **spec.get('environment', {}))
    records = []
    for trial in spec['trials']:
        for condition in args.conditions:
            baseline = args.out/f"case-{trial['case']}/baseline"
            out = args.out/f"case-{trial['case']}/{condition}"
            if out.exists():
                status = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                if status.get('status') != 'complete':
                    raise RuntimeError(f'Existing incomplete trial requires review: {out}')
                records.append(dict(condition=condition, **trial, **status, reused_completed_trial=True))
                continue
            if condition != 'baseline':
                status = json.loads((baseline/'status.json').read_text())
                if status['status'] != 'complete':
                    raise RuntimeError(f'Paired baseline incomplete: {baseline}')
            options = dict(spec['common'], **trial, out=str(out),
                           mode='baseline' if condition == 'baseline' else 'selection')
            if condition != 'baseline':
                options['reference_run'] = str(baseline)
                options['reuse_baseline_candidate_zero'] = True
            if condition == 'forced_zero':
                options['force_candidate_zero'] = True
            if condition == 'selection' and spec.get('reuse_control_imagination', True):
                options['reuse_imagination_from'] = str(out.parent/'forced_zero')
            command = [spec['python'], '-u', '-m', 'robowits.eef_loop']
            for key, value in options.items():
                if value is False or value is None:
                    continue
                command.append('--'+key.replace('_', '-'))
                if value is not True:
                    command.append(str(value))
            out.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.out/'runner-status.json', dict(status='running', condition=condition,
                       trial=trial, pid=os.getpid(), started_unix=time.time()))
            with (out.parent/f'{condition}.log').open('w') as log:
                code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            status = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {'status':'error'}
            records.append(dict(condition=condition, **trial, **status, returncode=code))
            summarize(args.out)
            if code:
                write_json(args.out/'runner-status.json', dict(status='error', trial=trial, condition=condition))
                raise RuntimeError(f'Trial failed; not counted as task failure: {out}')
    summarize(args.out)
    write_json(args.out/'runner-status.json', dict(status='complete', conditions=args.conditions,
               trials=len(records)))


if __name__ == '__main__':
    main()
