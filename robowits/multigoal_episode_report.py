"""Read-only Markdown timeline of a multi-subgoal episode, including partial runs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read(path):
    return json.loads(path.read_text())


def review_indices(count):
    """Fixed cadence, plus final decision; independent of rewards or outcomes."""
    return sorted(set(range(0, count, 25)) | ({count - 1} if count else set()))


def terminal_outcome(result, max_steps):
    if result.get('status') != 'complete' or type(result.get('success')) is not bool:
        return None
    if not (result.get('terminated') or result.get('truncated') or result.get('steps', 0) >= max_steps):
        raise ValueError('Complete episode has no recorded endpoint')
    return result['success']


def live(pid):
    try:
        command = Path(f'/proc/{int(pid)}/cmdline').read_text()
        # The config path may be outside the output directory. Check the process
        # identity as well as its status-file PID; never infer life from mtime.
        return 'robowits.multigoal_loop' in command
    except (OSError, ValueError, TypeError):
        return False


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def export(run, out):
    run, out = run.resolve(), out.resolve()
    protocol = read(run/'protocol.json')
    status = read(run/'status.json')
    result = read(run/'result.json') if (run/'result.json').exists() else status
    outcome = terminal_outcome(result, protocol['max_steps'])
    decisions = read(run/'decisions.json') if (run/'decisions.json').exists() else []
    # Decisions are written before execution. Do not label an in-flight decision
    # as an executed change or include it in action-count statistics.
    with np.load(run/'trajectory.npz') as data:
        steps = len(data['actions'])
        if len(data['joints']) != steps + 1:
            raise ValueError('Trajectory joint/action lengths disagree')
    executed = [d for d in decisions if d['step'] < steps]
    if outcome is not None and result['steps'] != steps:
        raise ValueError('Terminal result and trajectory disagree')
    primary = protocol['primary_judge']
    rows, changes, reasons, flat = [], [], Counter(), []
    for i, decision in enumerate(executed):
        selected = decision['selected']
        valid_control = [s for s in decision['no_wm'][primary] if s['valid']]
        control = max(valid_control, key=lambda s: s['score'])['id'] if valid_control else None
        scores = decision['wm'].get(primary, [])
        valid = [s['score'] for s in scores if s['valid']]
        if scores:
            flat.extend(scores)
            if not valid:
                reasons['all_invalid'] += 1
            elif max(valid) == 0:
                reasons['all_valid_zero'] += 1
            if valid and valid.count(max(valid)) > 1:
                reasons['top_tie'] += 1
            if selected != control:
                changes.append(i)
        distance = np.load(run/f'decision-{i:04d}'/'pairwise-position-distance-m.npy')
        rows.append({'decision': i, 'step': decision['step'], 'selected': selected,
                     'control': control, 'language': decision['candidate_map'][selected]['goal']['language'],
                     'scores': [s['score'] if s['valid'] else None for s in scores],
                     'max_mean_position_distance_mm': float(distance.max()*1000),
                     'paired': decision['paired_candidate_bank'], 'reason': decision['choice_reason']})
    summary = {'source': str(run), 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'status': result['status'], 'process_live': live(status.get('pid')),
        'native_success': outcome, 'executed_steps': steps, 'executed_decisions': len(executed),
        'recorded_decisions': len(decisions), 'changed_from_same_state_control': len(changes),
        'first_changed_step': executed[changes[0]]['step'] if changes else None,
        'score_diagnostics': dict(reasons), 'scored_candidates': len(flat),
        'invalid_candidates': sum(not s['valid'] for s in flat),
        'timing_seconds': {key: sum(d[key] for d in executed) for key in
            ('proposal_seconds', 'policy_seconds', 'proxy_seconds', 'wm_seconds')},
        'wall_seconds': result.get('wall_seconds'),
        'review_decisions': review_indices(len(executed))}
    audit_path = run/'contact-replay.json'
    if audit_path.exists():
        audit = read(audit_path)
        if audit['summary']['steps'] != steps or audit['summary']['success'] != outcome:
            raise ValueError('Contact audit and episode result disagree')
        summary['contact_audit'] = audit['summary']
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    (out/'timeline.json').write_text(json.dumps(rows, indent=2))
    def link(path):
        return os.path.relpath(path, out)
    lines = ['# 多子目标完整回合核查 / Episode review', '',
        f"更新 UTC：{summary['updated_utc']}", '',
        f"状态：`{summary['status']}`；进程存活：{summary['process_live']}；已执行 {steps} 步。",
        f'模拟器完整回合成功：**{outcome if outcome is not None else "未得到终局 / no terminal outcome"}**。', '',
        '单个初态不能估计总体成功率。运行中的 success=false 不计作任务失败。', '',
        f'模型：{protocol["proposer"]} 提议，{primary} 评分；模式：`{protocol["mode"]}`。',
        f'已执行 {len(executed)} 次选择；与同状态无 WM 选择不同 {len(changes)} 次。',
        '分叉后的“无 WM 选择”只是当前状态的对照判断，不是另一条完整轨迹的同一步。', '',
        f'评分异常计数：`{dict(reasons)}`；无效候选 {summary["invalid_candidates"]}/{len(flat)}。',
        f'已记录推理耗时（秒）：`{summary["timing_seconds"]}`。不含所有初始化和评分时间；不能当作端到端总耗时。', '',
        f'[原始配置]({link(run/"protocol.json")}) · [逐轮原始记录]({link(run/"decisions.json")})', '']
    if outcome is not None:
        lines += [f'[完整执行视频]({link(run/"execution.mp4")})', '']
    if 'contact_audit' in summary:
        audit = summary['contact_audit']
        lines += [f'物理重放：最大关节差 {audit["max_joint_error"]:.6g}；lemon 最大抬升 '
                  f'{audit["maximum_lift_m"]*1000:.1f} mm；指尖接触 {audit["finger_lemon_contact_frames"]} 帧。',
                  f'自动诊断：`{audit["failure_stage"]}`。接触和抬升不等于稳定抓握，仍需结合视频判断。',
                  f'[重放证据]({link(audit_path)})', '']
    images = []
    for i in summary['review_decisions']:
        folder = run/f'decision-{i:04d}'
        source = folder/f'no-wm-{primary}'/'current.png'
        if not source.exists() and protocol.get('reference_run'):
            source = Path(protocol['reference_run'])/f'decision-{i:04d}'/f'no-wm-{primary}'/'current.png'
        if source.exists():
            panel = Image.new('RGB', (336, 360), 'white')
            with Image.open(source) as im:
                panel.paste(im.convert('RGB').resize((336, 336)), (0, 24))
            ImageDraw.Draw(panel).text((5, 5), f"step {executed[i]['step']} | choice {executed[i]['selected']}", fill='black')
            images.append(panel)
    if images:
        canvas = Image.new('RGB', (336*len(images), 360), 'white')
        for i, panel in enumerate(images):
            canvas.paste(panel, (336*i, 0))
        canvas.save(out/'fixed-cadence.png')
        lines += ['固定每 25 次选择及末次选择前的真实观测；不按分数或成功与否挑帧。', '',
                  '![实际观测时间线](fixed-cadence.png)', '']
    lines += ['|原生步数|选中候选|同状态无 WM 选择|选中子目标|WM 分数（无效=null）|最大候选位置差 mm|',
              '|---:|---:|---:|---|---|---:|']
    for row in rows:
        lines.append(f'|{row["step"]}|{row["selected"]}|{row["control"]}|{cell(row["language"])}|{row["scores"]}|{row["max_mean_position_distance_mm"]:.2f}|')
    (out/'review.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.out), indent=2))
