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


def inference_usage(run):
    """Count saved attempts once, excluding proposals copied from a paired run.

    Raw provider metadata covers responses rejected by the JSON validator too.
    Missing responses have unknown usage, not zero usage. This is a log audit,
    not billing reconciliation, and includes unfinished-decision attempts.
    """
    totals, stages, issues = {}, Counter(), []
    excluded = 0
    for decision in sorted(run.glob('decision-*')):
        if not decision.is_dir():
            continue
        reused = (decision/'candidate-bank-reuse.json').exists()
        for request in sorted(decision.rglob('request.json')):
            relative = request.relative_to(decision)
            stage = relative.parts[0]
            if reused and stage == 'proposals':
                excluded += 1
                continue
            try:
                req = read(request)
                response_path = request.with_name('response.json')
                provider_path = request.with_name('provider-response.json')
                response = read(response_path) if response_path.exists() else {}
                provider = read(provider_path) if provider_path.exists() else {}
            except (OSError, json.JSONDecodeError) as exc:
                issues.append({'path': str(request.relative_to(run)), 'error': type(exc).__name__})
                continue
            name = req.get('backend') or ('qwen' if 'output_tokens' in response else 'unknown')
            group = totals.setdefault(name, {'requests': 0, 'responses_with_usage': 0,
                'requests_without_usage': 0, 'recorded_response_seconds': 0.,
                'token_counts': {}, 'models': []})
            group['requests'] += 1
            model = provider.get('modelVersion') or response.get('model') or req.get('model')
            if model and model not in group['models']:
                group['models'].append(model)
            seconds = response.get('seconds', 0.)
            group['recorded_response_seconds'] += seconds
            stages[stage] += seconds
            usage = provider.get('usageMetadata') or response.get('usage') or {}
            counts = {k: v for k, v in usage.items()
                      if k.endswith('TokenCount') and type(v) is int}
            if type(response.get('output_tokens')) is int:
                counts['localOutputTokens'] = response['output_tokens']
            if counts:
                group['responses_with_usage'] += 1
                for key, value in counts.items():
                    group['token_counts'][key] = group['token_counts'].get(key, 0) + value
            else:
                group['requests_without_usage'] += 1
    return {'providers': totals, 'recorded_response_seconds_by_stage': dict(stages),
            'copied_proposal_requests_excluded': excluded, 'unreadable_attempts': issues,
            'scope': 'saved local attempts including retries and unfinished decisions; '
                     'excludes reused proposal logs and external offline cross-scores; '
                     'missing usage is unknown; token fields are not additive'}


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
        'inference_usage': inference_usage(run),
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
    usage = summary['inference_usage']
    lines += ['## 已保存的模型调用用量 / Recorded inference usage', '',
              '包括重试及尚未执行的决策调用；排除从配对回合复制的提议记录和外部交叉评分。'
              '缺失用量不是零用量；各 token 字段可能重叠，不相加为总数。此处不是账单核对。', '',
              '|模型来源|记录请求数|有用量响应数|缺失用量请求数|Token 字段|记录响应秒数|',
              '|---|---:|---:|---:|---|---:|']
    for name, value in usage['providers'].items():
        lines.append(f'|{name}|{value["requests"]}|{value["responses_with_usage"]}|'
                     f'{value["requests_without_usage"]}|{value["token_counts"]}|'
                     f'{value["recorded_response_seconds"]:.1f}|')
    lines += ['', f'排除复制提议请求 {usage["copied_proposal_requests_excluded"]} 条；'
              f'未能读取的调用记录 {len(usage["unreadable_attempts"])} 条。'
              '响应时间可能已包含在上方提议耗时内，不与之相加。', '']
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
