"""Export a completed policy/WM selection pair to Markdown and review videos."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .closed_loop import same_state, write_json


def read(path):
    return json.loads(path.read_text())


def export(root):
    baseline, selection = root/'baseline', root/'selection'
    results = [read(p/'result.json') for p in (baseline, selection)]
    if any(r['status'] != 'complete' or type(r['success']) is not bool for r in results):
        raise ValueError('A full report requires two completed episodes with native success results')
    if not same_state(read(baseline/'initial.json'), read(selection/'initial.json')):
        raise ValueError('Different initial states')
    protocol = read(selection/'protocol.json')
    decisions = read(selection/'decisions.json')
    audit = read(selection/'contact-replay.json')['summary']
    if audit['success'] != results[1]['success'] or audit['steps'] != results[1]['steps']:
        raise ValueError('Replay result disagrees with execution')
    for d in decisions:
        if len(d['scores']) != 2:
            raise ValueError('This video layout requires exactly two scored candidates')
    with np.load(baseline/'trajectory.npz') as a, np.load(selection/'trajectory.npz') as b:
        common_steps = min(results[0]['steps'], results[1]['steps'])
        action_equal = bool(np.array_equal(a['actions'][:common_steps], b['actions'][:common_steps]))
        joint_diff = float(np.max(np.abs(a['joints'][:common_steps+1]-b['joints'][:common_steps+1])))
    switches = [i for i,d in enumerate(decisions) if d['selected'] != 0]
    summary = dict(status='complete', baseline=results[0], selection=results[1],
                   common_steps=common_steps, actions_equal_on_common_prefix=action_equal,
                   joint_max_abs_diff_on_common_prefix=joint_diff,
                   decisions=len(decisions), changed_choices=len(switches),
                   first_changed_step=decisions[switches[0]]['step'] if switches else None,
                   contact_audit=audit,
                   inference_seconds={k:sum(d[k] for d in decisions) for k in
                                      ('policy_seconds','wm_seconds','proxy_seconds','vlm_seconds')})
    write_json(root/'summary.json', summary)
    readers = [imageio.get_reader(p/'execution.mp4') for p in (baseline, selection)]
    try:
        for v,r in zip(readers,results):
            if v.count_frames() != r['steps']+1:
                raise ValueError('Video length differs from recorded actions')
        with imageio.get_writer(root/'execution-comparison.mp4',fps=1/.0333,codec='libx264') as writer:
            for step in range(max(r['steps'] for r in results)+1):
                canvas=Image.new('RGB',(896,480),'white');draw=ImageDraw.Draw(canvas)
                for i,(v,r,label) in enumerate(zip(readers,results,['Policy baseline','WM + Qwen selection'])):
                    ended=' | ended; last frame held' if step>r['steps'] else ''
                    canvas.paste(Image.fromarray(v.get_data(min(step,r['steps']))),(448*i,32))
                    draw.text((448*i+4,4),f'{label} | step {min(step,r["steps"])}{ended}',fill='black')
                writer.append_data(np.array(canvas))
    finally:
        for v in readers:v.close()
    lines=['# 完整 π0.5 + WM + Qwen 闭环结果','',
           '本回合已结束。成功判定来自模拟器；Qwen 分数不等于任务成功。两组是同一起始状态的一次配对，不是总体成功率估计。','',
           '| 组别 | 执行步数 | 原生成功 |','|---|---:|---|',
           f'| 纯策略 | {results[0]["steps"]} | {results[0]["success"]} |',
           f'| WM + Qwen | {results[1]["steps"]} | {results[1]["success"]} |','',
           f'共 {len(decisions)} 轮选择，选择候选 1 共 {len(switches)} 次；首次切换步数：{summary["first_changed_step"]}。',
           f'共同执行区间动作完全相同：{action_equal}；最大关节差：{joint_diff:.6g}。','',
           '[完整实际执行对比](execution-comparison.mp4) · [纯策略视频](baseline/execution.mp4) · [闭环视频](selection/execution.mp4)','',
           f'接触重放：最大关节误差 {audit["max_joint_error"]}；lemon 最大抬升 {1000*audit["maximum_lift_m"]:.1f} mm；指尖接触 {audit["finger_lemon_contact_frames"]} 帧。',
           f'原生结束标志：terminated={audit["terminated"]}，truncated={audit["truncated"]}；诊断：{audit["failure_stage"]}。','',
           '[完整逐轮评分及理由](selection/decisions.json) · [物理重放诊断](selection/contact-replay.json) · [运行配置](selection/protocol.json)','',
           '## 候选视频审核','']
    # Show the first and last decisions, plus up to three actual action changes.
    indices=sorted(set([0,len(decisions)-1]+switches[:3])) if decisions else []
    horizon=protocol['judged_future_frames']
    for index in indices:
        folder=selection/f'decision-{index:04d}';d=decisions[index]
        with np.load(folder/'candidate-0/input.npz') as z:current=z['history'][-1]
        preds=[np.load(folder/f'candidate-{i}/prediction.npy') for i in range(2)]
        name=f'candidates-step-{d["step"]}.mp4'
        with imageio.get_writer(root/name,fps=5,codec='libx264') as writer:
            for frame in range(horizon):
                canvas=Image.new('RGB',(1344,480),'white');draw=ImageDraw.Draw(canvas)
                labels=['Current observation']+[f'Candidate {i} | score {d["scores"][i]["score"]} | chosen={d["selected"]==i}' for i in range(2)]
                for i,(picture,label) in enumerate(zip([current,preds[0][frame],preds[1][frame]],labels)):
                    canvas.paste(Image.fromarray(picture),(448*i,32));draw.text((448*i+4,4),label,fill='black')
                writer.append_data(np.array(canvas))
        lines.append(f'- [第 {d["step"]} 步：当前画面 / 候选 0 / 候选 1]({name})')
    lines+=['','预测视频以 5 fps 播放，包含实际评分窗口；不能把预测画面当作模拟器真实未来。','',
            '实验到此结束，没有自动扩展至其他布局。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(export(args.root)),flush=True)


if __name__=='__main__':main()
