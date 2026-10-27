"""Build Markdown and videos for a bounded EEF review, without starting more trials."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .closed_loop import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    args = p.parse_args()
    runs = [args.baseline, args.batch/'forced_zero', args.batch/'selection']
    results = [json.loads((r/'result.json').read_text()) for r in runs]
    if any(r['status'] not in ('complete','paused_for_review') for r in results):
        raise ValueError('Review requires completed or explicitly paused runs')
    steps = min(r['steps'] for r in results)
    trajectories = [np.load(r/'trajectory.npz') for r in runs]
    equal = [bool(np.array_equal(trajectories[0]['actions'][:steps], t['actions'][:steps])) for t in trajectories[1:]]
    joint_errors = [float(np.max(np.abs(trajectories[0]['joints'][:steps+1]-t['joints'][:steps+1]))) for t in trajectories[1:]]
    if not equal[0] or joint_errors[0] > 1e-5:
        raise RuntimeError('Forced-zero control differs from baseline')
    readers = [imageio.get_reader(r/'execution.mp4') for r in runs]
    try:
        with imageio.get_writer(args.batch/'execution-comparison.mp4',fps=7.5,codec='libx264') as writer:
            for step in range(steps+1):
                canvas = Image.new('RGB',(448*3,480),'white')
                draw = ImageDraw.Draw(canvas)
                for i,(reader,label) in enumerate(zip(readers,['Policy baseline','WM scored, force candidate 0','WM selection'])):
                    canvas.paste(Image.fromarray(reader.get_data(step)),(448*i,32))
                    draw.text((448*i+5,5),f'{label} | step {step}',fill='black')
                writer.append_data(np.array(canvas))
    finally:
        for reader in readers:
            reader.close()
    decisions = json.loads((runs[2]/'decisions.json').read_text())
    protocol = json.loads((runs[2]/'protocol.json').read_text())
    margin = protocol['selection_margin']
    horizon = protocol.get('judged_future_frames', 6)
    summary = dict(status='paused_for_review', reviewed_steps=steps,
                   baseline_action_equal=equal, baseline_joint_max_abs_diff=joint_errors,
                   decisions=len(decisions), changed_choices=sum(d['selected']!=0 for d in decisions),
                   task_success_rate_available=False, automatic_continuation=False)
    write_json(args.batch/'review-summary.json',summary)
    lines = ['# Lemon：暂停供人工审核的短程对照','',
             f'本批仅检查前 {steps} 步（约 {steps/30:.2f} 秒仿真时间），不是完整任务成功率实验。三组均已停止，不会自动扩展到其余布局。','',
             '[实际执行三组对比视频，4 倍慢放](execution-comparison.mp4)。左：原策略；中：运行 WM/VLM 但固定候选 0；右：按评分选择。','',
             f"完成 {len(decisions)} 轮选择；选择非零候选 {summary['changed_choices']} 次。固定候选 0 的动作与基线完全相同；选择组动作与基线相同：{equal[1]}。",'',
             f'切换规则：须比有效候选 0 高出超过 {margin:.2f}；恰好等于门槛时保留候选 0（排除浮点舍入误差）。','',
             '当前结果不能证明任务成功率有提升。若没有改变动作，只能证明这批输入没有触发切换。相同输入复用了已有预测，墙钟时间不能当作完整部署推理成本。','',
             '| 步数 | 候选 0 分数 | 候选 1 分数 | 执行候选 |','|---:|---:|---:|---:|']
    for d in decisions:
        lines.append(f"| {d['step']} | {d['scores'][0]['score']:.2f} | {d['scores'][1]['score']:.2f} | {d['selected']} |")
    lines += ['','请审核：机器人是否在有效接近 lemon；候选视频是否存在物体消失/变形；评分理由是否有画面证据。看不清就记“无法判断”。','']
    for index in range(len(decisions)):
        folder = runs[2]/f'decision-{index:04d}'
        with np.load(folder/'candidate-0/input.npz') as z:
            current = z['history'][-1]
        predictions = [np.load(folder/f'candidate-{i}/prediction.npy') for i in range(2)]
        filename = f'candidates-step-{decisions[index]["step"]}.mp4'
        with imageio.get_writer(args.batch/filename,fps=5,codec='libx264') as writer:
            for frame in range(horizon):
                canvas = Image.new('RGB', (448*3,480), 'white')
                draw = ImageDraw.Draw(canvas)
                labels = ['Current observation'] + [f"Candidate {i} | score {decisions[index]['scores'][i]['score']:.2f}" for i in range(2)]
                for i,(picture,label) in enumerate(zip([current,predictions[0][frame],predictions[1][frame]],labels)):
                    canvas.paste(Image.fromarray(picture), (448*i,32))
                    draw.text((448*i+5,5),label,fill='black')
                writer.append_data(np.array(canvas))
        lines.append(f'- [第 {decisions[index]["step"]} 步候选视频：当前画面 | A（候选 0）| B（候选 1）]({filename})')
    lines += ['',f'[选择组逐轮评分及理由](selection/decisions.json)。视频包含 {horizon/20:.2f} 秒有效预测，不包含末端姿态填充。']
    (args.batch/'review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
