"""Build a Markdown audit of completed paired rollouts without rerunning models."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    root = args.run.resolve()
    protocol = json.loads((root/'protocol.json').read_text())
    status = json.loads((root/'status.json').read_text())
    rows = {}
    for path in sorted(root.glob('*/result.json')):
        r = json.loads(path.read_text())
        if r['status'] == 'complete':
            rows[r['mode'], r['case']] = r
    lines = ['# Lemon 闭环配对实验审计', '',
             f"运行状态：`{status['status']}`。预先固定场景：{protocol['cases']}。只统计完整回合，未完成和基础设施错误不记为失败。", '',
             '| 场景 | 原策略成功 | WM 筛选成功 | 原策略耗时 / s | WM 耗时 / s | 额外耗时 / s | 可比较 / 筛选次数 | 改选次数 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    pairs = []
    for case in protocol['cases']:
        a, b = rows.get(('baseline', case)), rows.get(('wm', case))
        if not (a and b):
            continue
        decisions = json.loads((root/f'wm-{case:02d}/decisions.json').read_text())
        wm = [d for d in decisions if d['wm_used']]
        comparable = sum(all(s['valid'] for s in d['scores']) for d in wm)
        delta = b['wall_seconds']-a['wall_seconds']
        pairs.append((a, b, delta))
        lines.append(f"| {case} | {'成功' if a['success'] else '失败'} | {'成功' if b['success'] else '失败'} | {a['wall_seconds']:.1f} | {b['wall_seconds']:.1f} | {delta:.1f} | {comparable}/{len(wm)} | {b['changed_choices']} |")
    if pairs:
        n = len(pairs)
        sa, sb = (sum(pair[i]['success'] for pair in pairs) for i in (0, 1))
        rescued = sum(not a['success'] and b['success'] for a, b, _ in pairs)
        harmed = sum(a['success'] and not b['success'] for a, b, _ in pairs)
        lines += ['', f'完整配对 {n} 组：原策略 {sa}/{n} 成功；WM 筛选 {sb}/{n} 成功，差值 {(sb-sa)/n*100:+.1f} 个百分点。',
                  '', f'由失败变成功 {rescued} 组；由成功变失败 {harmed} 组。平均额外耗时 {np.mean([d for _, _, d in pairs]):.1f} 秒/回合。',
                  '', '这是小样本探索结果，不能据此断言总体成功率提高。耗时含回合重置、策略推理、模拟器执行、录像以及 WM/机器人投影计算；不含首次模型加载。']
    lines += ['', '## 逐回合证据', '', '以下失败标签为自动诊断，抓空、碰撞、搬运掉落的最终归因应结合录像；“最高进度 0.5”不等于 50% 成功率。']
    for (mode, case), r in sorted(rows.items()):
        rel = f'{mode}-{case:02d}'
        directory = root/rel
        ds = json.loads((directory/'diagnostics.json').read_text())
        first, last = ds[0], ds[-1]
        initial_distance = np.linalg.norm(np.array(first['lemon'])[:2]-first['target'][:2])
        final_distance = np.linalg.norm(np.array(last['lemon'])[:2]-last['target'][:2])
        lines += ['', f'### {mode} / 场景 {case}', '', r['failure_type'], '',
                  f"运行 {r['steps']} 步；最高原生进度 {r['max_score']:.3f}。lemon 到目标中心的平面距离 {initial_distance:.3f} → {final_distance:.3f} m；采样最高高度 {max(s['lemon'][2] for s in ds):.3f} m。物体状态仅用于事后诊断。", '',
                  f'[执行录像]({rel}/execution.mp4) · [动作选择记录]({rel}/decisions.json) · [结果]({rel}/result.json)']
        # Contact sheets are evidence previews, not a substitute for full video.
        preview = directory/'contact-sheet.jpg'
        if not preview.exists():
            import av
            from PIL import Image, ImageDraw
            wanted = set(np.linspace(0, r['steps'], 12).round().astype(int).tolist())
            frames = []
            with av.open(str(directory/'execution.mp4')) as video:
                for i, frame in enumerate(video.decode(video=0)):
                    if i in wanted:
                        im = frame.to_image().resize((448, 448))
                        draw = ImageDraw.Draw(im)
                        draw.rectangle((0, 0, 120, 22), fill='white')
                        draw.text((4, 4), f'step {i}', fill='black')
                        frames.append(im)
            sheet = Image.new('RGB', (448*4, 448*3))
            for i, im in enumerate(frames):
                sheet.paste(im, ((i % 4)*448, (i // 4)*448))
            sheet.save(preview, quality=90)
        lines += ['', f'![{mode} 场景 {case} 执行采样]({rel}/contact-sheet.jpg)']
    lines += ['', '## 可复现性与边界', '',
              f"Genesis131 commit：`{protocol['sim_commit']}`；运行代码 commit：`{protocol['wap_commit']}`。模型、场景文件和评分代码 SHA256 见 [protocol.json](protocol.json)。", '',
              '现有 PI05 是多任务 EE_ABS 策略。本实验两组每 30 步重规划，区别于其默认 50 步配置。预测时长 1 秒，仅按图像内 lemon 与目标距离评分；无法充分衡量移开障碍物的长期收益。', '',
              '评分器遇到 lemon 遮挡、消失或重复时回退候选 0。若可比较次数很少或改选次数为零，应先解决观测/评分覆盖率问题，不能把结果解释为世界模型规划有效。', '',
              '配对初始状态和独立机器人投影对真实场景无修改均在运行时检查。没有使用任务物体的真实未来进行候选评分。']
    for filename, title in (('failure-review.md', '人工失败阶段复核'), ('contact-review.md', '原生接触与动作重放复核'), ('runtime-notes.md', '共享机器耗时说明')):
        if (root/filename).exists():
            lines += ['', f'[{title}]({filename})']
    (root/'audit.md').write_text('\n'.join(lines)+'\n')
    print(root/'audit.md')


if __name__ == '__main__':
    main()
