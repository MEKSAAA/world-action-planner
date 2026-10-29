"""Markdown, aligned GT/condition/WM videos, and diagnostic metrics for one bank."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .closed_loop import write_json


VIEWS={'ego':(slice(0,224),slice(0,224)),
       'right_wrist':(slice(224,448),slice(0,224)),
       'left_wrist':(slice(224,448),slice(224,448))}


def psnr(mse):
    return float(10*np.log10(255**2/max(mse,1e-12)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('run','wm','audit','out'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--decision',type=int,default=0)
    p.add_argument('--legacy-eef',action='store_true',
                   help='Use the recorded Qwen scores from a single-subgoal lemon EEF decision')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((args.run/'protocol.json').read_text())
    decision=json.loads((args.run/'decisions.json').read_text())[args.decision]
    scores=({'qwen':decision['scores']} if args.legacy_eef else json.loads((args.wm/'scores.json').read_text()))
    audit=json.loads((args.audit/'summary.json').read_text())
    task_id='robowits/03-gap-retrieve-v0' if args.legacy_eef else protocol['task_id']
    object_name='lemon' if '03-gap-retrieve' in task_id else 'cube'
    rows=[]
    with imageio.get_writer(args.out/'all-candidates.mp4',fps=5,codec='libx264') as combined:
        for row in audit:
            i=row['candidate'];folder=args.wm/f'candidate-{i}'
            pred=np.load(folder/'prediction.npy')[:6]
            truth=np.load(args.audit/f'candidate-{i}/truth.npy')
            current=np.asarray(Image.open(args.audit/f'candidate-{i}/current.png'))
            with np.load(folder/'input.npz') as source:
                future=source['future'][:6];past=source['past'][-1]
                if not np.array_equal(source['history'][-1],current):
                    raise ValueError('WM and actual branch current images differ')
            if truth.shape!=pred.shape or len(truth)!=6:
                raise ValueError('Need six aligned actual/predicted frames')
            physics=json.loads((args.audit/f'candidate-{i}/physics.json').read_text())
            initial=physics['initial']['objects'];final=physics['events'][-1]['state']['objects']
            before=np.asarray(initial[object_name]);after=np.asarray(final[object_name])
            target=np.asarray(initial['target_area'])
            item={'candidate':i,'language':row['language'],
                'native_success_within_10_steps':row['native_success'],
                'object_displacement_10_steps_mm':row['object_displacement_m'][object_name]*1000,
                'object_delta_xyz_10_steps_mm':((after-before)*1000).tolist(),
                'target_distance_xy_reduction_10_steps_mm':float((
                    np.linalg.norm(before[:2]-target[:2])-np.linalg.norm(after[:2]-target[:2]))*1000),
                'scores':{name:next(s for s in values if s['id']==i) for name,values in scores.items()},
                'pixel_metrics':{}}
            for view,(ys,xs) in VIEWS.items():
                gt=truth[:,ys,xs].astype(np.float32)
                mse=float(np.mean((pred[:,ys,xs].astype(np.float32)-gt)**2))
                copy=float(np.mean((current[ys,xs].astype(np.float32)-gt)**2))
                item['pixel_metrics'][view]={'wm_mse':mse,'copy_last_mse':copy,
                                             'wm_psnr_db':psnr(mse),'copy_last_psnr_db':psnr(copy)}
            rows.append(item)
            frames=[]
            for t in range(7):
                canvas=Image.new('RGB',(1344,496),'white');draw=ImageDraw.Draw(canvas)
                stamp='CURRENT' if t==0 else f'WM +{t/20:.2f}s / actual native offset {row["native_frame_indices"][t-1]}'
                draw.text((8,5),f'candidate {i} | {stamp} | {row["language"]}',fill='black')
                for col,label in enumerate(['SIMULATOR GROUND TRUTH','ROBOT-ONLY CONDITION','WORLD MODEL']):
                    draw.text((448*col+8,27),label,fill='black')
                tiles=(current,past,current) if t==0 else (truth[t-1],future[t-1],pred[t-1])
                for col,tile in enumerate(tiles):canvas.paste(Image.fromarray(tile),(448*col,48))
                frames.append(np.asarray(canvas));combined.append_data(frames[-1])
            with imageio.get_writer(args.out/f'candidate-{i}.mp4',fps=5,codec='libx264') as video:
                for frame in frames:video.append_data(frame)
    write_json(args.out/'metrics.json',rows)
    lines=['# 固定候选池诊断：真实执行 / 骨架条件 / WM 预测','',
        f'任务：`{task_id}`；开始于原生第 {decision["step"]} 步。', '',
        '[全部候选慢放视频](all-candidates.mp4)。每个候选单独从相同场景回放，三个面板依次为真实模拟器、机器人条件、WM。', '',
        '视频只展示被评分的 6 帧，5 fps 慢放；对应实际原生帧偏移 2/3/5/6/8/9。表中物体位移包含完整 10 步，两个时间范围相差最后一个控制步。', '',
        '这是局部诊断，不是独立的完整任务试验；物体靠近目标也不必然意味着动作更好，例如清障或抓起可能先增加距离。', '',
        '|候选|子目标|10 步物体位移 mm|平面目标距离减少 mm|Gemini|Qwen|视频|',
        '|---:|---|---:|---:|---:|---:|---|']
    for r in rows:
        gs=r['scores'].get('gemini',{}).get('score','—');qs=r['scores'].get('qwen',{}).get('score','—')
        lines.append(f'|{r["candidate"]}|{r["language"]}|{r["object_displacement_10_steps_mm"]:.2f}|'
            f'{r["target_distance_xy_reduction_10_steps_mm"]:.2f}|{gs}|{qs}|[查看](candidate-{r["candidate"]}.mp4)|')
    lines += ['', '完整逐视角像素指标见 `metrics.json`。PSNR/MSE 易受静态背景影响，不能替代接触正确性或任务成功判定。', '',
              '物理信息仅用于这份事后诊断，没有进入在线子目标提议或评分。']
    (args.out/'review.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
