"""Order-swap, identical-video and static-video controls for VLM rewards."""
import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np

from .vlm_reward import VLMReward, IndependentVLMReward
from .closed_loop import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--socket',type=Path,required=True)
    p.add_argument("--independent", action="store_true")
    p.add_argument('--examples', action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+300
    while not args.socket.exists():
        if time.monotonic()>deadline:raise TimeoutError('VLM socket did not appear')
        time.sleep(1)
    judge=(IndependentVLMReward if args.independent else VLMReward)(args.socket,examples=args.examples);rows=[]
    for i,d in enumerate(json.loads((args.source/'decisions.json').read_text())):
        folder=args.source/f'decision-{i:04d}'
        current=imageio.imread(folder/'reward/current.png')
        preds=[np.load(folder/f'candidate-{j}/prediction.npy') for j in range(2)]
        controls={'original':preds,'reversed':preds[::-1],'identical':[preds[0],preds[0]],
                  'static':[np.repeat(current[None],20,axis=0)]*2}
        for name,videos in controls.items():
            target=args.out/f'round-{i}-{name}'
            scores=judge.judge(current,videos,'Place the lemon on the green target area.',d['subgoal'],target)
            row={'round':i,'control':name,'scores':scores,'score_gap':scores[1]['score']-scores[0]['score']}
            rows.append(row);write_json(args.out/'results.json',rows)
            print(json.dumps(row),flush=True)
    write_json(args.out/'status.json',{'status':'complete','calls':len(rows)})


if __name__=='__main__':main()
