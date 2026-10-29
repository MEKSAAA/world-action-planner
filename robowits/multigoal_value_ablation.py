"""Fixed first-draw cases: isolate candidate-text priming in task-level value."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .closed_loop import write_json
from .multigoal_backend import Backend
from .multigoal_value import TaskValue


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--credential-file',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for label,run,frames in [
        ('initial_wm','lemon-gemini-action-probe-v3','lemon-saved-wm'),
        ('step350_actual','lemon-qwen-step350-probe-v2','lemon-step350-actual-branches')]:
        decision=json.loads((args.root/run/'decisions.json').read_text())[0]
        for i in (0,2,4,6):
            folder=args.root/frames/f'candidate-{i}'
            if label=='initial_wm':
                prediction=np.load(folder/'prediction.npy')[:6]
                with np.load(folder/'input.npz') as saved:current=saved['history'][-1]
            else:
                prediction=np.load(folder/'truth.npy')
                current=np.asarray(Image.open(folder/'current.png'))
            cases.append((f'{label}_candidate_{i}',current,prediction,
                          decision['candidate_map'][i]['goal']['language']))
    write_json(args.out/'manifest.json',{'cases':[c[0] for c in cases],
        'selection':'First seeded draw of EVERY subgoal, no outcome-based candidate filtering',
        'factor':'Candidate instruction included vs omitted; all other prompt and image inputs held fixed',
        'source_hint':'All inputs labeled predictions in both conditions to avoid a source-type cue',
        'scope':'Offline value diagnosis, not task success or WM-assisted execution'})
    rows=[]
    for name,current,prediction,language in cases:
        for provider in ('gemini','qwen'):
            for include in (True,False):
                out=args.out/name/provider/('with_text' if include else 'without_text')
                score=TaskValue(Backend(provider,socket=args.socket,credential_file=args.credential_file),
                    candidate_text=include).judge(current,[prediction],
                    'Place the lemon on the green target area.',[language],out,valid_future_frames=6)[0]
                rows.append({'case':name,'backend':provider,'candidate_text':include,'score':score})
                write_json(args.out/'results.json',rows)
                print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':main()
