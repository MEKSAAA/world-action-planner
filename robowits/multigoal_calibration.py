"""Predeclared real/static value checks; not evidence of closed-loop benefit."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .closed_loop import write_json
from .multigoal_backend import Backend
from .multigoal_value import TaskValue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--amber-run', type=Path, required=True)
    p.add_argument('--chunru-audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--socket', type=Path, required=True)
    p.add_argument('--credential-file', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # Last ten controls before simulator success; clip may contain no complete
    # release. Human labels remain pending, never assert an invented positive.
    result = json.loads((args.amber_run/'result.json').read_text())
    end = result['steps']
    start = end-10
    with imageio.get_reader(args.amber_run/'execution.mp4') as video:
        frames = np.stack([video.get_data(t) for t in range(start,end+1)])
    offsets = np.array([2,3,5,6,8,9])
    cases = [('amber_final_motion', frames[0], frames[offsets], 'Place the lemon on the green target'),
             ('amber_same_frame_repeated', frames[0], np.repeat(frames[:1],6,axis=0),
              'Place the lemon on the green target')]
    for i in (0,1):
        folder = args.chunru_audit/f'step-250-candidate-{i}'
        cases.append((f'chunru_first_switch_actual_{i}',np.asarray(Image.open(folder/'current.png')),
                      np.load(folder/'truth.npy')[:6], 'Pick up the lemon'))
    write_json(args.out/'manifest.json', {'cases':[c[0] for c in cases],
        'human_review':'pending; static query is exactly identical pixels',
        'amber_video_frames':[start,end], 'score_type':'actual observations, not WM predictions',
        'scope':'value diagnostic; cannot establish full episode success-rate improvement'})
    rows = []
    for name,current,prediction,language in cases:
        folder=args.out/name
        folder.mkdir()
        np.save(folder/'truth.npy',prediction)
        Image.fromarray(current).save(folder/'current.png')
        for provider in ('gemini','qwen'):
            judge=TaskValue(Backend(provider,socket=args.socket,credential_file=args.credential_file))
            scores=judge.judge(current,[prediction],'Place the lemon on the green target area.',
                              [language],folder/provider,valid_future_frames=6)
            rows.append({'case':name,'backend':provider,'score':scores[0]})
            write_json(args.out/'results.json',rows)
            print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':
    main()
