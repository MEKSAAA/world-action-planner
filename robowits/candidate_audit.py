"""Sample eight candidates at preselected recorded observations using an existing worker."""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .closed_loop import write_json
from .eef_loop import candidate_seeds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episode',type=int,default=105)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    folder=args.run/f'episode-{args.episode}';calls=json.loads((folder/'queries.json').read_text())
    requests=[json.loads(p.read_text()) for p in (args.run/'policy-queue').glob('request-*.json')]
    results=[]
    for n,index in enumerate([0,len(calls)//2,len(calls)-1]):
        original=next(r for r in requests if r['input']==str(folder/f'query-{index:04d}/policy-input.npz'))
        record=dict(original,seeds=candidate_seeds(args.episode,index,8),output=str(args.out/f'query-{index}.npy'))
        path=args.run/'policy-queue'/f'request-{900000+n}.json';write_json(path,record)
        response=path.with_name(path.name.replace('request-','response-'))
        deadline=time.monotonic()+180
        while not response.exists():
            if time.monotonic()>deadline:raise TimeoutError('Existing worker failed to respond')
            time.sleep(.2)
        status=json.loads(response.read_text())
        if status['status']!='complete':raise RuntimeError(status['traceback'])
        chunks=np.load(record['output'])
        pairs=list(itertools.combinations(range(8),2))
        # Sum translation deltas: intended endpoint translation relative to the query pose.
        endpoints=chunks[:,:, [0,1,2,7,8,9]].sum(axis=1).reshape(8,2,3)
        distances=[np.linalg.norm(endpoints[a]-endpoints[b],axis=1).mean() for a,b in pairs]
        original_zero=np.load(folder/f'query-{index:04d}/candidates.npy')[0]
        row={'episode':args.episode,'query':index,'step':calls[index]['step'],'prompt':calls[index]['prompt'],
             'candidate_count':8,'pairwise_mean_endpoint_distance_m':float(np.mean(distances)),
             'pairwise_max_endpoint_distance_m':float(np.max(distances)),
             'candidate_zero_repeat_max_abs_diff':float(abs(chunks[0]-original_zero).max()),
             'mean_native_delta_norm_m':float(np.linalg.norm(chunks[:,:,[0,1,2,7,8,9]].reshape(8,10,2,3),axis=-1).mean())}
        results.append(row);write_json(args.out/'results.json',results);print(json.dumps(row),flush=True)


if __name__=='__main__':main()
