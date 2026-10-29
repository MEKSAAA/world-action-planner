"""Cross-score saved WM or actual branch videos without rerunning policy or WM."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .closed_loop import file_sha256, write_json
from .multigoal_loop import backend
from .multigoal_value import TaskValue, select_outcome


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','videos','judge-config','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--kind',choices=['wm','actual'],required=True)
    p.add_argument('--decision',type=int,default=0)
    p.add_argument('--reuse-recorded-primary',action='store_true',
        help='Keep the actual online primary scores when cross-scoring a recorded WM decision')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    config=json.loads((args.run/'protocol.json').read_text())
    overrides=json.loads(args.judge_config.read_text());config.update(overrides)
    decision=json.loads((args.run/'decisions.json').read_text())[args.decision]
    if args.reuse_recorded_primary:
        if args.kind!='wm' or args.videos.resolve()!=(args.run/f'decision-{args.decision:04d}').resolve():
            raise ValueError('Recorded scores require the original online WM decision directory')
        original=json.loads((args.run/'protocol.json').read_text())
        for key in ('gemini_model','gemini_thinking_level','qwen_thinking','value_candidate_text','all_invalid_action'):
            if config.get(key)!=original.get(key):
                raise ValueError(f'Cannot reuse primary scores after changing {key}')
        primary=config['primary_judge']
        if not decision['wm'].get(primary):
            raise ValueError('The source decision has no online primary WM scores')
    languages=[d['goal']['language'] for d in decision['candidate_map']]
    predictions=[];current=None;bank_hashes=[]
    for i in range(len(languages)):
        folder=args.videos/f'candidate-{i}'
        if args.kind=='wm':
            prediction=np.load(folder/'prediction.npy')[:6]
            with np.load(folder/'input.npz') as data:initial=data['history'][-1]
        else:
            prediction=np.load(folder/'truth.npy')
            initial=np.asarray(Image.open(folder/'current.png'))
        if current is not None and not np.array_equal(current,initial):
            raise ValueError('Candidates do not share the same current image')
        current=initial
        if len(prediction)!=6:
            raise ValueError('Expected six frames within actual action horizon')
        predictions.append(prediction)
        bank_hashes.append({name:file_sha256(folder/name) for name in
                           (('prediction.npy','input.npz') if args.kind=='wm' else ('truth.npy','current.png'))})
    if hashlib.sha256(current.tobytes()).hexdigest()!=decision['observation_sha256']:
        raise ValueError('Saved prediction history does not match the recorded decision observation')
    write_json(args.out/'source-decision.json',decision)
    task={'robowits/03-gap-retrieve-v0':'Place the lemon on the green target area.',
          'robowits/02-retrieve-cube-v0':'Retrieve the cube from the container and place it on the green target area.'}[config['task_id']]
    write_json(args.out/'protocol.json',{'source_run':str(args.run),'videos':str(args.videos),
        'kind':args.kind,'decision':args.decision,'judge_config':overrides,
        'reuse_recorded_primary':args.reuse_recorded_primary,
        'candidate_bank_sha256':bank_hashes,
        'source_sha256':{name:file_sha256(Path(__file__).with_name(name)) for name in
                        ('multigoal_rescore.py','multigoal_value.py','multigoal_backend.py')},
        'scope':'offline cross-scoring; actual future frames must never enter online selection'})
    rows={}
    for name in config['judges']:
        recorded=args.reuse_recorded_primary and name==config['primary_judge']
        if recorded:
            no_wm=decision['no_wm'][name]
            values=decision['wm'][name]
        else:
            judge=TaskValue(backend(config,name),candidate_text=config.get('value_candidate_text',True))
            no_wm=judge.without_wm(current,task,languages,args.out/name/'no-wm')
            values=judge.judge(current,predictions,task,languages,args.out/name/'outcomes',valid_future_frames=6)
        selected,reason=select_outcome(values,no_wm,
            allow_abstention=config.get('all_invalid_action','error')=='hold')
        if recorded and selected!=decision['selected']:
            raise ValueError('Recorded primary scores do not reproduce the online choice')
        rows[name]={'no_wm':no_wm,'outcomes':values,'selected':selected,'choice_reason':reason,
                    'score_source':'recorded_online' if recorded else 'offline_cross_score'}
        write_json(args.out/'scores.json',rows)
        print(json.dumps({'backend':name,'scores':[s['score'] for s in values],
                          'selected':selected,'reason':reason}),flush=True)
    write_json(args.out/'result.json',{'status':'diagnostic_complete','native_task_success':None})


if __name__=='__main__':main()
