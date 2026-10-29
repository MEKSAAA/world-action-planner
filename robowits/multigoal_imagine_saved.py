"""Imagine a fixed, previously sampled bank and cross-score it offline.

This is a decision diagnostic, not a closed-loop task-success experiment.
The simulator is used only to restore the recorded past; candidate futures use
the robot-only proxy. Real task futures are produced separately by the audit.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np

from .closed_loop import Worker, snapshot, same_state, state16, write_json, file_sha256
from .eef_actions import EEFProxy
from .eef_loop import history_inputs
from .geometry import Marvin, pack
from .multigoal_loop import backend
from .multigoal_value import TaskValue, best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--decision',type=int,default=0)
    args=p.parse_args()
    config=json.loads((args.run/'protocol.json').read_text())
    decision=json.loads((args.run/'decisions.json').read_text())[args.decision]
    sim=Path(config['sim_root'])
    sys.path[:0]=[str(sim),str(sim/'third_party/lerobot/src')]
    import gs_gym
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'protocol.json',dict(source_run=str(args.run),decision=args.decision,
        source_protocol_sha256=file_sha256(args.run/'protocol.json'),
        source_sha256={name:file_sha256(Path(__file__).with_name(name)) for name in
                      ('multigoal_imagine_saved.py','multigoal_value.py','multigoal_backend.py')},
        online_oracle=False,scope='fixed candidate bank diagnostic; not task success'))
    env=wm=None
    try:
        env=gs_gym.get_task_class(config['task_id'])(control_mode='EE_DELTA',observation_mode='JOINT',
            max_episode_steps=config['max_steps'],genesis_backend='cpu',device='cpu',
            show_viewer=False,camera_type='rasterizer',eval_dataset_json=config.get('eval_json'))
        env.set_eval_dataset_idx(config['case'])
        obs,_=env.reset(seed=config['seed'])
        if not same_state(snapshot(env),json.loads((args.run/'initial.json').read_text())):
            raise ValueError('Initial state differs')
        fk=Marvin(Path(config['urdf']))
        history=deque(maxlen=180)
        def observe():
            history.append((env._step_count*env.dt,pack(obs['pixels']),pack(fk.skeleton(state16(env._robot)))))
        observe()
        with np.load(args.run/'trajectory.npz') as trajectory:
            for step,action in enumerate(trajectory['actions'][:decision['step']],1):
                obs,_,terminated,truncated,_=env.step(action)
                if terminated or truncated or not np.allclose(state16(env._robot),trajectory['joints'][step],atol=1e-5,rtol=0):
                    raise ValueError('Recorded prefix differs')
                observe()
        current=pack(obs['pixels'])
        if hashlib.sha256(current.tobytes()).hexdigest()!=decision['observation_sha256']:
            raise ValueError('Current observation differs')
        proxy=EEFProxy(env,fk)
        wm=Worker(SimpleNamespace(**dict(config,out=args.out,resume=False)))
        targets=np.load(args.run/f'decision-{args.decision:04d}/targets.npy')
        history_rgb,past=history_inputs(history)
        predictions=[]
        seconds=[]
        for i,target in enumerate(targets):
            folder=args.out/f'candidate-{i}';folder.mkdir()
            future=proxy.predict_targets(env,target)
            np.save(folder/'robot_only_states.npy',proxy.last_states)
            prediction,elapsed=wm.predict(folder,history_rgb,past,future,config['seed']+args.decision)
            predictions.append(prediction);seconds.append(elapsed)
            write_json(args.out/'status.json',{'status':'imagining','candidates_done':len(predictions),
                       'candidates_total':len(targets)})
        languages=[d['goal']['language'] for d in decision['candidate_map']]
        scores={}
        for provider in config['judges']:
            scores[provider]=TaskValue(backend(config,provider)).judge(current,predictions,
                env.task_description,languages,args.out/provider,valid_future_frames=6)
            write_json(args.out/'scores.json',scores)
        chosen={name:best(rows) if any(r['valid'] for r in rows) else None for name,rows in scores.items()}
        write_json(args.out/'result.json',{'status':'diagnostic_complete','wm_seconds':seconds,
            'wm_selected':chosen,'no_wm_selected':{name:best(rows) for name,rows in decision['no_wm'].items()},
            'native_task_success':None})
        write_json(args.out/'status.json',{'status':'diagnostic_complete'})
    except BaseException:
        write_json(args.out/'status.json',{'status':'error','traceback':traceback.format_exc()})
        raise
    finally:
        if wm is not None:wm.close()
        if env is not None:env.close()


if __name__=='__main__':main()
