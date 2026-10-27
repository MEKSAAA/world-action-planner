"""Run Amber's unmodified labeled evaluator, then replay its chunks via WAP's adapter."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from .closed_loop import write_json, snapshot, same_state, state16
from .eef_actions import chunk_targets, control_action
from .eef_loop import candidate_seeds
from .openpi_client import OpenPIClient


def run(args):
    sys.path[:0] = [str(args.sim_root), str(args.sim_root/'third_party/lerobot/src')]
    from hier_policy.rollout import run_labeled_trial, _write_mp4
    spec = importlib.util.spec_from_file_location('amber_eval', args.sim_root/'scripts/robowits/eval/eval_pi05_lemon.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    import pyarrow.parquet as pq
    meta = pq.read_table(args.dataset/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    rows = {int(r['episode_index']):r for r in meta}
    catalog = {'dataset':args.dataset, 'by_id':{}}
    for ep in args.episodes:
        rec=json.loads((args.annotations/f'reconstruction/{ep}/reconstruction.json').read_text())
        row=rows[ep]
        catalog['by_id'][ep]={'index':ep,'length':int(row['length']),
                              'recording_path':rec['recording_path'], 'file_index':int(row['data/file_index']),
                              'instruction':'Place the lemon on the green target area.'}
    args.out.mkdir(parents=True,exist_ok=False)
    env=native._make_env('cpu', max(r['length'] for r in catalog['by_id'].values()),'eef')
    scene=native.RecordedScene(env,catalog,args.out/'scenes','eef')
    client=None
    records=[]
    try:
        client=OpenPIClient(args.policy_python,args.runtime,args.checkpoint,args.out/'policy-queue',args.gpu)
        for ep in args.episodes:
            folder=args.out/f'episode-{ep}';folder.mkdir()
            scene.prepare(ep);scene.reset()
            initial=snapshot(env);write_json(folder/'initial.json',initial)
            labels=json.loads((args.annotations/f'labels/{ep}.json').read_text())['segments']
            calls=[];actions=[];joints=[state16(env._robot)]
            actual_step=env.step
            def capture(action):
                result=actual_step(action);actions.append(np.asarray(action).copy());joints.append(state16(env._robot))
                return result
            env.step=capture
            class Policy:
                def infer(self,obs):
                    d=folder/f'query-{len(calls):04d}';d.mkdir()
                    seed=candidate_seeds(ep,len(calls),1)
                    chunk,seconds=client.sample(obs,seed,d)
                    calls.append({'step':env._step_count,'chunk':chunk[0],'seconds':seconds,
                                  'prompt':obs['prompt']})
                    return chunk[0]
            start=time.monotonic()
            trial=run_labeled_trial(scene,Policy(),labels,chunk_steps=5,
                                    max_steps=catalog['by_id'][ep]['length'],record=True)
            elapsed=time.monotonic()-start
            env.step=actual_step
            _write_mp4(trial.frames,folder/'native.mp4',30)
            write_json(folder/'queries.json',[dict(c,chunk=c['chunk'].tolist()) for c in calls])
            np.savez_compressed(folder/'native.npz',actions=actions,joints=joints)
            # Replay precisely the same queried VLA chunks with WAP action math.
            scene.reset()
            if not same_state(initial,snapshot(env)):raise RuntimeError('Native/replay initial state differs')
            replay_actions=[];replay_joints=[state16(env._robot)]
            for i,call in enumerate(calls):
                count=(calls[i+1]['step'] if i+1<len(calls) else len(actions))-call['step']
                targets=chunk_targets(env.get_observations()['agent_pos_ee'],call['chunk'])
                for target in targets[:count]:
                    action=control_action(env.get_observations()['agent_pos_ee'],target)
                    obs,rew,term,trunc,info=env.step(action)
                    replay_actions.append(action);replay_joints.append(state16(env._robot))
            da=float(np.max(np.abs(np.asarray(actions)-replay_actions)))
            dq=float(np.max(np.abs(np.asarray(joints)-replay_joints)))
            result={'episode':ep,'conditioning':'ground-truth labeled subgoals, not autonomous planning',
                    'success':trial.success,'steps':trial.steps,'error':trial.error,'wall_seconds':elapsed,
                    'queries':len(calls),'adapter_action_max_abs_diff':da,'adapter_joint_max_abs_diff':dq,
                    'adapter_success':bool(info['is_success'])}
            write_json(folder/'result.json',result);records.append(result)
            write_json(args.out/'results.json',records)
            print(json.dumps(result),flush=True)
            if da!=0 or dq>1e-5 or result['adapter_success']!=trial.success:
                raise RuntimeError('WAP/native interface mismatch')
        write_json(args.out/'status.json',{'status':'complete','results':records})
    finally:
        if client is not None:client.close()
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sim-root','dataset','annotations','policy-python','runtime','checkpoint','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--episodes',nargs='+',type=int,default=[105,120,130,140,149])
    p.add_argument('--gpu',default='1')
    args=p.parse_args()
    try:run(args)
    except BaseException:
        import traceback
        if args.out.exists():write_json(args.out/'status.json',{'status':'error','error':traceback.format_exc()})
        raise


if __name__=='__main__':main()
