"""Offline simulator counterfactuals; never provide ground truth to action ranking."""
import argparse
import json
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np

from .closed_loop import snapshot, same_state, state16, write_json
from .geometry import pack
from .replay_contacts import contacts


def read(path):
    return json.loads(path.read_text())


def reset_prefix(env, run, trajectory, step, seed, case):
    env.set_eval_dataset_idx(case)
    obs,_=env.reset(seed=seed)
    if not same_state(snapshot(env),read(run/'initial.json')):
        raise ValueError('Initial state mismatch')
    max_error=0.
    for i,action in enumerate(trajectory['actions'][:step],1):
        obs,_,terminated,truncated,_=env.step(action)
        max_error=max(max_error,float(np.max(np.abs(state16(env._robot)-trajectory['joints'][i]))))
        if max_error>1e-5 or terminated or truncated:
            raise ValueError(f'Prefix replay mismatch at {i}: {max_error}')
    with np.load(run/f'decision-{step//10:04d}/policy-input.npz') as saved:
        current=dict(state=obs['agent_pos'],**obs['pixels'])
        matches={k:bool(np.array_equal(saved[k],v)) for k,v in current.items()}
    if not all(matches.values()):raise ValueError(f'Branch observations differ: {matches}')
    return obs,max_error


def physical(env,info):
    s=snapshot(env)
    return dict(step=env._step_count,success=bool(info['is_success']),objects=s['objects'],
                contacts=contacts(env),robot=s['robot'])


def branches(env,args,protocol,trajectory):
    decisions=read(args.run/'decisions.json')
    indices=[i for i,d in enumerate(decisions) if d['selected']!=0]
    rows=[]
    for index in indices:
        d=decisions[index];step=d['step'];source=args.run/f'decision-{index:04d}'
        bank=np.load(source/'candidates.npy')
        for candidate in range(2):
            folder=args.out/f'step-{step}-candidate-{candidate}';folder.mkdir()
            obs,error=reset_prefix(env,args.run,trajectory,step,protocol['seed'],protocol['case'])
            initial=snapshot(env);frames=[pack(obs['pixels'])];events=[];states=[]
            for action in bank[candidate,:30]:
                obs,_,terminated,truncated,info=env.step(action)
                frames.append(pack(obs['pixels']));events.append(physical(env,info));states.append(state16(env._robot))
                if terminated or truncated:break
            if len(states)!=30:raise RuntimeError('Counterfactual terminated before its one-second horizon')
            indices20=np.abs(np.arange(1,31)[:,None]*env.dt-np.arange(1,21)[None,:]/20).argmin(axis=0)
            gt=np.stack(frames)[indices20+1]
            np.save(folder/'truth.npy',gt);np.save(folder/'native-states.npy',states)
            imageio.imwrite(folder/'current.png',frames[0])
            with imageio.get_writer(folder/'truth.mp4',fps=20,codec='libx264') as video:
                for frame in gt:video.append_data(frame)
            proxy=np.load(source/f'candidate-{candidate}/robot_only_states.npy')
            heights=[e['objects']['lemon'][2] for e in events]
            fingers={'Link8_R','Link9_R','Link8_L','Link9_L'}
            row=dict(step=step,candidate=candidate,chosen=d['selected']==candidate,
                     score=d['scores'][candidate]['score'],prefix_joint_error=error,
                     success_within_1s=any(e['success'] for e in events),
                     max_lemon_lift_relative_branch_m=max(heights)-initial['objects']['lemon'][2],
                     finger_contact_frames=sum(bool(fingers.intersection(e['contacts'].get('lemon',{}).get('robot_links',[]))) for e in events),
                     proxy_actual_joint_max_abs_diff=float(np.max(np.abs(proxy-np.asarray(states)))),
                     native_frame_indices=(indices20+1).tolist())
            write_json(folder/'physics.json',dict(summary=row,initial=initial,events=events))
            rows.append(row);write_json(args.out/'results.json',rows)
            print(json.dumps(row),flush=True)
    return rows


def first_switch_only(env,args,protocol,trajectory):
    from .rot6d_loop import ReleasedPolicy
    from .eef_loop import candidate_seeds
    decisions=read(args.run/'decisions.json')
    first=next(i for i,d in enumerate(decisions) if d['selected']!=0)
    step=decisions[first]['step'];policy=ReleasedPolicy(Path(protocol['policy']))
    obs,error=reset_prefix(env,args.run,trajectory,step,protocol['seed'],protocol['case'])
    rows=[];actions=[];joints=[];started=time.monotonic()
    with imageio.get_writer(args.out/'continuation.mp4',fps=1/env.dt,codec='libx264') as video:
        video.append_data(pack(obs['pixels']));joints.append(state16(env._robot))
        for index in range(first,100):
            folder=args.out/f'decision-{index:04d}';folder.mkdir()
            if index==first:
                bank=np.load(args.run/f'decision-{index:04d}/candidates.npy')
                chunk=bank[args.intervention_candidate]
            else:
                bank,_=policy.sample(obs,env.task_description,candidate_seeds(protocol['seed'],index,1),folder)
                chunk=bank[0]
            for action in chunk[:10]:
                obs,_,terminated,truncated,info=env.step(action)
                actions.append(action);joints.append(state16(env._robot));rows.append(physical(env,info))
                video.append_data(pack(obs['pixels']))
                if terminated or truncated:break
            write_json(args.out/'status.json',dict(status='running',steps=env._step_count))
            if terminated or truncated:break
    initial=read(args.run/'initial.json');fingers={'Link8_R','Link9_R','Link8_L','Link9_L'}
    summary=dict(status='complete',success=bool(info['is_success']),steps=env._step_count,
                 intervention_step=step,intervention_candidate=args.intervention_candidate,
                 prefix_joint_error=error,post_intervention_policy='candidate 0, fresh observations, original decision-index seeds',
                 max_lemon_lift_from_episode_start_m=max(e['objects']['lemon'][2] for e in rows)-initial['objects']['lemon'][2],
                 finger_contact_frames_after_branch=sum(bool(fingers.intersection(e['contacts'].get('lemon',{}).get('robot_links',[]))) for e in rows),
                 terminated=bool(terminated),truncated=bool(truncated),wall_seconds=time.monotonic()-started)
    np.savez_compressed(args.out/'continuation.npz',actions=actions,joints=joints)
    write_json(args.out/'physics.json',rows);write_json(args.out/'result.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['branches','first_switch_only'],required=True)
    p.add_argument('--intervention-candidate',type=int,choices=[0,1],default=1)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    protocol=read(args.run/'protocol.json');sim=Path(protocol['sim_root'])
    if protocol['execute_steps']!=10 or protocol['max_steps']!=1000:raise ValueError('Audit targets the recorded 10/1000 experiment')
    sys.path[:0]=[str(sim),str(sim/'third_party/lerobot/src')]
    import torch,gs_gym
    torch.set_num_threads(4)
    env=gs_gym.get_task_class('robowits/03-gap-retrieve-v0')(
        control_mode=protocol['control_mode'],observation_mode=protocol['observation_mode'],
        max_episode_steps=1000,genesis_backend='cpu',device='cpu',show_viewer=False,
        eval_dataset_json=protocol['eval_json'])
    try:
        with np.load(args.run/'trajectory.npz') as trajectory:
            if args.mode=='branches':branches(env,args,protocol,trajectory)
            else:first_switch_only(env,args,protocol,trajectory)
    finally:env.close()


if __name__=='__main__':main()
