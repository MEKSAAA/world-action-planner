"""Read-only inference of the author's 20D LeRobot pi05 checkpoint with WAP/Qwen.

The policy proposes absolute native EE_ROT6D targets. The first 30 of its 50
actions condition a full second of WM prediction; no terminal padding is used.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from .closed_loop import Worker, RobotProxy, snapshot, same_state, state16, write_json, file_sha256
from .closed_loop_score import choose
from .eef_loop import candidate_seeds, history_inputs, reuse_prediction
from .geometry import Marvin, pack
from .illustrated_value import IllustratedValueReward


class ReleasedPolicy:
    def __init__(self, checkpoint):
        import torch
        from safetensors.torch import load_file
        from safetensors import safe_open
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        if cfg.input_features['observation.state'].shape != (20,) or cfg.output_features['action'].shape != (20,) or cfg.chunk_size != 50:
            raise ValueError('Require the released 20D, 50-step pi05 configuration')
        cfg.device = 'cuda'
        self.policy = PI05Policy(cfg)
        # The vendored from_pretrained catches load failures and may return an
        # uninitialized model. Explicit strict loading prevents that fallback.
        weights = load_file(checkpoint/'model.safetensors')
        weights = self.policy._fix_pytorch_state_dict_keys(weights,cfg)
        weights = {k if k.startswith('model.') else 'model.'+k:v for k,v in weights.items()}
        # save_model stores one tensor for tied embeddings and records the alias.
        # Restore only aliases that are also genuinely tied in the native model.
        native = self.policy.state_dict()
        with safe_open(checkpoint/'model.safetensors',framework='pt') as saved:
            for alias,target in (saved.metadata() or {}).items():
                if alias in native and alias not in weights and target in weights:
                    if target not in native or native[alias].data_ptr()!=native[target].data_ptr():
                        raise ValueError(f'Checkpoint alias is not tied in this runtime: {alias}')
                    weights[alias]=weights[target]
        self.policy.load_state_dict(weights,strict=True)
        del weights
        self.policy.to('cuda').eval()
        self.pre,self.post = make_pre_post_processors(cfg,pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor':{'device':'cuda'}})
        self.config = cfg

    def sample(self, obs, task, seeds, directory):
        import torch
        from lerobot.envs.utils import preprocess_observation
        np.savez_compressed(directory/'policy-input.npz',state=obs['agent_pos'],**obs['pixels'])
        write_json(directory/'policy-request.json',dict(task=task,seeds=seeds))
        batch = preprocess_observation(obs)
        batch['task'] = [task]
        batch = self.pre(batch)
        started = time.monotonic()
        candidates = []
        for seed in seeds:
            gen = torch.Generator(device='cuda').manual_seed(seed)
            noise = torch.randn((1,50,self.config.max_action_dim),device='cuda',generator=gen)
            with torch.inference_mode():
                output = self.post(self.policy.predict_action_chunk(batch,noise=noise))[0].float().cpu().numpy()
            if output.shape != (50,20) or not np.isfinite(output).all():
                raise ValueError('Invalid 50x20 policy output')
            candidates.append(output)
        values = np.stack(candidates)
        np.save(directory/'candidates.npy',values)
        return values,time.monotonic()-started


def paired_candidates(source, current, candidates, *, all_candidates=False):
    with np.load(source/'policy-input.npz') as a, np.load(current/'policy-input.npz') as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):
            raise RuntimeError('Paired policy inputs differ')
    a=json.loads((source/'policy-request.json').read_text());b=json.loads((current/'policy-request.json').read_text())
    if a['task'] != b['task'] or a['seeds'][0] != b['seeds'][0] or (all_candidates and a['seeds'] != b['seeds']):
        raise RuntimeError('Paired task or seeds differ')
    saved=np.load(source/'candidates.npy')
    if all_candidates and saved.shape!=candidates.shape:raise ValueError('Paired candidate count or shape differs')
    if not (current/'fresh-candidates.npy').exists():np.save(current/'fresh-candidates.npy',candidates)
    result=candidates.copy()
    if all_candidates:result[:]=saved
    else:result[0]=saved[0]
    write_json(current/('bank-reuse.json' if all_candidates else 'zero-reuse.json'),dict(source=str(source),exact_inputs=True))
    np.save(current/'candidates.npy',result)
    return result


def rollout(env, policy, wm, proxy, fk, args):
    import imageio.v2 as imageio
    env.set_eval_dataset_idx(args.case)
    obs,info=env.reset(seed=args.seed)
    initial=snapshot(env);write_json(args.out/'initial.json',initial)
    if obs['agent_pos'].shape != (20,) or not np.array_equal(obs['agent_pos'],obs['agent_pos_ee_rot6d']):
        raise ValueError('Native observation does not match EE_ROT6D')
    reference=args.reference_run
    ref_traj=None
    if reference is not None:
        if not same_state(initial,json.loads((reference/'initial.json').read_text())):raise RuntimeError('Initial states differ')
        ref_traj=np.load(reference/'trajectory.npz')
    source=args.reuse_imagination_from
    source_decisions=json.loads((source/'decisions.json').read_text()) if source else []
    judge=IllustratedValueReward(args.vlm_socket,args.value_examples) if args.mode!='baseline' else None
    history=deque(maxlen=180);decisions=[];actions=[];joints=[];changed=False
    started=time.monotonic();result=dict(status='running',success=None,steps=0)
    limit=min(args.max_steps,args.stop_after_steps)
    with imageio.get_writer(args.out/'execution.mp4',fps=1/env.dt,codec='libx264') as writer:
        def observe():
            state=state16(env._robot);rgb=pack(obs['pixels'])
            joints.append(state);history.append((env._step_count*env.dt,rgb,pack(fk.skeleton(state))));writer.append_data(rgb)
        observe()
        try:
            while env._step_count<limit:
                index=len(decisions);folder=args.out/f'decision-{index:04d}';folder.mkdir()
                seeds=candidate_seeds(args.seed,index,1 if args.mode=='baseline' else args.num_candidates)
                candidates,policy_seconds=policy.sample(obs,env.task_description,seeds,folder)
                if reference is not None and not changed:
                    candidates=paired_candidates(reference/f'decision-{index:04d}',folder,candidates)
                    if source:candidates=paired_candidates(source/f'decision-{index:04d}',folder,candidates,all_candidates=True)
                scores=[];wm_seconds=proxy_seconds=judge_seconds=0.;reused=0;selected=recommended=0
                if args.mode!='baseline':
                    rgb,past=history_inputs(history);predictions=[]
                    for i,chunk in enumerate(candidates):
                        target=folder/f'candidate-{i}';target.mkdir()
                        then=time.monotonic();future=proxy.predict(env,chunk[:30]);proxy_seconds+=time.monotonic()-then
                        np.save(target/'robot_only_states.npy',proxy.last_states)
                        pred=reuse_prediction(source/f'decision-{index:04d}/candidate-{i}',target,rgb,past,future) if source else None
                        if pred is None:
                            pred,seconds=wm.predict(target,rgb,past,future,args.seed+index);wm_seconds+=seconds
                        else:reused+=1
                        predictions.append(pred)
                    h=hashlib.sha256(pack(obs['pixels']).tobytes()).hexdigest()
                    cached=source_decisions[index] if index<len(source_decisions) else None
                    if reused==len(predictions) and cached and cached['observation_sha256']==h:
                        scores=cached['scores'];write_json(folder/'reward-reuse.json',dict(source=str(source),scores=scores))
                    else:
                        then=time.monotonic()
                        scores=judge.judge(pack(obs['pixels']),predictions,env.task_description,env.task_description,folder/'reward',valid_future_frames=20)
                        judge_seconds=time.monotonic()-then
                    if not any(s['valid'] for s in scores):raise RuntimeError('Every prediction was judged unreliable')
                    recommended=choose(scores,margin=args.selection_margin,require_baseline=False)
                    selected=0 if args.mode=='forced_zero' else recommended
                d=dict(step=env._step_count,seeds=seeds,scores=scores,selected=selected,recommended=recommended,
                       policy_seconds=policy_seconds,wm_seconds=wm_seconds,proxy_seconds=proxy_seconds,vlm_seconds=judge_seconds,
                       reused_wm_predictions=reused,observation_sha256=hashlib.sha256(pack(obs['pixels']).tobytes()).hexdigest())
                decisions.append(d);write_json(args.out/'decisions.json',decisions)
                changed |= selected!=0
                for action in candidates[selected,:args.execute_steps]:
                    obs,_,terminated,truncated,info=env.step(action);actions.append(action);observe()
                    if ref_traj is not None and not changed:
                        n=len(actions)
                        if not np.array_equal(action,ref_traj['actions'][n-1]) or not np.allclose(joints[-1],ref_traj['joints'][n],atol=1e-5,rtol=0):
                            raise RuntimeError('Unchanged paired execution diverged')
                    result.update(steps=env._step_count,success=bool(info['is_success']))
                    if terminated or truncated or env._step_count>=limit:break
                write_json(args.out/'status.json',dict(status='running',steps=env._step_count,decisions=len(decisions),pid=os.getpid()))
                print(json.dumps(d),flush=True)
                if terminated or truncated:break
            result['status']='complete'
            if not (terminated or truncated) and env._step_count<args.max_steps:
                result.update(status='paused_for_review',success_at_stop=result['success'],success=None)
        except BaseException:
            result.update(status='error',success=None,error=traceback.format_exc());raise
        finally:
            result.update(wall_seconds=time.monotonic()-started,decisions=len(decisions),changed_choices=sum(d['selected']!=0 for d in decisions))
            np.savez_compressed(args.out/'trajectory.npz',actions=actions,joints=joints)
            write_json(args.out/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['sim-root','policy','wm-python','wm-checkpoint','ckpts','wan','urdf','eval-json','out','vlm-socket']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['baseline','forced_zero','selection'],default='baseline')
    p.add_argument('--reference-run',type=Path);p.add_argument('--reuse-imagination-from',type=Path)
    p.add_argument('--value-examples',type=Path,required=True)
    p.add_argument('--case',type=int,default=0);p.add_argument('--seed',type=int,default=202610270)
    p.add_argument('--max-steps',type=int,default=1000);p.add_argument('--stop-after-steps',type=int,default=50)
    p.add_argument('--execute-steps',type=int,default=10);p.add_argument('--num-candidates',type=int,default=2)
    p.add_argument('--selection-margin',type=float,default=.1);p.add_argument('--wm-gpu',default='0')
    args=p.parse_args()
    if not 1<=args.execute_steps<=30 or min(args.max_steps,args.stop_after_steps)<1 or args.num_candidates<2:p.error('Invalid action/review budget')
    args.out.mkdir(parents=True,exist_ok=False);args.resume=False
    sys.path[:0]=[str(args.sim_root),str(args.sim_root/'third_party/lerobot/src')]
    import torch,gs_gym
    torch.set_num_threads(4)
    env=wm=None
    try:
        protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        protocol.update(control_mode='EE_ABS_ROT6D',observation_mode='EE_ROT6D',policy_chunk=50,wm_native_steps=30,
                        wm_future_frames=20,judged_future_frames=20,padding_frames=0,
                        sim_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.sim_root,text=True).strip(),
                        checkpoint_sha256=file_sha256(args.policy/'model.safetensors'),wm_sha256=file_sha256(args.wm_checkpoint),
                        examples_sha256=file_sha256(args.value_examples),
                        source_sha256={name:file_sha256(Path(__file__).parent/name) for name in
                         ['rot6d_loop.py','vlm_reward.py','illustrated_value.py','closed_loop_score.py','closed_loop_worker.py','prompts/value_examples_v1.txt']})
        write_json(args.out/'protocol.json',protocol)
        for reference in [args.reference_run,args.reuse_imagination_from]:
            if reference is None:continue
            before=json.loads((reference/'protocol.json').read_text())
            for key in ['checkpoint_sha256','wm_sha256','sim_commit','seed','case','eval_json','max_steps','execute_steps','selection_margin','source_sha256','examples_sha256']:
                if before[key]!=protocol[key]:raise ValueError(f'Paired provenance mismatch: {key}')
            status=json.loads((reference/'status.json').read_text())
            if status['status'] not in ('complete','paused_for_review') or status['steps']<min(args.max_steps,args.stop_after_steps):
                raise ValueError('Reference does not cover the requested review window')
        write_json(args.out/'status.json',dict(status='initializing',pid=os.getpid()))
        policy=ReleasedPolicy(args.policy)
        write_json(args.out/'policy-load.json',dict(strict=True,missing_keys=[],unexpected_keys=[],training=False))
        env=gs_gym.get_task_class('robowits/03-gap-retrieve-v0')(
            device='cpu',genesis_backend='cpu',show_viewer=False,max_episode_steps=args.max_steps,
            control_mode='EE_ABS_ROT6D',observation_mode='EE_ROT6D',eval_dataset_json=str(args.eval_json))
        if not np.isclose(env.dt,1/30,atol=5e-5,rtol=0):raise ValueError('This adapter requires native 30 Hz control')
        env.set_eval_dataset_idx(args.case);env.reset(seed=args.seed)
        fk=Marvin(args.urdf);proxy=None
        if args.mode!='baseline':proxy=RobotProxy(env,fk);wm=Worker(args)
        result=rollout(env,policy,wm,proxy,fk,args);write_json(args.out/'status.json',result)
    except BaseException:
        write_json(args.out/'status.json',dict(status='error',error=traceback.format_exc()));raise
    finally:
        if wm is not None:wm.close()
        if env is not None:env.close()


if __name__=='__main__':main()
