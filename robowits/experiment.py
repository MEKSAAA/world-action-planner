"""Run paired zero-shot probes, two-GPU finetuning and a matched final probe.

Launch with a persistent process (e.g. nohup). Each stage has its own log;
status.json records failures instead of silently advancing to the next stage.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .compare import compare
from .train import validate_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--ckpts', type=Path, required=True)
    p.add_argument('--wan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    gpus = args.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2:
        p.error('This experiment requires two distinct GPUs')
    if args.steps < 1:
        p.error('steps must be positive')
    args.out = args.out.resolve()
    args.data = args.data.resolve()
    args.ckpts = args.ckpts.resolve()
    args.wan = args.wan.resolve()
    rows = validate_data(args.data)
    if len({r['task_index'] for r in rows}) != 1:
        p.error('Use exactly one task for this experiment')
    checkpoints = {name: args.ckpts/name/'checkpoints/latest.ckpt'
                   for name in ('libero_90_base', 'robosuite_ft')}
    for path in (*checkpoints.values(), args.wan/'Wan2.1_VAE.pth',
                 args.ckpts/'robosuite_default_prompt.pt', args.ckpts/'robosuite_default_neg_prompt.pt'):
        if not path.is_file():
            raise FileNotFoundError(path)
    windows = args.out/'windows'
    def module(name, *values):
        return [sys.executable, '-u', '-m', 'robowits.'+name, *map(str, values)]
    def probe(name, checkpoint):
        return module('probe', '--windows', windows, '--checkpoint', checkpoint,
                      '--wan', args.wan, '--prompt', args.ckpts/'robosuite_default_prompt.pt',
                      '--negative-prompt', args.ckpts/'robosuite_default_neg_prompt.pt',
                      '--out', args.out/name, '--seed', 0, '--lpips')
    commands = {
        'windows': module('windows', '--data', args.data, '--out', windows, '--windows-per-episode', 4),
        **{name: probe(name, ckpt) for name, ckpt in checkpoints.items()},
        'train': module('train', '--data', args.data, '--init', checkpoints['robosuite_ft'],
                        '--wan', args.wan, '--run-dir', args.out/'train', '--gpus', args.gpus,
                        '--steps', args.steps, '--batch-size', 1, '--accum', 4,
                        '--workers', 4, '--save-every', 250, '--keep-every', args.steps,
                        '--val-every', 1000, '--sanity-batches', 0),
        'finetuned': probe('finetuned', args.out/'train/checkpoints/last.ckpt'),
    }
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'commands.json').write_text(json.dumps(commands, indent=2))
    state = {'pid': os.getpid(), 'started': datetime.now(timezone.utc).isoformat(),
             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'data': str(args.data), 'steps': args.steps, 'effective_batch': 8, 'stages': {}}
    def status(stage, value, **details):
        state['stages'][stage] = {'status': value, 'time': datetime.now(timezone.utc).isoformat(), **details}
        tmp = args.out/'status.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(args.out/'status.json')
        print(stage, value, details, flush=True)
    def run_group(jobs):
        children = []
        try:
            for name, gpu in jobs:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
                log = (args.out/(name+'.log')).open('w')
                try:
                    proc = subprocess.Popen(commands[name], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                finally:
                    log.close()
                children.append((name, proc))
                status(name, 'running', pid=proc.pid, gpu=gpu)
            while children:
                for name, proc in list(children):
                    code = proc.poll()
                    if code is None:
                        continue
                    status(name, 'complete' if code == 0 else 'failed', exit_code=code)
                    children.remove((name, proc))
                    if code:
                        raise RuntimeError(f'{name} exited {code}; inspect {args.out}/{name}.log')
                if children:
                    time.sleep(10)
        finally:
            for name, proc in children:
                proc.terminate()
                proc.wait()
                status(name, 'terminated_after_peer_failure')
    try:
        run_group([('windows', gpus[0])])
        run_group([('libero_90_base', gpus[0]), ('robosuite_ft', gpus[1])])
        run_group([('train', args.gpus)])
        import torch
        checkpoint = torch.load(args.out/'train/checkpoints/last.ckpt', map_location='cpu', weights_only=False, mmap=True)
        step = checkpoint['global_step']
        del checkpoint
        if step != args.steps:
            raise RuntimeError(f'Expected {args.steps} optimizer steps, checkpoint has {step}')
        run_group([('finetuned', gpus[0])])
        tuned = json.loads((args.out/'finetuned/metrics.json').read_text())
        report = ['# RoboWits single-task world-model evaluation', '',
                  f'Optimizer steps: {step}. Effective batch: 8. Seed: 0.', '',
                  'Three held-out episodes, four windows per episode; 80 history + 40 future frames at 20 Hz.',
                  'Measured future robot poses condition the model. This is not closed-loop policy success.',
                  'Robot-region radius is 15 px. Changed-region scores are not object/contact masks.',
                  'One seed and the same task/scene; generalization is not established.', '']
        for name in checkpoints:
            baseline = json.loads((args.out/name/'metrics.json').read_text())
            report += [f'## {name} versus finetuned robosuite_ft', '', compare(baseline, tuned), '']
        (args.out/'report.md').write_text('\n'.join(report))
        status('experiment', 'complete', report=str(args.out/'report.md'))
    except BaseException as exc:
        status('experiment', 'failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
