"""Read-only OpenPI checkpoint inference in a separate JAX environment.

Uses Amber's unmodified create_trained_policy and checkpoint normalization.
No trainer, optimizer, dataset writer, or checkpoint writer is invoked.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    args = p.parse_args()
    import jax
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config
    cfg = get_config('pi05_robowits_subgoal_eef')
    policy = create_trained_policy(cfg, args.checkpoint)
    args.queue.mkdir(parents=True, exist_ok=True)
    (args.queue/'ready').write_text(json.dumps({'config': cfg.name, 'horizon': cfg.model.action_horizon,
                                               'noise_dim': cfg.model.action_dim, 'checkpoint': str(args.checkpoint)}))
    while not (args.queue/'stop').exists():
        for request in sorted(args.queue.glob('request-*.json')):
            response = request.with_name(request.name.replace('request-', 'response-'))
            if response.exists():
                continue
            start = time.monotonic()
            try:
                record = json.loads(request.read_text())
                with np.load(record['input'], allow_pickle=False) as z:
                    obs = {k: z[k] for k in z.files}
                obs['prompt'] = record['prompt']
                candidates = []
                for seed in record['seeds']:
                    # Request order/count must not change candidate zero.
                    policy._rng = jax.random.key(seed)
                    noise = np.random.default_rng(seed).standard_normal(
                        (cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
                    actions = np.asarray(policy.infer(deepcopy(obs), noise=noise)['actions'], dtype=np.float32)
                    if actions.shape != (10, 14) or not np.isfinite(actions).all():
                        raise ValueError(f'Expected finite (10,14) EEF deltas, got {actions.shape}')
                    candidates.append(actions)
                np.save(record['output'], np.stack(candidates))
                result = {'status': 'complete', 'seconds': time.monotonic()-start}
            except Exception:
                result = {'status': 'error', 'traceback': traceback.format_exc()}
            temporary = response.with_suffix('.tmp')
            temporary.write_text(json.dumps(result))
            temporary.replace(response)
        time.sleep(.1)


if __name__ == '__main__':
    main()
