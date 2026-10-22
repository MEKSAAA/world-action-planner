"""File-based inference worker in WAP's separate model environment."""
import argparse
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--ckpts', type=Path, required=True)
    p.add_argument('--wan', type=Path, required=True)
    args = p.parse_args()
    from .probe import make_pipeline
    import torch
    pipeline = make_pipeline(args.checkpoint, args.wan,
                             args.ckpts/'robosuite_default_prompt.pt',
                             args.ckpts/'robosuite_default_neg_prompt.pt', 'cuda')
    args.queue.mkdir(parents=True, exist_ok=True)
    (args.queue/'ready').write_text('ready')
    while not (args.queue/'stop').exists():
        for request in sorted(args.queue.glob('request-*.json')):
            response = request.with_name(request.name.replace('request-', 'response-'))
            if response.exists():
                continue
            record = json.loads(request.read_text())
            start = time.monotonic()
            try:
                seed = record['seed']
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with np.load(record['input']) as z:
                    history, past, future = (z[k] for k in ('history', 'past', 'future'))
                with torch.inference_mode():
                    output = pipeline(list(history), list(past), list(future), return_type='np')
                pred = np.stack(output.pred_frames)
                if pred.shape != future.shape:
                    raise ValueError(f'Prediction shape {pred.shape}, expected {future.shape}')
                np.save(record['output'], pred)
                result = {'status': 'complete', 'seconds': time.monotonic()-start}
            except Exception:
                result = {'status': 'error', 'traceback': traceback.format_exc()}
            temporary = response.with_suffix('.tmp')
            temporary.write_text(json.dumps(result))
            temporary.replace(response)
        time.sleep(.2)


if __name__ == '__main__':
    main()
