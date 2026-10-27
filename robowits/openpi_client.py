"""Managed, seeded, file-based inference client; no policy-training dependency in simulator."""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .closed_loop import ROOT, Worker, write_json


class OpenPIClient:
    wait = Worker.wait
    close = Worker.close

    def __init__(self, python, runtime, checkpoint, queue, gpu):
        self.queue = Path(queue)
        self.queue.mkdir()
        self.calls = 0
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_MEM_FRACTION='.25',
                   PYTHONPATH=str(runtime/'src')+os.pathsep+str(ROOT), OMP_NUM_THREADS='4')
        command = [str(python), '-u', '-m', 'robowits.openpi_worker', '--queue', str(self.queue),
                   '--checkpoint', str(checkpoint)]
        with (self.queue.parent/'policy-worker.log').open('w') as log:
            self.proc = subprocess.Popen(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            self.wait(self.queue/'ready', 600)
        except BaseException:
            self.proc.terminate()
            self.proc.wait()
            raise
        self.metadata = json.loads((self.queue/'ready').read_text())

    def sample(self, observation, seeds, out):
        out = Path(out)
        inp, output = out/'policy-input.npz', out/'candidates.npy'
        np.savez(inp, **{k: v for k, v in observation.items() if k != 'prompt'})
        self.calls += 1
        name = f'{self.calls:06d}'
        write_json(self.queue/f'request-{name}.json', {'input': str(inp), 'output': str(output),
                   'prompt': observation['prompt'], 'seeds': seeds})
        response = self.queue/f'response-{name}.json'
        self.wait(response, 600)
        info = json.loads(response.read_text())
        if info['status'] != 'complete':
            raise RuntimeError(info['traceback'])
        return np.load(output), info['seconds']
