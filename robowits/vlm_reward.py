"""VLM outcome judging through a local image/text socket service.

Only current RGB, imagined RGB, task text and the current subgoal are sent.
No simulator reward, object position, or true future observation is accepted.
"""
import json
import math
from pathlib import Path
import socket

import numpy as np
from PIL import Image, ImageDraw


def generate(socket_path, prompt, images, out, *, examples=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    request = {'prompt': prompt, 'images': [str(Path(p).resolve()) for p in images]}
    if examples is not None:
        request['examples'] = examples
    (out/'request.json').write_text(json.dumps(request, indent=2))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(300)
        client.connect(str(socket_path))
        client.sendall((json.dumps(request)+'\n').encode())
        with client.makefile('rb') as stream:
            response = json.loads(stream.readline(8_000_000))
    (out/'response.json').write_text(json.dumps(response, indent=2))
    if 'error' in response or response.get('reached_token_limit'):
        raise RuntimeError('VLM failed or truncated its response; see response.json')
    return response['raw']


def parse_rewards(raw, count):
    text = raw.rsplit('</think>', 1)[-1].strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    data = json.loads(text, object_pairs_hook=pairs)
    if not isinstance(data, dict) or set(data) != {'candidates'} or not isinstance(data['candidates'], list):
        raise ValueError('Expected candidates JSON object')
    if len(data['candidates']) != count:
        raise ValueError('Missing or extra candidate judgments')
    scores = [None]*count
    for row in data['candidates']:
        if not isinstance(row, dict) or set(row) != {'id', 'valid', 'score', 'reason'}:
            raise ValueError('Unexpected reward fields')
        i = row['id']
        if type(i) is not int or not 0 <= i < count or scores[i] is not None:
            raise ValueError('Invalid or duplicate candidate ID')
        if type(row['valid']) is not bool or type(row['score']) not in (int, float):
            raise ValueError('Invalid reward types')
        if not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            raise ValueError('Score must be finite and within [0,1]')
        if not isinstance(row['reason'], str) or not row['reason'].strip() or row['reason'].strip().lower() == 'brief visual evidence':
            raise ValueError('Require a judgment reason')
        scores[i] = row
    return scores


class VLMReward:
    def __init__(self, socket_path, *, examples=False):
        self.socket_path = socket_path
        self.examples = examples

    def judge(self, current, predictions, task, subgoal, out, *, valid_future_frames=6):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        current_path = out/'current.png'
        Image.fromarray(current).save(current_path)
        images = [current_path]
        for i, pred in enumerate(predictions):
            # The rest is terminal-pose padding needed by the fixed WM horizon.
            if len(pred) < valid_future_frames:
                raise ValueError('Prediction shorter than candidate action horizon')
            indices = np.unique(np.linspace(0, valid_future_frames-1, 4).round().astype(int))
            sheet = Image.new('RGB', (448*len(indices), 472), 'white')
            draw = ImageDraw.Draw(sheet)
            for j, t in enumerate(indices):
                sheet.paste(Image.fromarray(pred[t]).resize((448, 448)), (448*j, 24))
                draw.text((448*j+4, 4), f'candidate {i}; +{(t+1)/20:.2f}s', fill='black')
            path = out/f'candidate-{i}.png'
            sheet.save(path)
            images.append(path)
        prompt = f'''Task: {task}\nCurrent subgoal: {subgoal}
You are an outcome judge, not an action proposer. The first image is the current observation.
Each following image is a labeled candidate's imagined future, ordered in time.
Each frame has duplicated ego views above, right wrist below-left and left wrist below-right.
Rank the predicted outcomes by useful progress toward the task and current subgoal.
Moving an obstacle to enable a later grasp can be useful even if lemon-target distance does not decrease.
Consider coherent robot/object motion, grasp/transport stability, visibility and apparent collision risk.
These are short, uncertain WM predictions, not ground truth. Do not infer unseen task completion.
Object disappearance, duplication, teleportation or inconsistent views are reasons to distrust a candidate.
Mark uninterpretable/unreliable predictions valid=false. Score trustworthy outcomes from 0 (worst) to 1 (best).
Judge all {len(predictions)} candidates, with integer IDs 0 through {len(predictions)-1}.
For an unchanged scene with no observable progress, score 0. Do not invent a grasp merely because the subgoal asks for one.
A grasp requires visible object contact/retention and subsequent object motion with the gripper. If that is not visible, say so.
Return a JSON object with one key, candidates, containing exactly one object per candidate.
Each object must have exactly these fields: id (integer), valid (boolean), score (number from 0 to 1),
reason (a specific description of what visibly changes or stays unchanged).
Do not echo instructions or use generic placeholder reasons. No markdown.'''
        if self.examples:
            prompt += '\n\n' + (Path(__file__).parent/'prompts/value_examples_v1.txt').read_text()
        raw = generate(self.socket_path, prompt, images, out/'vlm')
        return parse_rewards(raw, len(predictions))


class IndependentVLMReward(VLMReward):
    """Score each candidate in its own request; original IDs never enter the prompt.

    This removes cross-candidate display order as an input, but does not establish
    visual correctness or calibrate scores. All raw individual responses are kept.
    """
    def judge(self, current, predictions, task, subgoal, out, *, valid_future_frames=6):
        scores=[]
        for i,pred in enumerate(predictions):
            score=super().judge(current,[pred],task,subgoal,Path(out)/f'candidate-{i}',
                                valid_future_frames=valid_future_frames)[0]
            scores.append(dict(score,id=i))
        return scores
