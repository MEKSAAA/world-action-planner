"""Task-level value and no-WM control with backend-neutral text examples."""
import json
from pathlib import Path

from PIL import Image

from .illustrated_value import view_strips
from .vlm_reward import parse_rewards
from .multigoal_backend import validated_complete


RUBRIC = '''Evaluate progress toward the OVERALL task, not obedience to a candidate
instruction. A candidate subgoal is only a proposal and may be inappropriate.
Use visible evidence; do not assume the instruction was successfully performed.
Scores refer to NEW progress from the current state, not an absolute task stage.
Examples: unchanged object and arm -> 0.00; approach with no contact -> 0.10;
visibly moving an obstacle out of the grasp path -> 0.30; NEW stable grasp/lift
of the task object -> 0.60; visibly carrying it toward its target -> 0.80;
visibly releasing it to rest within the target -> 0.95.
Already holding an object without further movement is not a new grasp: 0.00.
Grasping a distractor or moving the task object away from its destination does
not earn grasp/transport credit. Camera/background movement alone is not progress.
Images may be uncertain. Object disappearance, teleportation, duplicated objects,
or incompatible views make the prediction invalid; score 0 and explain.
Do not infer a release, contact or future task success that is not visible.
Return JSON only: {"candidates":[{"id":0,"valid":true,"score":0.0,
"reason":"specific visible evidence and remaining uncertainty"}]}.
Include exactly the requested candidate IDs; no extra fields.'''


class TaskValue:
    def __init__(self, backend):
        self.backend = backend

    def judge(self, current, predictions, task, languages, out, valid_future_frames=6):
        if len(languages) != len(predictions):
            raise ValueError('Candidate language/prediction mismatch')
        scores = []
        for i, (prediction, language) in enumerate(zip(predictions, languages)):
            folder = Path(out)/f'candidate-{i}'
            images = view_strips(current, prediction, folder, valid_future_frames)
            prompt = (RUBRIC+f'\nOverall task: {task}\nCandidate instruction: {language}\n'
                'Judge only candidate id 0. Three images show ego, right wrist, left wrist. '
                'Within each image CURRENT is followed by timestamped imagined future frames. '
                'Compare each camera over time. These are model predictions, not observations.')
            parsed = validated_complete(self.backend, prompt, images, folder/'vlm', lambda raw: parse_rewards(raw,1))
            scores.append(dict(parsed[0], id=i))
        return scores

    def without_wm(self, current, task, languages, out):
        """Choose a subgoal from current RGB/text, then the first seeded policy draw.

        Equal-language samples receive exactly equal scores: no hidden access
        to predictions or low-level action outcomes in this control.
        """
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        image = out/'current.png'
        Image.fromarray(current).save(image)
        unique = list(dict.fromkeys(languages))
        prompt = (f'Overall task: {task}\nChoose the most useful immediate subgoal using '
                  'only this CURRENT observation and the proposed instructions. There are no '
                  'future images. Estimate appropriateness, not achieved progress.\n'+
                  json.dumps([{'id': i, 'instruction': s} for i,s in enumerate(unique)])+
                  '\nReturn JSON with exactly one key candidates, one entry per ID, with fields '
                  'id (integer), valid (boolean), score (0 to 1), reason (visible supporting evidence). '
                  'Exact structure: {"candidates":[{"id":0,"valid":true,"score":0.5,"reason":"specific evidence"}, ...]}. '
                  'The image contains duplicate ego panels above, right wrist below-left and left wrist below-right.')
        scores = validated_complete(self.backend, prompt, [image], out/'vlm', lambda raw: parse_rewards(raw,len(unique)))
        return [dict(scores[unique.index(s)], id=i) for i,s in enumerate(languages)]


def best(scores):
    valid = [s for s in scores if s['valid']]
    if not valid:
        raise RuntimeError('Every candidate was rejected; no action executed')
    # Candidate list is shuffled before sampling; ties favor the first saved draw.
    return max(valid, key=lambda s: s['score'])['id']
