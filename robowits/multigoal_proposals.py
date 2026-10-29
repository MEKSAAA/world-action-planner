"""Generate diverse grounded subgoals without consulting simulator object state."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .closed_loop import write_json
from .multigoal_backend import strict_json, validated_complete


GROUNDING = '''Use only the CURRENT ego RGB image. Anchor [x,y] is a visible object
surface in normalized image coordinates [0,1], origin top-left. Offset [dx,dy,dz]
is meters in robot base coordinates: +X away from robot, +Y robot's left, +Z up.
Choose arm left, right, both, or none. For none set anchor and offset to zero.
Each language is a short imperative of 2–12 words. Do not command joint values.
Every goal has exactly language, arm, anchor, offset. Do not invent invisible objects.
Return JSON only, with one key goals containing the requested list of goals.'''


def parse_goals(raw, count):
    from hier_policy.planner import parse_subgoal
    data = strict_json(raw)
    if set(data) != {'goals'} or not isinstance(data['goals'], list) or len(data['goals']) != count:
        raise ValueError('Wrong number or schema of subgoals')
    goals = []
    for row in data['goals']:
        if not isinstance(row, dict):
            raise ValueError('Subgoal must be an object')
        for key, length in (('anchor', 2), ('offset', 3)):
            value = row.get(key)
            if not isinstance(value, list) or len(value) != length or any(
                    type(v) not in (float, int) or not np.isfinite(v) for v in value):
                raise ValueError('Grounding coordinates must be finite numbers')
        goal = parse_subgoal('<ans>'+json.dumps(row)+'</ans>')
        if goal.arm == 'none' and any((*goal.anchor, *goal.offset)):
            raise ValueError('Unconditioned goal must use zero coordinates')
        goals.append(goal)
    if len({g.language.lower() for g in goals}) != count:
        raise ValueError('Duplicate subgoal languages')
    return goals


class Proposer:
    def __init__(self, backend, count=4, manual=None):
        if not 3 <= count <= 5:
            raise ValueError('Require 3–5 subgoals')
        self.backend, self.count = backend, count
        self.manual = json.loads(Path(manual).read_text()) if manual else None
        if self.manual and self.manual.get('review_status') != 'approved':
            raise ValueError('Manual subgoal catalogue needs user review before execution')

    def propose(self, env, out, decision_seed):
        from hier_policy.planner import compose_point
        from hier_policy.readers.robowits_replay import _read_ego_rgbd
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        rgb, depth, camera = _read_ego_rgbd(env)
        image = out/'ego.png'
        Image.fromarray(rgb).save(image)
        stage = None
        if self.manual:
            stages = self.manual['tasks'][env.task_description]['stages']
            # Only observational conditions enter this request. Never pass audit
            # labels, preferred-goal indices or counterfactual simulator futures.
            visible_stages = {name: item['visible_condition'] for name, item in stages.items()}
            raw = self.backend.complete('Classify the CURRENT visible scene. Task: '+env.task_description+
                '\nStages: '+json.dumps(visible_stages)+
                '\nReturn only {"stage":"one listed stage name"}.', [image], out/'stage')
            stage_data = strict_json(raw)
            if set(stage_data) != {'stage'} or stage_data['stage'] not in stages:
                raise ValueError('Invalid visual stage')
            stage = stage_data['stage']
            languages = stages[stage]['languages']
            if len(languages) != self.count:
                raise ValueError('Manual catalogue has wrong number of candidates')
            # Shuffle before grounding and keep private labels separate.
            order = np.random.default_rng(decision_seed).permutation(self.count)
            languages = [languages[i] for i in order]
            prompt = ('Task: '+env.task_description+'\nGround these exact candidate instructions '
                      'in this order. Preserve every language verbatim; do not replace unhelpful goals.\n'+
                      json.dumps(languages)+'\n'+GROUNDING)
        else:
            languages = None
            prompt = (f'Task: {env.task_description}\nPropose {self.count} distinct plausible next subgoals. '
                      'Consider different objects, prerequisites, or approaches, not synonymous descriptions. '
                      'Every candidate should be a sincere attempt to help. Do not deliberately inject errors.\n'+GROUNDING)
        goals = validated_complete(self.backend, prompt, [image], out/'grounded-goals',
                                   lambda raw: parse_goals(raw, self.count))
        if languages is not None and [g.language for g in goals] != languages:
            raise ValueError('Grounder changed manual candidate languages')
        if languages is None:
            order = np.random.default_rng(decision_seed).permutation(self.count)
            goals = [goals[i] for i in order]
        grounded = []
        for goal in goals:
            valid = goal.arm != 'none'
            point = compose_point(goal.anchor, goal.offset, depth, camera) if valid else np.zeros(3, np.float32)
            if not np.isfinite(point).all():
                raise ValueError('Non-finite RGB-D grounding')
            grounded.append((goal, point, valid))
        write_json(out/'goals.json', {'mode': 'manual' if self.manual else 'vlm', 'stage': stage,
            'goals': [dict(asdict(g), point_3d=p.tolist(), valid=v) for g,p,v in grounded],
            'source': 'current RGB + depth; no simulator object-state oracle'})
        return grounded
