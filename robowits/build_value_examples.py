"""Build provisional visual scoring examples from an existing training video.

This reads video only. It does not train or change a policy. The chosen source
episode and labels must be reviewed before using the reward for an outcome claim.
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .closed_loop import write_json
from .illustrated_value import PROMPT, view_strips
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True, help='ep000105/rgb.mp4 from the existing WM training set')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reader = imageio.get_reader(args.video)
    try:
        first = reader.get_data(440)
        later = np.stack([reader.get_data(i) for i in range(441,481)])
    finally:
        reader.close()
    prompt = PROMPT.replace('{task}', 'Place the lemon on the green target area.').replace(
        '{subgoal}', 'Pick up and carry the lemon toward the target')
    cases = [
        ('static_unheld', first, np.repeat(first[None],40,axis=0), 0.,
         'The lemon stays on the table; fingers and background do not change.'),
        ('new_grasp', first, later, .6,
         'The yellow lemon starts on the table and ends visibly between the left gripper fingers. Contact and retention appear across the left wrist frames; this is a new grasp, not completed placement.'),
        ('static_held', later[-1], np.repeat(later[-1: ],40,axis=0), 0.,
         'The lemon is already between the fingers in CURRENT and remains in the identical position in all future frames. There is no new movement or task progress.'),
    ]
    examples = []
    for name,current,future,score,reason in cases:
        examples.append(dict(images=view_strips(current,future,args.out/name,40), prompt=prompt,
            answer=json.dumps({'candidates':[dict(id=0,valid=True,score=score,reason=reason)]})))
    write_json(args.out/'manifest.json', dict(source_episode=105, source_frames=[440,480],
        source_video=str(args.video.resolve()),
        source_split='WM training; excludes current test layout and positive-control episode 101',
        label_status='provisional visual labels inspected by assistant, pending user review', examples=examples))


if __name__ == '__main__':
    main()
