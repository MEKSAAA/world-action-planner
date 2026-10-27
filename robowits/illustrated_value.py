"""Local Qwen value judging with real visual demonstrations and separate views."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .vlm_reward import generate, parse_rewards


def view_strips(current, prediction, out, count=20):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if count<1 or count>len(prediction):raise ValueError('Invalid scoring horizon')
    indices=np.unique(np.linspace(0,count-1,3).round().astype(int))
    frames=[current]+[prediction[i] for i in indices]
    labels=['CURRENT']+[f'+{(i+1)/20:.2f}s' for i in indices]
    boxes=[('ego',(0,0,224,224)),('right_wrist',(0,224,224,448)),('left_wrist',(224,224,448,448))]
    paths=[]
    for name,box in boxes:
        canvas=Image.new('RGB',(336*len(frames),360),'white');draw=ImageDraw.Draw(canvas)
        for i,frame in enumerate(frames):
            canvas.paste(Image.fromarray(frame).crop(box).resize((336,336)),(336*i,24))
            draw.text((336*i+5,5),name+' '+labels[i],fill='black')
        path=out/f'{name}.png';canvas.save(path);paths.append(str(path.resolve()))
    return paths


PROMPT = '''You evaluate a robot handling a yellow lemon on a white table.
The black shapes around the yellow object in wrist images are gripper fingers.
The three images are ego, right wrist, left wrist, respectively. Each image shows
ONE camera over time: CURRENT first, then future timestamps left to right.
Compare the current and future frames within each camera. Camera motion changes
the background; an object retained between moving fingers is evidence of carrying.
Assess incremental progress toward the task, not whether a grasp already existed.
No visible change -> 0.00. Small approach without contact -> 0.10. Clearing an
obstructing block -> 0.30. Visible new grasp/lift -> 0.60. Stable transport toward
target -> 0.80. Verified release/rest inside target -> 0.95.
Do not assign grasp credit merely because the instruction asks for grasping.
Unreliable disappearance/teleportation/inconsistent views -> valid=false, score=0.
An already-held object that stays stationary is not NEW progress: score 0.
Use the preceding visual examples as scoring demonstrations; they are different
episodes. Judge these new query images, not the example images. First locate the
lemon and fingers, then describe what changed across time in the reason.
Return only a JSON object whose candidates list has one object: id=0, valid as a
boolean, score as a number from 0 to 1, and reason as specific visible evidence.
Task: {task}
Current subgoal: {subgoal}'''


class IllustratedValueReward:
    def __init__(self,socket_path,examples_path):
        self.socket_path=socket_path
        self.examples=json.loads(Path(examples_path).read_text())['examples']

    def judge(self,current,predictions,task,subgoal,out,*,valid_future_frames=20):
        scores=[]
        for i,prediction in enumerate(predictions):
            folder=Path(out)/f'candidate-{i}'
            images=view_strips(current,prediction,folder,valid_future_frames)
            prompt=PROMPT.replace('{task}',task).replace('{subgoal}',subgoal)
            changes={}
            for name,(ys,xs) in {'ego':(slice(0,224),slice(0,224)),
                                  'right_wrist':(slice(224,448),slice(0,224)),
                                  'left_wrist':(slice(224,448),slice(224,448))}.items():
                delta=np.abs(prediction[:valid_future_frames,ys,xs].astype(np.float32)-current[ys,xs].astype(np.float32))
                changes[name]={'mean_absolute_rgb_change':round(float(delta.mean()),3),
                               'changed_pixel_percent':round(float((delta.max(axis=-1)>12).mean()*100),3)}
            prompt+='\nQUERY image measurements (computed from supplied images only): '+json.dumps(changes)
            prompt+='\nIf every mean_absolute_rgb_change is zero, these query frames are exactly unchanged: score 0 and do not claim motion. Nonzero pixel change alone does NOT prove grasping or progress.'
            raw=generate(self.socket_path,prompt,images,folder/'vlm',examples=self.examples)
            score=parse_rewards(raw,1)[0];scores.append(dict(score,id=i))
        return scores
