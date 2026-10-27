# Chunru pi05: complete lemon episode evaluation

**The full-episode evaluation is complete.** Two action execution horizons were
tested from the same initial state as the previous short review. Both reached
the predefined 1,000-step limit without native task success. Replanning every
10 actions achieved grasping and transport near the target but did not complete
placement. Executing 50 actions per query did not establish a meaningful lift.

[Full side-by-side execution video](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-comparison.mp4):
10-action replanning on the left, 50-action execution on the right. Each rollout
contains 1,001 frames, approximately 33.3 seconds of simulator time.

## Protocol

The newly shared LeRobot pi05 checkpoint and its normalization were loaded
strictly. The simulator was Genesis 1.3.1, RoboWits `db6acf3`, with native
EE_ROT6D observations and EE_ABS_ROT6D actions. The fixed novel layout was case 0,
seed 202610270. This is a layout variation, not a reproduction on a verified
author test distribution.

Each policy query produces 50 actions. One condition executes ten then observes
again; the other executes all 50, matching the checkpoint's configured
`n_action_steps=50`. Both run until native success or the experimental 1,000-step
limit. Native success requires the lemon to be on the table, overlap the green
target by at least 80%, be released, and satisfy the relevant boundary checks.

No WM or VLM participated. No pi05 training was performed, and Amber's training
code, configuration, data, and original weights were unchanged.

## Results

| Execution horizon | Steps | Policy queries | Success | Finger–lemon contact frames | Maximum lift |
|---|---:|---:|---|---:|---:|
| 10 actions | 1,000 | 100 | No; step limit | 654 | 70.6 mm |
| 50 actions | 1,000 | 20 | No; step limit | 126 | 4.9 mm |

The 10-action rollout moves a barrier, grasps the lemon, and transports it near
the target. At step 1,000 both left fingers still contact the lemon, with summed
contact-force norms approximately 9.28 N. The lemon center is approximately
0.808 m high above a table surface at 0.760 m, and 3.48 cm from the target center
in the horizontal plane. Video and contacts indicate retained grasp near the
target, without completed placement.

The 50-action rollout touches the lemon but remains below the diagnostic 25 mm
lift threshold. The lemon finishes far from the target. Its 816 robot–barrier
contact frames are not automatically collisions to penalize: moving a barrier
is part of the task.

Both action trajectories were replayed in the simulator with **zero maximum
joint error** and matching final success. Both have `terminated=false` and
`truncated=true`. Contact counts use a 0.01 N threshold and are frame counts,
not independent grasp attempts. Rollout wall times were 70.6 and 43.9 seconds,
excluding initialization and diagnosis; partial concurrent execution means
these are not controlled throughput benchmarks.

## Scope and artifacts

Initial states match exactly. The first 50 actions of the 10-action rollout
exactly match the earlier short baseline, with zero joint difference. Both
settings also share an identical first predicted 50-action chunk. Subsequent
differences arise from execution horizon and observation/resampling timing.

These are **two settings on one initial state**, not two independent samples
for estimating a success rate. The results do not establish that the checkpoint
fails all lemon scenes. They show incomplete placement within this budget and
more task progress with 10-action replanning on this layout.

Future WM evaluation should inspect barrier handling, grasping, and release /
placement rather than only the first 50 steps. Qwen's previously observed
overconfident scoring remains unresolved, so no extended WM loop was launched.

[10-action results and contact audit](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-case0/contact-replay.json),
[50-action results and contact audit](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-case0-execute50/contact-replay.json),
[pair/prefix validation](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-pair-validation.json).
The Chinese report links individual videos, keyframes, and run specifications.

WAP-only code changes allow 50-action execution in baseline mode and make the
contact audit honor recorded action/observation modes while saving termination
flags and final object positions. Validation: two full inference rollouts,
two exact physical replays, complete video decoding, and 28 tests passed /
6 skipped. All experiment runs have ended.
