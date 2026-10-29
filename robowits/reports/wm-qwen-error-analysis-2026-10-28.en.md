# Error analysis: full WAP / RoboWits loop

The strongest demonstrated failure is **Qwen's progress judgment under the
current visual-example context**. It assigns the highest progress score to
unfinished outcomes and selects an action at step 250 that is sufficient to
destroy later grasping progress in this fixed trial. Wrist-view prediction and
contact-dependent robot conditioning introduce additional errors.

Both the policy baseline and WM/Qwen run fail the task. This is a regression in
grasp/transport progress, not a measured drop from successful to failed task
execution. The evidence concerns one initial state and fixed noise seeds.

## Evidence collected

- All 100 decisions and 200 raw Qwen replies were audited.
- Predicted and executed images were compared over each decision's actually
  executed 0.30-second prefix, using three unique cameras.
- Both saved candidates at all six switching points were executed in the real
  simulator for a full second: 12 offline counterfactual branches.
- Two full continuations isolated the first switch and validated the diagnostic
  runner against the baseline.
- Qwen was tested with original requests, without visual examples, without
  image-change measurements, on real simulator video, on static images, and on
  a real transport positive control.
- At steps 250 and 760, WM prediction was repeated using actual future robot
  skeletons instead of the robot-only proxy's skeletons, for offline diagnosis.

No policy training or changes to Amber's training files were performed. Simulator
ground truth and actual future skeletons were never used in the original online
action ranking.

## The first switch is sufficient to cause the observed regression

Baseline and selection actions are identical through the first 250 actions,
and joints through frame 250. Joint trajectories first diverge at frame 251.
For the intervention, only actions 250–259 are taken from candidate 1; thereafter
the policy samples candidate zero from fresh observations with the original
decision-index seeds. This does not replay stale future actions in a changed state.

| Condition | Maximum lemon lift | Finger-contact frames | Task success |
|---|---:|---:|---|
| Policy baseline | 70.58 mm | 654 | No |
| Diagnostic runner, no switch | 70.58 mm | 654 | No |
| Only the step-250 switch | 4.43 mm | 91 | No |
| Full WM/Qwen, six switches | 2.97 mm | 107 | No |

The no-switch continuation exactly reproduces baseline actions with zero joint
error. The first ten actions of the single-switch continuation exactly reproduce
the actual intervention, also with zero joint error. There are no finger–lemon
contacts before the branch, so the continuation contact counts are comparable.

Baseline first contacts the lemon at step 320 and crosses a 25 mm diagnostic
lift threshold at step 471; contact is continuous from 450–1000. Full selection
first contacts at 369, has a longest contact interval of 61 frames, and never
crosses that lift threshold.

[Height plot](/data/mjw/world-action-planner/chunru-release/20261028/full-wm-qwen-case0/error-analysis/lemon-height-comparison.png)
and [causal comparison video](/data/mjw/world-action-planner/chunru-release/20261028/full-wm-qwen-case0/error-analysis/first-switch-causal-comparison.mp4).

## Qwen scores are effectively binary and overstate completion

Of 200 scores, **184 are 0.95 and 16 are zero**. No intermediate score or
`valid=false` occurs. Ninety decisions are ties. The six switches occur at steps
250, 570, 620, 760, 820, and 880, always comparing 0 against 0.95. Here 0.95 means
verified placement/release in the prompt; it is not a calibrated success probability.

All responses identify local Qwen3.5-9B with three visual examples. None is
truncated or fails schema parsing. Repeating the exact step-250 requests
reproduces the 0/0.95 scores. Some explanations equate remaining between the
fingers or disappearing from a wrist view with release and successful placement.

All 12 real simulator candidate branches fail to complete within their one-second
horizon. The six chosen candidates lift the lemon by at most 0.691 mm relative
to the branch state, with zero or one finger-contact frame. At step 250 both
grippers are commanded roughly 94 mm open; neither candidate contacts the lemon
within that second. Mean first-ten-action positional target differences between
candidates are 20.5 mm right / 23.6 mm left. Small nearby action changes can still
alter later contact paths in this narrow scene.

## Visual-example context contributes to the scoring error

| Step-250 scoring condition | Candidate 0 / 1 |
|---|---|
| Original request repeated | 0 / 0.95 |
| Visual example messages removed | 0 / 0 |
| Examples retained, change measurements removed | 0.95 / 0.95 |
| Full prompt, real simulator video instead of WM | 0.95 / 0.95 |
| Full prompt, static control | 0 |

Removing visual examples also produces 0/0 at the other five switch points.
A real transport positive control without examples scores 0.6, so this variant
does not simply output zero for every image. However, its stage descriptions
still contain errors. A static control can receive the correct score while its
reason incorrectly locates the lemon on the green target.

This establishes sensitivity to the current example context and a failure that
persists without WM image error. It does not identify the model's internal
attention mechanism, prove that example removal is a general fix, or demonstrate
useful ranking between two genuinely promising actions. Only known failed
candidates and one positive were tested. A full no-example loop was not run.

The earlier calibration overemphasized easy static/transport controls and lacked
occlusion, empty gripper motion, obstacle manipulation, and near-target but
unreleased-object negatives. The provisional example labels were not human
approved. That evaluation coverage was insufficient.

## WM predicts some motion, but wrist and contact behavior remain weak

For the 100 actually executed prefixes, 20 Hz prediction frames map to native
30 Hz action frames 2, 3, 5, 6, 8, and 9. This avoids comparing a one-second
open-loop prediction against a future that already contains replanning.

| View | Mean PSNR: WM / copy-last | Motion-region MSE wins |
|---|---|---:|
| Ego | 22.85 / 21.99 dB | 92/100 |
| Right wrist | 19.97 / 20.71 dB | 66/100 |
| Left wrist | 15.31 / 15.83 dB | 84/100 |

The motion mask is a maximum RGB change greater than 12 between current and true
future images; it is not an object segmentation mask. This comparison uses
compressed execution video and includes encoding error. Whole-view scores are
background-sensitive and cannot establish correct grasp physics.

On 12 full-second counterfactuals with raw rendered truth, mean WM / copy-last
PSNR is 21.95/20.23 dB ego, 18.72/19.27 right wrist, and 14.25/14.64 left wrist.

The robot-only proxy excludes task objects. Its maximum selected-candidate
left-tip position error is only 0.33 mm at the first switch, but reaches 51.80 mm
at step 760. Replacing its skeleton with the actual robot trajectory changes
step-250 PSNR by approximately zero; at step 760 it improves ego by 2.60 dB and
left wrist by 3.72 dB, while right wrist declines by 0.21 dB. Left-wrist MSE falls
from 3,890 to 1,651. Thus conditioning error contributes to later prediction
error, but does not explain the first wrong choice. Repeating both original proxy
predictions on the diagnostic GPU reproduces the saved videos pixel-for-pixel,
excluding rerun randomness or GPU changes as the explanation for this comparison.

These are oracle-robot-conditioning diagnostics, not deployable planning:
future robot joints are known only after the offline counterfactual. No future
object images are provided to WM.

## Priorities and limits

First validate the value model on human-approved, disjoint calibration/test
examples from the actual candidate distribution. Removing current examples is
one candidate variant, not an established fix. Require explicit observable
evidence of object location, contact, motion, and release; uncertainty should
have a validated conservative fallback rather than unsupported high scores.

Next test phase-aware progress judgments. The current `Current subgoal` repeats
the whole-task text, and candidates are nearby VLA samples rather than distinct
VLM plans. The causal contribution of this missing phase context was not isolated.
Separately improve wrist prediction and contact-dependent robot conditioning.
Only after offline checks should another full matched rollout be justified.

Do not interpret these results as a general success-rate estimate, as proof that
WM is wholly useless, or as evidence that pi05 must be retrained. The baseline
itself still fails final placement. No online selection logic was changed in
this analysis.

All detailed JSON, scripts, raw Qwen requests, plots, and videos are under
`/data/mjw/world-action-planner/chunru-release/20261028/full-wm-qwen-case0/error-analysis`.
The Chinese report links each artifact. Validation includes 12 exact-prefix
simulator branches, two complete continuations, and 29 tests passed / 6 skipped.
