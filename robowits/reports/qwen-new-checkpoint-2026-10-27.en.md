# WAP on RoboWits: released pi05 and illustrated Qwen value scoring

**The 50-step review is complete and paused. No selection benefit was observed:
all ten candidates scored 0.95, all five choices retained candidate zero, and
all three execution trajectories were identical.** This experiment uses the pi05 checkpoint newly shared by Chunru and local
Qwen3.5-9B for scoring. No pi05 training was performed. Amber's training code,
configuration, data, and original weights were unchanged. The experiment is
bounded to a 50-step paired review, not a task-success evaluation.

## Checkpoint and inference contract

[Author's download](https://drive.google.com/file/d/1hdynNWbjeJBkGvOe8Dcf2ZxpvNJzGg3U/view)
and [training entry point](https://github.com/UMass-Embodied-AGI/RoboWits/blob/main/scripts/robowits/train/train_pi05.sh).

The 11,745,224,387-byte archive contains pi05, pi0, and ACT. Only pi05 was
extracted, with ZIP CRC verification. It is stored separately at
`/data/mjw/world-action-planner/chunru-release/20261027/release_chekcpoints/pi05`.

The adapter uses the author's native LeRobot PI05 implementation and checkpoint
pre/postprocessors, including QUANTILES normalization. Inputs and outputs are
20-dimensional EE_ROT6D and EE_ABS_ROT6D, with 50 actions per chunk at native
approximately 30 Hz. The policy receives the three camera images, state, and
whole-task text, without annotated subgoals. The simulator is Genesis 1.3.1,
RoboWits commit `db6acf3`, in a read-only worktree.

Weights are loaded strictly. Safetensors tied-embedding aliases are restored only
when declared in the archive metadata and genuinely tied in the native model.
There is no random-weight fallback. The dataset configuration names
`local/pi05-EE_ABS_ROT6D-_full`; exact task coverage, saved training step, and joint
holdout coverage remain unverified. This differs from the old lemon/subgoal
checkpoint in conditioning and action format, so it is not a weights-only ablation.

## Qwen examples and calibration

Each candidate is scored independently using three camera-specific temporal
strips, each showing the current frame and three future timestamps. Three visual
demonstrations illustrate static/unheld (0), new grasp (0.6), and static/already
held (0). They come from WM training episode 105, frames 440–480. Labels are
provisional and pending human review. The real positive control is episode 101;
the paired run uses a predefined new layout.

The prompt also receives per-camera pixel-change measurements computed only from
the supplied images. These do not override Qwen's scores or use object-state
ground truth. The prompt states that pixel change alone is not task progress.

| Calibration variant | Real transport | Static control | Finding |
|---|---:|---:|---|
| Text examples only | 0 | 0 | Fails positive control |
| Visual examples and separate cameras | 0.8 | 0.8 | Hallucinates transport in static held-object images |
| Plus image-change measurements | 0.8 | 0 | Separates this small control pair |

For the final variant, scores follow content when order is reversed; identical
real clips score 0.8/0.8. Four near-static predicted candidates and a static image
with ±2 pixel noise score zero. Some explanations still misidentify objects or
invent camera motion. Passing these small controls does not establish reward
reliability on new imagined trajectories.

## Paired protocol

New layout case 0, seed 202610270; two candidates per decision. Thirty native
actions condition a full second of WM prediction (20 frames), without padded
actions. The selected candidate executes ten actions. Five decisions end at
50 steps, approximately 1.67 seconds of simulator time.

Eight samples at the initial observation have mean pairwise intended endpoint
distance 5.14 cm and maximum 8.73 cm at one second. Repeating candidate zero with
the same seed gives zero difference. These are command-space distances, not
physical displacement or success, and are not directly comparable to earlier
measurements with a different checkpoint, state, and horizon.

Conditions are policy baseline, WM/Qwen scoring while forcing candidate zero,
and score-based selection. An alternative must beat a valid candidate zero by
strictly more than 0.10. Inputs, seeds, initial state, actions, and joints are
checked before divergence. Matching candidates and predictions are explicitly
reused; fresh draws and cache usage are recorded. Cached wall time is not
deployment latency.

## Execution result

All three conditions stopped after 50 steps and five decisions, with success
reported as null. Both paired conditions exactly match baseline actions; maximum
joint differences are zero. Selection reused all ten predictions and scores
after verifying matching inputs. No task-success rate can be reported.

At steps 0, 10, 20, 30, and 40, Qwen scored both candidates 0.95. It repeatedly
claimed completed placement without sufficient visible evidence. At step 30 it
even treated the lemon disappearing from a wrist view as evidence of transport
and release; visual inspection instead shows the lemon near the barriers.
**The current value model fails this actual-candidate check despite passing the
small calibration controls.** This does not establish that WM prediction itself
is useless, but it prevents a claim of useful WM-based action selection.

Mean uncached time per decision: policy 0.61 s, robot proxy 6.22 s, two WM
predictions 89.46 s, Qwen 8.94 s—approximately 105 s to select 0.333 s of executed
actions. Baseline took 7.1 s, forced-zero scoring 534.5 s, and cached selection
40.8 s, excluding initialization. Cached selection is not deployment latency.

Human review should prioritize the
[step-30 candidates](/data/mjw/world-action-planner/chunru-release/20261027/review/candidates-step-30.mp4),
[three execution videos side by side](/data/mjw/world-action-planner/chunru-release/20261027/review/execution-comparison.mp4),
and [provisional visual examples](/data/mjw/world-action-planner/chunru-release/20261027/review/value-examples.mp4).
The [Markdown review](/data/mjw/world-action-planner/chunru-release/20261027/review/review.md)
links every decision's candidate video. Confirm whether placement is visible
and whether the demonstration labeled new grasp merits that label; mark ambiguous
clips as unjudgeable.

The next useful step is human-verified difficult negative examples, such as arm
motion without grasping or placement, followed by validation on different
candidate windows. Do not expand the success-rate experiment with this reward.
All experiment runs are paused; there is no automatic continuation. This batch
used local Qwen and made no Claude API calls.

## Artifacts and validation

[Positive controls](/data/mjw/world-action-planner/chunru-release/20261027/illustrated-measured-positive/results.json),
[negative controls](/data/mjw/world-action-planner/chunru-release/20261027/illustrated-negative/results.json),
[example manifest](/data/mjw/world-action-planner/chunru-release/20261027/visual-examples/manifest.json),
[run specification](/data/mjw/world-action-planner/chunru-release/20261027/rot6d-spec.json),
[checkpoint/config/example hashes](/data/mjw/world-action-planner/chunru-release/20261027/release-provenance.json).

`rot6d_loop.py` implements the inference adapter and paired run;
`illustrated_value.py` and `vlm_server.py` implement illustrated scoring;
`build_value_examples.py` rebuilds the examples from the existing training video.
All nine rebuilt example PNGs match this experiment byte-for-byte.

Tests in the simulator Python environment: 28 passed, 6 skipped. Strict real
checkpoint loading, native execution, and actual Qwen image processing were also
exercised. Skipped tests are not counted as passes. See the Chinese report for
the detailed review notes.
