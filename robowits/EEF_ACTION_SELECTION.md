# Lemon EEF VLA → WM → VLM action selection

The new entry point is `python -m robowits.eef_loop`. It uses Amber's **OpenPI/JAX**
checkpoint at `pi05_robowits_subgoal_eef/lemon_eef_b64/29999`, not the older LeRobot
`ckpt/pi05` model. All new implementation lives in WAP. Amber's source, training
configuration, normalization files, dataset and checkpoint are read-only.

## Loop

1. Obtain one current subgoal using the existing `hier_policy.planner` prompt and
   parser. Ground its image anchor using current ego depth. This is necessary
   because the supplied VLA was trained with subgoal language and spatial input.
   Refresh the subgoal every 50 executed steps by default. **The candidates being
   ranked are VLA action sequences under this same subgoal, not different plans.**
2. Sample K VLA action chunks from the same observation and subgoal. K defaults to
   4. Each uses an explicit independent noise seed; candidate 0's seed does not
   change with K or worker request order. Checkpoint normalization is unchanged.
3. Interpret each `(10,14)` chunk using the unmodified `hier_policy.eef` functions
   from `libero-recipe-eef`: compose per-step deltas from the query-time pose, then
   correct each target relative to the current measured pose for `EE_DELTA`.
   This matches Amber's `RecordedScene.begin_chunk/step` behavior.
4. Project candidates through a separate robot-only Genesis scene and render
   their future skeletons. Check that this does not mutate the actual task scene.
   WAP predicts RGB for every candidate with the same diffusion seed.
5. A VLM sees current RGB, the task, current subgoal, and time-labeled predicted
   multi-view frames. It scores each outcome for useful progress, motion coherence,
   grasp stability and apparent risk. This replaces the old HSV distance heuristic.
   It is a prompted VLM judge, not a separately trained reward model.
6. Execute the highest-scoring valid candidate's first five actions, acquire new
   observations, and repeat. Ties use candidate order. If all predictions are judged
   unreliable, stop with an explicit error instead of inventing a winner. Malformed
   or truncated VLM output is also an error, not a task failure.

Only native `is_success` establishes task success. VLM scores never override it.
`--force-candidate-zero` runs imagination and judging but executes candidate 0,
for control experiments. It does not by itself prove matched-trajectory causality.

## Horizon limitation

The VLA outputs 10 native steps (~0.333 s at 30 Hz) and executes five per query.
The current WAP checkpoint requires at least 20 future frames at 20 Hz (one second).
After the 10 predicted policy steps, the robot proxy **holds the last intended EE
pose** to fill the remaining 20 native steps. It does not repeat deltas or ask the
policy for fictitious observations. The VLM judge receives only the first six WM
frames (0.05–0.30 s); it never sees the padded tail. Padding is still part of the
WM conditioning and may affect its predictions. This short scoring horizon and
WM prediction errors remain limitations; it is not a full subgoal-completion test.

## Dependencies and worktree

Use the `libero-recipe-eef` branch, including its EE_DELTA correction. The first
local integration uses commit `db6acf3` in a separate detached worktree:

```bash
git -C /path/to/RoboWits fetch origin libero-recipe-eef
git -C /path/to/RoboWits worktree add --detach /path/to/RoboWits-wap-eef origin/libero-recipe-eef
git -C /path/to/RoboWits-wap-eef submodule update --init --recursive
```

The branch vendors modifications to OpenPI, not a complete installable checkout.
Prepare a **separate inference runtime**, never modify the source worktree:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git /data/wap/openpi-runtime
git -C /data/wap/openpi-runtime checkout --detach "$(cat /path/to/RoboWits-wap-eef/third_party/openpi/PINNED_COMMIT)"
cp -a /path/to/RoboWits-wap-eef/third_party/openpi/. /data/wap/openpi-runtime/
cd /data/wap/openpi-runtime
UV_PROJECT_ENVIRONMENT=/data/wap/openpi-venv uv sync --frozen --no-dev --python 3.11
```

The overlay copies Amber's files unchanged. Loading uses her `create_trained_policy`
and checkpoint-owned normalization; WAP's worker only supplies explicit inference
noise. No training script is launched. The original checkpoint must be readable;
do not copy, convert or overwrite it just to run inference.

Use three Python environments: Genesis simulator, OpenPI/JAX policy, and WAP WM.
The local Qwen3.5 service additionally uses a Transformers version supporting that
model, separate from the simulator's pinned Transformers. It accepts Unix socket
JSON requests containing `prompt` and local image paths; replies contain `raw`.
An existing compatible Qwen/Claude service can be used explicitly instead.

## Run

First start the local VLM (no API calls required):

```bash
CUDA_VISIBLE_DEVICES=1 /path/to/vlm-python -m robowits.vlm_server \
  --model /path/to/Qwen3.5-9B --socket /tmp/wap-vlm.sock
```

Then, from the WAP root, run a **two-decision integration check**, not a success-rate
benchmark. Replace paths for your machine and check free GPU memory first:

```bash
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl GS_GYM_ASSET_PATHS=/path/to/assets \
  /path/to/genesis-python -m robowits.eef_loop \
  --sim-root /path/to/RoboWits-wap-eef \
  --policy-python /data/wap/openpi-venv/bin/python \
  --openpi-runtime /data/wap/openpi-runtime \
  --policy-checkpoint /data/robowits-checkpoints/pi05_robowits_subgoal_eef/lemon_eef_b64/29999 \
  --wm-python /path/to/wm-python \
  --wm-checkpoint /path/to/wap-robowits-lemon/checkpoints/latest-step=2000.ckpt \
  --ckpts /path/to/world_models --wan /path/to/Wan2.1-T2V-1.3B \
  --urdf /path/to/assets/hf_assets/marvin_bimanual/urdf/marvin_pika.urdf \
  --eval-json /path/to/RoboWits-wap-eef/dataset/robowits/eval_dataset_seed/03.json \
  --vlm-socket /tmp/wap-vlm.sock --policy-gpu 1 --wm-gpu 0 \
  --num-candidates 2 --execute-steps 5 --max-steps 10 --case 10 \
  --out /data/wap/eef-smoke
```

The policy worker limits its JAX allocation fraction to 0.25; worker processes use
physical GPU indices through their own `CUDA_VISIBLE_DEVICES`. GPU requirements
include policy, WM, VLM and rendering. Avoid starting duplicate workers on resume.
This entry point currently requires a fresh output directory and does not resume
partial runs. Existing services are never stopped automatically; the loop closes
only its own policy and WM workers.

Saved evidence includes `protocol.json`, current subgoal inputs/output, policy
inputs and candidates, projected robot states, WM inputs/predictions, VLM contact
sheets/prompts/raw responses, scores, selection IDs, executed trajectories and
`execution.mp4`. A short successful integration check does not imply task success
or a statistically established benefit from WM selection.

## Verified integration (2026-10-27)

The actual `29999` checkpoint, finetuned lemon WM, Qwen3.5-9B judge and native
Genesis EE_DELTA controller completed two decision cycles on scene 10, with two
candidates each: four WM predictions, two VLM judgments, and 10 executed steps.
Both rounds selected candidate 0. Total rollout time was 230.9 s excluding initial
model loading. This 10-step check is not a task-success evaluation.

A separate request to the same real checkpoint with identical observation/seed
and one candidate reproduced candidate 0 exactly (maximum difference 0). The two
initial candidates differed by at most 0.000838 across their mixed-unit output
arrays; this is not a displacement in meters and does not establish semantic
strategy diversity. All 21 checkpoint files retained their size and modification
time; the source worktree stayed clean, and 60 copied OpenPI source files matched
Amber's files by SHA256. Tests: 24 passed, one Hydra-dependent test skipped in the
simulator environment; the real WM run used the separate Hydra-enabled environment.

**Observed scoring limitation:** Qwen described grasp/lift progress in the first
prediction where visual inspection showed nearly static motion and no clear grasp.
The loop is connected, but its reward judgments are not yet calibrated. Do not
interpret high VLM scores as actual task success. Inspect the saved images and raw
responses before running expensive evaluations or making performance claims.

[Machine-readable verification](validation/eef-loop-20261027.json).
Local complete evidence: `/data/mjw/world-action-planner/eef-loop/20261027/smoke-v1/`.

## Audit and paired trials (2026-10-27)

The initial Qwen reward failed static/order controls. See the new
[Chinese progress report](reports/progress-2026-10-27.md) and
[English progress report](reports/progress-2026-10-27.en.md) before interpreting
scores or running expensive studies. The current pilot explicitly uses the
existing Claude service, independent judgments and a 0.10 selection margin;
these are experiment settings, not a claim of a fully calibrated reward.

`eef_loop` now supports `--mode baseline`, `--reward-mode independent`,
`--selection-margin`, and `--reference-run`. A baseline skips all WM/judge calls
but uses the same single-subgoal planner. Matched trials check initial state,
pre-divergence observations and executed trajectories. Because independent JAX
processes were not bitwise identical, `--reuse-baseline-candidate-zero` pairs on
the saved baseline draw only after all inputs, prompt and seed match exactly;
fresh samples remain saved for audit. `--reuse-imagination-from` also reuses a
completed control's candidate set at identical states, and reuses WM outputs
only when every conditioning tensor and model/seed provenance match. Matching
judgments can then be reused. Report cached compute separately from wall time.

`python -m robowits.eef_study --spec study-spec.json --out study --conditions baseline`
runs the fixed baselines. Once they complete, the same specification with
`--conditions forced_zero selection` runs each paired control and selection.
The specification contains simulator Python, environment, common loop arguments,
and case/seed pairs. Completed trials are skipped; incomplete trials are never
overwritten. Each trial retains its own status, video, logs and protocol.
Errors are not task failures. The runner emits Markdown and JSON summaries.

Additional entry points: `eef_baseline` (native labeled evaluator + action replay),
`eef_contact_audit` (trajectory-checked post-hoc physical contacts),
`candidate_audit` (eight seeded draws at fixed states), and `reward_audit`
(order/identical/static controls). None invokes policy training.

## Pause for review and recover an interrupted control

`--stop-after-steps 50` keeps the native 1,000-step task budget but stops the
experiment after 50 executed steps. Unless the native environment has already
terminated, the result is `paused_for_review`, with `success=null`; it must not
be counted as a completed task failure.

`--recover-from /path/to/interrupted-forced-zero` requires a fresh output
directory and an interrupted forced-zero source. It reconstructs the trajectory
from reset and checks the paired baseline at every step. Saved policy draws,
predictions and judgments are reused only with matching inputs. Existing evidence
is preserved. Model, action-adapter and reward-prompt hashes must match; the old
and new orchestration hashes are explicitly recorded in `recovery.json`.

A selection trial can reuse a paused control only up to that control's review
limit. `python -m robowits.review_eef_batch --batch /path/to/review-batch
--baseline /path/to/completed-baseline` generates Markdown, a three-condition
execution video and candidate clips. It never starts another experiment.
