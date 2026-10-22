# RoboWits world-model probe and finetuning

This implementation follows RobertCHU01/world-action-planner's
[`kinder-wm-probe-finetune`, e7f1345](https://github.com/RobertCHU01/world-action-planner/tree/e7f1345e4a8f665a17a9c24be78df7c805655a1e/kinder):
paired RGB/skeleton videos, an episode-level holdout, true/frozen-condition probes,
copy-last comparison, weights-only initialization and full-state resume. The
DDP bucket-view and permanent-checkpoint changes in `exp_base.py` come from that
commit. KinDER data, results and machine-specific scripts are not bundled here.

All new implementation lives in this WAP repository. RoboWits Genesis 1.3.1
(`hierarchical-subgoal-genesis131`, `f5d68f2`) supplies the robot URDF and camera
conventions. No RoboWits checkout is modified.

Execution evidence is recorded in [validation/smoke-20261021.json](validation/smoke-20261021.json):
five converted episodes across three tasks, FK checked against Genesis, a real
optimizer update after full-state resume, validation sampling and paired probes.
These checks use the locally available `libero_90_base` initialization; they are
not a full finetuning experiment or evidence of improved prediction quality.

## Data contract

- Source: released LeRobot **v3.0** dataset, 30 FPS. Shared MP4 files are decoded
  using each episode's camera-specific `from_timestamp` and video presentation
  timestamps. Missing frames fail; no silent padding or cross-episode clipping.
- RGB: original released videos. Conditions: forward kinematics of measured
  `observation.state[t]`, not commanded `action[t]`, and **zero physics steps**.
  This avoids replay drift and the old/new simulator task-object discrepancies.
- State order: right 7 joints + total finger opening, then left 7 + opening.
  Each of the two prismatic joints receives half the opening.
- Cameras: ego and two attached wrist views from the Genesis131 implementation.
  Panels are `ego | ego ; wrist_right | wrist_left` at 448×448. The repeated ego
  supplies the fourth slot without inventing a camera. Scores count it **once**.
  Native 848×480 views are resized to square panels; skeleton projection uses
  the same scaling. Inspect saved overlays for every new source/camera setup.
- Finger lines extend to the URDF's gripper-tip plane, then clip to the image.
  They encode opening, but are not occlusion-aware segmentation masks.
- 30→20 FPS uses nearest source timestamps, with no duration stretching. RGB
  and skeletons share exactly the same indices. MP4 uses H.264 yuv444p CRF 12.
- 60 copies of the initial frame precede every episode, as in KinDER training.
  `source_indices.npy` records the exact original row for every output frame.
- By default, the first 3 episodes of **each dataset task index** are holdout,
  next 2 validation, and the rest training. These are source task indices, not
  RoboWits numeric task IDs. Splits are assigned before subset filtering and
  persisted for the entire source inventory. Probe windows only use holdout.

This is teacher-forced world-model evaluation with measured future robot poses.
It does not evaluate a closed-loop planner or predict future controls. The model
never receives future RGB or task-object poses. Region scores cover the skeleton
neighborhood and pixels changing from the last history frame; the latter includes
shadows/camera motion and is **not** a contact-event/object segmentation score.

## Setup and conversion

Run commands from the WAP repository root. Data conversion needs only CPU packages:

```bash
python -m pip install -r robowits/requirements-data.txt
export SOURCE=/data/datasets/RoboWits_lerobot_dataset
export URDF=/path/to/assets/hf_assets/marvin_bimanual/urdf/marvin_pika.urdf
export CKPTS=/path/to/downloaded/world_models
export WAN=/path/to/Wan2.1-T2V-1.3B
export DATA=/path/to/wap_robowits_dataset

python -m robowits.build_dataset --source "$SOURCE" --urdf "$URDF" \
  --out "$DATA" --prompt "$CKPTS/robosuite_default_prompt.pt" \
  --negative-prompt "$CKPTS/robosuite_default_neg_prompt.pt"
```

For a first check, add `--episode-ids 0 3 5` (one holdout, validation and training
episode respectively). `--task-indices 0 2` selects source dataset tasks.
`--plan-only` writes the split manifest without encoding; rerun with `--resume`
to execute the same plan. `--resume` verifies the manifest, source state/action
hashes and cached video counts. A partial episode with no `episode.json` is
deliberately rejected: remove that incomplete episode directory before resuming.
`metadata.csv` appears only after all selected episodes finish; partial progress
is written to `metadata.partial.csv`. Use a new output directory for new settings.

Inspect `ep*/overlay_*.png`, `ep*/episode.json` (skeleton coverage and hashes),
`manifest.json` and `metadata.csv`. The first source episode is a test of timing;
also check nonzero video offsets and multiple tasks before a full run.

Optional numerical FK verification against the unmodified Genesis131 checkout:

```bash
PYTHONPATH=/path/to/RoboWits-genesis131 \
  /path/to/RoboWits-genesis131/.venv/bin/python -m robowits.validate_geometry \
  --source "$SOURCE" --urdf "$URDF" --episode 0 --out outputs/fk-validation.json
```

## Probe

Install WAP's model dependencies using [its setup instructions](../world_model/README.md)
in a separate environment. The conversion environment may be different. In an
older WAP environment with NumPy 1.24, use `pyarrow==16.1.0`; PyArrow 26 requires
NumPy 2 and fails to import there. The probe also needs `scikit-image` and imageio.

```bash
python -m robowits.windows --data "$DATA" --out outputs/robowits-windows

# Run with the WAP environment's Python. Both true and frozen controls run.
CUDA_VISIBLE_DEVICES=0 python -m robowits.probe \
  --windows outputs/robowits-windows \
  --checkpoint "$CKPTS/robosuite_ft/checkpoints/latest.ckpt" --wan "$WAN" \
  --prompt "$CKPTS/robosuite_default_prompt.pt" \
  --negative-prompt "$CKPTS/robosuite_default_neg_prompt.pt" \
  --out outputs/robowits-zero-shot --seed 0
```

Default windows use 80 history and 40 future frames at 20 Hz, matching the KinDER
probe protocol. For a smoke test use `--windows-per-episode 1 --future 20` when
building windows. The official WAP pipeline samples history according to its
checkpoint. Future length must be at least 20 and divisible by 4.

The in-process pipeline uses the original WAP inference code, without a server.
It resets Python/NumPy/Torch RNGs before each true/frozen pair. This controls
sampling variation; it is not a claim of deterministic GPU kernels. Freeze mode
repeats the last history skeleton but is scored against the real future.
`metrics.json` contains PSNR, SSIM, region MSE and copy-last scores for each unique
view. `--lpips` adds optional CPU LPIPS using `pyiqa` and its pretrained weights.
Robot-region MSE uses a Euclidean 15-pixel radius around skeleton pixels above
intensity 24. Results from before this radius correction (including the original
smoke run) used a 15×15 square and should not be mixed into new comparisons.
Videos show `GT | condition | true prediction | frozen prediction`.

## Finetuning

```bash
CUDA_VISIBLE_DEVICES=0 python -m robowits.train \
  --data "$DATA" --init "$CKPTS/robosuite_ft/checkpoints/latest.ckpt" \
  --wan "$WAN" --run-dir /data/wap_robowits_runs/run01 --gpus 0 \
  --batch-size 1 --accum 8 --steps 2000
```

Defaults follow KinDER: LR 1e-5, constant schedule after 250 warmup steps; 81-frame
clips, 21 history frames at indices 0,3,...,60 and 20 future frames at 61,...,80.
WAP concatenates RGB and skeleton latent conditions. Validation occurs every
1000 optimizer steps, using 2 batches. Checkpoints retain rolling latest every
1000 steps and permanent copies every 2000. Full checkpoints are large (roughly
18 GB); select a run directory with sufficient space.

`--gpus 0,1` enables WAP's DDP; effective batch is batch-size × accum × GPU count.
`--dry-run` validates metadata and paths and prints the command. Offline W&B
logging needs no account; `--wandb-entity` changes the local namespace. The
launcher runs from WAP's `world_model/` and writes `launch.json` plus Hydra config.

A short execution check (not an experiment) uses:

```bash
python -m robowits.train --data "$DATA" \
  --init "$CKPTS/libero_90_base/checkpoints/latest.ckpt" --wan "$WAN" \
  --run-dir /data/wap_robowits_runs/smoke --gpus 0 \
  --steps 1 --batch-size 1 --accum 1 --workers 0 \
  --save-every 1 --val-batches 0 --sanity-batches 0
```

Resume into a **new** run directory with `--resume /old/run/checkpoints/last.ckpt`.
This uses `load=<checkpoint>` for full Lightning state, whereas fresh runs use
`load=<checkpoint>:model`. It does not use W&B's `resume=<run_id>`. Quoted Hydra
arguments support paths containing spaces or `=`.

## Evaluate the finetuned model

Use the same `robowits.probe` command and windows, replace `--checkpoint` with
the new run's `checkpoints/last.ckpt`, and choose a new `--out`. Keep `.hydra/`
next to `checkpoints/`. The probe also accepts `checkpoints/keep/step=2000.ckpt`
and locates the owning run's Hydra configuration without copying the checkpoint.
Then compare:

```bash
python -m robowits.compare --baseline outputs/robowits-zero-shot/metrics.json \
  --finetuned outputs/robowits-finetuned/metrics.json
python -m pytest tests/robowits -q
```

Comparison rejects different window sets, GT hashes or seeds. Evaluate multiple
seeds for quality claims and keep the final holdout separate from model selection.

## Single-task experiment pipeline

After conversion, run both official zero-shot checkpoints in parallel, then
two-GPU finetuning (batch 1 × accumulation 4 × two GPUs = effective batch 8),
then the final matched probe. This takes hours; launch it as a persistent job.
The output directory must be new. `--ckpts` points to the `world_models` directory
containing both checkpoint directories and the prompt embedding files.

```bash
python -m robowits.experiment --data "$DATA" --ckpts "$CKPTS" --wan "$WAN" \
  --out /data/wap_robowits_runs/lemon --gpus 0,1 --steps 2000
```

`--dry-run` validates inputs and prints commands. The actual run writes
`commands.json`, atomic `status.json`, individual stage logs and `report.md`.
It checks the final checkpoint's optimizer step before evaluation. Any stage
failure stops subsequent stages. Checkpoints roll every 250 optimizer steps;
the final step also has a permanent copy. Recover a failed training run using
the standalone train launcher's full-state `--resume` into a new directory.

For the released lemon task, use `--task-indices 2` during conversion. The fixed
split is episodes 100–102 holdout, 103–104 validation and 105–149 training.
The experiment uses four windows per held-out episode (12 total), 80 history +
40 future frames at 20 Hz, seed 0, PSNR/SSIM/LPIPS and region MSE. It evaluates
world-model prediction given measured robot motion, not task-success rate.

## Paired closed-loop lemon pilot

`robowits.closed_loop` runs the available LeRobot PI05 policy in the unmodified
Genesis131 task, then repeats the same evaluation scenes with WM action selection.
This is a small integration experiment, not the unreleased WAP planner.

- Default evaluation scenes: native seed JSON indices 10–14, seeds 1010–1014.
  Development checks used indices 0 and 1. Do not choose cases by observed success.
- Both arms replan every 30 native steps using candidate 0 from the same seeded
  PI05 sampler. The model generates 50 actions; only the first 30 are executed.
  This matched cadence differs from the checkpoint's default 50-step cadence.
- At steps 90,180,...,900 (if still running), the WM arm samples a second candidate.
  Each candidate gets 80 history / 20 future frames at 20 Hz. History is sampled
  from actual observations; unavailable initial history repeats the first frame.
- Future skeletons use the same FK renderer as finetuning, driven by the native
  controller in a separate robot-only scene. No task objects are present there.
  The evaluated scene is checked for mutation before/after candidate projection.
- Fixed image-only score: mean negative normalized 2D lemon/target centroid distance
  in the last five predicted ego frames. HSV yellow range 12–40 (S≥80,V≥35), green
  40–90 (S≥80,V≥35). Minimum component areas: yellow 4 pixels, green 12 pixels.
  Multiple yellow objects or missing objects/targets make a frame invalid. At least
  three frames must be valid. Both candidates must be valid, and candidate 1 must
  improve by >0.02 image widths; otherwise execute candidate 0. This simple score
  is not a success detector and does not fully value obstacle removal.
- Success comes exclusively from Genesis131's native termination/success flag.
  Simulator object state is saved for initial-state matching and failure diagnosis,
  never provided to the policy, WM or candidate scorer. Object orientation, robot
  position/velocity and object positions are checked when pairing initial states.
- Save each actual execution video, native actions, joint states, sampled object
  diagnostics, candidate predictions, decisions, timings and a Markdown report.
  Infrastructure errors remain errors, never counted as task failures.

Run from this WAP root using the Genesis131 Python, setting its asset root and
EGL environment as in that checkout's `LOCAL_SETUP.md`. Keep the WM on another GPU
and use the separate WAP Python for the worker:

```bash
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl GS_GYM_ASSET_PATHS=/path/to/assets \
  /path/to/Genesis131/.venv/bin/python -m robowits.closed_loop \
  --sim-root /path/to/Genesis131 --policy /path/to/ckpt/pi05 \
  --eval-json /path/to/Genesis131/dataset/robowits/eval_dataset_seed/03.json \
  --urdf /path/to/assets/hf_assets/marvin_bimanual/urdf/marvin_pika.urdf \
  --wm-python /path/to/wm-venv/bin/python --wm-checkpoint /path/to/step2000.ckpt \
  --ckpts "$CKPTS" --wan "$WAN" --wm-gpu 0 --out /data/lemon-paired-run
```

Use a new output directory. `--cases 1 --max-steps 120` is an integration check,
not task-success evaluation. `--baseline-only` runs the matched policy without WM.
Full pilot runtime is dominated by up to 100 WM predictions (two per selection,
ten selections per episode, five episodes), approximately 1.5 hours on this machine.

Generate a separate Markdown audit with paired time differences, comparable
prediction counts and execution contact sheets (also safe during an active run):

```bash
/path/to/Genesis131/.venv/bin/python -m robowits.summarize_closed_loop /data/lemon-paired-run
```

The audit excludes incomplete/error episodes from success-rate denominators.
Contact sheets sample 12 frames and do not establish collision or grasp causes
by themselves. Consult the full videos before assigning definitive failure labels.

## Single-task policy data

The available source dataset has 16D joint states/actions. Existing PI05 weights
expect 14D EE state/action. `robowits.prepare_policy_data` converts measured states
and commanded joint targets independently with the Marvin URDF, preserving every
timestamp, episode number and source video. It writes a separate dataset and
symlinks source videos. Source files are never modified.

```bash
python -m robowits.prepare_policy_data --source /path/to/RoboWits_lerobot_dataset \
  --out /data/lemon-policy-ee --urdf /path/to/marvin_pika.urdf
```

The native order is R position, L position, R rotation vector, L rotation vector,
R gripper width, L gripper width. Positions are world-frame `Gripper_Tip` poses;
rotations are canonical axis-angle. `validate_geometry` checks both EE position
and orientation against Genesis131 without advancing physics.

Global action/state normalization uses only episodes 105–149 (45 lemon demos).
For native LeRobot finetuning, pass episodes `[105,...,149,103,104]` in that order
and `dataset.val_frac=0.04`; its per-task end-of-list split then holds out 103/104.
Episodes 100–102 remain unused. Image timestamps and pixel values are unchanged.
The simulator's fixed initial scenes are a separate evaluation set, not source
episode numbers. Training loss is not evidence of task success: evaluate the
resulting checkpoint in Genesis131 after training.
