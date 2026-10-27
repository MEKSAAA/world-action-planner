# Qwen value examples

`value_examples_v1.txt` is the **unsuccessful text-only calibration variant**.
It scored both a real transport sequence and a static control as zero. Do not
assume that adding textual scoring examples makes the reward useful.

The current experiment uses `robowits.illustrated_value.IllustratedValueReward`:

- Local Qwen3.5-9B, deterministic generation, independent candidate scoring.
- Three separate camera strips: ego, right wrist, left wrist. Each strip contains
  the current frame and three timestamped future frames. The duplicate ego panel
  in the WM mosaic is not treated as a fourth independent camera.
- Three visual demonstrations: static unheld (0), new grasp (0.6), static already
  held (0). These are provisional labels requiring human review.
- Per-camera RGB-change measurements, computed only from the current observation
  and imagined frames. Qwen receives these measurements in its prompt; the code
  does not replace Qwen's scores with a motion heuristic. Pixel change alone is
  explicitly insufficient evidence of task progress.

Rebuild the image examples from the **existing WM training video**, without any
policy training:

```bash
python -m robowits.build_value_examples \
  --video /path/to/ep000105/rgb.mp4 \
  --out /path/to/new-value-examples
```

The builder uses frames 440–480, and generates absolute image paths in
`manifest.json`. Rebuild on a new machine or update those paths. The source is
episode 105, distinct from the real positive control (episode 101) and the fixed
new-layout review. This is a small reward calibration set, not a held-out task
success evaluation; the released policy's exact training coverage is unknown.

Pass the manifest with `robowits.rot6d_loop --value-examples .../manifest.json`.
The Qwen service (`robowits.vlm_server`) accepts the demonstrations as interleaved
user images/prompts and assistant scored answers. Each query saves its complete
request, raw response, and timing. Do not change the prompt or examples midway
through a paired experiment.

The release adapter is inference only and requires the author's 20D absolute
EE_ROT6D policy, a 50-step chunk, and native 30 Hz control. It uses the first 30
actions for 20 WM frames at 20 Hz, scores one second, and executes 10 actions by
default. It strictly loads weights and the checkpoint's own normalization. The
bounded review stops after 50 steps; it is not a success-rate measurement.
