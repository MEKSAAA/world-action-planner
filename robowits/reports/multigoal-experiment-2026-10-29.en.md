# Multi-subgoal Amber π0.5 + finetuned WM experiment

2026-10-29. Implementation and initial diagnostics are in progress. No new full-episode multi-subgoal result is available yet.

## Agreed design

The user chose plausible errors in timing or object choice, with one expected useful candidate, and independent Gemini/Qwen proposer-and-judge pipelines plus cross-scoring of identical predictions.

Each decision proposes four grounded subgoals. Frozen Amber π0.5 samples two action chunks per subgoal. The finetuned lemon WM imagines all eight candidates; a task-level judge chooses one to execute. Each chunk covers ten native steps, approximately 0.333 seconds. Only the first six predicted frames are scored; the rest of the fixed one-second WM horizon uses terminal-pose padding. Observe and repeat until native success or 1,000 steps.

Two proposal sources are implemented: natural VLM proposals and a human-authored stage-dependent catalogue. The latter remains a draft pending user review. Candidate order is shuffled and expected-correct labels never enter scoring prompts. Stage and grounding use current RGB/RGB-D, not simulator object states.

The no-WM control ranks subgoals from current RGB and language and uses the first seeded draw. Fixed-state diagnostics compare identical action banks with and without imagined outcomes. Full episodes share initial states and generation rules; after a decision changes the trajectory, they no longer have identical observations or candidate banks.

The second task is `robowits/02-retrieve-cube-v0`: retrieve a cube from a narrow container and place it on the target. Both checkpoints were finetuned on lemon, so this is a separate transfer test. No policy training code, normalization, or weights are modified.

## Initial evidence

Artifacts: `/data/mjw/world-action-planner/multigoal/20261029`.

- Gemini API connectivity confirmed; response model `gemini-3-flash-preview`. Local comparator: Qwen3.5-9B.
- Cube scene assets and RGB-D proposals work for both models. Qwen calls the green target a green cube; this possible recognition error is retained.
- Gemini's initial lemon candidates are left-block sliding, lemon grasping, right-block lifting, and right-block sliding. Without WM, Gemini favors clearing the left block; Qwen favors grasping the lemon directly.
- Across the eight sampled chunks, mean joint two-arm position-trajectory differences span only 0.36–1.94 mm. Rotation and gripper values are not mixed into this distance.
- Eight exact-prefix physical branches show 2.24–2.32 micrometers of lemon displacement and less than one micrometer of block displacement over ten steps. These initial short actions do not produce distinct object outcomes.
- Both judges score an exactly static image sequence zero. On actual videos of the previous Chunru first-switch candidates, Gemini scores 0.01/0.05 and Qwen 0/0. Both also score a short clip just before Amber's native success zero. Positive/negative ranking quality is therefore not established.
- Launcher virtualenv symlink handling was fixed. Malformed VLM JSON is retried at most three times with all raw responses retained; scores are never repaired or fabricated.
- Verification: 43 tests passed, 1 skipped. Same-bank WM imagination and cross-scoring are running. The manual catalogue awaits explicit review of the concrete chain.

Different instructions do not necessarily produce distinguishable short-term outcomes. Contact, transport and placement states must be examined before expanding the expensive full-episode matrix. These initial no-contact branches establish neither WM benefit nor WM uselessness.

## Outputs and limits

Report native success, failure type, grounding errors, action diversity, prediction/physical-branch disagreement, judge agreement, whether WM changes the choice and whether that helps, runtime and API tokens. Keep manual diagnostic results separate from natural proposals. Do not present one fixed initial state as a reliable aggregate success rate.

At roughly 45 seconds per WM candidate, eight candidates cost approximately six minutes per decision before scoring. A 100-decision episode may exceed ten hours. Terminal padding may affect jointly generated scored frames, and short-horizon predictions may not reveal the eventual benefit of clearing an obstacle.

Gemini integration follows the [GenerateContent API](https://ai.google.dev/api/generate-content). Credentials remain in memory and are excluded from experiment logs.
