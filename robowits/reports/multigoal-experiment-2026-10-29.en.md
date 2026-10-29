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

## Frozen protocol for subsequent full paired runs

The earlier full no-WM run is exploratory because prompts were subsequently improved. Formal pairs must use the same frozen code and model/task configuration. Until the first different selected action, reuse the entire verified reference bank and check complete policy inputs, subgoals, seeds, targets and executed joints. After divergence, each run observes its own state.

When WM maximum scores tie and include the no-WM choice, retain that choice. If all predictions are rejected, explicitly log fallback to the no-WM choice; never count this as WM approval or benefit. Otherwise choose the highest valid WM score.

At recorded step 350, Qwen initially repeated identical instructions while changing only the arm. Specific validation feedback resolved the literal duplication in a subsequent diagnostic. The eight action chunks now differ by up to 19.7 mm in mean two-arm position trajectories. Physical branches and WM predictions are pending. “Move towards” and “move to” remain semantically similar, so schema validity alone does not establish semantic diversity.

## Completed diagnostics and current limitations

Initial-state and step-350 banks now have eight physical branches, WM predictions and cross-scores each. In the initial state, low-thinking Gemini sometimes scores candidates 0.2–0.3 despite effectively zero real object movement; Qwen scores all zero. At step 350, actual lemon displacement spans 30.6–56.1 mm. Low-thinking Gemini changes its choice from candidate 4 without WM to candidate 3 with WM; Qwen retains its control choice on a complete tie. These are local observations, not task success rates.

Removing candidate language and presenting separate frames did not fix all visual judgments. Gemini high thinking recognizes some transport, but several positive scores contradict explanations claiming no movement. Qwen thinking still misclassifies one real transport clip as static. Reward reliability remains unproven; retain all cases rather than selecting favorable scores.

The exploratory full no-WM run stopped at step 230 due to repeated malformed Gemini JSON. It has status `error`, not simulator failure. Explicit JSON Schema passed a repeat of the same failed request. Subsequent paired runs use frozen code and new output directories.

For full runs, naturally generated banks may contain 3–5 distinct subgoals, preferably 3, with one sampled policy chunk each. Two samples per goal are retained in diagnostics. Outcome scoring hides candidate instructions and receives only task text and images. Gemini uses high thinking; Qwen uses its default non-thinking mode, with settings recorded. The next formal lemon pair uses case 1, seed 202610271, which was not used for the above reward-development checks. The exact manual chain remains pending user review.

The cube's eight physical branches completed, including a native integration check of candidate-bank reuse against complete inputs, seeds and targets. Its WM diagnostic is still running. Current tests: 46 passed, 1 skipped.

Review artifacts: `/data/mjw/world-action-planner/multigoal/20261029/initial-review/review.md` and `step350-review/review.md`.
