# Multi-subgoal Amber π0.5 + finetuned WM experiment

2026-10-29. Both formal lemon case-1 no-WM controls are complete; the paired WM loops are running. The manual version remains pending review.

## Completed no-WM controls

Both started from exactly identical native initial states and reached 1,000 steps without success (`terminated=false`, `truncated=true`). Each passed 100 decision-image checks and every joint-state check with zero error.

|Proposer/judge|Maximum object lift|Final planar distance to target|Episode wall time|
|---|---:|---:|---:|
|Gemini, no WM|39.2 mm|48.0 cm|67.7 min|
|Qwen, no WM|41.7 mm|56.2 cm|26.3 min|

Initial distance was 35.0 cm; the closest points during the entire episodes were still about 34.9 / 34.8 cm. Both contacted and briefly lifted the lemon without transporting it to the target. One episode per setting cannot estimate population success rates. Timing excludes startup, shares GPUs, and compares Gemini high thinking with nonthinking Qwen, so it is not an isolated model-efficiency benchmark.

The Gemini control recorded 201 API responses including retries: 1,089,902 provider-reported total tokens, comprising 262,087 prompt, 29,053 visible-output, and 798,762 thought tokens. These components must not be added again to the total. Qwen recorded 202 local responses and 38,373 local output tokens; comparable prompt and total counts are unavailable. Episode reports now audit raw usage, retries, missing responses, and response times by stage, excluding proposal logs copied from the paired control. Missing usage is unknown rather than zero, and this audit is not invoice reconciliation. Only offline reporting changed; frozen running inference code is unchanged.

[Actual trajectories, complete videos, and decision records](/data/mjw/world-action-planner/multigoal/20261029/control-comparison/review.md). Qwen has documented language/anchor mismatches. Gemini continues proposing transport/handover after the object becomes nearly stationary (less than 0.126 mm movement after step 700). These observations motivate separate stage, grounding, and planning analyses; they do not prove the policy entirely ignores spatial conditioning.

Gemini's WM loop switched from candidate 0 to 1 at the first decision; Qwen has not yet diverged from its control. Neither has a terminal WM outcome yet. The no-WM judge sees RGB and candidate language without explicit arm/anchor coordinates, whereas predictions convey action geometry, so any future gain needs to distinguish geometry from dynamics. Proposal/manual-stage inference currently uses ego RGB while value scoring uses multiple views. Historical sections below record earlier phases, not necessarily current status. The earlier successful single-subgoal case-0 pair is reported separately.

## Cube rejection stop and separate amended protocol

The original Gemini cube control stopped at step 750: the judge considered the cube and container invisible, rejected all three candidates, and `best()` raised. Qwen cube also stopped on all-rejected scores at step 130. Both have `status=error, success=null`, not native success or a 1,000-step endpoint; neither paired WM run started. Interrupted-episode reviews: [Gemini](/data/mjw/world-action-planner/multigoal/20261029/formal-review/cube-gemini-interrupted/review.md), [Qwen](/data/mjw/world-action-planner/multigoal/20261029/formal-review/cube-qwen-interrupted/review.md). Both lemon WM runs continue with the original frozen code.

Physical prefix replay confirms that **the objects did not fall out of the scene**. Final cube displacement was only 36.95 / 19.14 mm for Gemini / Qwen; container displacement was 35.41 / 17.50 mm. Cube height stayed around 0.782–0.787 m. The robot contacted the container for 474 / 92 steps, with no direct cube contact in either prefix. All joint states matched exactly and all 75 / 13 executed decision-image hashes passed. A judge's claim that an object is invisible or absent must be separated from physical disappearance; occlusion and recognition still require video review. Explicit abstention addresses the exception, not these perception or retrieval problems. [Actual object trajectories and audit](/data/mjw/world-action-planner/multigoal/20261029/formal-cube-interrupted-replay/review.md).

A separate amendment supports explicit abstention. With `all_invalid_action=hold`, a no-WM control rejecting every candidate records `selected=null`, holds the measured start-of-chunk end-effector pose for ten steps, then observes and proposes again. The WM arm can still execute a WM-valid candidate; it holds only if both judgements provide no valid candidate. Holds are counted separately and never labeled policy samples or WM approvals. The default remains an error, paired settings and source hashes must match, and the interrupted old trajectory is not spliced into an amended full episode.

Validation: 16 relevant tests passed. Native cube simulation with injected all-rejected scores reached 1,000 steps and 100 explicit abstentions in both modes, with exactly identical paired actions and joint states. Models were stubbed for this test: it verifies control flow, **not model performance**. [Evidence](/data/mjw/world-action-planner/multigoal/20261029/abstention-native-integration/verification.json). Both amended cube pairs use the separate frozen worktree `world-action-planner-multigoal-hold-eval` at `9c1ac29`, writing to `formal-cube-case1-hold`. The sequence is Gemini control→WM, then Qwen control→WM; the first control has started. Separate fixed-cadence cross-score and terminal-replay watchers are active.

## Fixed step-250 cross-score and physical check

At Qwen decision 25, the online Qwen judge scored all three predictions zero and retained its no-WM choice 2. Offline Gemini switched from its no-WM choice 0 to choice 2, with predicted-outcome scores 0 / 0.30 / 0.34. Gemini nevertheless described candidate 1 as stationary with no progress, contradicting its nonzero score. Agreement on the final index does not establish agreement about the predictions or a WM benefit.

All three saved action branches were replayed independently from native step 250, with zero prefix joint error and identical current RGB. The selected candidate 2 also exactly matched the ten online executed joint states. Lemon displacement was 5.07 / 5.93 / 8.07 mm; planar target distance increased by 2.66 / 3.74 / 5.41 mm. Candidate 2 instructed a right-arm grasp, but its ten-step lemon contacts involved only the left-side `Link8_L`, and maximum height increase was just 0.0041 mm. This short segment does not demonstrate a completed right-arm pickup. Contact alone is not a stable grasp, and local target-distance regression alone does not prove a poor long-term action.

[Cross-scores](/data/mjw/world-action-planner/multigoal/20261029/formal-crossscore/review.md), [actual / condition / WM videos for all candidates](/data/mjw/world-action-planner/multigoal/20261029/formal-qwen-step250-physical/review/review.md), and [contact and online-trajectory checks](/data/mjw/world-action-planner/multigoal/20261029/formal-qwen-step250-physical/physical-check.json). Physical branches are post-hoc diagnostics and never enter online scoring.

## Agreed design

The user chose plausible errors in timing or object choice, with one expected useful candidate, and independent Gemini/Qwen proposer-and-judge pipelines plus cross-scoring of identical predictions.

Initial diagnostics used four subgoals with two action samples each. Formal natural-proposal episodes allow 3–5 subgoals (prefer three), with one frozen Amber π0.5 action sample per subgoal; the finetuned lemon WM imagines all candidates and a task-level judge chooses one. The prepared manual version uses four fixed candidate instructions per stage with one sample each. Each chunk covers ten native steps, approximately 0.333 seconds. Only the first six predicted frames are scored; the rest of the fixed one-second WM horizon uses terminal-pose padding. Observe and repeat until native success or 1,000 steps.

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

## Running paired episodes

Runtime is pinned to detached worktree `/home/junwenmiao/code/world-action-planner-multigoal-eval`, commit `7caca41`, verified with 46 passed and 1 skipped. Subsequent edits in the development worktree do not alter running code.

Live Markdown: `/data/mjw/world-action-planner/multigoal/20261029/progress.md`. The `formal-case1` Gemini and Qwen supervisors each run a complete control, then automatically run its WM pair with verified shared-prefix candidates. `formal-cube-case1` queues the same comparisons on cube case 1, seed 202610291, after the older Amber job releases resources.

The extra Qwen thinking probe still scored one actual transport clip zero; its static query timed out. It is recorded as partial, and the dedicated thinking service was cleaned up. Formal Qwen remains non-thinking. A managed service handoff preserves access when the original experiment's Qwen server exits.

All three local GT/condition/WM reviews are available under `initial-review`, `step350-review`, and `cube-review`. An additional anchor audit finds two Qwen goals pointing at the robot hand, approximately 164.9 mm from the actual lemon center. Spatial conditioning was present and differed across goals; this is a proposal/grounding error rather than evidence that conditioning was absent. Simulator coordinates are used only offline. See `step350-grounding-review/review.md`.

Still required: completed native endpoints and paired outcome/runtime analysis, user review and execution of the manual chain, and the two-task/two-model comparative report. The overall goal remains active.

## First formal episode endpoint and cross-scoring plan

The lemon case-1 Qwen no-WM control reached **1,000 steps without success** (`terminated=false`, `truncated=true`; 100 decisions). Recorded episode wall time was 1,579.94 seconds, excluding startup initialization. Its paired WM episode has started but has no terminal result yet. This single control episode does not establish a population success rate or a WM benefit. The Markdown review, fixed-cadence observations, and full execution video are under `formal-review/qwen-without-wm/review.md` in the experiment directory.

Formal cross-scoring uses decision indices 0, 25, 50, 75, plus the last executed decision at the native endpoint; this schedule was saved before formal WM results. It retains the primary judge's actual online scores and selection, then asks the other judge to score identical saved predictions offline. Input/prediction hashes and observation identity are recorded. No offline score changes executed actions.

`robowits.multigoal_episode_report` exports Markdown timelines, excludes unexecuted decisions, and distinguishes running/technical-error states from native failures. `multigoal_rescore --reuse-recorded-primary` prevents a fresh primary rating from being mistaken for the actual online choice. Validation: 49 passed, 1 skipped. Formal execution remains frozen at `7caca41`; policy training and weights are untouched. Manual candidate-chain review and the remaining full pairs are still outstanding.

The completed Qwen control was physically replayed with zero joint error: 819 finger-contact frames, 41.7 mm maximum lemon lift. Its planar distance to the target increased from 350.1 to 561.6 mm. At step 750, two candidate anchors lay near the green target, while the selected “move towards the target” candidate pointed toward the opposite side, about 725 mm from the target center. See `formal-review/qwen-without-wm/grounding.md`. Simulator object coordinates are used only in this post-hoc audit.

Control limitation: the no-WM judge sees current RGB and candidate language, without explicit candidate arm/anchor/spatial coordinates. Predictions provide additional action geometry. Any eventual benefit therefore needs to distinguish geometry-informed action selection from better object-dynamics prediction.

The first formal same-bank cross-score is complete (Qwen case 1, decision 0). Online Qwen scores were `[0,0,0]`, retaining its no-WM choice 2. Offline Gemini scores were `[0.3,0.3,0]`, retaining its own no-WM choice 1. Neither judge changed its own choice because of WM; their disagreement already existed without WM. Gemini's first two explanations explicitly describe no motion/progress despite scores of 0.3, so score–rationale inconsistency persists on the new initial state. Evidence: `formal-crossscore/review.md`.

## Native replay verification for both tasks

`python -m robowits.multigoal_replay --run RUN --out AUDIT` replays completed recorded actions and checks the simulator revision, initial state, every joint state, each decision's real RGB hash, native success, and termination flags. Per-frame object positions/orientations and contacts are saved for post-hoc diagnosis only. It queries neither policy nor WM and cannot feed oracle data into online selection.

The full Qwen lemon control passed all 1,000 steps and 100 decision-image checks with zero joint error; contact/distance statistics agree with the earlier independent audit. The existing 10-step cube diagnostic also passed using explicit `--allow-diagnostic`, retaining `native_success=null`. An audit queue now watches all eight formal episodes and verifies each only after its native endpoint. Index: `formal-native-replay/review.md` under the experiment directory. These are reproducibility checks, not additional successful episodes or a remedy for grounding/scoring errors.

A separate [Chinese manual-chain review](manual-subgoals-review-2026-10-29.md) lists each stage and all four candidates, including reachability ambiguities. Eight manual-version configs (two tasks × two models × with/without WM) are prepared under `manual-prepared/`, but no jobs or model calls were started. The approved catalogue is intentionally absent and the source catalogue remains `draft` pending the requested human review.

## Earlier single-subgoal reference: first switched action

The earlier Amber case-0 single-subgoal loop first switched at step 310. Qwen assigned 0 / 0.6 to the two predictions, claiming only the second established a new grasp. Exact shared-prefix physical branches both had finger contact for all 10 steps and no lift; lemon displacement was 5.875 / 5.774 mm, with only 0.106 mm between final positions. This local evidence does not support a large grasp-quality difference. Videos are in `amber-release/20261029/full-wm-qwen-case0/switch-310-review/review.md` under the experiment root. This older illustrated-example configuration is reported separately from the new multi-subgoal text-example pairs.

The earlier single-subgoal case-0 pair is now complete: both succeeded, at 514 steps without WM and 526 with WM. Both passed zero-error physical replay; three WM switches did not reduce simulator steps. Recorded wall times were 64.9 s / 120.5 min, excluding startup and sharing GPUs with other experiments. See the [separate English result](amber-wm-qwen-2026-10-29.en.md). This does not replace the ongoing two-task, two-model multi-subgoal matrix. Resource cleanup enabled the cube Gemini control to start; the replacement Qwen service has already supported further online scoring.

Gemini's first formal WM switch (candidate 0 → 1 on the shared bank) now has actual branches for all three candidates. Lemon displacement over ten steps was about 0.00359 mm in each, and the blocks barely moved. The selected branch exactly matches online executed joints. This does not verify an immediate object-interaction benefit, but arm-approach value may emerge later. [Actual/condition/prediction videos](/data/mjw/world-action-planner/multigoal/20261029/formal-first-switch-gemini/review/review.md).
