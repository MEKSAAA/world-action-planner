# Amber lemon policy: completed WM + Qwen pair

Both runs succeeded in the native simulator: **514 steps without WM selection; 526 steps with WM + Qwen selection**. Both passed physical replay with zero joint error. This single paired initial state shows no success or step-count benefit from WM, and does not estimate population success rates.

|Setting|Native success|Steps|Recorded episode wall time|
|---|---|---:|---:|
|Amber policy + Qwen subgoals, no WM selection|Yes|514|64.9 s|
|Same policy + WM + Qwen selection|Yes|526|7,229.6 s (120.5 min)|

The WM run used 12 more simulator steps and about 111.4 times the recorded wall time. Startup initialization is excluded; concurrent GPU experiments affected timing, so this is not an isolated hardware benchmark. Recorded WM inference totaled 6,069.2 s, Qwen scoring 723.8 s, and robot-proxy work 326.5 s.

The policy is Amber's existing `lemon_eef_b64/29999`, with the existing 2k-step lemon world model. This earlier case-0 setting refreshes one Qwen subgoal every 50 steps, samples two 10-step EEF action candidates, and uses illustrated scoring examples. Only the first six WM frames (0.30 s) are scored; terminal-pose padding fills the remaining prediction horizon. It is distinct from the newer case-1 multi-subgoal/text-example experiment.

WM changed actions at steps 310, 320, and 400. The first 310 executed actions were identical to the control, with zero joint difference. After branching, each rollout continued from its own observations. The WM run recorded 64.5 mm maximum lemon lift, 182 finger-contact frames, `terminated=true`, and `truncated=false`.

At the first switch, Qwen scored the two predictions 0 / 0.6, claiming only the latter established a new grasp. Exact physical branches showed contact throughout both ten-step windows, no lift, and final lemon positions only 0.106 mm apart. This short-window evidence does not support such a clear physical-quality difference. Neither this local audit nor the successful full run proves a WM benefit.

- [Full paired execution video](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/execution-comparison.mp4)
- [Completed run report and candidate videos](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/report.md)
- [Paired outcome evidence](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/paired-outcome-audit.json)
- [First-switch actual/condition/WM review](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/switch-310-review/review.md)
- [Chinese report](amber-wm-qwen-2026-10-29.md)

No policy training, training configuration, training data, or checkpoint was modified. The completed job released its model workers; a replacement local Qwen service supports ongoing experiments, and the cube Gemini control has started.

All three switches now have exact physical counterfactual branches, and each selected branch matches the actual executed ten-step joint trajectory with zero error. At steps 310 and 320, the 0 / 0.6 score separation did not correspond to a clear grasp-quality difference. At step 400, the selected candidate scored 0.95 with a completion claim, while its actual branch remained unsuccessful and the object was still about 65.4 mm from the target center. None of the three selected candidates reduced short-window target distance more than its alternative, but differences were small and distance alone does not establish long-term harm. [All-switch videos and evidence](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/switches-review.md).
