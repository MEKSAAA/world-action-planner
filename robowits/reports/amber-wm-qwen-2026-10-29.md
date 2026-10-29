# Amber lemon π0.5：重新测试完整 WM + Qwen 闭环

**完整配对已结束：无 WM 对照在第 514 步成功，WM + Qwen 组在第 526 步成功。** 两者均通过原生物理重放，最大关节误差为 0。这一个初态未显示成功率或执行步数收益；不能从一对回合估计总体成功率。

|设置|原生成功|执行步数|回合墙钟耗时|
|---|---|---:|---:|
|Amber + Qwen 子目标，无 WM 筛选|是|514|64.9 秒|
|同策略 + WM + Qwen 筛选|是|526|7,229.6 秒（120.5 分钟）|

WM 组多执行 12 步，实测墙钟约为对照的 111.4 倍。该耗时不含启动初始化，且与其他实验共享 GPU，不能当作独占硬件性能基准。其中 WM 推理累计 6,069.2 秒，Qwen 评分 723.8 秒，机器人代理 326.5 秒。

[完整实际执行对比视频](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/execution-comparison.mp4) · [完整结果与候选视频](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/report.md) · [配对证据](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/paired-outcome-audit.json)

WM 组共 53 次决策，在第 310、320、400 步改选。前 310 步动作完全一致、关节差为 0；分叉后各自继续闭环。WM 组最大抬升 64.5 mm、182 帧指尖接触，原生 `terminated=true, truncated=false`。这些事实说明最终仍可完成任务，没有证明 WM 使任务更好。

这套是 case 0、每 50 步更新一个子目标、每次采样两个动作、带图片评分示例的旧设置；不混入 case 1 的新多子目标/文字示例实验。

[成功对照完整视频](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/baseline/execution.mp4) · [关键帧](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/baseline/keyframes.png) · [对照结果](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/baseline/result.json) · [物理复核](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/baseline/contact-replay.json)

## 实际使用的设置

| 项目 | 设置 |
|---|---|
| checkpoint | `/data/robowits-checkpoints/pi05_robowits_subgoal_eef/lemon_eef_b64/29999` |
| 推理实现 | 既有 OpenPI/JAX 环境，原 `create_trained_policy` 和 checkpoint 归一化 |
| 仿真 | Genesis 1.3.1；RoboWits `db6acf3` 只读 worktree |
| 初始状态 | 固定新布局 case 0、seed 202610270，与 Chunru 实验同一布局 |
| 策略输入 | 多相机图像、机器人状态、子目标文字、手臂标记、RGB-D 提升的 3D 目标 |
| 子目标规划 | 本地 Qwen3.5-9B，每 50 步更新；沿用原生 planner 的 prompt / 解析 / 深度提升 |
| 动作 | 10 × 14 维 EEF 增量；使用 Amber 原生坐标变换与执行函数 |
| 执行频率 | 每轮执行 10 步，然后重新观察并规划动作 |
| WM | 既有 lemon 微调 2,000 步 checkpoint，不重新训练 |
| 候选 / 评分 | 两个候选；本地 Qwen，沿用上轮图片示例、图像变化量提示、独立评分 |
| 选择规则 | 有效候选 1 比有效候选 0 高出超过 0.10 才切换，同分保留 0 |
| 结束 | 原生成功或 1,000 步；没有 50 步暂停 |

没有训练 π0.5，也没有修改 Amber 的训练代码、配置、数据或权重。没有调用 Claude API。

## 与 Chunru 的比较边界

Amber 需要显式子目标和 3D 位置，因此这里的对照是 **Amber + Qwen 子目标规划器，不含 WM 动作筛选**；不是只有完整任务文本的策略。两组在动作分叉前复用同一份经观测核对的子目标和候选 0，并检查执行轨迹一致。

Amber 的动作窗口约 0.333 秒，而 WM 接口需要一秒。机器人代理在前 10 步之后保持最后目标姿态，用于满足接口；评分器仅查看前 **6 个预测帧 / 0.30 秒**，排除填充部分。但填充仍可能影响联合生成的视频，不能把它当成真正的一秒动作计划。

因此与 Chunru 的整任务输入、50 步 chunk、完整一秒有效预测相比，策略条件、动作表示和评分窗口均不同，不能当作只换权重的严格消融。之前 Amber 的旧实验还使用过不同规划器和 5 步执行窗口；本次成功不能被解释为同配置结果突然反转。

为观察换用 Amber 后的闭环行为，本次保留了上轮 Qwen 的评分示例，**没有把“移除示例”的诊断版本同时混入**。已知评分误判仍是解释限制，最终成功只由模拟器判定。

## 无 WM 对照结果

- 第 514 步原生成功，52 次策略查询；约 17.1 秒仿真时间。
- 最大 lemon 抬升 64.7 mm，181 帧有效指尖接触；重放关节最大误差 0。
- 原运行墙钟约 64.9 秒，不含初始化；这是一个固定初始状态的单次成功，不能当成总体成功率。

## 完整闭环产物

[实时 Markdown 进度](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/progress.md) · [状态 JSON](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/job-status.json) · [已完成的结果报告](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/report.md)

后台运行目录：`/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0`。每轮保存动作候选、子目标、WM 条件与预测、Qwen 原始请求 / 回复、评分和执行选择。本次已正常结束，自动物理重放、完整对比视频及 Markdown 均已生成。

代码只修改 WAP 的推理入口，使 EEF 模型可使用现有 illustrated Qwen reward，并在报告中标明子目标规划器。新增测试验证 0.30 秒之后的填充图像既不会显示给评分器，也不会进入图像变化量提示。相关测试：35 passed、1 skipped；原生策略已实际加载并完成成功回合。

## 首次动作改选：局部真实后果核查

选择组在第 310、320、400 步改选候选 1。首次改选时 Qwen 给候选 0 / 1 的分数为 0 / 0.6，声称后者新建立抓握。从共同历史精确恢复第 310 步后，两个真实候选都全程有指尖接触、均未抬升；lemon 位移为 5.875 / 5.774 mm，最终位置只差 0.106 mm。该短窗口没有验证如此明显的收益差异，不能把改选次数当作提升。

[真实/条件/预测视频与核查说明](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/switch-310-review/review.md)。首次分歧之前使用已结束对照的共同轨迹，严格核对当前图像哈希、历史关节与候选目标。该局部诊断执行时，旧入口尚未写出完整选择轨迹；回合结束后已落盘，并另外验证前 310 步动作及关节确与对照相同。此分析属于旧单子目标 case 0，与新多子目标 case 1 分开解释。

## 全部三次改选的分支检查

第 310、320、400 步的两个候选都已在相同历史状态真实执行；三个选中分支与原执行关节轨迹完全一致。第 310、320 步的 0 / 0.6 分差没有对应明显的抓取收益差异。第 400 步选中候选得 0.95、解释称任务完成，但真实 10 步后仍未成功，物体距目标中心约 65.4 mm。

三个选中候选在短窗口内都没有比另一候选更接近目标，但差距较小，距离也不是唯一质量指标，不能据此断言每次切换都损害长期结果。这里更明确的证据是评分理由不可靠，且完整回合没有省步数。

[全部三次真实/条件/WM 视频与统计](/data/mjw/world-action-planner/amber-release/20261029/full-wm-qwen-case0/switches-review.md)。
