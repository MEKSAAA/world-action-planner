# Chunru π0.5：完整 lemon 回合测试

**已完成完整回合测试。** 在与此前 50 步检查完全相同的初始状态下，分别测试每 10 步重新规划、按 checkpoint 配置每 50 步执行一个 chunk。两者均达到预设 1,000 步上限，原生成功判定为 false。10 步设置已经抓起并搬运 lemon 到目标附近，最后仍未完成放置；50 步设置未有效抬起 lemon。

[完整双列执行视频，正常速度](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-comparison.mp4)：左为 10 步重规划，右为 50 步执行。每条包含初始画面及 1,000 个执行步，共 1,001 帧、约 33.3 秒仿真时间。

## 设置与成功判定

- 使用 Chunru 新分享包中的 LeRobot π0.5，严格加载原权重和配套归一化；状态 / 动作格式为原生 EE_ROT6D / EE_ABS_ROT6D。
- Genesis 1.3.1，RoboWits `db6acf3`；沿用固定新布局 case 0、seed 202610270。这是原任务的布局变化，不能当作作者原始数据分布上的复现。
- 策略每次预测 50 个动作。第一组执行前 10 步后重新观察；第二组执行完整 50 步，对齐 checkpoint 的 `n_action_steps=50`。
- 两组均执行到原生成功终止或预设 1,000 步上限，取消此前 50 步人工审核暂停。1,000 是本实验设置的上限。
- 使用模拟器原生条件：lemon 在桌面上、其平面轮廓与绿色目标重叠至少 80%、不再被夹持，并通过边界检查。没有用 Qwen 判断任务成功。
- 本次是纯策略测试，没有调用 WM、Qwen 或 Claude，也没有训练或修改 π0.5 / Amber 的训练相关文件。

## 结果

| 设置 | 执行步数 | 策略查询次数 | 成功 | 结束原因 | 指尖接触 lemon 帧数 | 最大抬升 |
|---|---:|---:|---|---|---:|---:|
| 每 10 步重规划 | 1,000 | 100 | 否 | 步数上限 | 654 | 70.6 mm |
| 每 50 步执行 | 1,000 | 20 | 否 | 步数上限 | 126 | 4.9 mm |

**10 步组：具备明显任务进展，失败集中在最终放置。** 视频显示移开挡板、抓取、搬运；第 1,000 步仍有两根左手指与 lemon 接触，接触力范数合计约 9.28 N。lemon 中心高度约 0.808 m、桌面高度 0.760 m，平面中心距目标中心约 3.48 cm。结合视频，它已到目标附近但仍被夹持，未满足完整放置条件；这不是“完全不会抓取”。

**50 步组：未建立有效抬升。** 虽有指尖接触，但最高只比初始位置高约 4.9 mm，低于本诊断使用的 25 mm 抬升阈值，最后仍远离目标。视频主要表现为挡板附近的操作与抓取尝试。机器人与挡板接触 816 帧；移开挡板本就是任务的一部分，不能把所有这些接触都当作碰撞失败。

两组都经记录动作的物理重放复核：最大关节误差 **0**，成功判定一致，`terminated=false`、`truncated=true`。接触统计要求原生接触力大于 0.01 N，连续多帧接触不是独立抓取次数。

原策略回合墙钟约 70.6 秒 / 43.9 秒，不含初始化和事后诊断；两组部分并行运行，这些时间不是严格吞吐量基准。

## 对照与解释范围

两个完整回合及先前短程检查的初始状态完全一致。10 步完整回合的前 50 个动作与先前短程基线逐元素一致，关节差为 0。两种执行设置的首个 50 步候选 chunk 也完全一致；后续差异来自执行窗口与重新观察 / 采样的时机变化。

这是**同一个初始状态、两种执行设置**，不能当成两条独立测试样本估计成功率，也不能由此断言该 checkpoint 在所有 lemon 场景都会失败。此次只证明：在这个布局与 1,000 步预算下，两种设置都未完成任务，且 10 步重规划的任务进展明显更多。

后续若评估 WM 是否有用，应重点检查挡板处理、抓取以及最终释放 / 放置阶段，而不能仅看起始 50 步。当前 Qwen 在 WM 候选上的高分误判仍未解决，因此本次没有扩展 WM 闭环实验。

## 审核材料

- [10 步组完整视频](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-case0/execution.mp4) · [关键帧](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-case0/keyframes.png) · [结果](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-case0/result.json) · [接触诊断](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-case0/contact-replay.json)
- [50 步组完整视频](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-case0-execute50/execution.mp4) · [关键帧](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-case0-execute50/keyframes.png) · [结果](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-case0-execute50/result.json) · [接触诊断](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-case0-execute50/contact-replay.json)
- [初始状态 / 前缀一致性验证](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-pair-validation.json)
- [10 步运行参数](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-spec.json) · [50 步运行参数](/data/mjw/world-action-planner/chunru-release/20261027/full-episode-execute50-spec.json)

代码只调整 WAP：纯策略模式允许完整执行 50 步 chunk；接触审核读取运行记录中的动作 / 观测模式，并保存原生终止标志及最终物体位置。验证包括两条完整真实推理、两条逐步一致的物理重放、视频完整读取，以及 `tests/robowits` 的 28 passed / 6 skipped。所有本次实验已结束。
