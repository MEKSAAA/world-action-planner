# 完整 π0.5 → WM → Qwen 闭环运行

> 后续状态：本回合已跑满 1,000 步，未成功，发生 6 次切换。已完成[详细错误分析](wm-qwen-error-analysis-2026-10-28.md)，包含首次切换的因果干预、Qwen 提示消融和 WM 条件诊断。以下为原始启动协议。

本文件记录用户确认后的启动协议，**不是完成结果**。本次主实验是加入 WM 和 Qwen 的动作选择闭环，取消 50 步人工审核暂停。

[实时 Markdown 进度](/data/mjw/world-action-planner/chunru-release/20261028/full-wm-qwen-case0/progress.md) · [任务状态 JSON](/data/mjw/world-action-planner/chunru-release/20261028/full-wm-qwen-case0/job-status.json)

## 固定配置

- Chunru 新分享的 LeRobot π0.5；原生 EE_ABS_ROT6D 动作，沿用 checkpoint 归一化，没有训练。
- Genesis 1.3.1；固定新布局 case 0、seed 202610270，与前一天的完整纯策略回合一致。
- 每轮从 π0.5 采样 **两个** 50 步动作 chunk。
- 每个候选的前 30 步通过仅含机器人的代理生成骨架，WM 预测 **20 帧 / 1 秒**。沿用 lemon 微调 2,000 步的 WM。
- 本地 **Qwen3.5-9B**，沿用图片评分示例、三相机时间条和图像变化量提示，不使用 Claude API。
- 按既有规则选择动作：有效候选 1 必须比有效候选 0 高出 **超过 0.10** 才切换；相同分数保留候选 0。
- 执行选中候选的前 **10 步**，再观察并重复闭环。
- 停止条件：**模拟器原生成功，或达到 1,000 步**。基础设施 / 模型响应错误会单独标记为错误，不伪装成完成回合。

完整回合重新生成同配置纯策略对照用于配对核验。首次动作分叉前，严格匹配观测和种子后复用对照候选 0，并核对实际动作与关节轨迹。全部 WM 预测和 Qwen 评分重新计算，没有复用前一天 50 步实验的预测。

此前 Qwen 对短程候选的评分出现过饱和与视觉误判。本次保留该设置，观察完整闭环行为；成功与否始终使用模拟器原生条件，不能由 Qwen 分数代替。

## 自动收尾与结果位置

运行目录：`/data/mjw/world-action-planner/chunru-release/20261028/full-wm-qwen-case0`。

- `spec.json`：运行参数。
- `supervise.py`、`supervisor.log`：独立运行的启动、监控、收尾流程；对话结束不停止任务。
- `baseline/`：完整纯策略对照。
- `selection/`：完整 WM + Qwen 闭环；保存每轮原始候选、WM 输入 / 预测、Qwen 请求 / 回复、选择理由和执行轨迹。
- `progress.md`：每 5 秒更新执行进度和评分表。
- `job-error.json`：若出现运行错误，保留错误原因。
- 正常结束后自动重放实际动作、检查接触和任务判定，再生成 `summary.json`、`report.md`、`execution-comparison.mp4` 及首轮 / 末轮 / 前三次切换的候选视频。

收尾会关闭本次 Qwen 服务。不会自动扩大到其他初始状态。按此前约 105 秒 / 轮估算，100 轮需要约 3 小时；实际耗时取决于运行负载和是否提前成功。

报告生成器 `robowits/full_loop_report.py` 拒绝把暂停回合或与原生重放判定不一致的记录输出为完整结果。相关测试：29 passed、6 skipped。Amber 的训练代码、配置、数据、权重均未修改。
