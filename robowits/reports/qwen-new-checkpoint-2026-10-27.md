# RoboWits / WAP：新 π0.5 checkpoint 与 Qwen 评分示例

> 后续完整回合已补测：同一起始状态的 10 步 / 50 步执行设置都跑满 1,000 步；10 步组已抓取并搬到目标附近但未放置，50 步组未有效抬升。[完整回合报告](chunru-full-episode-2026-10-27.md)。

本批按用户要求，使用本地 Qwen3.5-9B 作为 value model，并换用 Chunru 新分享的 π0.5。**50 步三组检查已完成并暂停，仍未体现 WM 选择收益：10 个候选均被 Qwen 评为 0.95，5 轮均未切换，三组执行完全一致。** 没有训练 π0.5，没有修改 Amber 的训练代码、配置、数据或原 checkpoint。本批不能报告任务成功率。

## 新 checkpoint

[作者提供的下载](https://drive.google.com/file/d/1hdynNWbjeJBkGvOe8Dcf2ZxpvNJzGg3U/view) · [作者提供的训练入口](https://github.com/UMass-Embodied-AGI/RoboWits/blob/main/scripts/robowits/train/train_pi05.sh)

下载包 `release_chekcpoints.zip` 为 11,745,224,387 字节，包含 π0.5、π0 和 ACT；本批仅解压 π0.5，逐文件通过 ZIP CRC 校验。独立目录：

`/data/mjw/world-action-planner/chunru-release/20261027/release_chekcpoints/pi05`

| 项目 | 本次实际使用 |
|---|---|
| 策略实现 | 作者 worktree 中的 LeRobot PI05 |
| 仿真 | Genesis 1.3.1；RoboWits `db6acf3`，只读 |
| 状态 / 动作 | 原生 20D EE_ROT6D / EE_ABS_ROT6D |
| 动作顺序 | 右手位置、左手位置、右手旋转 6D、左手旋转 6D、两手夹爪 |
| chunk | 50 步，原生约 30 Hz |
| 归一化 | checkpoint 自带 QUANTILES pre/postprocessor |
| 输入 | ego、双腕图像、状态、完整任务文字；无标注子目标输入 |
| checkpoint 配置中的数据名 | `local/pi05-EE_ABS_ROT6D-_full` |

严格加载所有参数；原生库加载失败可能返回不完整模型，所以本适配器显式执行 `load_state_dict(strict=True)`。对于 safetensors 去重的 tied embedding，只恢复文件元数据声明且原生模型确实共享存储的别名。未使用随机初始化替代缺失参数。`_full` 是配置名称，不能据此确认全部训练任务、实际保存步数或联合留出划分。

新模型与旧 lemon/subgoal checkpoint 的动作格式、任务条件和推理窗口均不同，因此本批不是只换权重的受控消融。三组闭环内部使用同一个新模型。

## Qwen 评分修改与校准

最终版本把三个相机分别展示为“当前 + 三个未来时刻”的时间条，并加入三个真实图片示例：未抓取且静止 0 分、新抓取 0.6 分、已抓住但静止 0 分。示例来自 WM 训练 episode 105 的帧 440–480，标签由助手初看，**尚待用户审核**。正对照来自 episode 101；实际闭环使用预先固定的新布局。

还在 prompt 中给出各相机当前 / 未来画面的像素变化量，提醒模型静止画面没有新进展。变化量仅从实际输入图片计算；不使用仿真物体真值，也没有用规则覆盖 Qwen 输出。非零变化本身不代表任务进展。

| 校准版本 | 真实搬运 | 静止对照 | 判断 |
|---|---:|---:|---|
| 仅加文字评分示例 | 0 | 0 | 未通过正对照 |
| 图片示例 + 分相机时间条 | 0.8 | 0.8 | 把已持物的静止画面误判为搬运 |
| 再加入图像变化量说明 | 0.8 | 0 | 初步区分正负对照 |

最终版本交换候选顺序后分数仍跟随内容；相同真实片段重复得 0.8 / 0.8。两个近静止 WM 窗口的四个候选均为 0，静止画面加 ±2 像素噪声也为 0。**部分理由仍误认物体或声称不存在的相机运动，不能把文字理由当作可靠视觉证据。** 样本很少，这不是全面的 reward 验证。

[正对照原始结果](/data/mjw/world-action-planner/chunru-release/20261027/illustrated-measured-positive/results.json) · [负对照原始结果](/data/mjw/world-action-planner/chunru-release/20261027/illustrated-negative/results.json) · [示例及标签](/data/mjw/world-action-planner/chunru-release/20261027/visual-examples/manifest.json)

## 候选与配对协议

固定新布局 case 0、seed 202610270；每轮采样两个候选。每个候选的前 30 个原生动作，经不包含任务物体的机器人代理转换为骨架条件，供 WM 预测 20 帧 / 1 秒；不填充动作尾部。Qwen 评分后执行所选候选前 10 步，共 5 轮、50 步后暂停。

同一起始观测采样 8 个候选，1 秒末端目标的平均两两距离 **5.14 cm**，最大 **8.73 cm**，同种子重采样差异 0。这是指令空间距离，不是物理位移、不同任务计划或成功率。不能与旧 checkpoint 在不同状态和窗口下的距离直接作等条件比较。

三组分别是原策略、WM/Qwen 都运行但强制候选 0、按 Qwen 分数选择。切换须比有效候选 0 高出 **超过 0.10**；不为增加切换次数临时调整阈值。分叉前核对完整策略输入、种子、初始状态和动作 / 关节轨迹；相同条件下显式复用候选，保留新采样输出。预测缓存要求完整输入匹配。缓存节约时间单独记录，不能视作部署延迟。

## 本批执行结果

三组均执行 50 步、5 轮后暂停，成功字段为 `null`，不是完整失败回合或 0% 成功率。固定候选 0 组、选择组相对策略基线的动作逐元素相等，最大关节差均为 **0**。选择组复用了严格匹配的全部 10 段预测及其评分。

| 原生步数 | 候选 0 | 候选 1 | 执行候选 |
|---:|---:|---:|---:|
| 0 | 0.95 | 0.95 | 0 |
| 10 | 0.95 | 0.95 | 0 |
| 20 | 0.95 | 0.95 | 0 |
| 30 | 0.95 | 0.95 | 0 |
| 40 | 0.95 | 0.95 | 0 |

**当前 Qwen value model 未通过实际候选检查。** 它反复声称 lemon 已放置在绿色目标区；在第 30 步甚至把“lemon 从腕部视野消失”当作搬运与释放证据。初看对应预测，lemon 仍在挡板附近，缺少完成放置的画面证据。因此不能因为静止 / 真实搬运控制通过，就宣布评分器有效；也不能据此判断 WM 的预测完全无用。此次首先暴露的是评分分数饱和与视觉误判。

固定候选 0 组每轮平均耗时：两候选策略采样 0.61 秒、机器人代理 6.22 秒、WM 89.46 秒、Qwen 8.94 秒，合计约 **105 秒 / 轮**，每轮仅执行约 0.333 秒仿真动作，离实时控制很远。整段基线约 7.1 秒，完整评分组约 534.5 秒，缓存选择组约 40.8 秒；这些不含模型 / 仿真初始化，缓存选择组不能当作实际部署耗时。

### 请优先人工审核

1. [第 30 步候选视频](/data/mjw/world-action-planner/chunru-release/20261027/review/candidates-step-30.mp4)：是否真的能看到 lemon 到达绿色区并释放？两候选是否有明确优劣？看不清就标“无法判断”。
2. [三组实际执行对比，4 倍慢放](/data/mjw/world-action-planner/chunru-release/20261027/review/execution-comparison.mp4)：这里三列相同是核对后的实际结果。
3. [提示词中使用的评分示例视频](/data/mjw/world-action-planner/chunru-release/20261027/review/value-examples.mp4)：中间的“新抓取 0.6”标签是否合理？

[完整 Markdown 审核表及全部五轮候选视频](/data/mjw/world-action-planner/chunru-release/20261027/review/review.md) · [原始评分理由](/data/mjw/world-action-planner/chunru-release/20261027/review/selection/decisions.json)

下一步应先用人工确认的“手臂在动但未抓取 / 未放置”等困难负例校准 Qwen，并在另外的候选窗口验证。当前不应扩展长程成功率实验或继续训练策略。实验已暂停，不会自动继续；本次使用本地 Qwen，没有调用 Claude API。

## 代码与复现

- `robowits/rot6d_loop.py`：新 checkpoint 原生严格加载、推理、三组短程对照。
- `robowits/illustrated_value.py`、`vlm_server.py`：Qwen 多模态示例和独立评分。
- `robowits/build_value_examples.py`：从现有训练视频重建示例；九张示例图片重建后与本批逐字节相同。
- `robowits/review_eef_batch.py`：按实际评分窗口导出 Markdown 和视频。
- `robowits/prompts/README.md`：示例构建与迁移说明。

[固定运行参数](/data/mjw/world-action-planner/chunru-release/20261027/rot6d-spec.json) · [运行日志](/data/mjw/world-action-planner/chunru-release/20261027/run-review.log)

测试：本机仿真 Python 下 `tests/robowits` 为 28 passed、6 skipped；另实际验证新权重严格加载、原生 50 步执行和 Qwen 图像输入。跳过项不能视为通过。
