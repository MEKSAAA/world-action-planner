# 多子目标 × Amber π0.5 × 微调 WM：实验方案与人工候选审核

状态：实现与首轮诊断进行中，尚无新多子目标完整回合结果。2026-10-29。

## 已确认的设置

- 人工候选以动作合理、但时机或对象错误为主，每轮保留一个预期合适的候选。
- Gemini、Qwen 分别提出子目标并评价结果；另用同一批候选预测交叉评分。
- 诊断阶段每次提出 4 个子目标，各采样 2 个动作块。后续完整回合按用户要求允许 3–5 个不同子目标（优先 3 个），每个采样 1 个动作块，共 3–5 个候选；保留多动作选择，同时降低重复预测成本。
- 每个动作块 10 个原生控制步；WM 预测 20 帧，评分仅看动作覆盖的前 6 帧（约 0.30 秒）；选中后执行 10 步，然后重新观察、提议、采样、预测、评分。
- 策略及 WM checkpoint 全程冻结。完整回合以原生模拟器成功或 1,000 步终止。
- 人工“正确”是待检验的预期；若物理分支显示另一候选也可行，记录歧义，不改标签来迎合结果。

## 候选链草案：需要用户确认后运行人工版本

这些是按当前画面选择的阶段，不是按固定步数强制推进。实际输入模型的候选随机排序，隐藏下表中的预期类别。

| 当前可见阶段 | 预期合适的候选 | 时机错误 | 对象或方向错误 | 时机或对象错误 |
|---|---|---|---|---|
| lemon 仍在窄缝，挡板阻挡夹爪 | Move the obstructing block away from the lemon | Pick up the lemon | Move the other block toward the lemon | Place the lemon on the green target |
| lemon 已可接近但尚未抓住 | Pick up the lemon | Place the lemon on the green target | Pick up the nearby block | Move the block back beside the lemon |
| lemon 已抓住，尚未到目标上方 | Carry the lemon above the green target | Put the lemon down beside the gap | Carry the lemon back toward the gap | Pick up the nearby block |
| lemon 在目标上方，尚未放下 | Place the lemon on the green target | Carry the lemon back toward the gap | Put the lemon down beside the target | Pick up the nearby block |

不采用“搬动绿色目标垫”：原生任务中的目标垫是固定标记，不适合作为合理可执行的错误动作。

第二任务选 `robowits/02-retrieve-cube-v0`（容器取方块）。先检查资产与策略执行能力；同一个 lemon checkpoint 在此属于跨任务迁移，必须独立报告。

| 当前可见阶段 | 预期合适 | 其余三个候选 |
|---|---|---|
| 方块在深窄容器内 | Tilt the container to release the cube | Pick up the cube / Place the cube on the green target / Move the container away from the target |
| 方块已在桌面，尚未抓住 | Pick up the cube | Place the cube on the green target / Pick up the container / Put the container over the cube |
| 方块已抓住 | Carry the cube above the green target | Return the cube to the container / Put the cube down beside the target / Pick up the container |
| 方块已到目标上方 | Place the cube on the green target | Return the cube to the container / Carry the cube away from the target / Pick up the container |

## 公平对照与评分

1. 同状态诊断：固定画面、子目标、动作采样种子和 WM 预测，分别由 Gemini / Qwen 看当前画面及候选文字评分，再看预测评分。之后在独立模拟器分支执行已保存的动作，检查预测和选择是否正确。
2. 完整回合：比较不看 WM 与看 WM 的选择策略，使用同一批初始状态。分叉后观测不同，不能再声称整回合候选池相同；只保证生成规则和种子方案一致。
3. 自然 VLM 候选与人工混合候选分表报告，避免把人为错误检测能力当成真实任务提升。
4. 评分依据整个任务的可见增量进展，不给“完成错误子目标”额外奖励。静止、靠近、清障、抓起、搬运、放下分别给文字示例；分清“已经抓住”和“新抓住”，不把手移动当作物体移动。
5. 之前视觉 few-shot 发生评分失真，本轮先用文字评分示例；在人工检查真实/预测视频后再决定是否加入视觉示例。全零、全高分、模型认为不可判读等均单独统计。

指标包括：原生成功/步数、失败类型、候选动作差异、阶段/锚点错误、WM 预测与实际分支差异、两位评分者一致性、WM 是否改变选择、改变是否有益、耗时/API token 用量。

## 初步规模和限制

先做 lemon 与第二任务的观测/动作可执行性及离线评分检查；人工候选审核通过后，做两种候选来源 × 两位评分者的关键状态分支诊断。再开展每任务每设置至少一个完整配对回合；初步结果不可称为可靠总体成功率。

8 个候选约需 6 分钟 WM 时间/轮，完整 100 轮可能超过 10 小时。先检查候选是否产生可区分动作、短视野是否能区分后果，避免在无信号设置下直接铺满实验。短诊断不替代完整回合结果。

当前 WM 固定一秒预测，而 Amber 单块约三分之一秒，需要末端保持填充。评分不看填充段，但联合视频生成仍可能受它影响。清障的长期价值可能超出可见短视野；若区分不了，应报告限制，而非强迫模型选预设答案。

Gemini 接口依据 [Google GenerateContent 文档](https://ai.google.dev/api/generate-content)，模型与用量将记录在实际响应中；凭据不进入代码或实验日志。

## 首轮实际检查

实验根目录：`/data/mjw/world-action-planner/multigoal/20261029`。

- Gemini API 已连通，响应模型为 `gemini-3-flash-preview`；本地 Qwen 为 Qwen3.5-9B。
- 第二任务资产加载、RGB-D 提议成功；Gemini 和 Qwen 都给出 4 个候选。Qwen 把绿色目标区域称为 green cube，记录为潜在识别错误，未人工修正候选。
- lemon 的 Gemini 提议为左移左挡板、右手抓 lemon、抬起右挡板、右移右挡板，每个采样 2 组动作。Gemini 不看 WM 时偏向左移左挡板（0.9），Qwen 偏向直接抓 lemon（0.9）。
- 8 个候选之间，10 步末端位置轨迹的平均两臂联合欧氏差异约 0.36–1.94 毫米。这个统计只含位置，未混合旋转和夹爪开合。
- 8 个真实模拟器分支均从完全相同观测出发。10 步内 lemon 移动约 2.24–2.32 微米，挡板位移不到 1 微米：本次起始状态的短动作没有产生可区分的物体后果。
- 真实短视频评分预检中，两位评分者都给严格静止对照 0；旧 Chunru 首次分歧的真实视频评分为 Gemini 0.01/0.05、Qwen 0/0。Amber 完成前的短片也都为 0，因此目前没有证据说明新的评分器已具有有效的正负区分能力。
- 遇到两种接口错误并保留日志：启动器解析虚拟环境 Python 符号链接导致丢失包；VLM 返回裸数组/JSON 后多余文本。前者已修复，后者使用最多三次重新请求并保留每次原文，绝不修改输出分数来通过校验。
- 测试：43 passed, 1 skipped。真实动作诊断已经完成；同一候选池的 WM 预测、Gemini/Qwen 交叉评分正在进行。人工版本仍等待具体候选链确认。

这些结果表明“候选文字不同”不等于“短期动作后果不同”。下一步必须加入接近物体、抓取、搬运、放置状态的检查，再决定完整矩阵的规模。不能从当前初始状态的八个无接触分支推断 WM 有用或无用。

## 运行与当前未完成项

入口：`python -m robowits.multigoal_loop --config CONFIG.json`；配置示例为 `robowits/configs/multigoal-example.json`。每次必须设置新的输出目录。`mode` 为 `selection` 或 `without_wm`；人工版本再传审核后的 `manual` 文件路径。

首轮八候选真实执行视频：`/data/mjw/world-action-planner/multigoal/20261029/lemon-eight-actual-candidates.mp4`，5 fps 慢放，原生 10 步，属于诊断而非完整回合。

完整新链路无 WM 对照已启动：`lemon-gemini-full-without-wm`，成功或 1,000 步结束；同池初始状态 WM 预测位于 `lemon-saved-wm`。原有 Amber 单子目标完整 WM 回合仍独立运行。

仍需完成：人工链确认与运行；接触/搬运阶段的候选及评分检查；Gemini/Qwen 与人工/VLM 来源的对照；第二任务的动作、预测和完整回合检查；相同初态的有无 WM 完整配对；汇总耗时、原生成功及失败归因。当前目标未完成。

## 后续完整配对的冻结规则

正式多子目标完整配对使用相同代码版本；对照完成后，WM 组在未分叉的相同状态直接复用整批候选，核对所有策略输入、子目标、随机种子、目标姿态及执行关节轨迹。两组一旦选择不同动作，后续各自根据新观测生成候选。源代码/模型/任务配置不匹配则拒绝配对。当前较早启动的完整对照仅作为探索回合，不与更新后的提示词版本混作正式配对。

预先固定 WM 无信息时的规则：最高分并列且包含无 WM 选择时，保留无 WM 选择；全部预测被判无效时，明确记录回退到无 WM 选择，不能计作 WM 批准或改进。其他情况选择最高有效分。

第 350 步状态的 Qwen 提议曾重复相同语言、只改 arm；保留失败记录后，在重试中加入具体校验错误。后续一次诊断通过，八动作的末端位置轨迹平均差异最大约 19.7 毫米，显著大于起始状态；实际分支及 WM 预测待完成。四条语言中“move towards”和“move to”仍有语义近似，需计入候选多样性不足，不能仅靠 JSON 校验声称语义不同。

## 已完成的局部对照与当前限制

- 起始状态八组 WM 预测及两模型评分已完成。低推理 Gemini 会给没有真实物体运动的候选 0.2–0.3；Qwen 全零。不能认为这证明 WM 改善了动作选择。
- 第 350 步八个真实分支与 WM 预测均已完成。lemon 实际移动 30.6–56.1 mm。原低推理 Gemini 的 WM 选择为候选 3，无 WM 选择为候选 4；Qwen 因全零保留无 WM 选择。这里的局部位移不是任务成功率。
- 视频与表格：`/data/mjw/world-action-planner/multigoal/20261029/initial-review/review.md` 和 `step350-review/review.md`。所有真实分支都由同一历史状态重放，物理结果仅用于事后诊断。
- 隐藏候选文字与拆分帧图的检查没有解决全部评分问题。Gemini 开启 high 推理后能识别部分真实搬运，但整批评分仍有“分数非零、解释却说静止”的矛盾；Qwen 开启推理后仍把一个真实搬运片段判为静止。评分器可靠性尚未证实。
- 不依据某个有利分数筛选结果。后续完整回合采用盲评分：WM 评分只看总体任务和画面，不显示候选指令；Gemini high、Qwen 默认非推理均记录实际配置。两者表现可能不同，不能把设置变化混作 WM 增益。
- 早期完整无 WM 探索回合在第 230 步因 Gemini 连续输出 JSON 后多余文本而中断，状态为 `error`，没有模拟器终局结果。严格 JSON Schema 已用同一失败请求验证通过；后续完整配对使用冻结代码和新输出目录。
- 第二任务八个真实短分支完成，并实际验证了整批候选复用的完整策略输入、种子、目标姿态匹配；其 WM 预测仍在运行。
- 最新测试：46 passed, 1 skipped。

后续正式配对使用未用于上述评分调试的 lemon 初态 case 1（seed 202610271）。评分调试仍属于开发集诊断，不作为正式成功率样本。人工版本的具体链条仍待用户确认。
