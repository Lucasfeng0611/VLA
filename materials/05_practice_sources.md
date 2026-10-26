## 16｜把论文知识映射到 openpi 实现

### 16.1 固定版本，再讨论“默认参数”

本讲义核对的 openpi commit 为 215abfb217dbac7d5f1273282331b9b1866c0479（2026-08-24）。读取日期为 2026-10-24 至 25。源码来自官方仓库，本文保存读取文件的 SHA-256 清单用于追溯。未来主分支可能变化，因此以下参数是这个快照的结果，不是永远不变的产品规格。[S12]

|位置|阅读目标|需要避免的误解|
|---|---|---|
|README.md|模型清单、支持范围、环境与微调示例|论文全部能力不等于仓库全部实现|
|src/openpi/models/pi0_config.py|action_dim、action_horizon、pi05 开关|32 维默认容量不等于真实机械臂维度|
|src/openpi/models/pi0.py|prefix / suffix、compute_loss、sample_actions|flow 的 t 与论文 τ 方向不同|
|src/openpi/training/config.py|数据变换、checkpoint、训练配置|类默认值可能被具体 recipe 覆盖|
|src/openpi/transforms.py|归一化、delta 动作、token 化等|输入处理与输出逆变换必须成对|
|scripts/train.py|实际 batch、优化器更新与日志|分片设备数不直接等于数据副本数|

### 16.2 已核对的几个参数

|字段 / 行为|快照中的值或说明|实际含义|
|---|---|---|
|Pi0Config.action_dim|32|模型动作向量容量，可含填充|
|Pi0Config.action_horizon|50|每次预测 50 个动作时间位置|
|max_token_len|π0 为 48，π0.5 为 200|对应文本等 tokenizer 的长度设定，不是全部视觉 token 上限|
|dtype|bfloat16|模型配置中的计算类型，不表示所有状态都占 2 字节|
|sample_actions.num_steps|默认 10|一次动作块生成的积分次数|
|TrainConfig.batch_size|默认 32，标注为 global|需确认具体训练配置和后端是否覆盖|
|num_train_steps|默认 30,000|优化更新预算，不能当成 30,000 个 episode|
|ema_decay|默认 0.99|参数移动平均，可被 recipe 改动或禁用|

README 明确说明，这个版本对 π0.5 的训练与推理支持 flow matching head。不要把持有该 checkpoint 等同于获得论文中全部高层子任务生成、FAST 联合训练和非机器人数据混合管线。不同 JAX / PyTorch 后端支持程度也可能不同，必须分别看对应脚本，不能仅凭相同配置字段推断功能一致。[S12]

### 16.3 设备预算怎样估算

显存至少分成权重、梯度、优化器状态、激活、临时工作区和缓存。以 3.3B 模型作教学估算：若所有权重仅按 2 字节存储，约 6.6 GB（十进制），但训练显然不止这些；还要梯度、动量 / 方差、可能的主权重和多模态激活。LoRA 可以减少优化器状态，但视觉 token 与长历史激活仍可能占很大空间。

官方快照的单卡估计是推理 >8 GB、LoRA 微调 >22.5 GB、全量微调 >70 GB，并说明其测试环境是 Ubuntu 22.04。这是该项目给出的配置估计，不能作为所有 batch、相机、精度和实现下的保证。当前 Windows 电脑可用于阅读、数据准备或远程开发；本文未在用户电脑上安装训练环境或启动机器人训练。[S12]

## 17｜微调与验证的可执行路线

### 17.1 第一步：定义任务和成功标准

教学项目：让双臂机器人把两类物体分别放入指定容器。首先定义输入相机、关节与夹爪状态、动作空间、控制频率、超时时间、成功 / 失败标准。再决定是要同场景成功、未见物体成功还是新场景成功。任务定义决定数据划分与评测，不应训练完成后才挑一种最好看的口径。

保留失败原因标签：识别错对象、语言理解错、接近失败、夹持失败、释放失败、队列停顿、长程漏步骤。它们对应不同改进方向。一个总成功率不足以告诉你应该增加数据、改归一化、引入 MEM 还是 RTC。

### 17.2 第二步：验证数据与接口

先可视化几个完整 episode，同时画关节状态、目标动作、夹爪通道与时间轴。完成一次归一化再反归一化的往返检查；完成动作输入变换到输出逆变换的往返检查；检查 episode 尾部是否跨越下一 episode 取未来动作；确保缺失相机与填充位置处理正确。

再取极小、确定的数据子集验证训练管线是否能拟合，目的在于排查接线问题，而非证明泛化。若连小数据都拟合不了，应先检查标签方向、mask、时间错位、动作单位和冻结参数范围。只增加训练步数可能掩盖根本错误。

### 17.3 第三步：建立最简单有效基线

先用可核对的公开 checkpoint 和原始推理方式建立基线，保存完整配置与闭环日志。微调时一次只调整少数因素，分别比较全量与 LoRA、不同数据覆盖、不同学习率或更新预算。改变 batch 时同时记录样本呈现次数和 wall-clock，避免只按 step 对比。

教学试验表：

|试验|保持一致的条件|唯一主要变化|要回答的问题|
|---|---|---|---|
|A|任务、评测集、控制定义|基础 checkpoint 直接运行|迁移起点如何？|
|B|同上|加入目标任务示范微调|领域适应是否有效？|
|C|B 的数据与预算|LoRA / 全量更新方式|资源节省的质量代价是什么？|
|D|同一模型与评测|RTC 开 / 关|瓶颈是否来自延迟？|
|E|相同数据覆盖|历史记忆建模|任务是否确实依赖过去？|
|F|相同起点|经验 / 纠正 / RL 改进|收益是否超过额外交互成本？|

这里 E、F 是研究设计，不意味着公开 openpi 中已有一键 MEM / RECAP。若缺少对应组件，应先实现并验证接口，不要把一个模块名称当成可直接打开的参数。

### 17.4 第四步：度量真实闭环

至少报告：任务成功次数 / 总次数、场景与对象范围、最大时限、成功与失败耗时、总墙钟吞吐、人工介入、模型延迟分布和队列耗尽。不同初始状态应按预先制定的规则抽样，测试时不要只挑训练过的摆放。

![图17：验证必须覆盖数据、策略和运行系统](figures/17_eval.png)

教学统计例：20 次试验成功 16 次，点估计是 80%，但样本很少，不能把它讲成精确到个位数的真实成功率。模型 B 成功 17 次也不足以直接证明比 A 显著更好。需要更多试验、相同条件的配对比较或适当置信区间，并明确失败与超时计入规则。

吞吐也可能与成功率相冲突。教学例：策略 A 每次平均 20 秒、成功率 80%；策略 B 每次平均 10 秒、成功率 70%。忽略重置的近似成功吞吐分别为 2.4 和 4.2 次 / 分钟；若 B 的失败需要长时间人工重置，实际优势可能消失。真正应报告全程测得的成功数 / 总时间。

### 17.5 常见故障排查

|现象|优先检查|不要先做什么|
|---|---|---|
|loss 低，机器人方向相反|动作坐标系、单位、delta 逆变换|盲目加模型规模|
|每个 chunk 边界停顿|同步调用、队列、端到端延迟|只加训练 epoch|
|切块抽动|时间对齐、旧前缀、模式切换|直接假定平均动作一定安全|
|完全不听语言|任务与场景捷径、指令多样性、遗忘|只扩大图像分辨率|
|验证好，换房间失败|场景泄漏、相机视角、数据覆盖|只报告随机帧划分结果|
|重复同一种失败|历史条件、失败恢复数据、状态覆盖|把所有问题都归因于采样随机性|
|RL 越训越激进|奖励尺度、critic 偏差、参考正则、数据支持|仅继续提高学习率|
|接上 MEM 后超时|视频编码成本、缓存、K、调度|把固定输出 token 当零额外成本|

## 18｜贯穿案例：整理厨房时，各方法分别在做什么

总指令：“把杯子放进柜子，擦干台面，最后关闭所有柜门。”初始视觉包含杯子、湿台面和柜子，关节状态告知机械臂位置。VLM 表征提供对象与语义关系，高层策略选择“拿起杯子”，动作专家生成连续动作块。

控制器以 50 Hz 跟踪动作目标；推理服务在旧块执行期间计算下一块。RTC 使新块与实际已排队动作兼容。若杯子被另一只手遮挡，短期视频历史有助于保持杯子位置与运动线索；长期文本记忆记录“杯子已放入，柜门还开着”。

夹取失败后，短期历史让同一模型看到刚才的失败，可能改变接近姿态；这一步不必更新权重。若经过多次部署，发现“从这个角度抓”长期失败，则收集失败和纠正，使用微调或 RECAP 改变未来策略。若瓶颈只在关门锁扣的精密对准阶段，可以研究 RLT 局部优化，而不必从头训练整个通用模型。

π0.7 式的目标图像可进一步给出希望的台面或柜子状态，metadata 可指定期望行为风格。但最终是否真正放好了杯子、擦干了台面、关上柜门，仍需要真实观测与任务评测确认。计划文本或生成图片不能代替物理结果。

![图18：厨房任务中的方法分工](figures/18_case.png)

这个案例也展示兼容性边界：MEM 记忆应根据实际执行更新；RLT 修改动作后，RTC 应读取修改后的承诺前缀；高层切换子任务时，要决定旧 chunk 继续多久；RL replay 应记录真实动作与对应观测版本。把每个模块独立跑通，只是系统整合的开始。

## 19｜检查理解：十五问与答案

### 问题

1. B=16、H=50 是否意味着 batch size=800？
2. 梯度累计 8 次是否等于把一个动作块执行 8 次？
3. 50 Hz 控制是否要求完整大模型推理每次少于 20 ms？
4. Transformer 的所有 token 都是离散 ID 吗？
5. flow matching 的 velocity 一定是关节物理速度吗？
6. 为什么论文目标 A−ε 和 openpi 目标 ε−A 都可能正确？
7. FAST 训练快，为什么推理可能慢？
8. stop-gradient 是否等于冻结整个 VLM？
9. π0.5 的公开 flow checkpoint 是否自动包含全部论文体系？
10. RTC 为什么不能简单用两块动作平均代替？
11. MEM 的 15 分钟任务是否意味 VLM 同时读入 15 分钟原始视频？
12. 失败一次后换策略，是否证明模型进行了在线 RL？
13. RECAP 为什么保留负优势数据？
14. RLT 的 RL token 是否就是一个奖励 token？
15. 为什么 online RL 的 15 分钟数据不等于 15 分钟完成训练？

### 答案与解释

1. 不是。B=16 个观测锚点样本，每个标签含 50 个相关未来动作。800 是动作时间位置数。
2. 不是。前者发生在参数优化循环，后者是环境执行；它们的时间尺度与变量完全不同。
3. 不一定。可以低频预测动作块，由控制器高频消费；但必须有实时调度与足够队列余量。
4. 不是。图像 patch、连续状态和连续动作可以用向量 token；离散文字和 FAST token 只是其中一类。
5. 不一定。它是生成时间中的动作样本变化率；要看动作本身定义才能判断物理单位。
6. 两者采用相反的生成时间方向。插值、速度符号和积分方向一起转换才一致。
7. teacher-forced 训练并行、信息密度高；推理仍有串行 token 解码和大骨干成本。
8. 不是。KI 只阻断动作专家到骨干的特定梯度，骨干仍可由离散动作与 VLM 目标训练。
9. 不是。公开实现、训练数据、辅助目标、高层策略和完整研究系统的可用范围不同。
10. 多峰有效策略的平均可能无效，且不能保证时间对齐。RTC 在生成时引入已承诺动作条件。
11. 不是。短期观测保持细节，长期文本压缩事件；最终骨干 token 数也经过控制。
12. 不是。MEM 的上下文适应可以在权重不变时发生；在线 RL 涉及经验驱动参数更新。
13. 用条件区分行为质量，同时保留状态覆盖和对低质量行为的建模；部署偏向正优势条件。
14. 不是。它是从 VLA 内部表示中学习到的连续状态特征，供小 actor-critic 使用。
15. 因为重置、准备、人工干预、计算、通信与评估都消耗墙钟时间。

## 20｜符号、术语与公式速查

|符号 / 术语|本讲义含义|容易混淆的对象|
|---|---|---|
|t|真实机器人控制时间索引|flow 生成时间|
|τ|噪声到动作的生成时间，0→1|openpi 反向时间 t，1→0|
|B|训练样本 batch 数|机器人自由度、动作块长度|
|K|短期历史帧数|attention 的 Key 矩阵，需看上下文|
|H|VLA 预测动作步数|实际执行步数 s|
|d_a|每步动作维数|RTC 延迟步数 d|
|d|RTC 推理延迟，以控制步计|动作 embedding 宽度|
|s|执行 / 重规划间隔所用 horizon|MDP 状态 s，需看章节|
|C|RLT 实际动作块步数|前文相机数量 C，需看章节|
|θ、φ、ψ|策略、辅助模块、价值模型等参数|固定数学常数|
|V、Q、A|状态价值、动作价值、优势|动作块 A，需看章节|
|BC|Behavioral Cloning，行为克隆|所有形式的强化学习|
|VLM / VLA|视觉语言 / 视觉语言动作模型|普通文本模型|
|KI|Knowledge Insulation，知识隔离|全骨干冻结|
|RTC|Real-Time Chunking，实时分块执行|缩短所有前向计算的加速器|
|MEM|Multi-Scale Embodied Memory|KV cache 或参数记忆的同义词|
|RECAP|经验与纠正驱动的优势条件策略学习|单纯成功轨迹筛选或通用 PPO|
|RLT|RL Token 路线|给奖励数值加个文字 token|

关键公式汇总：有效 batch = b×D×G；固定集合每轮更新约 N/B_eff；动作覆盖时间 H/f；RTC 等待覆盖条件 d≤H−s；flow 路径 Xτ=(1−τ)ε+τA；flow 目标 A−ε；优势 Aπ=Qπ−Vπ；chunk TD bootstrap 使用 γ^C。所有公式都依赖正文说明的计数与时间约定。

## 21｜原始资料、阅读顺序与证据边界

建议先读 π0 的架构与 flow 部分，再读 FAST、π0.5、KI；随后读 RTC 和训练时 RTC；最后按 MEM、π*0.6、RLT、π0.7 的顺序理解系统整合。论文的附录往往比摘要更能回答“具体怎么训练”。

[S1] Physical Intelligence，π0: A Vision-Language-Action Flow Model for General Robot Control。重点：§IV、§V、附录B-D。[论文 PDF](https://www.pi.website/download/pi0.pdf)；[官方介绍](https://www.pi.website/blog/pi0)。

[S2] Kim 等，OpenVLA: An Open-Source Vision-Language-Action Model，2024。用于对照离散动作 VLA 与预训练迁移。[论文](https://arxiv.org/abs/2406.09246)。

[S3] Pertsch 等，FAST: Efficient Action Tokenization for Vision-Language-Action Models，2025。重点：DCT、量化、BPE、训练与推理权衡。[论文全文](https://arxiv.org/html/2501.09747v1)；[官方介绍](https://www.pi.website/research/fast)。

[S4] Physical Intelligence，π0.5: a Vision-Language-Action Model with Open-World Generalization，2025。重点：§IV 的数据混合、两阶段与高低层分解。[论文 PDF](https://www.pi.website/download/pi05.pdf)；[官方介绍](https://www.pi.website/blog/pi05)。

[S5] Driess 等，Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better，2025。重点：§5 的分支、mask 与 stop-gradient。[论文 PDF](https://www.pi.website/download/pi05_KI.pdf)；[官方介绍](https://www.pi.website/research/knowledge_insulation)。

[S6] Black、Galliker、Levine，Real-Time Execution of Action Chunking Flow Policies，2025。重点：§3-4、式5、算法1。[论文全文](https://arxiv.org/html/2506.07339v1)；[官方介绍](https://www.pi.website/research/real_time_chunking)。

[S7] Black 等，Training-Time Action Conditioning for Efficient Real-Time Chunking，2025。重点：§IV、§V-B。本文伪代码使用自洽的噪声→动作时间约定，复现请核对原算法符号。[论文全文](https://arxiv.org/html/2512.05964v1)。

[S8] Physical Intelligence，π*0.6: a VLA that Learns from Experience，2025。重点：§IV-V 的 RECAP、奖励和价值估计，附录F。[论文 PDF](https://www.pi.website/download/pistar06.pdf)；[官方介绍](https://www.pi.website/blog/pistar06)。

[S9] Torne 等，MEM: Multi-Scale Embodied Memory for Vision Language Action Models，2026。重点：§III-A-D、记忆监督和视频编码。[论文 PDF](https://www.pi.website/download/Mem.pdf)；[官方介绍](https://www.pi.website/research/memory)。

[S10] Xu 等，RL Token: Bootstrapping Online RL with Vision-Language-Action Models，2026。重点：§IV-V、actor-critic 目标和关键阶段切换。[论文全文](https://arxiv.org/html/2604.23073v1)；[官方介绍](https://www.pi.website/research/rlt)。

[S11] Physical Intelligence，π0.7: a Steerable Model with Emergent Capabilities，2026。重点：§IV-VII、metadata、目标图像、历史编码与异步运行。[论文 PDF](https://www.pi.website/download/pi07.pdf)；[官方介绍](https://www.pi.website/blog/pi07)。

[S12] Physical Intelligence，openpi 官方代码。本文快照 commit：215abfb217dbac7d5f1273282331b9b1866c0479。[固定版本仓库](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)；[模型实现](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)；[模型配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)；[训练配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)。

[S13] Physical Intelligence，Frontier Models for Every Robot: Announcing our API beta，2026-10-13。仅支持文中的发布状态说明，不提供未披露模型参数。[官方发布](https://www.pi.website/blog/api-beta)。

[S14] Lipman 等，Flow Matching for Generative Modeling，2022/2023。用于概率路径和向量场原理。[论文](https://arxiv.org/abs/2210.02747)。

[S15] Vaswani 等，Attention Is All You Need，2017。用于 scaled dot-product attention、Transformer 基础；现代 VLM 的具体 norm 与 mask 以各模型实现为准。[论文](https://arxiv.org/abs/1706.03762)。

[S16] Hu 等，LoRA: Low-Rank Adaptation of Large Language Models，2021。用于低秩更新机制，不把原论文语言任务结果当成 VLA 保证。[论文](https://arxiv.org/abs/2106.09685)。

[S17] PyTorch 官方文档，Automatic Mixed Precision examples。用于梯度累计、缩放与裁剪等实现原则。[官方文档](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)。

[S18] Zhao 等，Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，2023。用于 ACT、action chunking 与 temporal ensembling 的背景。[论文](https://arxiv.org/abs/2304.13705)。

[S19] Chi 等，Diffusion Policy: Visuomotor Policy Learning via Action Diffusion，2023/2024。用于动作分布、多峰生成和滚动执行背景。[论文](https://arxiv.org/abs/2303.04137)。

证据边界：官方演示与自报实验用于解释方法和其报告的结果；本文未开展独立机器人复现实验，也未将不同论文的成功率拼成统一排行榜。表格中数值算例和全部方法示意图为原创教学内容。对未公开的全量预训练配比、完整生产部署配置或未发布权重，本文不作补猜。
