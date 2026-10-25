# Legged Robot Learning｜足式机器人强化学习

**从四足运动控制到全身控制（WBC），再到 PPO 训练机制诊断。**

[English](README.md) · [简体中文](README_CN.md) · [完整实验结果](RESULTS.md) · [复现指南](docs/REPRODUCIBILITY.md)

这是 [chao142857](https://github.com/chao142857) 独立研究工作的公开整理，涵盖四足运动控制、移动操作、自适应任务采样及 PPO 更新诊断。贯穿这些实验的问题是：**任务设计、采样分布、反馈机制和策略更新，如何影响机器人运动与操作能力的学习及可靠性？**

这里既有能够学会目标行为的训练，也有未通过验收的策略、负结果和用于解释训练现象的分析工具。**这是多个研究项目的集合，不是一个已经统一训练、统一评测的完整算法。**

> **研究快照：2026 年 10 月。** 所有机器人实验结果均来自**仿真**；当前没有真机验证。公开仓库保留精选源码及数值证据，但不声称完整端到端复现，也不声称 SGS 已稳定优于 Uniform 采样。

## 研究路线

**四足 Locomotion → Whole-Body Control → 任务采样与 PPO 优化**

- **四足运动控制：** 研究楼梯、接触行为、机身高度奖励，以及受控策略续训。
- **全身控制：** 在 Go2 + Piper 上训练并评估底盘运动与末端执行器（EE）跟踪的耦合控制。
- **训练机制分析：** 检验自适应采样是否提供有效学习信号，并分析学习失败时梯度、KL、PPO clipping 和 Adam 更新的表现。

这些方向共享研究问题，但使用的**环境、任务定义和评测协议并不相同**。不能把各项目的结果拼成一个排行榜。

## 代表性实验结果

| 研究方向 | 已观察到的结果 | 必须同时说明的限制 |
|---|---|---|
| **Z1 四足 Locomotion** | 10 月 10 日 CP5700 仿真策略完成 **32/32** 次八级、每级 10 cm 的上楼梯测试。 | 另一项 10 月 16 日下楼研究中，原策略及两组续训策略的严格安全通过率均为 **0/48**；小腿接触问题仍然存在。 |
| **Go2 + Piper：原生 WBC** | CP3000 确定性评测 **768/768** 次存活 20 秒，EE 位置 RMSE 为 **8.07 cm**；训练约 **2.949 亿 transitions**。 | 单独设置的精度诊断门槛仅 **0/768** 通过。存活不代表高精度或安全控制。 |
| **Go2 + Piper：Recipe v1** | CP1200 的正式 T1 任务 **192/192** 完成，EE 平均 RMSE **2.42 cm**；训练约 **1.180 亿 transitions**。 | 全回合严格安全仅 **15/192**，小幅右转一致性与动态 X 轴跟踪未达验收要求。**正式 T1 门槛未通过。** |
| **SGS 采样实验** | 固定 48 任务的 Uniform 与 SGS，在训练前均为 **40/48** 联合成功，后续训练检查点均降至 **0/48**。 | 未观察到 SGS 收益；训练反馈中没有正的联合成功样本，不能由此推断 SGS 在所有任务中无效。 |
| **PPO Gradient Lab** | 基于冻结模型和状态，分析 **120** 个独立的单 minibatch 更新候选，检查原生 GAE、KL、clipping 与 Adam。 | 较小学习率在主要窗口中减小了 held-out clipped-surrogate 偏差；这**不是**机器人真实回报提升的证据。 |

**上述成功率不能横向直接比较。** 原生 WBC、Recipe T1、Z1、SGS 和冻结梯度分析的任务、初始化条件或评测定义不同。此处列出的机器人学习实验均只有**一个训练 seed**；多个评测 seed 不等于多次独立训练。

![不同仿真协议下的数值结果，不能作为跨方法排行榜](docs/figures/simulation_results.svg)

具体分母、评测规则、失败条件、原始数值文件链接，见 [**完整结果与证据说明（RESULTS.md）**](RESULTS.md)。

## 项目导航

| 项目 | 开放内容 | 正在回答的问题 |
|---|---|---|
| [**Z1 Locomotion**](projects/z1_locomotion/README.md) | 楼梯课程、接触相关任务/奖励配置、站立采样、高度奖励干预、评测与接触统计 | 为什么能够完成路径穿越，却仍然无法满足严格下楼接触安全条件？ |
| [**Go2 + Piper WBC**](projects/go2_piper_wbc/README.md) | 原生 PPO WBC 基线、任务适配器、T1 scratch v2、Recipe v1、A+ 准备及受约束训练入口 | 任务、奖励、控制与启动阶段约束如何影响移动和 EE 跟踪的联合学习？ |
| [**Success Guided Sampling（SGS）**](projects/sgs/README.md) | 本地 SGS 实现、反馈/reset 生命周期测试、Uniform 对照及因子实验准备 | 在什么样的成功反馈条件下，任务采样才能提供有区分度的学习信号？ |
| [**PPO Gradient Lab**](projects/ppo_gradient_lab/README.md) | 冻结梯度探针、原生 GAE 校准、KL/clipping 分析与 Adam 更新显微镜 | 哪些 PPO 更新现象与训练不稳定相关，哪些只是诊断相关性？ |

**方法来源：** SGS 研究基于 [Zhang 等人的 *A Balanced Data Diet*](https://arxiv.org/abs/2610.12465v1)。本仓库**不宣称发明 SGS 算法**；贡献体现在相应的本地实现、系统集成、实验协议与分析中。

## 快速开始：仅运行 CPU 检查

在仓库根目录执行：

~~~bash
python tools/audit_publication.py
python tools/cpu_smoke.py
python projects/z1_locomotion/run.py train --check
python projects/go2_piper_wbc/run.py native-train --check
~~~

这些命令**不会启动 GPU 训练，也不会运行仿真器**。部分 ML 测试需要额外安装依赖，可用 `python tools/cpu_smoke.py --require-ml` 强制要求相关依赖齐备。公开前的实际验证记录为 **74 项测试通过，1 项涉及私有 checkpoint 的一致性测试跳过**，另有源码与入口检查；这并不代表重新完成了 Isaac GPU 仿真实验。

实际运行 Z1 与 Go2 + Piper，需要**两个不同版本的仿真/训练环境**、独立获取并获准使用的机器人模型；历史续训和 Recipe 实验还需要与原始来源绑定的 checkpoint 或运行证据。训练前请阅读 [复现指南与完整命令](docs/REPRODUCIBILITY.md)。

## 当前限制与待研究问题

- **Z1：** 合作团队提供的 URDF/meshes 及衍生可视化未公开。电机参数仍是暂定仿真设定，下楼严格接触安全问题尚未解决。
- **WBC：** 原生 WBC 与 Recipe 的任务协议不同，**不能**将两者的 EE 误差直接解释为算法优劣。Recipe v1 正式 T1 未通过；被检查的 A+ 快照中**新增完成更新为 0**。
- **SGS：** 历史对照未显示收益；B1/B2/B3 仍处于实验准备阶段，**不是已完成的训练结果**。
- **PPO 诊断：** 原生 advantage 校验、冻结 surrogate 分析并不能单独证明历史训练失败的因果机制，也不构成梯度手术或策略回报提升的证据。
- **复现范围：** 私有 checkpoint、模型资产、原始 rollout 和部分运行凭据未再分发；公开的统计结果不能替代这些原始输入。

当前重点是针对**训练稳定性、任务采样、动态末端跟踪以及严格安全性**建立可比较、可审计的实验，而非宣布一个已经解决的问题。

## 文档与来源

| 文档 | 内容 |
|---|---|
| [RESULTS.md](RESULTS.md) | 各实验指标、分母、协议差异及对应证据 |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | 环境版本、CPU 检查、仿真依赖及运行命令 |
| [VALIDATION.md](VALIDATION.md) | 公开前实际验证了什么、跳过了什么 |
| [SOURCE_MAP.md](SOURCE_MAP.md) | 来源快照与项目演变记录 |
| [PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md) | 公开范围、排除项与 Git 历史隔离审计 |
| [THIRD_PARTY.md](THIRD_PARTY.md) · [NOTICE](NOTICE) · [LICENSES/](LICENSES/) | 第三方归属与许可范围 |

## 致谢与许可

研究基于 [IIT DLS Lab basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab)、[LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab)、[Isaac Lab](https://github.com/isaac-sim/IsaacLab)、[RSL-RL](https://github.com/leggedrobotics/rsl_rl) 等开源研究与软件。合作团队提供了 Z1 机器人模型，**模型文件未随本仓库公开**。此处不表示上述机构或作者对本项目的认可或背书。

原创部分采用 **Apache-2.0** 许可；第三方材料仍受各自声明和[许可证](THIRD_PARTY.md)约束。源码许可不包含获取或再分发被排除的模型资产、checkpoint 与策略权重的权利。
