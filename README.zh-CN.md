# 腿式机器人学习

[<kbd>English</kbd>](README.md) · **简体中文**

**面向腿式运动与全身控制的强化学习研究**

[chao142857](https://github.com/chao142857) 的独立研究项目，涵盖四足运动、移动操作、自适应任务采样和 PPO 更新诊断。仓库收录研究实现、实验设计与分析，并标明上游来源和各项结果的适用范围。

## 当前研究 nSGS

nSGS 是用于 Go2–Piper 全身控制的状态条件神经任务采样器。在四个 Student 种子的初步仿真实验中，训练至 10,000 次 PPO 更新时，修复后的 CleanMeta8 Teacher 平均严格联合成功率为 **33.50%**，略高于 Uniform 采样的 **31.88%**。Fresh Teacher 的平均值为 **33.30%**。

![Uniform、Fresh Teacher 和修复后的 CleanMeta8 在四个 Student 种子上的平均严格联合成功率曲线](https://raw.githubusercontent.com/chao142857/chao142857.github.io/main/images/projects/nsgs-four-seed-strict-success.png)

*在 1,024 条固定测试序列上按 5 cm / 80% 判据评估严格联合成功率，对 Student 种子 42、47、48、49 等权平均。阴影表示种子间的 ±1 个样本标准差。*

我们正在探索如何从多个 Student 的学习轨迹中训练共享 Teacher。后续实验将研究自适应采样、跨 Student 泛化，以及采样与学习动态和性能退化之间的关系。

[早期 CleanMeta8 复现实验](projects/sgs/cleanmeta8_20261109/README.md) · [11 月 6 日 V0.2 历史快照](https://github.com/chao142857/legged-robot-learning/tree/da583b33b10351a8e5c9f19e61b75ba929ea026d/projects/sgs/nsgs_v02_20261106)

## 研究概览

研究关注任务分布、奖励设计、重置方式和策略更新如何影响机器人控制。仓库记录训练结果、基线表现和受控干预，也提供分析 PPO 更新何时导致策略退化的工具。

各项目沿用各自的框架版本和实验协议。固定任务的暖启动试验、原生任务分布下的从零训练基线和带约束的训练方案分别评估。

## 项目

| 项目 | 研究内容 | 当前结果 |
|---|---|---|
| [nSGS](#当前研究-nsgs) | 全身控制中的状态条件神经任务采样 | 修复后的 CleanMeta8 在四个 Student 种子上训练至 10k 更新时，平均严格联合成功率为 33.50% |
| [Z1 腿式运动](projects/z1_locomotion/README.md) | 训练配置、楼梯课程、接触奖励、站立采样、高度奖励 A/B 实验及评测 | 已完成仿真通行与接触诊断，严格下楼安全要求仍待解决 |
| [Go2 + Piper 全身控制](projects/go2_piper_wbc/README.md) | 原生 WBC 基线、固定任务集、T1 从零训练，以及 Recipe v1 和 A+ 的准备工作 | 原生方案已展示学习效果，Recipe v1 的正式 T1 评测仍未通过 |
| [SGS 研究](projects/sgs/README.md) | 成功引导采样和早期神经采样器实验 | 历史采样实验及评测协议 |
| [PPO 梯度诊断](projects/ppo_gradient_lab/README.md) | 冻结策略梯度、分量价值探针、原生 GAE、KL、裁剪、Adam 和留出集代理目标诊断 | 已完成 CPU 诊断，控制性能的改善仍待验证 |

## 方法

- **腿式运动与全身控制**使用原生 RSL-RL 和 Isaac 训练栈中的 PPO，并通过独立的任务适配器检查重置、动作、接触和观测。
- **受控干预**在实验协议允许时匹配初始状态和训练预算，用于比较奖励高度、命令采样和原生任务分布。记录任务、重置、动作维度及控制器先验的差异。
- **自适应采样**包含基于 [Zhang 等人的 *A Balanced Data Diet*](https://arxiv.org/abs/2610.12465v1) 实现的 SGS，记录回合与序列的归属，处理截尾反馈，隔离随机数和评测。未访问任务采用 0.5 的成功率先验。
- **更新分析**校准原生总优势函数，隔离 PPO 和 Adam 更新，并在冻结的留出集上测量代理目标。分量探针用于近似诊断。

## 实验结果

| 实验 | 主要观测 | 解读 |
|---|---|---|
| Z1 10 月 16 日高度与接触实验 | 原策略和两组续训的严格下楼安全均为 **0/48**。小腿接触占比 A 为 10.17%，B1 为 6.88% | 小腿接触减少，早期足部负载增加。下楼安全仍待改善 |
| Go2 + Piper 原生 WBC CP3000 | 确定性策略存活 **768/768**，XY 速度 RMSE **0.1189 m/s**，末端位置 RMSE **8.07 cm** | 策略在仿真中学会了移动和位姿跟踪，跟踪精度与安全性仍有改进空间 |
| Recipe v1 CP1200 正式 T1 | 完成 **192/192**，末端 RMSE **2.42 cm**，全回合严格安全 **15/192** | 任务分布与原生 WBC 不同，转弯、起步超速和动态 X 跟踪仍是限制 |
| SGS 固定 48 任务先导实验 | 两组初始联合成功均为 **40/48**，更新 32、64 和 128 次后均为 **0/48** | 成功反馈缺乏区分度，限制了该对照的解释 |
| PPO 更新诊断 | 120 个独立候选中，较低学习率降低了主要分析窗口内的留出集裁剪代理目标偏差 | 结果反映代理目标的变化，控制收益和历史训练退化的原因仍待验证 |

原生 WBC 的误差统计使用首个回合中第一秒后的有效样本。Recipe 的结果在三个评测种子上等权平均。此表中的早期研究各使用一个训练种子和各自的评测协议。完整定义及结果链接见 [RESULTS.md](RESULTS.md)。

## 复现

可先阅读[运行环境与入口指南](docs/REPRODUCIBILITY.md)。CPU 检查使用合成输入和已发布的数值摘要。

```bash
python tools/audit_publication.py
python tools/cpu_smoke.py
```

`cpu_smoke.py --require-ml` 需要可选的 Torch 和 RSL 依赖。脚本会报告缺少的依赖项以及真实 checkpoint 一致性检查的状态。两个机器人项目都提供可在无仿真器环境下运行的 `--check` 入口。

完整仿真复现需要自行获取上游软件、兼容的 Linux/GPU 环境和获准使用的机器人资产。公开仓库提供代码与数值摘要。历史续训实验还需要对应的原始输入、策略权重和回合数据。

[源码索引](SOURCE_MAP.md)、[发布审计](PUBLICATION_AUDIT.md)、[文件来源记录](provenance/files.json)和[验证报告](VALIDATION.md)标注了历史结果与本次发布检查的来源。公开仓库使用独立的 Git 历史。

## 实验范围

- 当前控制结果来自仿真。硬件验证、仿真到现实迁移和连续时间安全性仍待检验。
- Z1 执行器参数是临时仿真设置，模型资产及可视化由合作团队提供。
- 原生 WBC 基线和 Recipe v1 的任务域与安全统计口径不同。
- Recipe A+ 在记录的快照中完成了 **0 次额外更新**，续训处于规划阶段。
- CPU 检查覆盖导入、数值约定和运行入口。CleanMeta8 的 GPU 训练在记录的环境中完成。
- 公开数值摘要支持查阅和比较。端到端复现需要原始轨迹、权重和模型几何。

## 致谢

实验基于 [IIT DLS Lab 的 basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab)、[LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab)、[Isaac Lab](https://github.com/isaac-sim/IsaacLab) 和 [RSL-RL](https://github.com/leggedrobotics/rsl_rl)。SGS 归功于原论文作者。Z1 机器人模型由合作团队提供。

原创贡献采用 Apache-2.0 许可。上游声明见 [NOTICE](NOTICE)、[LICENSES](LICENSES/) 和 [THIRD_PARTY.md](THIRD_PARTY.md)。模型和策略资产遵循各自的授权条款。
