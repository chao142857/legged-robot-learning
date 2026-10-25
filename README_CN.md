# Legged Robot Learning｜足式机器人强化学习

**面向四足运动控制与全身控制的强化学习研究**

[![English](https://img.shields.io/badge/English-64748b?style=for-the-badge)](README.md) [![简体中文](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-2563eb?style=for-the-badge)](README_CN.md)

这里整理了我在机器人学习方向的研究项目，从四足运动控制、全身控制到强化学习训练机制分析。我主要关注**机器人如何学会协调运动和操作**，以及奖励设计、任务采样与 PPO 更新如何影响最终表现。

## 研究方向

- **四足运动控制（Legged Locomotion）：** 楼梯和复杂地形运动、接触行为、奖励函数及课程学习。
- **全身控制（Whole-Body Control）：** 使用学习策略联合控制四足底盘和机械臂，实现移动与操作的协同。
- **强化学习方法：** 任务采样、训练稳定性、梯度分析和 PPO 优化过程。

## 研究项目

| 项目 | 完成的工作 | 部分实验进展 |
|---|---|---|
| [**Z1 Locomotion**](projects/z1_locomotion/README.md) | 四足训练任务、楼梯课程、接触/机身高度奖励研究及评测工具 | CP5700 策略在仿真中完成 **32/32** 次八级、每级 10 cm 的上楼梯测试；进一步研究了下楼接触问题 |
| [**Go2 + Piper WBC**](projects/go2_piper_wbc/README.md) | 原生 WBC 基线、腿臂联合任务、T1 从零训练和奖励/控制策略实验 | 原生 CP3000：确定性评测 **768/768** 次存活 20 秒；Recipe v1 T1：**192/192** 次任务完成，EE 平均位置 RMSE **2.42 cm** |
| [**Success Guided Sampling**](projects/sgs/README.md) | SGS 本地实现、回合反馈管理、Uniform 对照及因子实验准备 | 完成固定任务 SGS/Uniform 对照实验，正在研究什么条件下成功反馈能有效指导采样 |
| [**PPO Gradient Lab**](projects/ppo_gradient_lab/README.md) | 原生 GAE、KL、clipping、Adam 更新与冻结策略梯度诊断工具 | 分析了 **120** 个独立单 minibatch 更新候选，探索 PPO 参数更新的实际行为 |

*以上机器人结果均来自仿真；任务完成率与存活率不等于严格安全验收通过。详细指标、未成功的实验和评测协议见 [RESULTS.md](RESULTS.md)。*

## 当前研究重点

我目前重点研究**动态末端执行器跟踪、移动操作耦合任务的稳定训练，以及不同任务采样策略的严格对照**。希望弄清楚哪些训练设计真正有效，以及某些看似合理的方法为何在实验中失效。

## 代码与实验

四个项目目录分别包含精选算法实现、实验配置、评测工具和数值结果。可以从各项目 README 开始阅读，也可以运行以下不启动仿真器的 CPU 检查：

~~~bash
python tools/audit_publication.py
python tools/cpu_smoke.py
~~~

完整仿真需要另外获取对应版本的框架和获得授权的机器人模型资产。

**更多资料：** [实验结果](RESULTS.md) · [复现指南](docs/REPRODUCIBILITY.md) · [验证报告](VALIDATION.md) · [来源记录](SOURCE_MAP.md)

## 致谢

本项目基于 [IIT DLS Lab basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab)、[LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab)、[Isaac Lab](https://github.com/isaac-sim/IsaacLab) 和 [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 等开源工作。SGS 方法参考 [*A Balanced Data Diet*](https://arxiv.org/abs/2610.12465v1)。合作团队提供了 Z1 机器人模型，相关模型资产未收入公开仓库。

原创代码采用 **Apache-2.0** 许可，第三方代码及许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
