# nSGS V0.2 — 反事实教师学习

本目录归档 Go2-Piper WBC 的 nSGS V0.2 已执行源码、工程测试和轻量进展证据。八个起点的三臂实验与八次 Teacher 更新已经完成；新 Student seed 47 的三组 10k 迁移对照仍在运行。结果和采集时间见 [PROGRESS.md](PROGRESS.md)。

这是探索性仿真实验。目前未证明预训练 Teacher 优于 Uniform。此目录与早期固定48任务 SGS、E1宽松标签和E3聚类协议分别绑定，不能混用。

## 实验设计与实际配置

| 项目 | 实际配置 |
|---|---|
| Student | 原生 GO2-PIPER-WBC PPO，4096 env，rollout 24，18维腿臂动作 |
| 原生代码 | LeggedManip_Lab `4e12109d1eccb1d9da2e4a35f189fabea3f13677` |
| Isaac Lab | `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` |
| 执行设备 | YangLab，NVIDIA GeForce RTX 3060 12 GB |
| 池 | 32,768训练序列；256条开发序列×2固定重置；独立1,024条正式测试序列 |
| 元训练起点 | seed 43、44各取CP1000/3000/6000/9000，按阶段再按seed顺序；1k来自prefix，其余来自B |
| 分叉 | 同一可恢复状态分别运行Uniform、q+、q−，每条100 PPO updates |
| Teacher | 全局状态10→32 tanh→8维偏好；Teacher seed1048；固定投影seed30261106 |
| 探索和概率 | 对称Gaussian探索σ=0.5；20% Uniform mixing；有界logit=4 |
| Teacher更新 | Adam LR=0.01，梯度裁剪1，Gaussian偏好KL上限0.05，最多8次减半回退 |
| 迁移对照 | 新Student seed47；Uniform→Fresh Teacher→预训练Teacher，分别从零10,000 updates |
| 迁移Teacher | 两种Teacher参数均冻结、不加探索噪声；每100更新根据Student状态重新计算采样分布 |
| 评测 | 每100更新开发评测；每1000更新正式测试；均值动作 |

所有奖励、观测、动作、终止规则、域随机化、网络和PPO参数保持该原生WBC实验的绑定配置。完整运行协议在 [PROTOCOL.json](evidence/PROTOCOL.json)，起点路径/哈希在 [sources.json](sources.json)。没有加入WM、SuperMimic或动态聚类。

**实际配置入口是 `teacher_v02.py::V02_CONFIG` 与控制器协议。** `code/protocol.py::CONFIG` 保留V0的兼容定义，其中seed45和旧Teacher参数不是V0.2的生效配置。当前执行源码及33个文件的哈希见 [source_manifest.json](evidence/source_manifest.json)。

## 教师如何学习

每条序列的命令特征与真实历史反馈经固定投影构成8维特征。动态输入包含控制误差、严格成功率、联合达标时间占比、学习进展、反馈年龄和证据量；未观测项保留显式mask。全局Student状态条件网络输出偏好均值。序列logit来自特征与偏好的点积，经过有界变换后softmax，并混入20% Uniform。

在同一Student起点，取 `z± = μ(s) ± σ ε`，分别生成q±并执行100次真实PPO更新。开发收益是训练后减训练前的连续联合得分；Uniform的同起点收益作为基线。Teacher损失为两个已采样Gaussian偏好log-prob与脱离计算图的后续收益的加权和。对均值的上升梯度为 `(R+ − R−) ε / (2σ)`。不会通过PhysX或PPO链条反向传播，也不会用旧决策进行无修正的on-policy重放。

每次保存Teacher版本、偏好、概率、实际序列抽样、物理曝光、开发结果及参数更新。已在分支开始前存在的回合是共同的条件状态，单独记录；不能把这些曝光误算成新分布抽样。正式测试结果不进入Teacher输入、奖励或参数选择。

## 指标定义

严格成功需要：无原生失败终止、至少100个有效计分控制步、四项误差同时达标的时间占比≥80%，且四项有效窗口RMSE均不超过各自阈值。

| 误差 | 物理阈值 |
|---|---:|
| 底盘XY速度向量误差 | 0.20 m/s |
| 偏航角速度误差 | 0.30 rad/s |
| EE位置向量误差 | 0.05 m |
| EE姿态旋转距离 | π/12 rad（15°） |

EE和速度命令均距各自最近切换至少2秒才计分；重采样重新计时，终止步在reset前读取。WBC位置使用原生link0坐标的XY与世界坐标Z。超时本身不算成功。

Teacher主目标是每次重置中**四项同时达标的有效时间占比**，随后对全部512次重置平均；原生失败或不足100计分步记0。它不是“最长连续达标秒数”，后者另作诊断。开发集用于教师学习，其结果不等同于独立泛化成绩。

## 文件与验证

- [code/](code/)：已执行的Teacher、序列适配、反馈、分叉恢复、评测和报告源码。
- [tests/](tests/)：36项CPU检查；[测试记录](evidence/jobs/cpu-tests-r03/console.txt)与[退出收据](evidence/jobs/cpu-tests-r03/result.json)。
- [controller.py](controller.py)、[host.py](host.py)：持久队列与隔离容器执行。
- [evidence/](evidence/)：聚合结果、实际配置、源码和资产哈希、smoke验收及修复保全收据。
- [PACKAGE_MANIFEST.json](PACKAGE_MANIFEST.json)：本次归档文件大小与SHA256；[verify_archive.py](verify_archive.py)作标准库完整性核对。

CPU测试需要已有NumPy和PyTorch环境，不启动仿真：

```bash
python -m unittest discover -s projects/sgs/nsgs_v02_20261106/tests -v
python projects/sgs/nsgs_v02_20261106/verify_archive.py
```

本次公开包的运行源码与远端已通过36项测试的文件逐一哈希相等；归档阶段只复核语法、文件、引用和证据一致性，未重复启动GPU实验。

## 外部资产与恢复边界

这是实验源码归档，尚非仅clone即可运行的自包含工程。真实执行依赖既有不可变Docker镜像、Isaac Lab、LeggedManip_Lab、USD资产，以及YangLab上的以下文件：

- `/data/go2_piper_sgs_e2_20261104/code/docker_job.py` 和E2训练/测试池。
- `/data/go2_piper_nsgs_v0_20261106/prepared/` 下的命令特征、归一化和固定开发池，以及绑定的原生配置参照。
- `sources.json` 列出的八个Student checkpoint、历史反馈与更新日志。
- 镜像 `sha256:8ee9b2bb8e6aae08dace71ee1e2141a4d62bcc63e4cf0eac28b347111a486a87`。

运行根目录是 `/data/go2_piper_nsgs_v02_20261106/`，完整经验库在 `/data/nsgs_teacher_experience_v02/`，tmux会话为 `go2-nsgs-v02`。大checkpoint、完整概率/抽样张量、原始物理日志和视频保留在执行主机；公开摘要不能替代这些原始证据。

CP0明确是初始化记录，正更新checkpoint才包含环境可恢复状态。`--transfer-only`是三组均未训练时的专用启动恢复入口，不能用于任意中断续训。显式环境、管理器、优化器与随机数状态可恢复；PhysX隐藏solver缓存不能保证逐位恢复。初始两起点的已执行重复测试差异为0，也不能推及所有未来运行。

## 来源与限制

本实验沿用仓库已有的原生WBC适配和SGS反馈基础；`vendor_sampler.py`保留早期SGS实现的文献注释。V0.2实际采样分布由神经偏好生成，不使用固定SGS甜点或硬聚类。上游软件许可及资产许可仍分别适用，参考仓库根目录的来源和许可说明。

一个Teacher、两个已有Student种子的八个起点、每起点一个对称扰动对，以及一个新迁移Student种子，只支持可行性观察。偏好变化、正梯度与工程通过都不构成算法优越性证据；正式迁移对照完成前不宣称优于Uniform。
