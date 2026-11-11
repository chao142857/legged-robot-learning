# Teacher Adam 回溯 BUG：修复与历史影响审计

修复提交：`58a1e1a972f6e2404ff2d696e054d1eaed811a2b`；独立分支：`fix/teacher-adam-audit-20261109`。审计日期：2026-11-09。

## 结论

已确认真实实现错误，不是 step 日志显示错误：V0.2 在 CPU 上将同一 Adam state_dict 快照反复传给 load_state_dict；加载后的 step、exp_avg、exp_avg_sq 张量与快照共享存储。试探 optimizer.step() 会改写用于下次回溯的“旧状态”。网络参数回到旧值，但 Adam 历史未回到旧值。全拒绝时也可能留下被改写的 moments。

从每个历史决策的原始状态，逐点重建 24 次原更新，网络、优化器及完整 Teacher 状态均与已存 checkpoint 精确一致；相同历史输入与真实收益改用修复版时，15 次产生不同权重。**Meta8、Meta16A、Meta16B 均受影响。** 这证明执行偏离预期算法，但不证明该 BUG 导致任何特定成功率升降。

修复已通过 15 项独立 CPU 测试及真实 GPU 闭环 smoke。未启动 TR0 12k，未修改 TV=2%，未开展新的 Meta-Training。YangLab 尚在等待的 Meta24/16 后续队列已暂停，既有冻结 Meta8 学生训练未被中断。

## 1. 原因、修复与保证

原写法先 deepcopy 一次 optimizer.state_dict，但每次 load_state_dict(old_optimizer) 会重新引入共享。正确做法是在**每一次加载**时使用 deepcopy(old_optimizer)，包括拒绝回滚和普通 checkpoint 恢复。TR0 的 isolated_adam_loads 已实现这一隔离；本轮将其落实到独立 canonical Teacher 实现，不再依赖调用方记得套上下文。

本轮另加异常事务回滚：梯度非有限或 step 后发生异常，恢复网络、optimizer、encoder、私有 RNG、版本集合和原梯度。正常拒绝/零信号保留原参数、Adam、param_groups、encoder、RNG 与原梯度；逻辑版本仍消费一次，防止相同决策 replay。这是记账语义，不是真实 optimizer step。

多次试探在计算上会调用 optimizer.step 多次，但每次都从完全相同旧状态出发；最终接受的状态严格等价于只执行一次、使用被接受学习率的 Adam 更新。接受后恢复预定基础学习率，未改梯度、奖励、探索 σ、网络、KL 门槛或 Student PPO。

`build_fixed_runtime.py` 在全新目录构建修复运行时，将 teacher_v02.py 指向修复实现。实测 teacher_ops、prepare_continual、prepare_matched 导入同一个修复 Teacher。V0/V0.1 基类也加强 checkpoint/回滚快照隔离；原运行目录与正在运行的进程没有热替换。未来正式使用必须显式选该新 runtime。

## 2. Meta8 的 8 次更新为何得到 15 步

| 逻辑更新完成数 | 回溯候选数 | 旧 Adam step（前→后） | 同一历史起点修复后 step | 原/修复局部 q 的 TV |
|---:|---:|---|---|---:|
| 1 | 2 | [0] → [1] | [1] | 0.00000% |
| 2 | 2 | [1] → [3] | [2] | 0.25915% |
| 3 | 2 | [3] → [5] | [4] | 0.27906% |
| 4 | 2 | [5] → [7] | [6] | 0.17104% |
| 5 | 2 | [7] → [9] | [8] | 0.28765% |
| 6 | 2 | [9] → [11] | [10] | 0.23249% |
| 7 | 2 | [11] → [13] | [12] | 0.14548% |
| 8 | 2 | [13] → [15] | [14] | 0.34295% |

第一次更新前 Adam state 为空，每次 load 会回到空字典，因此候选虽有两次，最终仍为 step=1，原/修复**完整状态相同**。从第二次起存在已分配状态张量，七次更新各错误累计两步，得到 1+7×2=15。首次污染 checkpoint 为 point_01（逻辑更新2）。

局部 TV 是在各点真实固定命令/反馈/global_state 上、保留20% Uniform 混合后计算；它不是从干净 Teacher 重跑全部训练后的累计差异，也不是性能损失。

## 3. 两条 Meta16 谱系

| Teacher | 来源 | 逻辑数 | 接受的非零更新数（累计） | 实际 Adam step | 额外累计 step |
|---|---|---:|---:|---:|---:|
| Meta8 | seed43/44，各1k/3k/6k/9k | 8 | 8 | 15 | 7 |
| Meta16A | Meta8 + seed46八阶段 | 16 | 15 | 30 | 15 |
| Meta16B | Meta8 + seed43/44八阶段 | 16 | 16 | 24 | 8 |

Meta16A 第一条新反馈为零对比，消耗逻辑版本但不 step；其后7次分别有2/2/2/2/3/2/2个候选，实际增加15步，而应只接受7步。Meta16B 第一次续学有两个候选，增加2步；后7次无回溯，各增加1步。这7次局部新旧一致，但它们继承了已污染的网络和 moments，不能认定为干净 Teacher。

| 谱系 / 新点 | 旧 step 前→后 | 候选数 | 局部权重 L2 差 | 局部 q TV |
|---|---|---:|---:|---:|
| Meta16A/point_00 | [15] → [15] | 0 | 0.00000000 | 0.00000% |
| Meta16A/point_01 | [15] → [17] | 2 | 0.01021220 | 0.17759% |
| Meta16A/point_02 | [17] → [19] | 2 | 0.00786941 | 0.14045% |
| Meta16A/point_03 | [19] → [21] | 2 | 0.01253159 | 0.31662% |
| Meta16A/point_04 | [21] → [23] | 2 | 0.00942678 | 0.25835% |
| Meta16A/point_05 | [23] → [26] | 3 | 0.01014240 | 0.26737% |
| Meta16A/point_06 | [26] → [28] | 2 | 0.03709088 | 0.73553% |
| Meta16A/point_07 | [28] → [30] | 2 | 0.01127994 | 0.25775% |
| Meta16B/point_00 | [15] → [17] | 2 | 0.01608269 | 0.17363% |
| Meta16B/point_01 | [17] → [18] | 1 | 0.00000000 | 0.00000% |
| Meta16B/point_02 | [18] → [19] | 1 | 0.00000000 | 0.00000% |
| Meta16B/point_03 | [19] → [20] | 1 | 0.00000000 | 0.00000% |
| Meta16B/point_04 | [20] → [21] | 1 | 0.00000000 | 0.00000% |
| Meta16B/point_05 | [21] → [22] | 1 | 0.00000000 | 0.00000% |
| Meta16B/point_06 | [22] → [23] | 1 | 0.00000000 | 0.00000% |
| Meta16B/point_07 | [23] → [24] | 1 | 0.00000000 | 0.00000% |

## 4. 哪些资产能复用

| 资产 | 分类与使用边界 |
|---|---|
| Fresh Teacher、Meta8 point_00 | 对本 BUG 干净；point_00 已验证新旧完整状态相同，可作正确算法的最近干净起点 |
| Meta8 point_01–07、Meta8最终、Meta16A/B全部续学产物 | 确认受影响；保留作原实现的冻结复现/历史基线，不可称为修复算法产物 |
| TR0 修复后的 Original/Sampling-TR | 本次局部更新正确（15→16），但继承污染 Meta8；不是干净 Meta9 |
| TR0 早期 Adam alias 中间结果 | 已归档为无效中间候选，不参与最终结论 |
| Uniform / Fresh 学生、原生与 E1–E3 基线 | 不经该 V0.2 Teacher Adam 路径；本 BUG 不直接影响这些训练与评测 |
| Meta 驱动的冻结学生及其曲线 | 测得的结果仍有效，但采样 Teacher 来自旧实现；seed47/48/49相关比较应标注，不能冒充修复版 Meta 的证据 |
| 训练池、开发池、正式测试池、学生起点 | 无本 BUG 的数据写入；仍可用于配对实验，须保留原哈希 |
| 旧真实分叉数据 | 可做逐点审计与历史分析；策略变更后不可直接当新 Teacher 的 on-policy 收益重复训练 |

三个最终资产的 SHA256：

```text
Meta8    8929d0a13c4aaa7b98c70acf584d04390711bbc9ea6df8b61f3bcb556cdbd3b9
Meta16A  803eb25da1fb3ed31c4a03c5b103aa4f24d57dd5c4f73f7f8d3645d507ec5968
Meta16B  a2e81e5a672628ad9f5463c61b6318a0f6bde6c80b714ab252a3ad0bd8f640e6
```

最近干净 Teacher：`/data/nsgs_teacher_experience_v02/meta-eight-starts/point_00/teacher_after.pt`，SHA256 `c6f37915da62e820f1f555eceba5816a5e1649f3a8b04c3fb4deafcbcd9c2eea`。它不是 Meta8，而是仅完成一次正确更新的状态。完整逐 checkpoint 清单见证据 JSON；没有修改任何历史 step、moment 或权重。

## 5. V0/V0.1、续学与恢复审计

V0/V0.1 采用单次 step 后拒绝，快照在 step 前单独 deepcopy，未在 step 前反复 load 同一快照，因此不存在同一多次回溯累积机制。两者普通 load(state) 仍有调用方内存快照共享的 API 风险，已在新运行时隔离；不能把此潜在风险等同于已证实的历史训练污染。

检查 Ubuntu 可获得的 64 个 V0/V0.1 回合记录：57个实际更新均 step+1；7个显式 teacher_frozen 验证的完整状态不变。未发现错误 step 转换，未含实际拒绝案例；拒绝行为由独立测试补充。这一结论限于已检查的保存记录，不宣称扫描了所有未留存运行。

V0.2 的更新、Meta8、16A、16B 使用的 teacher_v02.py 源哈希均为 `c83b3943dd4fb509094a94eaa404ddf4a715635394f85c9009771fbad78a794c`。续学入口复制整个 Teacher/optimizer，因而忠实继承污染；普通恢复未清洗历史，也不应清洗。私有 RNG 与特征编码器沿原状态恢复。源文件清单与两机 canonical checkpoint 清单均已保存。

## 6. 验收、成本与限制

15项 CPU 测试涵盖：无/已有 Adam 状态、全部拒绝、多个连续回溯、独立 Adam 方程 oracle、exp_avg/exp_avg_sq/step/param_groups、AMSGrad 与多参数组、无回溯新旧一致、零信号、非有限梯度、step后异常回滚、输入快照不共享、序列化后下一决策和更新一致、污染 step 不伪修，以及 V0/V0.1 的拒绝/恢复。

GPU：RTX 5060 Laptop；三臂 Uniform/q+/q−各2次学生更新，共 **6次PPO、589,824条训练转移**；3次开发评测共1,536,000条额外转移。端到端 smoke **278.70秒（4.65分钟）**，包含启动/训练/评测/保存，不是积分 GPU 利用率。Teacher 在CPU计算，Isaac Lab和Student在GPU执行。

真实开发连续得分（before/Uniform/q+/q−）：`[0.1265941969951834, 0.12481403154077463, 0.12807163499420193, 0.12772796047094911]`。Teacher梯度范数 `0.0021648414`，参数 L2 变化 `0.0888114074`。全步 KL=.080754超出原.05门槛，半步 KL=.019267接受，Adam仅 **15→16**；保存恢复完整状态精确一致，源checkpoint哈希未变。

这是测试闭环是否执行正确，不能把两次 PPO 的微小收益差当算法优劣或因果效果。diagnostic_state.pt仅作审计，没有注册/输出新 Meta；不用于未来正式实验。TV门槛没有参与本 smoke，也没有调参。

## 7. 是否必须重做 Meta-Training

**若结论要针对正确回溯算法，必须从干净起点重新进行真实分叉。** 可从 Fresh重跑，或从已验证干净的point_00接续，保留其历史RNG与一次更新状态。不能从Meta8改step=8或清空moments来恢复：网络权重已经改变，后续采样与真实收益也随之改变。即使只清空optimizer，也是另一个新实验。

point_01的旧决策发生在干净point_00之后，仍可作同起点诊断；本轮没有把它导出为新Teacher。更新一旦改正，point_02及之后的旧q、学生训练结果就不再对应新Teacher，必须重新收集真实分叉。不能把24次独立局部重建当作修复后的连续Meta-Training。

本轮不启动上述重训，也不根据本审计调整学习率、KL或TV。旧结果保留；建议后续将“正确实现复现”与新信赖域机制分开，先恢复可信基线。

## 凭据与保存位置

- Ubuntu：`/home/alex/dog/teacher_adam_audit_20261109/`；smoke完整物理/训练/开发日志、学生checkpoint、诊断Teacher与独立fixed_runtime均保留。
- YangLab：`/data/teacher_adam_audit_20261109/`；历史审计只读输入，单独CPU容器、未占GPU。
- 队列暂停凭据：`/data/go2_piper_nsgs_v02_seed49_meta_yanglab_20261109/ADAM_AUDIT_HOLD.json`；原队列文件、checkpoint和结果保留。
- 本目录evidence只含轻量摘要、哈希、测试与smoke收据；Git不含checkpoint、完整原始反馈、凭据或私有密钥。
- 最终重新核对 Ubuntu 129 个、YangLab 16 个历史文件路径的 SHA256，全部一致；Ubuntu GPU 无剩余计算进程，暂停的 YangLab 新 Meta 队列无残留控制进程。
- 历史结果未覆盖、未删除。尚未验证“修复后性能会变好还是变差”，也未将旧成功率提升全部归因于该BUG。
