# Teacher Adam 回溯修复与历史审计

本目录为独立修复版；历史 V0–V0.2、E1–E3、TR0 源文件与 checkpoint 均不改写。
不启动 TR0 12k，不调整 TV 半径，不注册新的 Meta Teacher。

- `code/teacher_fixed.py`：V0.2 Teacher 完整 API；每次 Adam 快照加载先 deepcopy，普通 checkpoint 加载同样隔离。更新异常回滚全部 Teacher 状态和梯度；拒绝或零信号恢复原梯度。
- `code/build_fixed_runtime.py`：从已验证原运行时构建一个**新目录**，将 canonical `teacher_v02.py` 替换为修复版；teacher_ops、续学、恢复入口无需改变其调用协议。V0/V0.1 基类的 optimizer 加载也隔离。不会启动任何队列。
- `code/legacy_load_guard.py`：V0/V0.1 的显式 opt-in 包装；不修改进程全局 PyTorch 或运行中的任务。
- `code/audit_history.py`：从每个历史决策的真实起点，分别诊断原实现与修复实现的一次更新。不会把旧反馈串成修复后的 on-policy 训练，不输出可续学的新 Teacher。
- `code/audit_legacy.py`：区分真实更新与显式冻结，审计 V0/V0.1 每次 Adam 状态转换。
- `code/gpu_smoke.py`：Ubuntu 现有环境，三臂各 2 次 Student PPO；同源恢复、真实开发集反馈、一次诊断 Teacher 更新和 checkpoint round-trip。生成 `diagnostic_state.pt`，没有可误用于正式 transfer 的顶层 `teacher` 键。

所有拒绝/零信号决策仍消费逻辑版本，防止 replay；因此逻辑版本数不一定等于实际接受的 Adam step。已有污染 step 不会自动归零或篡改。加载旧 Meta 能用于冻结复现，但不能因此宣称其历史已经修复。

运行（依赖既有环境与原代码，不新装环境）：

```bash
PYTHONPATH=/path/to/audit/code:/path/to/native_experiment/code python -m unittest discover -s /path/to/audit/tests -v
python code/build_fixed_runtime.py /path/to/native_experiment/code /path/to/NEW_runtime/code
```

未来正式启动前，必须将新实验指向该独立 runtime，并核验代码 manifest；不能继续从历史目录启动原 teacher_v02。干净 Meta-Training 是否重跑由用户决定，本轮不启动。
