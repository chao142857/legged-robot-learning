> Revision 02：用户取消重复基线训练。后续仅新增 Ubuntu seed48 CleanMeta8 10k，以及 YangLab seed49 CleanMeta8 10k；Uniform/Fresh/旧Meta不重跑、不自动补跑。下文初始双学生计划作为历史方案保留，以 revision02/PROTOCOL.json 为准。

# CleanMeta8 — 原 V0.2 正确实现复现

独立根目录：`/home/alex/dog/go2_piper_cleanmeta8_20261109`（ubuntu-robot）。大输出保存在 `/media/alex/Sharedata/go2_piper_cleanmeta8_20261109`。tmux：`cleanmeta8-reproduction`。

从原 Fresh 完整状态开始，依次使用 seed43/44 × 1k/3k/6k/9k 八起点；每点重新运行 Uniform/q+/q− 各100 PPO updates，合计2400。4096 env、rollout24，原V0.2全部超参数保持不变，使用已验证 fixed runtime。不使用 TV 上限，不续学 Meta24，不将旧反馈重放为新 Teacher 收益。

每点保存 `decision.pt`、真实分叉 checkpoint/日志/开发评测、`adam_full_trace.pt`、`adam_acceptance.json`。Trace包含更新前后完整Teacher状态、每次候选前后network和Adam、梯度、接受/拒绝、学习率、原KL、实际导入源码路径和SHA。观察包装不改变算法，配有更新结果等价测试。每个被接受更新只累积一次Adam状态；逻辑版本可因零信号或拒绝而多于真实Adam步数。

完成后保存 `teachers/CleanMeta8.pt` 并自动核验，再串行运行 seed48 的 OldMeta8 与 CleanMeta8 两个冻结Teacher学生，各从零10k。相同本机原生PPO、初始化和独立评测；每1000更新正式测试、每100更新原有开发反馈，不训练冻结Teacher。核对已有本机Uniform配置、初始化、池、评测身份和checkpoint；匹配则复用，不匹配自动补跑同预算Uniform。

旧Meta8/16与学生结果不改写；新 `LEGACY_LABELS.json` 记录旧实现来源。待执行Meta24队列保持暂停。

原Meta8的Teacher分叉在YangLab，CleanMeta8在Ubuntu；尽管起点状态/命令和算法参数一致，PhysX跨硬件及分叉反馈噪声不保证逐位确定性。同机学生比较能控制学生执行硬件，不能将所有最终性能差异唯一归因于Adam修复。单seed验证不宣称统计显著，修复复现与新算法效果分开报告。

原Fresh文件SHA：`3c068821ffc58e29463c61b515602a4207b16edb0bafde7b00bbd86f7c1b3279`。
Ubuntu缓存容器文件SHA可不同，但已要求**完整Teacher状态**SHA完全一致：`b312ce7657bb6a7497e4dfe0062fe996fff0c6c653bd4836cd929d74b47fe2c2`。

程序/数据完整性问题会停止队列并保留FAULT与全部输出；不因性能差停止或缩短预算。正式长队列不依赖Mac在线。
