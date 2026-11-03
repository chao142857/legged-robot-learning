E2: WBC 命令序列分组与共享 SGS 统计（离线研究）

分支 e2/grouped-sgs-offline；baseline-frozen 提交保留原严格 SGS 源码与预注册判据。
32768训练序列 -> 4096组x8条。Random Group seeds20261104/05/06，Command Cluster seed20261104；主Random为20261104。
独立1024评测池不参与聚类或训练。聚类只读取训练池和共享的reset参考状态，不读取成功率、checkpoint表现或评测池内容。
特征根据真实native时序：自动reset后的velocity slot2在最后一个控制步有效，纳入精确驻留特征；无效第四槽和固定timer随机列排除。
特征含归一化位置/速度、站立模式、命令顺序、SO(3)矩阵和精确切换/驻留。不是对原始数组直接运行KMeans。

测试：python -m unittest discover -s tests -v（需要numpy）。真实规模Torch检查在冻结Isaac镜像中运行code/production_checks.py。
分组采样器：code/grouped_sampling.py；Isaac接入：code/grouped_adapter.py。Random和Command使用同一代码，只有mapping不同。
history100表示每组最近100个真实结束回合，不是每条100x8。空历史p=0.5，不创建伪反馈。
P(group k)=size_k*f(p_k)/sum_j(size_j*f(p_j))；组内均匀抽真实sequence。保存pool/mapping哈希、窗口、时间、成员、计数、事件归属和RNG。
集成顺序：native WBC环境 -> GroupedAdapter(raw,'wbc','sgs',train_pool,out,mapping) -> RslRlVecEnvWrapper（会reset）-> 原生RSL Runner。
奖励、观测、18D动作、终止、随机化、网络和PPO保持原样。独立评测继续用原SequenceAdapter evaluation模式，不调用GroupedAdapter回填。
工程synthetic histories/checkpoint仅在tests_only中，禁止用于任何正式初始化。

离线结果区分：固定策略面板、变策略训练历史、同时间窗口成员混合。历史p不等于冻结难度。
结构修订：v1只追踪初次reset；精确时钟模型识别auto-reset最后一帧速度槽，v2第二回合追踪确认后修正特征。v1结果保留远端revisions，不参与最终判定。未按成功表现搜索权重或超参数。
原六组15k目录在所有E2容器中只读，六个模型哈希再次核验。
原始反馈日志、特征大数组、工程checkpoint、逐步结构追踪保留YangLab；交付包仅含必要映射、汇总、图表、源码、测试及报告。
远端 /data/go2_piper_sgs_e2_20261104；tmux go2-sgs-e2。
正式训练仅在预注册GO条件全部满足后按用户授权自动启动两组各10k；当前判定见E2_GO_NO_GO.json。最小冻结补充方案为未执行的方案文件。
