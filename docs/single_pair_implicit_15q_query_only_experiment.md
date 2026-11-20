2026-11-20，single_pair_implicit_15q_en 去锚点复核。

本次删除生成结果中的 `new_state_anchor`，四条探针直接以原始 query 为既定事实。主方法 Top-3/5/10 为 **8/9/11**，上轮为 **4/7/11**；NDCG@10 从 0.3780 升至 0.4057。前排排序改善，Top-10 总覆盖未增加。60 条探针的保守核读结果由 **27/22/11** 变为 **42/3/15**（明确冲突／相容或复述／歧义），仍有不合格探针。

代码只改提示词、结构校验及相关审计、测试、历史重放适配。`build_knn_transition`、`compute_bipolar_diffusion_scores` 的源码哈希与修改前相同，`cli.py`、`experiment.py` 文件哈希相同；没有增加过滤、重新生成、调参或新打分。CLI 保留三个 `anchored_*` 臂名。

数据是原有 15 条英文查询、225 条共享旧记忆、15 对金标（每条查询一个金标）。三份输入与历史基线逐字节一致；本次使用冻结快照，通过主包 `run_cli` 只运行三个臂。六个基线复用已核验的完整排名，未重新运行，SparseCL 加载数为 0。所有方法均在完整 225 条旧库上排序。

BGE 为 `BAAI/bge-base-en-v1.5`，revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`，CLS、L2、CPU FP32、batch=4、threads=4、seed=0、max_length=512。本次三个臂共享一次加载的 BGE 和同一批探针；编码器身份与历史对照一致。历史 SparseCL 沿用其三个训练检查点及各自原参数。

LLM 使用当前 `.env` 中的服务配置，模型 `deepseek-v4.1-flash`，温度 0、JSON 模式、`reasoning_effort=none`、4 并发。英文提示词不含生活领域例词或测试集示例，保留四个纯逻辑维度；输出仅 `violation_probes` 四条非空字符串。要求同语种、肯定陈述、每条不超过 30 词，禁止复述 query。实际遵循情况单独报告，不据此剔除探针。

图扩散仍为正余弦 Top-k 有向图、对角线为零、行归一化，`k=5`，平局按 memory_id 升序；`top_seeds=5, alpha=0.2, beta=0.3`。`r0=max_j(X@u_j)`，`z_pos` 保留 r0 前五节点原分数，`z_neg=max(0, X@v_q-r0)`；正扩散 `(1-alpha)*r0+alpha*(P@z_pos)`，双极扩散再减 `beta*(z_neg+alpha*(P@z_neg))`。

下表各 Top-K 单元格为命中数/15（召回率）。NDCG@10 按 15 条查询平均，全部九行均由 `pytrec_eval` 从完整排名复核，与包内指标误差 <1e-12；平均金标排名越低越好。

| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 平均金标排名 |
|---|---:|---:|---:|---:|---:|
| BM25（历史） | 0/15（0.0%） | 0/15（0.0%） | 1/15（6.7%） | 0.0201 | 90.33 |
| BGE Dense（历史） | 0/15（0.0%） | 1/15（6.7%） | 5/15（33.3%） | 0.1177 | 14.47 |
| Hybrid（历史） | 0/15（0.0%） | 1/15（6.7%） | 3/15（20.0%） | 0.0681 | 27.27 |
| SparseCL ArguAna（历史） | 0/15（0.0%） | 1/15（6.7%） | 5/15（33.3%） | 0.1185 | 14.33 |
| SparseCL MS MARCO（历史） | 2/15（13.3%） | 4/15（26.7%） | 8/15（53.3%） | 0.2734 | 17.60 |
| SparseCL HotpotQA（历史） | 3/15（20.0%） | 4/15（26.7%） | 6/15（40.0%） | 0.2762 | 44.33 |
| anchored_probes_only（本次） | 5/15（33.3%） | 9/15（60.0%） | 10/15（66.7%） | 0.3711 | 10.47 |
| anchored_pos_diffusion（本次） | 7/15（46.7%） | 9/15（60.0%） | 11/15（73.3%） | 0.4011 | 7.40 |
| anchored_bipolar_diffusion（本次主方法） | 8/15（53.3%） | 9/15（60.0%） | 11/15（73.3%） | 0.4057 | 7.27 |

与紧邻上轮（保留锚点、强化极性约束）相比，探针臂由 4/5/8 变为 5/9/10，正扩散由 4/7/11 变为 7/9/11，双极扩散由 4/7/11 变为 8/9/11；主方法平均金标排名由 9.27 降至 7.27。本次双极扩散比仅正扩散多一个 Top-3 命中，Top-5/10 相同。这是同一探索集上的修复复核，不能单凭本轮证明未见数据泛化。

逐条核读以原始 query 及其时间范围为准，由助手完成，非独立人工标注。沿用上轮保守口径：未来计划或点餐未执行、时间/物品/类别条件不足，归入歧义，不算已证实冲突；能够直接共存的陈述归入相容。每条理由和原文均保存。

| 版本 | 明确冲突 | 相容或复述 | 存在歧义 |
|---|---:|---:|---:|
| 上轮 | 27 | 22 | 11 |
| 本次 | 42 | 3 | 15 |

去掉锚点后，锚点反转计数已不适用。此前六条反转查询均产生至少两条明确冲突探针，未再出现围绕错误锚点成组复述原句的情况，但这不等于四条探针全部恢复合格。下表的三元组均为明确冲突／相容或复述／歧义。

| 查询 | 上轮四条探针 | 本次四条探针 | 剩余问题 |
|---|---:|---:|---|
| q_hc_02 | 0 / 4 / 0 | 3 / 0 / 1 | 下一次计划自驾不等于过去已经驾车。 |
| q_hc_05 | 0 / 4 / 0 | 3 / 0 / 1 | 第 3 条未明确排除浮具；四条仍带 can't 等显式否定。 |
| q_ic_01 | 1 / 3 / 0 | 3 / 0 / 1 | 未来侵入式体检计划不能推翻既往全部非侵入。 |
| q_ic_04 | 1 / 2 / 1 | 3 / 0 / 1 | 未来先带回家再付款的计划不能推翻既往付款顺序。 |
| q_bs_01 | 0 / 4 / 0 | 4 / 0 / 0 | 四条均明确指向近期实际进入地下。 |
| q_bs_02 | 0 / 4 / 0 | 2 / 1 / 1 | 第 3 条仍是计划；第 4 条有信号不等于实际联网。 |

三条相容探针分别是允许范围内的含干果浸泡饮品（q_ic_02 第 3 条）、公园内具备联网信号（q_bs_02 第 4 条）、天亮时由闹钟叫醒的计划（q_bs_05 第 3 条）。15 条歧义中有 10 条来自第 3 个维度，常见问题是把未来计划写成对过去行为的反例；其他例子有电脑包未必装电脑、眼罩露眼的时间未限定在入睡前。另有两条合肥探针虽然表达了出市事实，但句内措辞不够自洽，逐条记录中已说明。

自动审计：15/15 响应结构有效；60 条均为英文且未超过 30 词，无未知 BGE 词元、LLM 截断或 BGE 截断。显式否定命中 4/60，全部来自 q_hc_05；原样保留。原句余弦 >0.85 为 2/60，均是 q_bs_04 的明确冲突探针，高余弦不等于相容漂移；逐字复述原句为 0/60。

实验总墙钟时间 **16.25 秒**（不含单测及核读），探针生成墙钟时间 **15.28 秒**，单查询平均 3.55 秒、中位数 3.40 秒。LLM 与本地 BGE 重叠执行，分项耗时不能直接相加。

15 次 API 请求全部一次成功，缓存命中 0、无重试、无语义筛选或重生成。输入 **5,470 tokens**，输出 **1,212 tokens**，合计 **6,682 tokens**；服务端上报 reasoning tokens 为 0。新目录及新缓存与全部历史结果隔离。

全部 **41 项 pytest 单测通过**（45.67 秒）。历史 STALE 已保存向量的零推理重放也通过：完整排名相同，分数与 NDCG@10 最大差均为 0。没有为该检查调用 LLM 或加载编码器。

当前提示词 SHA256：`a38b198018c62846780914f1ef90fe2461ee1554c421d42996f54525f93217bf`。

[本轮汇总与逐查询排名](../runs/experiments/single-pair-15q-query-only-bge-20261120/comparison.json) · [60 条探针逐条核读](../runs/experiments/single-pair-15q-query-only-bge-20261120/semantic_review.zh.md) · [结构化核读记录](../runs/experiments/single-pair-15q-query-only-bge-20261120/semantic_review.json) · [原始探针](../runs/experiments/single-pair-15q-query-only-bge-20261120/results/anchored_probes.json) · [自动审计](../runs/experiments/single-pair-15q-query-only-bge-20261120/results/anchored_probe_audit.json)

[输入及代码清单](../runs/experiments/single-pair-15q-query-only-bge-20261120/input_manifest.json) · [提示词原文](../runs/experiments/single-pair-15q-query-only-bge-20261120/prompt.txt) · [主包 CLI 启动脚本](../runs/experiments/single-pair-15q-query-only-bge-20261120/launch.py) · [完整排名与报告](../runs/experiments/single-pair-15q-query-only-bge-20261120/results/report.json) · [指标与完整性核验](../runs/experiments/single-pair-15q-query-only-bge-20261120/verification.json) · [图算法历史重放](../runs/experiments/single-pair-15q-query-only-bge-20261120/historical_graph_consistency.json)

[六个历史基线来源](../runs/experiments/single-pair-15q-anchored-bge-20261120-retry1/comparison.json) · [上轮结果](../runs/experiments/single-pair-15q-anchor-polarity-fix-bge-20261120/comparison.json) · [上轮探针核读](../runs/experiments/single-pair-15q-anchor-polarity-fix-bge-20261120/semantic_review.json) · [主算法代码](../memory_retrieval/bipolar_graph.py)
