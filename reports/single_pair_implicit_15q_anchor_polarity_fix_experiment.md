# 锚点极性保真提示词修改与 15 查询复核

2026-11-20。按要求修改提示词后，仅重跑 `anchored_probes_only`、`anchored_pos_diffusion`、`anchored_bipolar_diffusion` 三个实验臂。六个基线直接复用上轮已核验结果，没有重新运行。**本次提示词修改没有解决模型的锚点极性反转：逐条核读为 6/15，上一轮为 5/15，未降至 0/15。**

## 修改与测试

`memory_retrieval/bipolar_graph.py` 中仅替换指定的两处提示词内容：

> new_state_anchor: one objective declarative sentence of at most 20 words that faithfully preserves the exact polarity, restrictions, and any explicit negations of query (never remove negations or flip the meaning in the anchor). This anchor is solely a comparison reference.

> Avoid explicit negation (such as "not", "never", "no longer", and negative contractions) in violation_probes only, never in new_state_anchor.

其余提示词及打分公式未改。新提示词 UTF-8 SHA256：`0df129c07977b5a71b73e116d4950a1ba63e4910b793b4faa5eabd73fcdc708f`。单测验证与历史提示词相比恰好只有这两处替换；全部 **39 项 pytest 测试通过**，耗时 51.65 秒。测试只能确认实现和协议，不能保证 LLM 遵循语义约束。

历史一致性脚本现区分当前和历史提示词哈希，只重放旧向量的评分；复核所得历史完整排名、分数及 NDCG 均一致，差值为 0，无模型推理。历史实验文件和缓存均保留。

## 数据与实验配置

当前 `single_pair_implicit_15q_en`：15 条英文查询、225 条共享旧记忆、15 对金标，每条查询一个目标。输入快照与历史基线逐文件哈希相同。这是经过多轮构造和实验的探索诊断集。

- 本轮由主包 CLI 运行 `--arms anchored_probes_only anchored_pos_diffusion anchored_bipolar_diffusion`。独立新目录、新空缓存；没有运行六个基线、旧 OCSP 或其他额外方法。
- 冻结 `BAAI/bge-base-en-v1.5`，revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`；CLS/L2、CPU FP32、batch=4、4 线程、seed=0、max_length=512。BGE 加载一次，三个臂共用原句、旧库和同一批 60 条探针向量，锚点不编码。未加载 SparseCL。
- 使用用户更新后的服务配置；实际请求和响应模型均为 `deepseek-v4.1-flash`，4 并发、温度 0、JSON 模式、`reasoning_effort=none`。
- 锁定 `k=5, top_seeds=5, alpha=0.2, beta=0.3`。正相似有向邻边、零对角、按行归一化，按原公式计算 `P@z`。无调参、输出筛选、结果驱动重生成或额外规则。
- 历史基线来自 [上轮完整运行](../runs/experiments/single-pair-15q-anchored-bge-20261120-retry1/comparison.json)。已检查数据、基础 BGE 版本和编码配置一致；对六个历史方法重新读取完整排名并复算指标，不重新执行模型。

## 九个方法统一对照

前六行是历史基线，后三行是本次新提示词复跑。Top-K 为命中数 / 15，括号为召回率。NDCG@10 越高越好，平均金标名次越低越好；全部九行的 NDCG 经 `pytrec_eval` 复核。

| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 平均金标名次 |
|---|---:|---:|---:|---:|---:|
| BM25（历史） | 0/15（0.0%） | 0/15（0.0%） | 1/15（6.7%） | 0.0201 | 90.33 |
| BGE Dense（历史） | 0/15（0.0%） | 1/15（6.7%） | 5/15（33.3%） | 0.1177 | 14.47 |
| Hybrid（历史） | 0/15（0.0%） | 1/15（6.7%） | 3/15（20.0%） | 0.0681 | 27.27 |
| SparseCL ArguAna（历史） | 0/15（0.0%） | 1/15（6.7%） | 5/15（33.3%） | 0.1185 | 14.33 |
| SparseCL MS MARCO（历史） | 2/15（13.3%） | 4/15（26.7%） | 8/15（53.3%） | 0.2734 | 17.60 |
| SparseCL HotpotQA（历史） | 3/15（20.0%） | 4/15（26.7%） | 6/15（40.0%） | 0.2762 | 44.33 |
| anchored_probes_only | 4/15（26.7%） | 5/15（33.3%） | 8/15（53.3%） | 0.2558 | 12.73 |
| anchored_pos_diffusion | 4/15（26.7%） | 7/15（46.7%） | 11/15（73.3%） | 0.3780 | 9.13 |
| anchored_bipolar_diffusion | 4/15（26.7%） | 7/15（46.7%） | 11/15（73.3%） | 0.3780 | 9.27 |

本批探针下，加入正扩散使 Top-5 从 5 增至 7、Top-10 从 8 增至 11。负源抑制没有进一步改善这些预算指标和 NDCG；平均金标名次由 9.13 小幅变为 9.27，具体是 `q_bs_01` 从第 26 降至第 28。

双极方法本轮仍优于六个基线，但较上轮旧提示词的 Top-3/5/10 `6/9/13` 下降为 `4/7/11`，NDCG 从 0.4408 降至 0.3780，平均金标名次从 6.27 变为 9.27。两次调用的输出不同，这一轮对照不足以证明提示词修改必然导致下降，也不能声称已修复语义反转。

## 逐条语义核读

已核读全部 15 个锚点及 60 条探针，判断始终以原始 query 为准，而非以生成锚点为准。核读由助手完成，不是独立人工复标。

| query_id | 原句核心含义 | 本轮锚点核心含义 | 反转 |
|---|---|---|---|
| q_hc_01 | 近期饮食全素、未吃动物来源食物 | 保留该含义 | 否 |
| q_hc_02 | 从未亲自驾驶汽车 | 至少亲自驾驶过一次 | **是** |
| q_hc_03 | 猫狗从未进入当前住所 | 保留该含义 | 否 |
| q_hc_04 | 该期间外出从未带笔记本 | 保留该含义 | 否 |
| q_hc_05 | 无法触底时从未无支撑游过一米 | 可以在该条件下无支撑游泳 | **是** |
| q_ic_01 | 体检始终非侵入、从未破皮 | 至少一次侵入性破皮体检 | **是** |
| q_ic_02 | 该期间只喝白开水与花果茶 | 保留该限制 | 否 |
| q_ic_03 | 近期饮食全部无麸质 | 保留该含义 | 否 |
| q_ic_04 | 一直付清全款才带商品回家 | 先带回家再付清 | **是** |
| q_ic_05 | 从上床前至入睡眼罩一直遮住双眼 | 保留该限制 | 否 |
| q_bs_01 | 近期外出从未进入地下 | 至少一次外出进入地下 | **是，较上轮新增** |
| q_bs_02 | 从未在公园用随身设备联网 | 发生过这种行为 | **是** |
| q_bs_03 | 整个假期都在合肥市界内 | 保留该限制 | 否 |
| q_bs_04 | 场地访问全程在轮椅、左脚在脚踏上 | 保留该含义，但锚点 27 词超限 | 否 |
| q_bs_05 | 从昨晚早睡到天亮连续睡眠未醒 | 保留该含义 | 否 |

60 条探针逐条核读结果：27 条有明确冲突，22 条与原句相容或复述原状态，11 条需要额外条件或存在时间/计划执行歧义。例如，“计划先把商品带回家再付款”并不等于已经执行，不能直接否定此前所有购买均先付清。逐条文本、判断及原因全部保存在 [语义核读明细](../runs/experiments/single-pair-15q-anchor-polarity-fix-bge-20261120/semantic_review.json)。这些判断未进入检索流程，未修改任何探针或排名。

自动审计为：与原句余弦 >0.85 的探针 **0/60**，原样重复 query/anchor **0/60**，显式否定标记 **5/60**，锚点超 20 词 **1/15**。探针超 30 词、非拉丁文字、未知 BGE 词元及截断均为 0。自动相似度标记没有发现上述语义反转，不能将其解释为输出语义正确。

## 时间和用量

| 指标 | 本轮数值 |
|---|---:|
| 三个实验臂总墙钟耗时 | 13.97 秒 |
| 探针生成总耗时，含并发排队 | 13.02 秒 |
| 单查询平均 / 中位耗时，不含排队 | 3.19 / 3.00 秒 |
| BGE 加载校验 | 6.97 秒 |
| BGE 旧库 / 查询 / 探针编码 | 4.93 / 0.24 / 0.72 秒 |
| 图构建及三个臂打分 | 0.047 秒 |
| API 尝试 / 成功 / 缓存命中 | 15 / 15 / 0 |
| 输入 / 输出 / 总 tokens | 6,790 / 1,492 / 8,282 |
| 服务端报告 reasoning tokens | 0 |

LLM 与本地编码重叠，环节耗时不能直接相加。总墙钟时间不含单测、历史指标离线复核或人工核读。

[统一指标与逐查询名次](../runs/experiments/single-pair-15q-anchor-polarity-fix-bge-20261120/comparison.json) · [运行配置](../runs/experiments/single-pair-15q-anchor-polarity-fix-bge-20261120/results/run_config.json) · [新提示词](../runs/experiments/single-pair-15q-anchor-polarity-fix-bge-20261120/prompt.txt) · [原始锚点与探针](../runs/experiments/single-pair-15q-anchor-polarity-fix-bge-20261120/results/anchored_probes.json) · [指标核验](../runs/experiments/single-pair-15q-anchor-polarity-fix-bge-20261120/verification.json)
