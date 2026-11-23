2026-11-20，STALE 扩增版全新 10 场景九方法对照；抽样种子 `20261121`（种子数值不是运行日期）。

锁定方法在本轮主方法 Top-10 命中 **51/61（83.6%）**，官方原生 M_old 命中 **9/10**；Dense 分别为 **32/61（52.5%）** 和 **3/10**。相比仅探针，正扩散与双极扩散均增加 6 个 Top-10 目标、2 个原生目标。双极负源在本轮只比正扩散多一个 Top-5 命中，总体 NDCG@10 略低（0.7660 对 0.7664），不能把本轮改善全部归因于负源。

主包 `memory_retrieval/` 全部源码哈希前后一致。提示词 SHA256 保持 `a38b198018c62846780914f1ef90fe2461ee1554c421d42996f54525f93217bf`，参数保持 `k=5, top_seeds=5, alpha=0.2, beta=0.3, probe_workers=4`。没有新增方法、修改图公式、筛选探针、按结果重生成或调整参数。

排除过程覆盖本仓库全部历史实验的输入、排名、报告、日志及 API 缓存，并检查此前讨论与 README 示例。直接排除 24 个场景：Austin 1 个、Dev 10 个、此前 Test 10 个、树皮蝎/龙舌兰 2 个示例，以及 README 中另一个点名示例。再排除原生 M_old 规范化正文相同的 7 个场景，共 31 个。规范化采用 NFKC、casefold、弯引号折叠和空白合并。另逐条复核历史输入中 31 条原生记录（21 个不同正文）与当前语料一致。

剩余 T1 185、T2 184 个场景。每层先按 scenario_id 排序，使用同一个 `random.Random(20261121)`，依次打乱 T1、T2，取前五个全局 M_old 正文不重复者。抽样在推理前写入并哈希冻结，复算完全一致。历史 selection 中仅作为候选清单出现的 eligible/shuffled IDs 不算已经评测；处理规则和来源证据已记录。10 个入选 ID 与排除集无交集，原生正文与历史排除正文无交集，样本内部正文也互不重复。

本轮 10 场景共 1,133 条旧记忆，各自库大小 98–120 条；每个方法只在对应场景旧库内排序。T1 有 33 个已标注目标，T2 有 28 个，共 61 个：10 个官方原生目标、46 个合成复核目标、5 个历史补标目标。所有正文和标签均从语料原样抽取。本轮衡量已有标注的召回，未把未标注项重新判为负例。

使用当前 `.env` 服务，`deepseek-v4.1-flash`、温度 0、JSON 模式、`reasoning_effort=none`、4 并发。三个新臂共享 40 条探针与一次加载的冻结 BGE；探针请求与本地编码重叠。BGE 为 `BAAI/bge-base-en-v1.5`，revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`，CLS、L2、CPU FP32、batch=4、threads=4、seed=0。SparseCL 三个原有训练检查点及参数完整运行，身份与已核验历史配置一致。BM25 k1=1.2、b=0.75；Hybrid 等权 RRF 常数 60。

通过主包 `run_cli` 实际运行的九个臂为 `bm25 dense hybrid sparsecl_arguana sparsecl_msmarco sparsecl_hotpotqa anchored_probes_only anchored_pos_diffusion anchored_bipolar_diffusion`。下表“仅探针／正扩散／双极扩散”分别对应上述三个 `anchored_*` 臂。Top-K 单元格为命中数（组内目标微平均召回率）；NDCG@10 按查询平均。

**T1：5 场景、33 目标**

| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 原生 M_old Top-10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 7（21.2%） | 7（21.2%） | 10（30.3%） | 0.3193 | 0/5 |
| BGE Dense | 5（15.2%） | 12（36.4%） | 19（57.6%） | 0.4832 | 2/5 |
| Hybrid | 6（18.2%） | 11（33.3%） | 16（48.5%） | 0.4587 | 1/5 |
| SparseCL ArguAna | 5（15.2%） | 12（36.4%） | 19（57.6%） | 0.4797 | 2/5 |
| SparseCL MS MARCO | 8（24.2%） | 9（27.3%） | 17（51.5%） | 0.5143 | 0/5 |
| SparseCL HotpotQA | 3（9.1%） | 6（18.2%） | 7（21.2%） | 0.1874 | 1/5 |
| 仅探针 | 10（30.3%） | 15（45.5%） | 25（75.8%） | 0.6759 | 4/5 |
| 正扩散 | 9（27.3%） | 16（48.5%） | 28（84.8%） | 0.7324 | 5/5 |
| 双极扩散（主方法） | 9（27.3%） | 17（51.5%） | 28（84.8%） | 0.7351 | 5/5 |

**T2：5 场景、28 目标**

| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 原生 M_old Top-10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 2（7.1%） | 3（10.7%） | 6（21.4%） | 0.2098 | 0/5 |
| BGE Dense | 5（17.9%） | 10（35.7%） | 13（46.4%） | 0.3963 | 1/5 |
| Hybrid | 4（14.3%） | 6（21.4%） | 8（28.6%） | 0.2834 | 0/5 |
| SparseCL ArguAna | 5（17.9%） | 10（35.7%） | 13（46.4%） | 0.3641 | 1/5 |
| SparseCL MS MARCO | 8（28.6%） | 10（35.7%） | 13（46.4%） | 0.5038 | 1/5 |
| SparseCL HotpotQA | 4（14.3%） | 5（17.9%） | 7（25.0%） | 0.2632 | 2/5 |
| 仅探针 | 11（39.3%） | 16（57.1%） | 20（71.4%） | 0.7790 | 3/5 |
| 正扩散 | 12（42.9%） | 16（57.1%） | 23（82.1%） | 0.8004 | 4/5 |
| 双极扩散（主方法） | 12（42.9%） | 16（57.1%） | 23（82.1%） | 0.7970 | 4/5 |

**总体：10 场景、61 目标**

| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 原生 M_old Top-10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 9（14.8%） | 10（16.4%） | 16（26.2%） | 0.2646 | 0/10 |
| BGE Dense | 10（16.4%） | 22（36.1%） | 32（52.5%） | 0.4397 | 3/10 |
| Hybrid | 10（16.4%） | 17（27.9%） | 24（39.3%） | 0.3710 | 1/10 |
| SparseCL ArguAna | 10（16.4%） | 22（36.1%） | 32（52.5%） | 0.4219 | 3/10 |
| SparseCL MS MARCO | 16（26.2%） | 19（31.1%） | 30（49.2%） | 0.5091 | 1/10 |
| SparseCL HotpotQA | 7（11.5%） | 11（18.0%） | 14（23.0%） | 0.2253 | 3/10 |
| 仅探针 | 21（34.4%） | 31（50.8%） | 45（73.8%） | 0.7275 | 7/10 |
| 正扩散 | 21（34.4%） | 32（52.5%） | 51（83.6%） | 0.7664 | 9/10 |
| 双极扩散（主方法） | 21（34.4%） | 33（54.1%） | 51（83.6%） | 0.7660 | 9/10 |

全部 27 行 NDCG@10 已用 `pytrec_eval` 从完整排名复核，最大绝对误差 `1.11e-16`。为保留主包的既定平局顺序，复核使用严格递减的排名分数。九方法的每条排名均完整覆盖该场景旧库，且不存在跨场景候选。

40 条探针按原始 query 逐条核读，归为明确冲突（含直接失效）、相容或复述、歧义。核读由助手完成，非独立人工标注；不依据检索命中来决定类别。时段、规则执行或必要条件不足的条目归入歧义。结果如下：

| 分组 | 明确冲突/直接失效 | 相容或复述 | 歧义 | 显式否定词探针 | cos >0.85 |
|---|---:|---:|---:|---:|---:|
| T1 | 16 | 0 | 4 | 7 | 0 |
| T2 | 10 | 4 | 6 | 2 | 1 |
| 总体 | 26 | 4 | 10 | 9 | 1 |

相容项主要是：不必逐张阅读食品标签仍可依赖安全食物清单或团队核准；尝试新菜仍可只去熟悉餐馆；尝过且惦记某道菜不代表喜欢它。歧义项包括未来计划与过去行程不重叠、固定手腕不等于双手完全不能参与、违反医嘱的行为不必然被医嘱本身否定。听力场景中的部分探针只是反写复诊细节，虽可直接冲突，但未必覆盖需要检索的长期旧状态。

10/10 响应 JSON 结构有效，40 条均为英文、均未超过 30 词。BGE 未知词元 **0**、BGE 截断 **0**、LLM 截断 **0**，逐字复述 query **0**。9 条含 no/without/cannot 等显式否定，说明快速模型仍未完全遵循肯定句要求。唯一 cos>0.85 的探针是医生允许左腕承重，与 query 的禁止承重相反；高余弦不等于相容漂移。所有格式或语义问题均原样保留，没有改变本轮评分。

全部九方法总墙钟耗时 **117.03 秒**（约 1 分 57 秒，不含抽样和事后核读）。LLM 探针生成墙钟 **10.95 秒**，单条请求平均 **3.69 秒**、中位数 **3.78 秒**。请求与本地模型工作重叠，耗时不应直接相加。

请求 10 次，全部一次成功；新空缓存命中 **0**，重试 **0**。输入 **3,834 tokens**，输出 **852 tokens**，总计 **4,686 tokens**，服务端上报 reasoning tokens **0**。

本轮支持锁定探针加图扩散在这批未用于此前实验的样本上提高召回，尤其是官方原生目标；仍只是 10 场景小样本结果。正扩散贡献清楚，负源的额外收益在本轮很小且随指标变化。方法保持锁定，没有用本轮结果反向修改提示词或参数。

入选场景完整记录：

| 类型 | scenario_id | 原生 memory_id |
|---|---|---|
| T1 | 6ff5a576-b0e1-4127-80b7-0d3466526bcc | stale_m_89000fdc6475153317db4b75 |
| T1 | 9ce11bb3-9680-4e92-86fc-5f450d36405b | stale_m_6d526896aadc0a0fe0b69481 |
| T1 | dae22057-9b78-4647-abee-a6834b6cb84e | stale_m_b9cfb3becfcc7d70076bafe6 |
| T1 | 7966499b-2362-48ad-976b-eeb09595be37 | stale_m_2059a02a35138c960c16de38 |
| T1 | f3259991-79c1-445f-8421-42f9e9fcb78a | stale_m_c7f9b04e23892781f1619cbc |
| T2 | f57f6037-c95a-40ef-a7ab-6f5b396ba927 | stale_m_8c1b0a88a55f4af950044e02 |
| T2 | 48707e03-300e-4835-afe4-ae24edb26e18 | stale_m_83e125cb585c9a480c5bfa0e |
| T2 | 16763588-459c-4dca-95ec-f7ff02c5a940 | stale_m_ce8ce77e9d78b9cfea017806 |
| T2 | ef4af969-d7b1-447a-8bc8-378b933fef3a | stale_m_c29b5ab440e22f1477de1ae5 |
| T2 | 463b4ca9-a781-49fa-b831-e36d61f11b58 | stale_m_962fd46ef95d1b35113f3c0c |

复现入口（在已配置依赖的 WSL Python 环境运行；需要复制脚本并使用新的独立目录，原目录禁止覆盖）：

```text
python launch.py
python verify.py
```

[完整抽样记录](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/selection.json) · [排除来源记录](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/exclusions.json) · [历史原生正文核验](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/historical_native_text_verification.json) · [锁定协议](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/protocol.json) · [主包 CLI 启动脚本](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/launch.py)

[全部指标及金标排名](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/comparison.json) · [数值与完整性核验](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/verification.json) · [原始完整报告](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/results/report.json) · [执行日志](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/execution.log)

[40 条探针逐条核读](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/semantic_review.zh.md) · [结构化核读](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/semantic_review.json) · [原始探针](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/results/anchored_probes.json) · [自动质量审计](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/results/anchored_probe_audit.json) · [提示词原文](../runs/experiments/stale-unseen-10scene-query-only-bge-seed20261121/prompt.txt)
