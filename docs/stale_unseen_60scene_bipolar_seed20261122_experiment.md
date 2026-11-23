# STALE 未见 60 场景对照实验

2026-11-20，在服务器执行；随机种子 20261122 是抽样标识。

从 400 场景池排除 43 个历史使用、提及或原生正文重复场景，从剩余 357 个中固定抽取 30 T1＋30 T2。样本 scenario_id 和规范化原生 M_old 正文与排除集均无交集，样本内部原生正文也互不重复。
本轮 6673 条旧记忆，340 个已标注目标，含 60 个官方原生目标。其余目标来自既有扩增与模型复核；不是全量人工金标，未改写正文或标签。每个方法只在对应场景的旧库内排序。

主包原样调用：query-only 四维探针，提示词 SHA256 `a38b198018c62846780914f1ef90fe2461ee1554c421d42996f54525f93217bf`；k=5、Top-5 种子、α=0.2、β=0.3。LLM 为当前配置的 deepseek-v4.1-flash，temperature=0、JSON、reasoning_effort=none、16 并发；仅传输错误按 APIClient 退避重试，所有响应原样保存，无筛选、改写或重抽。
BGE 为冻结 BAAI/bge-base-en-v1.5（revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a），CLS/L2、FP32、max_length=512、batch=4、每进程 CPU threads=4、seed=0。三个 SparseCL 检查点及 α=0.18/2.19/4.82 沿用主包锁定配置；BM25 k1=1.2、b=0.75；Hybrid 等权 RRF=60。四个编码器任务与探针请求重叠运行，BGE 三个新臂共用同一批向量和图。

## T1：30 场景、182 目标

| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 原生 M_old@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 15/182（8.2%） | 30/182（16.5%） | 53/182（29.1%） | 0.2284 | 6/30 |
| BGE Dense | 32/182（17.6%） | 46/182（25.3%） | 95/182（52.2%） | 0.4295 | 17/30 |
| Hybrid | 28/182（15.4%） | 47/182（25.8%） | 83/182（45.6%） | 0.3872 | 12/30 |
| SparseCL ArguAna | 31/182（17.0%） | 46/182（25.3%） | 96/182（52.7%） | 0.4371 | 17/30 |
| SparseCL MS MARCO | 40/182（22.0%） | 58/182（31.9%） | 100/182（54.9%） | 0.4946 | 20/30 |
| SparseCL HotpotQA | 28/182（15.4%） | 40/182（22.0%） | 54/182（29.7%） | 0.2990 | 9/30 |
| 完整方法 | 62/182（34.1%） | 96/182（52.7%） | 153/182（84.1%） | 0.7839 | 26/30 |

## T2：30 场景、158 目标

| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 原生 M_old@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 16/158（10.1%） | 21/158（13.3%） | 33/158（20.9%） | 0.1714 | 4/30 |
| BGE Dense | 17/158（10.8%） | 38/158（24.1%） | 84/158（53.2%） | 0.3860 | 9/30 |
| Hybrid | 13/158（8.2%） | 33/158（20.9%） | 67/158（42.4%） | 0.3151 | 7/30 |
| SparseCL ArguAna | 16/158（10.1%） | 38/158（24.1%） | 82/158（51.9%） | 0.3789 | 9/30 |
| SparseCL MS MARCO | 33/158（20.9%） | 47/158（29.7%） | 69/158（43.7%） | 0.4098 | 11/30 |
| SparseCL HotpotQA | 17/158（10.8%） | 26/158（16.5%） | 40/158（25.3%） | 0.2360 | 6/30 |
| 完整方法 | 47/158（29.7%） | 76/158（48.1%） | 123/158（77.8%） | 0.6808 | 21/30 |

## 总体：60 场景、340 目标

| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 原生 M_old@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 31/340（9.1%） | 51/340（15.0%） | 86/340（25.3%） | 0.1999 | 10/60 |
| BGE Dense | 49/340（14.4%） | 84/340（24.7%） | 179/340（52.6%） | 0.4077 | 26/60 |
| Hybrid | 41/340（12.1%） | 80/340（23.5%） | 150/340（44.1%） | 0.3512 | 19/60 |
| SparseCL ArguAna | 47/340（13.8%） | 84/340（24.7%） | 178/340（52.4%） | 0.4080 | 26/60 |
| SparseCL MS MARCO | 73/340（21.5%） | 105/340（30.9%） | 169/340（49.7%） | 0.4522 | 31/60 |
| SparseCL HotpotQA | 45/340（13.2%） | 66/340（19.4%） | 94/340（27.6%） | 0.2675 | 15/60 |
| 完整方法 | 109/340（32.1%） | 172/340（50.6%） | 276/340（81.2%） | 0.7323 | 47/60 |

## 逐场景配对显著性

差值为完整方法减基线的平均 NDCG@10。Wilcoxon 双侧、Pratt 处理零差、正态近似，差值仅在检验秩时舍入至 12 位。Bootstrap 按场景成对重采样 20,000 次，种子 20261122；总体保持 T1/T2 各 30 个，取百分位 95% CI。配置在推理前冻结。p 值未做多重比较校正；总体为主要比较，T1/T2 为探索性分组。

| 分组 | 对照 | 平均 ΔNDCG@10 | Wilcoxon p | Bootstrap 95% CI | 胜/平/负 |
|---|---|---:|---:|---|---|
| T1 | BGE Dense | +0.3543 | 1.56327e-05 | [+0.2545, +0.4493] | 27/1/2 |
| T1 | SparseCL MS MARCO | +0.2893 | 0.000160464 | [+0.1713, +0.4064] | 25/0/5 |
| T2 | BGE Dense | +0.2948 | 9.75984e-06 | [+0.2152, +0.3734] | 27/1/2 |
| T2 | SparseCL MS MARCO | +0.2710 | 8.91873e-05 | [+0.1432, +0.3825] | 26/0/4 |
| all | BGE Dense | +0.3246 | 7.0238e-10 | [+0.2611, +0.3867] | 54/2/4 |
| all | SparseCL MS MARCO | +0.2802 | 5.32134e-08 | [+0.1939, +0.3624] | 51/0/9 |

两个消融臂的完整排名和指标均保存在 comparison.json；与主方法使用相同探针和图，无额外 LLM 调用。

总实际运行 227.74 秒，探针生成 134.12 秒，评分与评测 2.61 秒；并行阶段不能相加当作总耗时。
Token / API：`{"api_attempts": 65, "cache_hits": 0, "invalid_outputs": 0, "known_prompt_tokens": 22991, "known_completion_tokens": 5774, "known_total_tokens": 28765, "responses_without_usage": 0, "cost_currency": null}`。
重试统计：`{"status_counts": {"http_error": 3, "ok": 60, "network_error": 2}, "http_status_counts": {"429": 3}, "successful_requests_after_retry": 4, "max_recorded_attempt": 3}`。
探针基础审计（机械标记，非人工语义判定）：`{"probes": 240, "structurally_valid_responses": 60, "probe_query_cosine_gt_085": 0, "exact_query_repeats": 0, "probes_over_30_words": 0, "probes_with_explicit_negation": 42, "probes_with_non_latin_letters": 0, "llm_truncated_responses": 0, "anchor_encoded": false, "filtered_or_regenerated": false, "token_audit": {"texts": 240, "tokens_excluding_special": 4233, "unknown_tokens": 0, "unknown_ratio": 0.0, "max_tokens_with_special": 32, "would_truncate": 0}}`。

全部 27 个方法×分组的 NDCG@10 由 pytrec_eval 从完整排名独立复算，误差 <1e-12；核验所有候选均属于对应场景。排名分数严格递减，仅用于保留原方法的平局顺序。主包、数据池和历史实验文件哈希均保持不变。

原始目录：`/home/dzp/paper/memory-comparison-retrieval/runs/experiments/stale-unseen-60scene-bipolar-seed20261122`。`selection.json` / `exclusions.json` 保存抽样与排除证据；`protocol.json` 冻结参数和统计方案；`comparison.json`、`per_query_ndcg.json`、`arms/` 保存指标及完整排名；`probe_cache.sqlite3`、`probe_records/`、`api_events/` 保存原始响应及调用记录；`encoders/`、`graphs/` 支持断点恢复。
