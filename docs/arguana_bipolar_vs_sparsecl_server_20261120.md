# ArguAna：完整双极图扩散方法与 SparseCL 对照

2026-11-20，在服务器 `/home/dzp/paper/memory-comparison-retrieval` 完成。当前锁定方法在本次完整 ArguAna 测试中低于此前完整复现的 SparseCL；这次结果不能支持“我们的方法在该数据集上优于 SparseCL”。仅测试完整 `anchored_bipolar_diffusion`，没有新增变体或针对测试集调参，SparseCL 没有重跑。

| 方法 | NDCG@1 | NDCG@10 | Recall@10 | 来源 |
|---|---:|---:|---:|---|
| SparseCL（BGE cosine + ArguAna SparseCL Hoyer） | 0.49360 | 0.70406 | 0.91038 | 已核验的历史完整复现，Dev 选定 α=0.18 |
| 四维失效/冲突探针 + 双极图扩散 | 0.19061 | 0.37326 | 0.59744 | 本次服务器完整测试 |

我们的方法 Top-1 命中 268/1,406，Top-10 命中 840/1,406。表中结果使用官方 BEIR/pytrec_eval 评测，保留五位小数。另从保存的分数、Top-10、qrels 独立复算，结果与未舍入指标的差异小于 1e-12；全体原始 Top-10 排名与保存的完整分数一致。

数据与评测：使用 SparseCL 官方 ArguAna train→dev→test 合并语料，共 8,674 篇文档、1,406 条 Test 查询。公开文件中 5 条查询的正例不在语料中，保留这些查询并计零分，与历史复现一致。先按完整得分取原始 Top-10，再由 BEIR 排除与 query_id 相同的文档，不从第 11 名补位。本次有 650 条查询触发这一排除。语料仅使用正文，与历史编码一致。历史 SparseCL 只有汇总指标，不能据此做逐查询配对显著性检验。

方法与配置：直接调用主包的探针生成、k-NN 图和双极扩散实现。提示词 SHA256 为 `a38b198018c62846780914f1ef90fe2461ee1554c421d42996f54525f93217bf`；k=5、Top-5 种子、α=0.2、β=0.3。LLM 为当前配置的 `deepseek-v4.1-flash`，温度 0、JSON 模式、`reasoning_effort=none`，恢复阶段按用户要求采用 16 并发。冻结 BGE 为 `BAAI/bge-base-en-v1.5`，revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`，CLS/L2、FP32、最大 512 tokens、batch=4、seed=0。CPU 编码使用 16 线程，固定小批量验证与 4 线程输出一致；主包文件没有修改。

恢复与保留：在原 `arguana-bipolar-vs-sparsecl-bge-20261120-163016-r3` 目录继续，恢复前元数据和缓存保存在该目录 `resume_history/`。16 并发恢复开始时复用了 357 份有效响应，并复用全库向量、已完成的查询向量和 r2 保存的原图，未重新编码查询或重建图。后续异常退出仍在同一目录恢复，所有有效响应保留。原始失败记录、原实验目录和历史 SparseCL 结果均保留。

1,406 条查询对应 1,298 个不同的请求正文；重复正文复用精确请求缓存，评测仍包含全部查询。最终保存 1,406 组、5,624 条探针。共记录 3 条格式异常输出，按完全相同请求重试；未根据金标或语义效果筛选、改写、裁剪探针。恢复执行和适配器测试共 51 项通过。

探针机械审计：0 条逐字复述原查询；5 条与原句 cosine > 0.85；7 条超过 30 词；1,219 条命中主包的显式否定词正则；0 条包含非拉丁字母；探针 BGE 未知词元及截断均为 0。该统计不是人工语义正确性判定，不能据此声称全部探针都是有效反例。最终 1,298 份缓存均为 `finish_reason=stop`；其中 29 份响应包含上游返回的 reasoning_content，虽然所有请求均发送 `reasoning_effort=none`。未更改请求参数以处理此服务行为。

耗时：最后一次恢复约 20.07 分钟，其中探针生成约 16.96 分钟、探针编码 77.76 秒、完整打分 99.35 秒；各次实际运行累计约 54.12 分钟，含失败与恢复，未计中间人工排查间隔。首次启动约北京时间 16:36，完整任务约 17:37 完成。复用的历史全库编码成本不计入本次，因此不能与本地历史 SparseCL 的总耗时直接比较。

累计已记录 prompt tokens 781,387、completion tokens 182,099，总计 963,486。日志记录 460 次 HTTP 429、26 次网络异常；有 17 份成功响应的用量从中断前缓存恢复，单次延迟未保存。停止进程时尚未返回的请求，其上游是否计费未知，因此上述为可核验的已记录用量，不是账户账单总额。

原始输出目录：`/home/dzp/paper/memory-comparison-retrieval/runs/experiments/arguana-bipolar-vs-sparsecl-bge-20261120-163016-r3`。

关键文件为 `report.json`、`raw_top10.json`、`per_query_metrics.json`、`scores.npy`、`anchored_probes.json`、`anchored_probe_audit.json`、`api_events_all_attempts.json`、`resume_info.json`、`independent_verification.json` 和 `completion.json`。完整分数、排名、提示词、探针与向量均可追溯。历史 SparseCL 记录位于 `/home/dzp/paper/baselines/SparseCL/test_results/repro_arguana`。
