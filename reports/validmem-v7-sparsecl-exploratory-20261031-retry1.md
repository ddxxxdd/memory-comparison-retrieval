# ValidMem v7 原生 baseline 预算比较

- 运行：`validmem-v7-sparsecl-exploratory-20261031-retry1`（评估标记：`exploratory`）
- 数据：`validmem_v7`；181 条查询、301 个冲突金标；旧记忆 1225 条。
- SparseCL 保存 min(1000, 旧库条数) 初召回及完整融合排序；预算网格：1, 2, 3, 5, 10, 20, 30, 50。
- proposed 动态阈值：未运行 proposed。

SparseCL 保留 Top-1000 初召回后的完整融合排序。各行使用同一查询、旧库与冲突金标。 SparseCL 按 cosine + alpha × Hoyer 融合重排；没有增加主体/事项筛选或 LLM 判断。

## 召回—候选负担曲线数据

SparseCL 的预算点从完整融合排序截取，不额外调用模型。 recall_micro 按冲突配对汇总，recall_macro 对每条查询等权；完整找全率是所有冲突金标均被找回的查询占比。NDCG@10 对空冲突金标查询记 0，并保留在查询平均分母；召回等统计包含全部查询。

| 方法 | 变体 | 设置 | NDCG@10 | micro 召回 | macro 召回 | 完整找全率 | 候选纯度 | 候选均值 | p95 | max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparsecl | bge_initial_recall_at_1000 | initial-top-1000 | 0.918 | 1.000 | 1.000 | 1.000 | 0.002 | 1000.00 | 1000.0 | 1000 |
| sparsecl | arguana_alpha_0.18 | top-1 | 0.701 | 0.502 | 0.665 | 0.536 | 0.834 | 1.00 | 1.0 | 1 |
| sparsecl | arguana_alpha_0.18 | top-2 | 0.843 | 0.711 | 0.857 | 0.801 | 0.591 | 2.00 | 2.0 | 2 |
| sparsecl | arguana_alpha_0.18 | top-3 | 0.882 | 0.787 | 0.923 | 0.884 | 0.436 | 3.00 | 3.0 | 3 |
| sparsecl | arguana_alpha_0.18 | top-5 | 0.907 | 0.857 | 0.971 | 0.950 | 0.285 | 5.00 | 5.0 | 5 |
| sparsecl | arguana_alpha_0.18 | top-10 | 0.917 | 0.914 | 0.990 | 0.978 | 0.152 | 10.00 | 10.0 | 10 |
| sparsecl | arguana_alpha_0.18 | top-20 | 0.917 | 0.963 | 0.995 | 0.983 | 0.080 | 20.00 | 20.0 | 20 |
| sparsecl | arguana_alpha_0.18 | top-30 | 0.917 | 0.990 | 0.999 | 0.989 | 0.055 | 30.00 | 30.0 | 30 |
| sparsecl | arguana_alpha_0.18 | top-50 | 0.917 | 1.000 | 1.000 | 1.000 | 0.033 | 50.00 | 50.0 | 50 |
| sparsecl | bge_cosine | top-1 | 0.705 | 0.505 | 0.668 | 0.536 | 0.840 | 1.00 | 1.0 | 1 |
| sparsecl | bge_cosine | top-2 | 0.844 | 0.711 | 0.857 | 0.801 | 0.591 | 2.00 | 2.0 | 2 |
| sparsecl | bge_cosine | top-3 | 0.883 | 0.787 | 0.923 | 0.884 | 0.436 | 3.00 | 3.0 | 3 |
| sparsecl | bge_cosine | top-5 | 0.908 | 0.857 | 0.971 | 0.950 | 0.285 | 5.00 | 5.0 | 5 |
| sparsecl | bge_cosine | top-10 | 0.918 | 0.914 | 0.990 | 0.978 | 0.152 | 10.00 | 10.0 | 10 |
| sparsecl | bge_cosine | top-20 | 0.918 | 0.963 | 0.995 | 0.983 | 0.080 | 20.00 | 20.0 | 20 |
| sparsecl | bge_cosine | top-30 | 0.918 | 0.990 | 0.999 | 0.989 | 0.055 | 30.00 | 30.0 | 30 |
| sparsecl | bge_cosine | top-50 | 0.918 | 1.000 | 1.000 | 1.000 | 0.033 | 50.00 | 50.0 | 50 |

## 按场景/模板族分组 Bootstrap 95% 区间

未计算：运行输入没有经核验的 `query_groups.json`。查询可能共享场景或模板，不能假设每条查询相互独立。

## 运行成本记录

| 方法 | 查询时延口径 | LLM 网络尝试 | 嵌入网络尝试 | 缓存命中 | LLM prompt/completion token | 已知总 token | 查询 p50/p95 ms | 可作方法间时延比较 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| sparsecl | batch_amortized_per_query | 0 | 0 | 0 | 0 / 0 | 0 | 27.6 / 32.2 | 不与 API baseline 横向比较 |

### SparseCL 计算成本

| 成本阶段 | 秒 |
|---|---:|
| 加载两套模型 | 3.118 |
| 离线 BGE 旧库编码 | 11.619 |
| 离线 SparseCL 旧库编码 | 12.199 |
| 离线 HNSW 建索引 | 0.028 |
| 在线 BGE 查询编码 | 2.530 |
| 在线 SparseCL 查询编码 | 1.854 |
| 在线 HNSW 搜索 | 0.177 |
| 在线 Hoyer 打分与融合重排 | 1.218 |
| dev alpha 搜索 | 0.000 |
| SparseCL 各查询总时延之和 | 5.779 |

旧库缓存命中：`False`；索引缓存：`/mnt/c/Users/dzp/project/paper/memory-comparison-retrieval/runs/experiments/validmem-v7-sparsecl-exploratory-20261031-retry1/index/sparsecl/d5cddb65b0481b8066a0e8fb55e8ab3d7101f79bad8da022c371414783e517fb`。查询分阶段时延是按编码批次分摊的每条查询时间；成本表中的分阶段总秒数是实测值。

## 离线索引构建成本

- 本次未构建项目描述索引或 BM25 索引；SparseCL HNSW 成本见上表。
- SparseCL 双编码与 HNSW：23.845 秒；缓存命中时该值为 0，成本明细见 SparseCL 计算成本表。

## 解释边界

- 这 181 条查询已用于数据修复、提示词和流程诊断，结果属于探索性分析；不能把其中的最优 top-k 或阈值当作独立测试结果。
- 正式测试应使用新建且人工核验的查询/金标，按场景或模板族整组留出；在开发集选定 top-k 与动态阈值后冻结设置。测试集需与当前 v7 查询 ID 和正文完全不重叠。
- 本报告的多 top-k 指标是对已存排序的离线截断，适合比较召回与候选数；每个预算点不是一次独立运行的时延测量。
- 时延来自单次运行，未做重复运行或置信区间估计；即使新运行缓存隔离，也只作成本诊断。
- 本次没有运行依赖项目描述索引的方法臂，因此未构建该索引。

逐点机器可读数据见同名 `.json` 和 `.csv` 文件。
