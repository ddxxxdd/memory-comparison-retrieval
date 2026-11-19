# 写入时的记忆冲突检索

项目主方法是 **OCSP-QR**。给定一条新记忆和对应的旧记忆库，检索器把可能与新记忆冲突的旧记录排在前面。旧记忆直接编码原文。写入时，模型只读新记忆，生成两条相容改述和三条违例探针。OCSP-QR 用新记忆与相容改述构造 QR 正交基，从每条探针中减去相容方向，再结合原始探针相似度对同库旧记忆打分。正式对照使用冻结 BGE 和 `γ=0.65`。[方法说明](docs/ocsp_qr_method_summary.md) · [研究问题](docs/research_problem_and_objectives.md) · [提示词与评分实现](memory_retrieval/ocsp.py)

| 实验参数 | 方法 |
|---|---|
| `ocsp_qr` | 项目主方法，默认运行 |
| `bm25` | BM25，k1 = 1.2，b = 0.75 |
| `dense` | BGE-base-en-v1.5 原文余弦 |
| `hybrid` | BM25 和 Dense 的等权 RRF，常数 60 |
| `sparsecl_arguana` | SparseCL ArguAna，α = 0.18 |
| `sparsecl_msmarco` | SparseCL MS MARCO，α = 2.19 |
| `sparsecl_hotpotqa` | SparseCL HotpotQA，α = 4.82 |
| `probes_only_no_ortho` | 消融，仅按违例探针与旧记忆的最大余弦排序 |

## 运行实验

在本机从仓库根目录打开 PowerShell。下面的命令运行 Austin 单场景的主方法与六个基线。当前机器的 WSL 启动脚本使用已有 CPU 模型环境，模型文件来自同级的 `../baselines/SparseCL`。新查询使用项目 `.env` 的 `LLM_*` 配置生成探针。

```powershell
$scene = 'data/stale_scenarios/augmented/scenarios/T1/7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f'
wsl -d Ubuntu-22.04 -u dzp -- bash scripts/run_wsl.sh --dataset $scene --arms all
```

复现已保存的 Austin 探针时，先复制历史缓存，再把副本传给 `--cache`。需要同时看消融时传 `--arms all probes_only_no_ortho`。全量对照可以把 `--dataset` 换成版本根目录。按 ID 选多个场景也可以直接从 Windows 运行 BM25。

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --dataset data/stale_scenarios/augmented --scenario-ids 7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f 01f694b2-c555-4b6e-8ca0-5f11210e2227 --arms bm25
```

在配好模型依赖的 Python 3.11 环境中，执行 `python -m pip install -e .` 后即可使用 `memory-retrieval --dataset <目录>` 或 `python -m memory_retrieval --dataset <目录>`。六个基线均使用本地计算。BGE 和 SparseCL 的本机模型栈为 CPU PyTorch、Transformers、NumPy 和 FAISS，权重版本、文件哈希和各检查点长度见[模型配置](memory_retrieval/model_lock.json)。模型放在其他位置时传 `--model-root`。早期使用向量 API 的 OCSP 对照可以传 `--embedding-backend api`，并设置 `EMBEDDING_*`。[环境变量示例](.env.example)

## 数据与输出

输入目录含 `memories.jsonl`、`queries.jsonl` 和 `gold.jsonl`。每条查询按 `scenario_id` 选择自己的完整旧库，选择发生在 BM25 统计、SparseCL 初召回和其他方法打分之前。没有分场景字段的数据集共用一份旧库。金标通过 `query_id` 对应查询，`conflict_old_memory_ids` 标出应检索的旧记忆 ID。

- STALE 原版与扩增版各有 400 个场景。索引列出新记忆、场景 ID、目录和旧库条数。单场景目录可以直接作为输入。[原版场景索引](data/stale_scenarios/base/scenarios/index.json) · [扩增版场景索引](data/stale_scenarios/augmented/scenarios/index.json)
- 重建 STALE 时依次使用原版转换、扩增和场景导出脚本。扩增项的来源及冲突或过时标签保存在金标内。[转换](scripts/convert_stale_scenarios.py) · [扩增](scripts/augment_stale_scenarios.py) · [扩增提示词](scripts/stale_augmentation_prompts.py) · [导出](scripts/export_stale_scenarios.py)
- 每次运行新建一个结果目录，默认位于 `runs/experiments/`，也可通过 `--output` 指定。查看 `report.zh.md` 可比较 Top-10 命中数与 NDCG@10，`report.json` 含各预算的召回率和逐条金标名次，`arms/` 保存方法实际返回的完整排名。探针及其对应排名在 `per_query_details.json`。当标注未覆盖整个旧库时，命中和 NDCG 均以已有金标计算。

## 已保留的实验

| 实验 | 结果 |
|---|---|
| 正交投影早期试验与零样本 QR | [初版](runs/experiments/ocsp-implicit-pilot-20261118/report.zh.md) · [OCSP-QR](runs/experiments/ocsp-qr-zeroshot-20261118/report.zh.md) |
| 11 查询英文基线对照 | [BGE 等通用基线](runs/experiments/single-pair-english-general-bge-20261118/summary.md) · [三个 SparseCL 检查点](runs/experiments/single-pair-english-sparsecl-20261118/summary.md) |
| 15 查询实验与数据修订 | [英文试验](runs/experiments/single-pair-15q-english-20261119/comparison.zh.md) · [两版对比](runs/experiments/single-pair-bge-v2-and-15q-20261119/comparison.zh.md) · [定向修订](reports/single_pair_implicit_15q_revision_20261119/targeted_followup.zh.md) · [修订前后排名](runs/experiments/single-pair-15q-targeted-20261119/comparison.json) · [冻结 BGE 复测](docs/single_pair_implicit_15q_bge_experiment.md) |
| ValidMem v7，181 查询 | [实验报告](docs/validmem_v7_bge_ocsp_experiment.md) |
| STALE 扩增版 Austin，1 查询 | [实验报告与金标名次](runs/experiments/stale-augmented-t1-austin-1q-20261119/summary.md) |

`data/` 和 `runs/` 沿用仓库已有的本地存储规则。实验结果、输入快照和探针缓存都可以在本机对应目录查看。
