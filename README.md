# 写入时的隐式冲突与状态失效检索

项目主方法是 **四维失效/冲突探针 + 旧记忆库双极图扩散**。写入一条新记忆时，LLM 只读这条原句，把它当作既定事实，生成四条可能与它矛盾或因它而失效的旧陈述，输出字段只有 `violation_probes`。冻结 BGE 编码新记忆、旧记忆和探针。每条旧记忆先取与四条探针的最大余弦相似度作为初分，再从自己选出的相似邻居汇聚一步信号。初分最高的 5 条提供正种子，原句相似度超过探针初分的部分提供负向惩罚。默认 `k=5`、`α=0.2`、`β=0.3`。最终按分数返回候选，供下游核查。[方法说明](docs/anchored_bipolar_diffusion_method_summary.md) · [提示词与公式](memory_retrieval/bipolar_graph.py)

英文提示词要求每条探针与原句语种相同，以不超过 30 词的肯定陈述表达矛盾或失效的旧事实，并避免复述原句。提示词 UTF-8 SHA256 为 `a38b198018c62846780914f1ef90fe2461ee1554c421d42996f54525f93217bf`。旧方法 `ocsp_qr` 使用相容改述、三条违例探针和 QR 正交投影，默认 `γ=0.65`。[旧方法实现](memory_retrieval/ocsp.py) · [研究问题](docs/research_problem_and_objectives.md)

| `--arms` 取值 | 方法 |
|---|---|
| `anchored_bipolar_diffusion` | 新主方法，默认运行 |
| `anchored_probes_only` | 四条探针最大余弦，去掉图扩散 |
| `anchored_pos_diffusion` | 仅正源扩散，去掉相容负源及其传播 |
| `ocsp_qr` | 保留的旧方法，QR 正交投影 |
| `bm25` | BM25，k1 = 1.2，b = 0.75 |
| `dense` | BGE-base-en-v1.5 原文余弦 |
| `hybrid` | BM25 和 Dense 的等权 RRF，常数 60 |
| `sparsecl_arguana` | SparseCL ArguAna，α = 0.18 |
| `sparsecl_msmarco` | SparseCL MS MARCO，α = 2.19 |
| `sparsecl_hotpotqa` | SparseCL HotpotQA，α = 4.82 |
| `probes_only_no_ortho` | 旧方法消融，使用原三条违例探针最大余弦 |

## 运行实验

在服务器的仓库根目录，以下命令运行 Austin 单场景的主方法与六个基线。主包默认从同级 `../baselines/SparseCL` 加载冻结权重。新方法从 `.env` 读取 `LLM_BASE_URL`、`LLM_API_KEY` 和超时、重试配置，固定使用 `deepseek-v4.1-flash`、温度 0、JSON 模式和 `reasoning_effort=none`。旧 OCSP-QR 使用 `.env` 的 `LLM_*` 设置。

```bash
scene='data/stale_scenarios/augmented/scenarios/T1/7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f'
.venv/bin/python -m memory_retrieval --dataset "$scene" --arms all
```

本机 PowerShell 可从仓库根目录通过 WSL 运行同一场景。

```powershell
$scene = 'data/stale_scenarios/augmented/scenarios/T1/7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f'
wsl -d Ubuntu-22.04 -u dzp -- bash scripts/run_wsl.sh --dataset $scene --arms all
```

`all` 运行主方法和六个基线。要同时运行两项消融，传 `--arms all anchored_probes_only anchored_pos_diffusion`。探针请求可与本地 BGE、SparseCL 编码重叠执行。三个 `anchored_*` 臂共享同一批探针、同一个 BGE 编码器及同一张场景图。

CLI 默认参数为 `--graph-k 5 --graph-seeds 5 --graph-alpha 0.2 --graph-beta 0.3 --probe-workers 4`。服务器 60 场景 BGE 实验将探针并发设为 16，图参数沿用默认值。每次运行默认在新结果目录创建 `probe_cache.sqlite3`。要复用已有响应，可将缓存复制后用 `--cache` 指向副本。缓存按请求 URL 和完整请求正文匹配。[60 场景 BGE 实验](reports/stale_unseen_60scene_bipolar_seed20261122_experiment.md)

全量对照可以把 `--dataset` 换成版本根目录。按 ID 选多个场景也可以直接从 Windows 运行 BM25。

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --dataset data/stale_scenarios/augmented --scenario-ids 7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f 01f694b2-c555-4b6e-8ca0-5f11210e2227 --arms bm25
```

在配好模型依赖的 Python 3.11 环境中，执行 `python -m pip install -e .` 后即可使用 `memory-retrieval --dataset <目录>` 或 `python -m memory_retrieval --dataset <目录>`。默认六个基线均在本地计算。BGE 和 SparseCL 使用 CPU PyTorch、Transformers、NumPy 和 FAISS，权重在其他位置时传 `--model-root`。旧 OCSP 对照支持 `--arms ocsp_qr --embedding-backend api` 和 `EMBEDDING_*` 配置。主包 CLI 的三个 `anchored_*` 臂使用本地 BGE。Qwen 底座对照通过独立实验脚本复用已生成的探针，以 Qwen 向量重新构图，调用同一套评分函数。[模型配置](memory_retrieval/model_lock.json) · [环境变量示例](.env.example) · [Qwen 对照报告](reports/stale_unseen_60scene_qwen_emb_seed20261122_experiment.md)

安装 `pytest`、`pytrec_eval` 后，可运行单元测试并用历史 Test 保存的向量复核评分。

```bash
python -m pytest -q
python scripts/verify_anchored_graph_consistency.py
```

一致性检查读取 `runs/experiments/stale-anchored-dev-test-bge-20261120/test`，用保存的探针向量重算仅探针和完整方法的排名。它校验完整排名一致、T1/T2 NDCG@10 误差小于 `1e-12`，并用 `pytrec_eval` 复核，同时分别记录历史和当前提示词哈希。

## 数据与输出

`docs/` 存放研究问题、方法设计和数据构建说明，`reports/` 存放实验结果与案例分析。每次运行的排名、向量、日志和原始报告保存在 `runs/experiments/`。文件组织约定见 [AGENTS.md](AGENTS.md)。

输入目录含 `memories.jsonl`、`queries.jsonl` 和 `gold.jsonl`。每条查询按 `scenario_id` 选择自己的完整旧库，选择发生在 BM25 统计、SparseCL 初召回和其他方法打分之前。没有分场景字段的数据集共用一份旧库。金标通过 `query_id` 对应查询，`conflict_old_memory_ids` 标出应检索的旧记忆 ID。

- STALE 原版与扩增版各有 400 个场景。索引列出新记忆、场景 ID、目录和旧库条数。单场景目录可以直接作为输入。[原版场景索引](data/stale_scenarios/base/scenarios/index.json) · [扩增版场景索引](data/stale_scenarios/augmented/scenarios/index.json)
- 重建 STALE 时依次使用原版转换、扩增和场景导出脚本。扩增项的来源及冲突或过时标签保存在金标内。[转换](scripts/convert_stale_scenarios.py) · [扩增](scripts/augment_stale_scenarios.py) · [扩增提示词](scripts/stale_augmentation_prompts.py) · [导出](scripts/export_stale_scenarios.py)
- STALE V2 收录写入前每场会话的所有合格用户句子，仅作为旧库背景。离线继承 V1 全部记忆、合成标签和原有待定项，移除 74 个历史目标；金标为官方 400 个加合成 1,818 个，共 2,218 个。V2 构建不调用模型，命令和验收见[构建说明](docs/stale_v2_server_run.zh.md)。
- 主包 CLI 每次新建一个结果目录，默认位于 `runs/experiments/`，也可通过 `--output` 指定。`report.zh.md` 和 `report.json` 给出总体、以及有 `source_ref.type` 时的 T1/T2 分组 Top-3/5/10 命中数、NDCG@10 和原生 `stale_native` 在 Top-10 的命中数。指标按已有金标计算。
- `arms/` 保存各方法实际返回的排名。主方法及其两项消融对场景内全部旧记忆排序。三个 SparseCL 基线先用 BGE 初筛最多 1,000 条，再在这些候选中排序。[SparseCL 候选与排序实现](memory_retrieval/experiment.py)
- `anchored_probes.json` 保存探针，`anchored_probe_audit.json` 记录完全复述、显式否定词、超长及与原句余弦大于 0.85 等标记。这些探针原样参与评分。响应被截断、JSON 无法解析或结构不符时，主包报错并停止本次运行。旧 OCSP 的逐查询明细在 `per_query_details.json`。报告保留服务端返回的实际用量。[探针校验与审计](memory_retrieval/bipolar_graph.py)

## 代码入口

| 入口 | 用途 |
|---|---|
| `python -m memory_retrieval`、`scripts/run_experiment.py` | 当前主方法、基线和消融的统一入口 |
| `scripts/convert_stale_scenarios.py`、`augment_stale_scenarios.py`、`export_stale_scenarios.py`、`audit_stale_v2.py` | STALE 抽取、合成或继承、场景导出与 V2 验收 |
| `scripts/run_stale_unseen_60scene.py`、`run_stale_qwen_embedding.py`、`run_stale_llm_rerank.py`、`run_stale_native_bank_audit.py` | 固定 60 场景的主方法、编码器、重排和原生旧库对照 |
| `scripts/run_arguana_bipolar_comparison.py` | ArguAna 对照 |
| `scripts/verify_anchored_graph_consistency.py` | 用已保存向量核对图打分与指标 |

`run_orthogonal_conflict_retrieval.py` 和 `run_revised_pilot_baselines.py` 是历史 OCSP 与基线复现实验使用的兼容入口。早期四探针、图参数搜索和旧锚点 Dev/Test 的独立探索脚本已清理，当前算法与消融通过统一入口运行。STALE V2 的构建及验收测试集中在 `tests/test_stale_v2.py`。

## 已保留的实验

表中四项 60 场景实验共用 `seed=20261122` 选出的 30 个 T1 和 30 个 T2 场景。扩增旧库的 340 个目标由 60 条官方 `M_old`、273 条模型复核的合成目标、7 条模型补标的原有历史记忆组成。原生旧库对照仅按 60 条官方 `M_old` 评测。

| 实验 | 结果 |
|---|---|
| STALE 60 场景，BGE 主方法、两项消融与六个基线 | [九方法对照](reports/stale_unseen_60scene_bipolar_seed20261122_experiment.md) |
| 同批场景，BGE Top-20 加单次 LLM 重排 | [召回、排序与调用成本](reports/stale_unseen_60scene_llm_rerank_comparison_seed20261122.md) |
| 同批场景，Qwen 替换 BGE，复用 240 条探针 | [编码器对照](reports/stale_unseen_60scene_qwen_emb_seed20261122_experiment.md) |
| 同批场景，原生旧库与扩增库的模块分析 | [官方目标评测与扩散收益、误伤](reports/stale_unseen_60scene_native_bank_and_module_audit_seed20261122.md) |
| STALE 10 个未见场景，四条探针与双极图扩散 | [10 场景对照](reports/stale_unseen_10scene_query_only_seed20261121_experiment.md) |
| 15 条英文查询，直接从原句生成四条探针 | [主方法与两项消融](reports/single_pair_implicit_15q_query_only_experiment.md) |
| ArguAna 完整测试集，与训练过的 SparseCL 对照 | [反论点检索结果](reports/arguana_bipolar_vs_sparsecl_server_20261120.md) |
| 正交投影早期试验与零样本 QR | [初版](runs/experiments/ocsp-implicit-pilot-20261118/report.zh.md) · [OCSP-QR](runs/experiments/ocsp-qr-zeroshot-20261118/report.zh.md) |
| 11 查询英文基线对照 | [BGE 等通用基线](runs/experiments/single-pair-english-general-bge-20261118/summary.md) · [三个 SparseCL 检查点](runs/experiments/single-pair-english-sparsecl-20261118/summary.md) |
| 15 查询实验与数据修订 | [英文试验](runs/experiments/single-pair-15q-english-20261119/comparison.zh.md) · [两版对比](runs/experiments/single-pair-bge-v2-and-15q-20261119/comparison.zh.md) · [定向修订](reports/single_pair_implicit_15q_revision_20261119/targeted_followup.zh.md) · [修订前后排名](runs/experiments/single-pair-15q-targeted-20261119/comparison.json) · [冻结 BGE 复测](reports/single_pair_implicit_15q_bge_experiment.md) |
| ValidMem v7，181 查询 | [实验报告](reports/validmem_v7_bge_ocsp_experiment.md) |
| STALE 扩增版 Austin，1 查询 | [实验报告与金标名次](runs/experiments/stale-augmented-t1-austin-1q-20261119/summary.md) |

`data/` 和 `runs/` 在 Git 忽略列表中。上述 60 场景实验的完整排名、输入快照和缓存保存在服务器项目的 `runs/experiments/` 下，各报告列出了对应目录。
