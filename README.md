# 写入时的记忆冲突检索

项目主方法是 **四维失效/冲突探针 + 旧记忆库双极图扩散**。给定新记忆及对应旧库，LLM 只读新记忆，直接以原句为既定事实，生成四条与之矛盾或被其使之失效的旧状态探针。输出仅含 `violation_probes`，不再生成 `new_state_anchor`。冻结 BGE 编码原文和探针，旧库内构建正余弦有向 k-NN 图，按行归一化；探针最大相似度提供正种子，原句相似度超过探针相似度的部分提供相容负源。默认 `k=5`、Top-5 种子、`α=0.2`、`β=0.3`，使用 `P@z` 单步扩散。[提示词与公式](memory_retrieval/bipolar_graph.py)

当前英文提示词要求四条探针直接与原句矛盾或失效，禁止复述原句，并保持肯定陈述、同一语种、每条不超过 30 词。UTF-8 SHA256 为 `a38b198018c62846780914f1ef90fe2461ee1554c421d42996f54525f93217bf`。图扩散公式、参数和 `anchored_*` CLI 臂名保持不变。旧 **OCSP-QR** 及其三探针消融保留原提示词、QR 和评分逻辑，仍可按名称运行，默认 `γ=0.65`。[旧方法说明](docs/ocsp_qr_method_summary.md) · [研究问题](docs/research_problem_and_objectives.md)

| 实验参数 | 方法 |
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

在本机从仓库根目录打开 PowerShell。下面的命令运行 Austin 单场景的新主方法与六个基线。当前机器的 WSL 启动脚本使用已有 CPU 模型环境，模型文件来自同级的 `../baselines/SparseCL`。新方法从 `.env` 读取 `LLM_BASE_URL`、`LLM_API_KEY` 和网络超时/重试配置；模型固定 `deepseek-v4.1-flash`、温度 0、JSON 模式、`reasoning_effort=none`。旧方法继续遵循 `.env` 的 `LLM_*` 设置。

```powershell
$scene = 'data/stale_scenarios/augmented/scenarios/T1/7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f'
wsl -d Ubuntu-22.04 -u dzp -- bash scripts/run_wsl.sh --dataset $scene --arms all
```

`all` 运行新主方法与六个基线；同时运行两项新消融可传 `--arms all anchored_probes_only anchored_pos_diffusion`。新探针默认 4 并发，与 BGE/SparseCL 本地编码重叠执行；所有新臂共享同一批四条探针和同一个 BGE 实例，旧库/查询各编码一次。

新方法的可选参数为 `--graph-k 5 --graph-seeds 5 --graph-alpha 0.2 --graph-beta 0.3 --probe-workers 4`。复现锁定协议时保持默认值，Dev/Test 共用设置。复用探针时，先复制历史 `probe_cache.sqlite3`，再通过 `--cache` 指定副本；只有提示词、模型与请求正文完全相同才会命中缓存。去掉锚点输出后的探针需使用新缓存生成。旧 Austin 缓存需配合 `--arms ocsp_qr probes_only_no_ortho` 使用。

全量对照可以把 `--dataset` 换成版本根目录。按 ID 选多个场景也可以直接从 Windows 运行 BM25。

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --dataset data/stale_scenarios/augmented --scenario-ids 7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f 01f694b2-c555-4b6e-8ca0-5f11210e2227 --arms bm25
```

在配好模型依赖的 Python 3.11 环境中，执行 `python -m pip install -e .` 后即可使用 `memory-retrieval --dataset <目录>` 或 `python -m memory_retrieval --dataset <目录>`。六个基线均使用本地计算。BGE 和 SparseCL 的本机模型栈为 CPU PyTorch、Transformers、NumPy 和 FAISS，权重版本、文件哈希和各检查点长度见[模型配置](memory_retrieval/model_lock.json)。模型放在其他位置时传 `--model-root`。早期使用向量 API 的 OCSP 对照可以传 `--arms ocsp_qr --embedding-backend api`，并设置 `EMBEDDING_*`；三个新臂固定使用本地 BGE。[环境变量示例](.env.example)

在安装 `pytest`、`pytrec_eval` 的 Python 环境中可运行全部单测及历史 Test 的零推理一致性检查（不调用 LLM、不加载编码器、不改写历史目录）：

```bash
python -m pytest -q
python scripts/verify_anchored_graph_consistency.py
```

一致性检查读取 `runs/experiments/stale-anchored-dev-test-bge-20261120/test` 保存的探针与 BGE 向量，断言完整排名一致、T1/T2 NDCG@10 误差小于 `1e-12`，并用 `pytrec_eval` 复核。该检查只重放历史评分，分别记录历史和当前提示词哈希，不把历史探针当作修复后生成的探针。

## 数据与输出

输入目录含 `memories.jsonl`、`queries.jsonl` 和 `gold.jsonl`。每条查询按 `scenario_id` 选择自己的完整旧库，选择发生在 BM25 统计、SparseCL 初召回和其他方法打分之前。没有分场景字段的数据集共用一份旧库。金标通过 `query_id` 对应查询，`conflict_old_memory_ids` 标出应检索的旧记忆 ID。

- STALE 原版与扩增版各有 400 个场景。索引列出新记忆、场景 ID、目录和旧库条数。单场景目录可以直接作为输入。[原版场景索引](data/stale_scenarios/base/scenarios/index.json) · [扩增版场景索引](data/stale_scenarios/augmented/scenarios/index.json)
- 重建 STALE 时依次使用原版转换、扩增和场景导出脚本。扩增项的来源及冲突或过时标签保存在金标内。[转换](scripts/convert_stale_scenarios.py) · [扩增](scripts/augment_stale_scenarios.py) · [扩增提示词](scripts/stale_augmentation_prompts.py) · [导出](scripts/export_stale_scenarios.py)
- 每次运行新建一个结果目录，默认位于 `runs/experiments/`，也可通过 `--output` 指定。`report.zh.md` 和 `report.json` 输出全集、以及存在 `source_ref.type` 时的 T1/T2 分组 Top-3/5/10 命中、NDCG@10 和 `stale_native` 原生金标命中；`arms/` 保存完整排名。新探针和格式/漂移标记在 `anchored_probes.json`、`anchored_probe_audit.json`，旧 OCSP 明细仍在 `per_query_details.json`。余弦 >0.85、复述及格式标记只用于审计，不触发筛选或重生成；机械标记不能证明严格矛盾。报告保留 API 原始用量，发送 `reasoning_effort=none` 不保证服务端推理用量为零。当标注未覆盖整个旧库时，命中和 NDCG 均以已有金标计算。

## 已保留的实验

| 实验 | 结果 |
|---|---|
| 正交投影早期试验与零样本 QR | [初版](runs/experiments/ocsp-implicit-pilot-20261118/report.zh.md) · [OCSP-QR](runs/experiments/ocsp-qr-zeroshot-20261118/report.zh.md) |
| 11 查询英文基线对照 | [BGE 等通用基线](runs/experiments/single-pair-english-general-bge-20261118/summary.md) · [三个 SparseCL 检查点](runs/experiments/single-pair-english-sparsecl-20261118/summary.md) |
| 15 查询实验与数据修订 | [英文试验](runs/experiments/single-pair-15q-english-20261119/comparison.zh.md) · [两版对比](runs/experiments/single-pair-bge-v2-and-15q-20261119/comparison.zh.md) · [定向修订](reports/single_pair_implicit_15q_revision_20261119/targeted_followup.zh.md) · [修订前后排名](runs/experiments/single-pair-15q-targeted-20261119/comparison.json) · [冻结 BGE 复测](docs/single_pair_implicit_15q_bge_experiment.md) |
| ValidMem v7，181 查询 | [实验报告](docs/validmem_v7_bge_ocsp_experiment.md) |
| STALE 扩增版 Austin，1 查询 | [实验报告与金标名次](runs/experiments/stale-augmented-t1-austin-1q-20261119/summary.md) |

`data/` 和 `runs/` 沿用仓库已有的本地存储规则。实验结果、输入快照和探针缓存都可以在本机对应目录查看。
