# 可比较记忆检索：先选择已有描述，再生成未覆盖事项

此项目实现已定方法的首版框架：固定旧库、每条新记忆独立查询、完整目录选择、描述向量粗召回、假设句细筛、固定逐对冲突判定。LLM 和向量均通过 `.env` 中独立配置的 OpenAI 兼容 API 接入。没有硬编码真实模型、服务地址或 Key。

当前主评估集为 `data/validmem_v7/`，用于同一冻结记忆库上的冲突候选检索比较；主实验协议见第 5 节。`data/dialogue_nli_mixed_v1/` 是另一条早期来源构造流程，不作为 ValidMem v7 的默认数据集。

历史的 `data/combined_sources_v1/` 存在候选遗漏和新旧边界问题，不作为当前默认数据集。它早期写入的 `fully_reviewed=true` 不能证明金标完整，审计说明见工作区 `research_notes/dataset_audit_2026-10-15.md`。

## 1. 安装与配置

在 PowerShell 中运行：

```powershell
cd C:\Users\dzp\project\paper\memory-comparison-retrieval
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

根目录已准备好空白 `.env`，填写：

```dotenv
LLM_BASE_URL=https://your-llm-provider.example/v1
LLM_API_KEY=你的Key
LLM_MODEL=你的模型名称

EMBEDDING_BASE_URL=https://your-embedding-provider.example/v1
EMBEDDING_API_KEY=你的Key
EMBEDDING_MODEL=你的多语向量模型名称
```

上面是地址示意，不是真实服务。向量模型应支持数据原文的语言；可填服务提供的 BGE-M3 或其他冻结多语编码器。两个模型可来自不同服务。不在代码中下载或选择模型，不使用聊天模型代替向量模型。

服务前缀自动追加 `/chat/completions` 或 `/embeddings`；填完整端点也可以。当前仅支持这两个兼容端点，不是任意厂商的专有协议。服务不支持 `response_format` 时，将 `LLM_JSON_MODE=false`；不支持 `temperature` 时将 `LLM_TEMPERATURE=` 留空。两种情况仍执行程序端 JSON 和编号校验。

`.env.example` 是配置参考。主方法粗筛和细筛都固定保留余弦相似度 `>= 0.7` 的全部候选，数量可变；该阈值不通过环境变量或命令行调整。`COARSE_K=50`、`FINAL_K=10` 仅供 top-k 诊断方法使用。进程环境变量优先于 `.env`；也可在子命令前用 `--env 路径` 指定文件。缓存路径默认相对于项目根目录，其他 CLI 路径相对于当前目录。

配置完成后，先运行真实接口连通性测试：

```powershell
.\.venv\Scripts\python.exe check_api_connection.py
# 可选：指定环境文件或单次请求超时
.\.venv\Scripts\python.exe check_api_connection.py --env .env --timeout 60
```

脚本分别显示 LLM、向量模型的 `[成功]` 或 `[失败]`、耗时和失败原因；向量成功时显示维度。即使一项失败，也会继续测试另一项。两项都成功退出码为 `0`，任一失败为 `1`，PowerShell 可通过 `$LASTEXITCODE` 查看。配置文件无法读取时明确报告两项均未执行。

每次使用临时缓存，发送真实 API 请求，不读取已有响应，不输出 Key。LLM 测试使用与主流程相同的 JSON mode 和 temperature 设置，校验 JSON 回复；网络错误不重试，JSON 输出无效时沿用主流程的一次格式修复。只发送固定测试文本，不上传旧记忆库；此测试验证接口兼容性，不验证方法的语义效果。

## 2. 运行完整流程

以下命令可直接使用安装后的虚拟环境 Python；无需激活虚拟环境。

```powershell
# 不调用 API，检查示例与两层标签
.\.venv\Scripts\python.exe -m memory_retrieval validate-data --memories examples/memories.jsonl --queries examples/queries.jsonl --gold examples/gold.jsonl

# 一次离线建库，默认批式复用：每批 16 条共用一次请求，先引用已有描述，批内同一事项收敛到同一份文本
.\.venv\Scripts\python.exe -m memory_retrieval build --input examples/memories.jsonl --adapter text --output artifacts/text-index

# 每条新记忆先选择旧描述，再生成未覆盖事项；运行完整流程和下游判定
.\.venv\Scripts\python.exe -m memory_retrieval query --index artifacts/text-index --queries examples/queries.jsonl --output runs/proposed.jsonl --judge

# 评测，不调用 API
.\.venv\Scripts\python.exe -m memory_retrieval evaluate --gold examples/gold.jsonl --results runs/proposed.jsonl --output runs/proposed-metrics.json
```

去掉 `--judge` 只返回候选，保留相同检索流程。主方法固定使用阈值 0.7；top-k 诊断可用 `--coarse-k` 和 `--final-k`。没有达标候选时返回空集合。

索引目录、查询结果和报告不会覆盖已有文件。重复实验使用新的输出名称；相同请求可复用缓存。批量查询若中途失败，会保留已经完成的结果并非零退出；修复配置后用新输出路径重跑即可，不完整结果不能通过评测。

## 3. 数据与接口约定

文本旧库为 JSONL，每条一行：

```json
{"memory_id":"m1","content":"2026年10月13日，小林唯一的工作单位是腾讯。","source_ref":"text:m1"}
```

适配器输出完整 `content` 与可追溯的 `source_ref`，不生成描述。检索表示的语义内容仍是 `description、content、source_ref`；`memory_id` 和目录编号只是记录映射。一个旧记录可有多个描述入口；返回候选按原记录去重，不将相同内容的不同记录合并。

查询 JSONL 只需要 `query_id、content`，见 `examples/queries.jsonl`。将主体、必要上下文和明确时间写入可读内容；系统不凭空恢复原对话。代码仅将查询 content 传给模型，不把金标传入模型。

金标独立存放：

```json
{"query_id":"q1","fully_reviewed":false,"human_reviewed":false,"gold_compare_ids":["m1","m2"],"conflict_labels":{"m1":"conflict","m2":"no_conflict"}}
```

当前新数据集的 `conflict_labels` 只含 `conflict、no_conflict`；构造时无法确定的来源配对不入选，合库后任何配对仍无法确定的查询整条排除。需要比较不等于冲突：同事项重复、不同答案和历史状态可进入应比较集合；明确不同有效时间的状态不自动冲突。历史数据与判定器仍兼容 `uncertain`，本次没有修改冻结的实验判定器。

`fully_reviewed` 和 `human_reviewed` 是数据集中的审核状态字段，由数据维护者更新。评估代码对提供的全部金标行计算指标，不以这两个字段筛除查询；标签是否适合支撑正式结论由数据维护者核对后判断。更换或扩充旧库后，候选集合需要重新确认。

图适配器读 `examples/graph.json` 展示的明确 schema：

- `nodes` 保存可读实体/事件名称；`edges` 保存方向明确的关系及 `qualifiers`；`memories` 用边引用组合同一原记录的完整主张。
- `relation_templates` 按后端关系语义表达主语、宾语及所有限定属性；逆向关系可在模板中交换表达顺序。否定、条件、有效时间、语气必须显式保存在限定属性并由模板表达。
- 模板遗漏限定属性、边引用不存在或记录 ID 重复时拒绝输入，不静默删信息。模板的语义正确性仍需人工核查。
- 当前只接这个明确 schema，不声称能自动无损读取任意图数据库；也不沿图推导其他旧记忆未直接表达的结论。

示例文本与图适配后具有相同 `memory_id、content`，来源引用不同。可用下面的命令建立等价图索引，再运行同一批查询：

```powershell
.\.venv\Scripts\python.exe -m memory_retrieval build --input examples/graph.json --adapter graph --output artifacts/graph-index
```

## 4. 方法对应实现

```text
离线：文本/图 → content + source_ref →（按批，默认每批 16 条）旧描述目录 → 描述向量与完整内容向量 → 冻结索引

查询：新记忆 content
  → LLM 先返回适用旧描述编号，再返回未覆盖事项的新描述（每条新增附对照编号与差异维度）
  → 程序按编号复制目录原文，对照后无差异的新增也换回原文，组合本次描述入口
  → 新旧描述向量匹配，对每条旧记录取最高分，保留分数 >= 0.7 的全部记录
  → 根据新记忆与描述生成同值改述/异值假设；自动保留原文探针
  → 仅在粗候选中计算探针与旧完整内容的最高相似度，再保留分数 >= 0.7 的全部记录
  → 可选的固定判定器只读取新旧完整 content
```

离线建库默认 `batch_reuse`：把旧记录按输入顺序分批（默认每批 16 条），同一批放进同一次请求，模型同时看到该批全部记忆和当前全部描述目录。批内涉及同一事项时要求输出完全相同的描述文本，本地按精确文本合并成一条；批间直接引用已有编号，不重新生成。批次必须顺序推进（后一批要用前一批结束时的目录），因此加速来自批内一次请求处理多条，请求次数由 O(n) 降为 O(n/批大小)。批次末按记忆原始顺序合并，保证条目编号与原序一致、结果可复现。某条记忆没有结果会被拦下并让整次建库失败，不静默丢记忆。

选中已有描述时直接复用其冻结向量；新描述才调用向量 API。共享描述只代表共享检索入口。时间、条件、答案是否相容仍由完整内容决定。

查询侧沿用同一条描述提示（`value`／`description` 两栏），只是不写回：**目录永远不因查询而改变**，查询只决定本次使用哪几条描述入口。

假设生成每个描述最多一条同值改述、两条异值探针。原文始终作为同值基本探针。假设不读取候选旧内容，不作为真实事实或冲突证据。最终排序是最高余弦相似度；平分时依次按粗分和记录 ID 确定顺序。粗分平分按记录 ID，不依赖随机文件顺序。

默认采用精确矩阵相似度，先验证方法机制，未加入 ANN、目录预筛、在线写回、训练、自动更新或额外语义判定模型。目录过大导致服务拒绝时明确失败，不自动截断目录；方案中的前置目录召回属于后续规模阶段。

异常处理：非法目录编号/JSON 最多修复一次；请求超时、429 或服务器错误按 `.env` 重试。旧记录没有可识别描述时停止建库以免漏收旧记录；新查询没有描述时返回空候选并标记 `no_descriptions`。没有隐藏的原文补检或语义回退。JSON 校验不能保证模型选对事项，真实效果需实验验证。

建库阶段另有两道可判定检查，只依赖库内数据，不读数据集标注、不带领域词表：

零、事项—取值显式分离（默认开启，不需要开关）。每个 `new_items` 必须给出两栏：
`value` 是原文对该事项的回答（含必要的否定、状态或不确定信息），`description` 只保留
主体、待比较事项与必要实例限定。**只有 `description` 进入描述目录、描述向量与目录去重**；
`value` 不参与检索，仅与 `description` 一起留在 `extraction.new_items` 与 `values.items` 里，
供事后核对"描述是否仍含着取值"。

生成标准是“换值不换事项”：固定主体、事项与实例范围，只换答案时描述应保持不变。
例如教材 A/B 都对应“微积分课程推荐的教材”，而不是“是否推荐 A”。日期、人名、数字有时
是主体或实例限定，不能机械删除。首批空目录、单条、批量及重写沿用同一要求，在原有一次
生成中完成；已有项仍只返回编号。字段校验只保证非空字符串，**不能证明语义上没有值泄露**；
不会按字符串重合自动删值，也不增加第二个语义复核模型。

**这里不再有新增准入（`contrast_id`／`differs`／`difference` 与 `unverified` 已撤除）。**
撤除依据是实测：协议 5 下强制每条新增给出对照编号与差异理由，程序只能保证"给了理由"，
不能保证"理由成立"；抽查新增项，约 15–21% 被判为其实与对照是同一事项，而收敛率相对
上一版持平（76.4% 对 76.7%）。换言之该机制把错判变成了**可审计**，但没有**拦住**它，
却额外引入了重写轮与待核查状态。现在只保留提示词里的复用指引（答案、措辞、语言、数值
不同都不是新增理由，另一种语言的已有条目也应选中其编号）与 `value`／`description` 的审计留档。

一、抽取结果校验与重写（M1，默认开启）。每批结果返回后按可判定规则复核：每条记忆至少要有一条描述；描述不能是纯标点，也不能比原文长一倍以上；同一批里内容相似度达到 `MEMORY_TEMPLATE_THRESHOLD`（0.90）的两条记忆属于同一事项，必须至少共用一条描述，各写一份措辞不同的就算违规。违规时把具体条目和违反的规则发回，让模型只改这些描述重写一轮；重写后仍违规的保留下来，并把违规清单写进索引供事后审计，既不静默接受也不中断建库。校验阈值是固定先验，与是否启用归并无关；`build --no-repair` 可关掉重写（仍然照常校验并记录）。

二、目录归并（M2，**默认关闭**）。按两条判据合并条目：两条描述向量相似度达到 `DESCRIPTION_DUPLICATE_THRESHOLD`（0.95），即同一事项的两种措辞；两个条目各自挂着的旧记忆里存在一对内容相似度达到 0.90，即同一事项的两个取值——取值写进描述时条目会被拆开，这一条负责收回。归并只改变目录呈现方式：被收掉的措辞保留在条目的 `aliases` 里继续参与打分，条目得分取其全部别名的最大值，条目文本取挂着记忆最多的那一份。

  默认关闭是实测结果，不是保守设置：在同一份抽取结果上只加归并（`tmp/index_v6base_m2only`，771 → 716 条目），37 条写入的候选召回从 0.980 掉到 0.961，而同一配置两次独立运行的召回都稳定在 0.980。原因是**归并只保留一份条目文本**：模型在查询时看不到被收掉的那些措辞，就失去了选中它的入口（别名只救打分、救不了选择）。归并确实换来了更小的目录（706 vs 771 条）和更集中的同一事项（世界 topic 收敛到单条目 31/84 vs 24/84），但代价是召回。`build --consolidate` 可开启，用于消融。

索引 `format_version=3` 额外保存 `quality`（重写批次数与违规按规则计数）、`values`（每条记忆抽取到的 `value` 与 `description`，供事后核对取值泄漏）和 `consolidation`（归并前后条目数、每组归并的编号与触发判据）。归并后描述向量的行数不再等于目录条目数，条目通过 `aliases` 指向自己的全部向量行；`format_version=1/2` 的旧索引仍可加载，按一一对应处理（2 版里已撤除的 `admission`／`unverified` 字段只是被忽略）。

**当前建库协议为 `BUILD_PROTOCOL=7`（撤除新增准入，保留事项与取值分离），旧建库哈希不可再用于描述检索（`proposed`／`independent`）**：已有带值目录不会自动清理，应在重新建库时应用新约束；不会在查询时改写冻结目录。索引身份只绑定建库用的描述提示与协议；仅修改查询阶段的 hypothesis/judge 提示无需重建索引。`dense`/`bm25`/`hybrid` 不读描述，仍可用旧索引。

历史 v5 机制消融（37 条写入、102 个旧金标对）：

> 下表只记录旧版工程诊断，不是 ValidMem v7 的主 baseline 结果。数据、金标、索引和运行缓存均与当前协议不同；不得用它支持 v7 的方法优劣结论。

| 配置 | 索引 | 候选召回 | 候选纯度 | 冲突 F1 | 误报率 | 平均候选 | 运行目录 |
|---|---:|---:|---:|---:|---:|---:|---|
| M1 关 / M2 关 | `validmem-v6-base` | 0.980 | 0.407 | 0.933 | 0.062 | 6.6 | `validmem-v6base-37w` |
| M1 关 / M2 关（空缓存重跑，测噪声） | `validmem-v6-base` | 0.980 | 0.383 | 0.910 | 0.081 | 7.1 | `validmem-v6base-37w-r2` |
| M1 开 / M2 关（当前默认） | `validmem-v7` | 0.980 | **0.426** | 0.933 | 0.067 | **6.4** | `validmem-v7-37w` |
| M1 关 / M2 开 | `tmp/index_v6base_m2only` | 0.961 | 0.386 | 0.906 | 0.090 | 6.9 | `validmem-v6base-m2-37w` |
| M1 开 / M2 开 | `validmem-v6` | 0.961 | 0.387 | 0.906 | 0.090 | 6.84 | `validmem-v6-37w` |

读法：召回在两次同配置运行里都稳定在 0.980，纯度和候选数有运行间波动（0.407/0.383，6.6/7.1）。M1 在同召回下把纯度抬到 0.426、候选压到 6.4，超出该波动；M2 在两种 M1 状态下都把召回压到 0.961，方向一致，因此默认关闭。第一次跑出的 v5 数字（0.941 / 0.405）来自旧网关，与上表不可直接比较：换网关本身就把召回从 0.941 抬到 0.980。

两点必须一起读的测量前提：一、`reference` 网关在 `temperature=0` 下**不可复现**——同一条请求原样重发，9 条结果里有 5 条描述文本不同（`tmp/probe_determinism.py`），所以任何单次建库之间的目录差异都只有一半来自机制本身（两次建库的目录文本重合率仅 46%–50%）；判断某个机制是否有效，要看同一份抽取结果上的固定输出对照，或多次重复后的均值。二、`proposed` 的端到端时延在不同运行间不可比，因为缓存命中状态不同（上表 p50 从 3.0 秒到 64 秒都是同一套代码）。

主要模块：`adapters.py` 读取后端，`prompts.py` 固定提示，`index.py` 离线索引，`quality.py` 可判定校验与目录归并，`pipeline.py` 查询，`api.py` 两类接口与缓存，`evaluation.py` 评测。`index.json` 保存目录、完整旧内容、描述提取结果、模型/提示指纹；`vectors.npz` 保存归一化向量。加载时校验文件完整性，查询不改动二者。更换向量服务/模型必须重建索引。

## 5. v7 主 baseline 比较协议

主问题是比较“动态阈值候选”与原生 top-k 检索在候选召回和候选负担上的权衡。所有方法使用完全相同的冻结旧记忆库、查询和冲突金标；保持各方法原生候选策略，不给 baseline 添加主体判断或 LLM 筛选。

| 方法 | 候选策略 | 比较设置 |
|---|---|---|
| `proposed` | 描述检索与细筛按冻结相似度阈值返回变长候选 | 阈值只在开发集选择，测试前冻结；报告一个工作点 |
| `bm25` | 对完整记忆正文做 BM25 原生排序 | top-k ∈ `{1, 2, 3, 5, 10, 20, 30, 50}` |
| `dense` | 对完整记忆正文做向量排序 | 使用与 proposed 相同的嵌入模型，报告同一组 top-k |
| `hybrid` | BM25 与向量排序按 RRF（常数 60）融合 | 报告同一组 top-k，不加额外语义筛选 |
| `sparsecl` | BGE cosine HNSW Top-1000 初召回，官方 SparseCL Hoyer 融合重排 | dev 选择并冻结 alpha；报告同一组 top-k、BGE 初召回 Recall@1000 与 NDCG@10 |

统一报告：冲突配对 micro 召回、逐查询 macro 召回、完整找全率、候选纯度，以及候选数 mean/p95/max。无冲突金标查询的 macro 召回记 1，NDCG@10 记 0 并保留在查询平均分母；micro 召回仍按冲突配对计分。SparseCL 另报告 NDCG@10、初召回 Recall@1000 和同一 BGE 的纯 cosine 排序对照。主图用平均候选数作横轴、召回作纵轴，baseline 展示完整预算曲线，proposed 标出其阈值工作点。不同方法返回数量不同是设计本身；结论应表述为召回—候选负担权衡，不写成相同预算下的胜负。

SparseCL 使用 `baselines/SparseCL` 中已固定版本的 `BAAI/bge-base-en-v1.5` 与官方 `SparseCL/BGE-SparseCL-arguana` 权重，属于官方预训练权重的迁移评估，不在 ValidMem 上重新训练。运行器通过现有 WSL 独立环境一次批量处理整组查询；默认 Ubuntu-22.04、用户 `dzp`、`/home/dzp/.venvs/sparsecl/bin/python`，可用 `--sparsecl-*` 参数指定本机路径。

单独使用 `--arms sparsecl` 时不要求项目 LLM/embedding API 配置，也不加载或构建项目描述索引；SparseCL 使用自己的双编码与 HNSW 缓存。其 Top-B 由 `--final-k` 控制，与项目 `--coarse-k` 无关。

SparseCL 只读取记忆和查询的 `content.strip()`，不拼接主体、标签、描述、标题或 instruction。BGE 用 CLS 池化，SparseCL 沿用官方 `our_BertForCL` attention-mask 平均池化，两套向量均 L2 归一化、FP32、最大长度 512。初召回为归一化 BGE 向量上的 `IndexHNSWFlat(768, 64)`，默认 L2，`efSearch=1000`，深度 `min(1000, 旧库条数)`；之后严格按 `cosine + alpha × Hoyer(归一化 SparseCL 向量差)` 对整个初召回集合重排，Hoyer 差向量 L2 范数小于 `1e-3` 时取 `-1e9`。项目 `--coarse-k` 不会缩小 SparseCL 的初召回深度。已复现模型是英文配置，而 v7 含有中文文本；完整共同查询结果会保留并应披露语言限制。若另做英文切片，须先人工核验切片，并让所有方法使用相同查询、原有完整旧库和金标，不由 SparseCL 单独删样本。

完整 `dev` 运行以官方 `(0,10)`、10 bins、`eps=0.01` 搜索选择 NDCG@10 最优 alpha，并保留官方返回最终区间左端点的行为；test 只接受 dev 冻结的 alpha、模型/代码/资源签名和同一个旧库向量及 HNSW 缓存。test 配置缺失或不匹配时会在推理前失败。探索性运行同时使用已复现的 Arguana `alpha=0.18`，不读取探索集标签调参。开发结果另外保留 Arguana alpha=0.18 与 BGE cosine 的离线预算曲线；各变体都从同一 Top-1000 初召回集合产生，不另行运行模型。

在线成本另列查询时延 p50/p95、API 尝试、缓存命中与已知 token；SparseCL 的编码批次耗时均摊到批内查询，只表示摊销成本，不作为单条查询响应时延。BM25 索引和描述索引的离线建库时间单独记录。缓存按方法臂隔离，避免先运行的方法为后运行的方法预热。单次时延仍是诊断值；正式成本结论应重复运行并报告波动。当前项目的 BM25 与 Hybrid 在查询前各构建一次 BM25 索引，不把索引构建时间重复算进每条查询。

当前 181 条 v7 查询已经用于数据修复和流程调试，因此只能作为探索/开发结果。正式测试需要新建、人工核验的查询与金标，并按来源场景或模板族整组留出；不能把同一模板的近似句拆到 dev 和 test。开发集确定 top-k 与动态阈值后冻结设置，测试集完整运行一次。测试集查询 ID 和正文必须与当前 v7 不重叠，金标需将 `fully_reviewed`、`human_reviewed` 设为 `true`；运行脚本会检查这些条件。

开发阶段应对完整 dev 查询运行五个主方法，并保存 BM25、Dense、Hybrid 的完整 top-50 排序、SparseCL 的完整融合排序和覆盖每条查询的场景/模板族映射。可以用 v7 作为探索/开发数据；它不能再充当独立 test。仓库当前没有该分组映射，运行前需由数据维护者准备并核验 `<dev-group-map.json>`：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --dataset data\validmem_v7 --limit 0 --evaluation-split dev --query-groups <dev-group-map.json> --workers 4 --arms proposed bm25 dense hybrid sparsecl --coarse-k 50 --final-k 50 --similarity-threshold 0.7 --run-name validmem-v7-dev
```

先在 dev 上选定动态阈值与报告预算网格，再固定 `.env` 中的模型配置、提示、索引、阈值和并发设置。独立 test 使用新查询与金标，仅完整运行一次；查看 test 结果后不得据此改参数并把同一批数据继续称为未触碰测试集。

对已保存且包含 top-50 排序的运行，可离线算完整预算曲线，无需重新请求模型：

```powershell
.\.venv\Scripts\python.exe scripts\analyze_v7_experiment.py --run-dir runs\experiments\v7-20261029 --dataset data\validmem_v7 --output reports\v7_experiment_2026-10-29.md
```

脚本会核对输入文件哈希和查询集合，生成 Markdown、JSON、CSV；如果运行保存了 `query_groups.json`，还会按场景/模板族做分组 bootstrap。新运行的标准入口为 `scripts/run_experiment.py`；`independent`、`no-rerank` 和目录建库参数属于其他诊断，不纳入本次主 baseline 比较。`dense`、`bm25`、`hybrid` 是本文实现的原生检索对照；SparseCL 是基于已复现官方推理代码和固定权重实现的完整迁移 baseline，不包含重新训练。

`batch_reuse` 是当前描述索引的建库方式：旧库描述在批内合并、批间复用；描述目录只在建库期间变化，查询生成的新描述永远不写回。查询结果保留排序、分数、提示/索引版本、阶段耗时、API 调用和 token 统计。

`memory_retrieval evaluate` 可评估一次运行的固定候选集合；`scripts/analyze_v7_experiment.py` 从原生排序生成多预算曲线。候选检索主比较不报告逐对冲突分类 F1；如需评价下游冲突判定，应另设统一判定器实验。

当前查询检索采用完整候选库精确扫描。单一库容点不能证明时延随库容次线性；只有完成多个库容的重复测量后，才能作规模复杂度结论。

默认 `--cache-policy isolated` 为各方法臂创建独立临时缓存；`shared` 仅用于复用缓存的调试运行，不应与 isolated 的时延混在同一张比较表。输出记录 API 尝试、缓存命中和已知 token；服务未提供 usage 时单独计数。项目未固定单价，因此不猜测货币成本。

## 6. 数据转换与实验

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试使用本地 HTTP 模拟服务，不需要真实 Key，也不调用外部付费模型；它验证数据流、接口、去重、预算、冻结、缓存、错误处理和指标计算，不验证语义效果。

以下命令按当前 v7 主协议运行探索性比较：完整使用共同旧库和 181 条查询，保存 BM25、Dense、Hybrid top-50 原生排序及 SparseCL Top-1000 初召回后的完整融合排序，再离线汇总预算曲线。探索性 SparseCL 使用固定 Arguana alpha=0.18；这个已反复查看的数据集不能作为独立测试集：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --dataset data\validmem_v7 --limit 0 --evaluation-split exploratory --workers 4 --arms proposed bm25 dense hybrid sparsecl --coarse-k 50 --final-k 50 --run-name validmem-v7-exploratory
```

结果写入 `runs/experiments/<run-id>/`，包含数据哈希、评估用途、冻结索引、逐查询排序和指标摘要。proposed 的阈值可用 `--similarity-threshold` 指定；在开发集确定后必须冻结。BM25/Hybrid 索引每次运行只构建一次，时间与在线查询分开；默认各方法臂隔离 API 缓存。此命令会发起 API 请求，但本次代码修复不运行它。

正式测试应准备独立数据目录：旧记忆库与 dev 使用相同哈希，查询和逐条人工核验金标全新。准备 JSON 分组映射 `query_id -> scene/template_family`，dev 与 test 的分组名不得重叠；该映射同时供分组 bootstrap 使用。测试金标必须设置 `fully_reviewed=true`、`human_reviewed=true`；ID 和正文不能与 v7 或 dev 重叠。运行时使用 `--evaluation-split test --dev-run runs/experiments/validmem-v7-dev --index-dir runs/experiments/validmem-v7-dev/index --query-groups <test-group-map.json> --limit 0 --arms proposed bm25 dense hybrid sparsecl`，并保持 dev 的 LLM、嵌入模型、查询/索引提示、阈值、top-50 深度、并发数、缓存策略、API 重试/超时设置、嵌入批大小、SparseCL WSL/批量/线程设置和五个方法臂一致。test 只接受带 `completion.json` 且未包含 `failure.json` 的完整 dev；完成标记绑定各方法结果哈希，并逐臂核验查询覆盖。脚本会核验这些可机器检查的条件；场景/模板组是否真正隔离仍由数据构造者确认。

以下内容仅描述早期 Dialogue NLI 补充集的构造流程，不是 ValidMem v7 的实验划分方案。其源文件为工作区 `tmp/dialogue_nli/verified_test.jsonl`，来自 Dialogue NLI 已有三份标注的子集；原文件名中的 test 只是来源文件名称。该补充集构造流程本身不划分 train/dev/test，也不训练或调参。先筛出共享同一句查询、已有两种关系且三份标注一致的来源组；排除相同句对标签矛盾的记录，优先核对明确数量对照，避免将不同职业等可并存值直接判为冲突。

经用户确认，入选原句统一重组为同一测试主体的记忆；保留原文，不补造人物、时间或事实，也不声称恢复了原始人物时间线。模型先独立核对来源组，只保留与来源标签一致的配对以及仍有两类候选的查询；再合库，并核对每条入选查询与全部旧记忆，补齐跨组候选。任一配对未解决、来源关系在全库复核中不一致的查询整条排除，不强制输出二元标签。

```powershell
.\.venv\Scripts\python.exe scripts\build_mixed_dataset.py --groups 24 --max-memories 2500 --workers 4
```

输出目录 `data/dialogue_nli_mixed_v1/` 只包含 `memories.jsonl`、`queries.jsonl`、`gold.jsonl`。24 是首次核对的候选组数，实际通过数量以构造结果为准；2,500 是旧库硬上限，不以填满库容为目标。超过上限时跳过整个候选组，不截断已确认的必需候选。全库复核按 40 条旧记忆分块请求，索引映射回同一个完整旧库；接口无法核对的组或查询会排除并写入审计目录，不强行补标签。来源与全库复核依据保存在 `runs/dataset_builds/dialogue_nli_mixed_v1/`，不增加第四个数据文件。

构造完成后 `fully_reviewed` 和 `human_reviewed` 都为 `false`；全库模型复核不等于人工审核。数据维护者完成人工核对后再更新这两个字段。实验不依据它们跳过查询，而是按提供的金标计算指标。构造请求使用现有 API 缓存，中断重跑复用成功响应，不覆盖非空数据目录。实验默认使用此共同旧库，随机抽 12 条查询并行运行；不同库容的召回率-时延曲线由 `scripts/scale_curve.py` 汇总多次运行，单个库容点不能支持规模复杂度结论。


## 7. ValidMem：统一旧记忆库与同主体冲突

当前修订版为 `data/validmem_v7`。任务是从完整旧记忆库中，为每条新写入找出“同一主体、同一事项、不同取值”的旧记忆。数据不再按 personal/world 或来源 `user_id` 切分；不同值包含后续更新，同值复述不算冲突。事项的适用条件和实例范围仍需一致。

主体按正文判定：完全相同的姓名跨场景视为同一人；不同姓名视为不同人，不做近似姓名合并。某类主语只有匿名记录时，全库统一改写成一个姓名；如果该类同时有匿名和具名记录，则根据每条匿名记录原始的场景元数据为其分配新姓名，不与记录者姓名自动合并。映射保存在 `entity_aliases.jsonl`。

从 v6 生成 v7：

```powershell
.\.venv\Scripts\python.exe scripts\repair_validmem_v7.py
```

当前 v7 保留 1,225 条旧记忆；v6 原有 188 条写入中，7 条在主体复核后已无有效冲突，移出活动集并保留在 `provenance.jsonl`，因此活动集有 181 条写入和 301 个冲突金标配对。全部 288 个继承金标都已重新核对正文主体与事项：6 对因姓名不同而排除，另有 2 对处方记录经用户确认是不同患者；两对新会员折扣与通用折扣也按用户裁定为暂不冲突。另新增 23 个主体、事项和值均可明确对齐的冲突配对。Physics 101 教材版本与春季清仓商品范围两对按用户裁定保留为冲突；春季清仓旧记忆改写为“仅部分商品打 5 折”，原文保存在 `provenance.jsonl`。计划学习 Biology 与最终确定 Data Science 专业按用户裁定为不冲突。`gold_audit.jsonl` 记录了全部 288 个继承配对、23 个自动新增配对和 Biology 计划事项的用户裁定；当前没有待审配对。继承金标中可自动核对出不同取值的保留项按规则保留，人工裁定的例外逐条记录；这不代表全体金标已完成人工审核。脚本只会更新带有本脚本标识的 `validmem_v7` 目录中的固定生成文件；若目录不是本脚本生成的 v7，则会停止，不覆盖。`gold.jsonl` 的 `fully_reviewed`、`human_reviewed` 均为 `false`。

v7 的 `memories.jsonl` 和 `queries.jsonl` 不含 `scope`、`user_id`、`source_role`。原始来源字段与正文保存在 `provenance.jsonl`，金标配对理由保存在 `pair_label_rationales.jsonl`。匿名主体使用独立新姓名，不会因为与记录者处于同一场景而被自动并成记录者；`John Doe` 在 v7 全局改名为 `Mara Voss`，与 John 明确区分；其他相同姓名保持统一。

`data/validmem_v5`、`data/validmem_v6` 是历史快照，不覆盖。此前基于 personal/world、owner 和 topic 边界得出的指标或索引均不代表 v7。此次只修订运行协议，并离线重算已保存排序的预算指标；没有重新调用模型或向量服务。

配套脚本：

| 脚本 | 用途 |
|---|---|
| `scripts/build_validmem_dataset.py` | 重建历史 v5 基础数据 |
| `scripts/add_user_names.py` | 从 v5 生成历史 v6 |
| `scripts/repair_validmem_v7.py` | 从 v6 生成统一库 v7、来源记录、逐对金标审计和待审配对 |
| `scripts/verify_conflict_labels.py` | 按 v7 正文主体与事项抽样复核正负例；兼容 v5/v6 元数据采样，不在本次运行 |
| `scripts/score_validmem_run.py` | 汇总候选冲突召回与纯度 |
