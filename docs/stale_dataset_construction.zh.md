# STALE 场景检索数据集构建规范

本文对应 2026-11-23 的代码和已发布 V2。它说明原始对话如何转换为检索输入；服务器命令见[构建与验收](stale_v2_server_run.zh.md)，指标解释见[评测口径](evaluation_protocol.zh.md)。项目的 `base` 是转换后的基础版，不是官方直接发布的记忆库。

## 1. 一个样本是什么

一个场景对应一次写入：新记忆 `M_new` 是查询，写入前的该场景历史形成旧库，目标是从旧库中检索已标注的冲突或失效记忆。400 个场景对应 400 个独立旧库，T1/T2 各 200；T1/T2 沿用官方分类，本项目不把它们直接重命名为“事实冲突/状态过时”。

官方文件每条记录包含一对 `M_old/M_new`、50 场会话、三个测试问题，以及解释和位置。我们使用的固定原始文件为 `data/stale_sources/T1_T2_400_FULL.json`，SHA256 为 `5f3ec375179e20e2e94469e018189188f34e2e7e5f21cbecbd99fcfa648c1876`。转换脚本中的来源 URL 指向 `main`，本地实际输入以文件哈希为准，重新下载时应检查哈希。

| 原始字段 | 转换用途 |
| --- | --- |
| `uid` | 场景 ID；查询 ID 为 `stale_q_<uid>` |
| `M_new` | 查询 `content`，不改写、不使用原三个问答题替代 |
| `M_old` | 单独加入旧库，保留官方目标标签 |
| `relevant_session_index = [old_session, new_session]` | 校验旧句先于新句；确定历史取材截止位置 |
| `haystack_session` | 抽取截止位置之前的用户原句 |
| `type` | 写入查询的 `source_ref.type`，供场景目录和分组评测使用 |
| `timestamps` | 不参与当前截断规则；以官方会话索引确定先后 |
| `explanation`、`probing_queries` | 不进入检索正文、探针请求或评分特征；原文件留存供溯源 |

官方指定的一对目标不等于官方对我们摘出的全部历史句提供了相容标签。`M_old` 参与数据构建和金标建立，但主方法运行时不会按它的来源标签选种子或加分。

## 2. 历史取材边界与句子筛选

实现入口：[convert_stale_scenarios.py](../scripts/convert_stale_scenarios.py) 的 `history_claims` 和 `convert_scenario`。

1. 要求 `0 <= old_session < new_session < 会话数`。只遍历 `haystack_session[:new_session]`，也就是新记忆所在会话之前已经结束的会话。新记忆所在的整场会话都不抽取，包括该会话中出现在新句之前的发言；后续会话也不抽取。
2. 只处理 `role == "user"` 的发言，不抽取助手回复。`session_index`、`turn_index` 均从 0 开始；发言索引包含助手轮次。
3. 用脚本的 `SENTENCE_BOUNDARY` 正则按句末标点后的空白或换行切分，并对单个大写字母缩写及 `Mr.`、`Ms.`、`Dr.` 做局部排除。不是模型分句，也不是完整英文句法分析。
4. 候选须满足：按空白切分有 7–55 个词；以 `.` 或 `!` 结尾；不含 `?`；含独立词 `I/my/we/our` 之一，不区分大小写。
5. 对弯撇号转直撇号后的文本应用以下句式过滤。过滤只决定是否收录，不进行语义真假判断。

| 过滤位置 | 脚本检查的表达 |
| --- | --- |
| 句首 | `thanks`、`thank you`、`if`、`what if`、`imagine`、`suppose`、`let's` |
| 句首回应语 | `yeah/yes/right/okay/ok/cool/great/sure/wow` 后跟指定标点或空白 |
| `I` 后 | `think/wonder/see/agree/guess/suspect/might/may/would/need/want/should/could` |
| `I'm` 后 | `thinking/considering/trying/looking/wondering/curious/not sure` |
| 句首请求 | `please/could/can/would you`，以及 `could/can/should/do/did/will I` |
| 句首代词 | `it/this/that/these/those/they` |
| 全句元对话 | `follow-up question(s)`、`thanks for`、`your advice/reply/response/suggestion(s)`、`could/can/would you`、`do you think` |

这些是字面规则。例如 `I'm actually thinking ...` 不直接匹配 `I'm thinking ...`，仍可能通过；引述、依赖上下文的指代也可能留下。因此“全部合格句”是“全部通过上述规则的句子”，不是“所有真实、独立、长期有效的事实”。历史正文只去除句子首尾空白，不做改写、指代消解或摘要。官方 `M_old` 单独保留，不受这套历史筛选规则限制。

## 3. 去重、ID 与 V1/V2 差异

`text_key(text)` 先做 `casefold()`、统一弯撇号，再提取 `\w+` 并用空格连接。它用于场景内的字面规范化去重，不做词干还原或语义去重。除重复句外，排除与 `M_new` 规范化后完全相同的句子，以及与 `M_old` 规范化文本互为子串的句子；后者是字符串包含检查，不是逻辑等价判断。

| 规则 | V1 | V2 |
| --- | --- | --- |
| 每次用户发言 | 首个通过句式规则的句子；该句后续被去重拒绝时也不回退到同轮下一句 | 所有通过句式规则的句子 |
| 每场已完成会话 | 最多收录 3 条去重后的背景句 | 取消条数上限 |
| 重复句的代表 | 按 V1 顺序选择 | 先保留 V1 已选句的正文、ID、出处，再补充其余合格句 |
| 官方旧句 | 单独收录，不占每场 3 条名额 | 原样继承 |
| 历史背景标签 | 已冻结扩增版曾有历史模型补标 | 新增历史仅作背景；不调用模型、不新增标签 |

V2 的两遍收录使得某条重复句可以保留较晚会话的 V1 出处，而不是重新选择最早出现处。最终旧库按 `memory_id` 排序，文件行号不表示发生时间或检索名次。

ID 计算为 `stale_m_` 加 `fingerprint([scenario_id, text_key(content)])` 的前 24 个十六进制字符。`fingerprint` 对规范 JSON 做 SHA256，定义见 [data.py](../memory_retrieval/data.py)。场景 ID 参与计算，因此相同正文在不同场景不会被跨场景合并。

## 4. 合成记忆怎样产生，V2 怎样复用

合成构建入口为 [augment_stale_scenarios.py](../scripts/augment_stale_scenarios.py)，完整生成提示词和评审提示词位于 [stale_augmentation_prompts.py](../scripts/stale_augmentation_prompts.py) 的 `GENERATION`、`PAIR_RULES`、`REVIEW`。它们独立于检索探针提示词，二者用途如下。

| 阶段 | 模型可见信息 | 作用 |
| --- | --- | --- |
| 构造合成数据 | 新记忆、官方旧句、生成轮次和反馈 | 构造可在写入前成立的同话题相容句，以及被新记忆推翻的隐式冲突/失效句 |
| 复核合成候选 | 新记忆、官方旧句、打乱顺序的候选 ID/正文、已接受的合成条目 | 给每个候选判定关系和原因；不传生成器给出的预期组别 |
| 检索时生成探针 | 仅新记忆正文 | 生成用于检索的四条旧陈述，不读取官方旧句、候选库或金标 |

新建合成数据时，每个场景最终接受 3–6 条 `compatible` 和 3–6 条 `implicit`，共 6–12 条。生成提示词每轮请求尚未达最低数目的组各 8 个候选；解析最多读取每组 12 个。默认最多 6 轮，每组达到至少 3 条即停止补齐，接受上限为 6；不能补齐则构建失败，不发布缺额版本。

候选先规范化空白、按 `text_key` 去重，长度要求为 6–65 个规范化词。对 `implicit` 组另做形式筛选：

- 拒绝 `no/not/never/neither/nor/none/nothing/nobody/cannot/without/nope`、`used to`、`instead of` 和否定缩写等正则命中的表达。
- 去除脚本固定停用词后，以词集合计算共有词数、Jaccard 和包含率。共有词数大于 1，且 Jaccard 大于 0.15 或包含率大于 0.25 时拒绝。此规则允许少量重合，不保证零重合，也不等于低向量相似度。
- 与新记忆用词相似的要求主要施加在相容组的生成和语义评审上；相容组没有统一的向量相似度硬阈值。

共有词数是两边去停用词后的词集合交集大小；Jaccard 的分母是并集大小，包含率的分母是较小词集合的大小，分母至少为 1。停用词集合和否定正则以脚本常量为准，不采用外部可变词表。合成目标保存的 `lexical_jaccard` 供形式审计，不进入检索评分。

评审关系包括 `conflict/stale/compatible/uncertain/unsuitable`。相容组仅接受 `compatible`；隐式组仅接受 `conflict` 或 `stale`，且评审原因不能命中脚本的“仍可相容”保留意见正则。其余候选不进入最终合成库。通过评审不等于人工金标或逻辑证明。

已发布 V1 的生成模型标识为 `deepseek-v4.1-flash`，评审模型为 `glm-5.3-flash`；运行参数见其 `manifest.json`。当前在线探针也使用前者，因此提示词独立不能消除同模型的表达偏好。当前脚本若没有指定 `--review-env` 或 `--review-model`，会复用生成配置，不能仅从脚本默认值推断实际采用独立评审模型。构建中的补齐、格式修复及失败批次拆分，也不同于检索主包对合法探针不筛选重抽的策略。

**当前 V2 不重新执行上述生成与评审。**它继承冻结 V1 的全部 44,622 条记忆和 4,095 条合成记录，再追加 63,086 条历史背景。金标只删除 `source=history_review` 的 74 项，其旧记忆正文仍然在库；官方及合成标签原样继承。原有 4,820 个待定 ID 及其他金标元数据保留，不因扩库增加待定 ID。

历史 V1 标签政策与当前新建 V1 脚本不同：当前脚本仅评审合成候选，不会重建历史补标。复现已发布 V2 必须使用冻结 V1 文件；重新生成合成数据不能保证得到相同正文、数量和标签。

## 5. 输入文件与字段

每个版本的 `base/`、`augmented/` 均含三个 UTF-8 JSONL 文件及 `manifest.json`。导出后每个 `scenarios/T1/<scenario_id>/` 或 `scenarios/T2/<scenario_id>/` 下也有同样三份 JSONL；`scenarios/index.json` 是目录索引，其 `path` 相对于 `scenarios/`。

| 文件 | 字段 | 含义 |
| --- | --- | --- |
| `queries.jsonl` | `query_id`、`scenario_id`、`content` | 写入 ID、对应旧库、`M_new` 正文 |
| 查询 | `source_ref` | `dataset=STALE`、`record_type=M_new`、官方 `type`、新句 `session_index` |
| `memories.jsonl` | `memory_id`、`scenario_id`、`content` | 旧记忆 ID、所属旧库、用于编码的正文 |
| 记忆 | `source_ref.record_type` | `M_old`、`history` 或 `synthetic` |
| 原始记忆出处 | `session_index`、历史句另有 `turn_index` | 原始会话及用户发言位置；合成句没有伪造的对话位置 |
| 合成记忆出处 | `augmentation_group` | `compatible` 或 `implicit`；来源数据集仍写作 STALE，需结合 `record_type` 区分 |
| `gold.jsonl` | `query_id` | 关联查询；场景通过查询定位 |
| 金标 | `conflict_old_memory_ids` | 检索目标 ID 列表，兼含冲突与状态失效 |
| 金标 | `target_details` | 每个目标的 `source`、`relation`；合成目标还保留 `reason`、`lexical_jaccard` |
| 标签来源 | `stale_native / official_stale` | 官方 `M_old`；不据此推断更细的冲突类别 |
| 标签来源 | `synthetic_review / conflict或stale` | 合成且经模型接受的目标 |
| 复核字段 | `fully_reviewed`、`human_reviewed`、`review_status` | 当前库并非全量复核或全量人工标注；`model_reviewed` 不表示每条历史背景都审过 |
| 待定字段 | `unjudged_old_memory_ids` | 继承的部分待定 ID，**不是全部未经标注背景的完整列表**；空数组也不证明旧库已完整标注 |

基础版每个场景一个官方目标。V2 扩增版每个场景有 1 个官方目标加 3–6 个合成目标，即 4–7 个已知目标。正文之外的来源、解释和标签用于构建溯源、检查及评测，不作为主方法文本编码或图打分特征。

聚合文件的物理合并不等于检索合库。[dataset.py](../memory_retrieval/dataset.py) 先按所选 `scenario_id` 取库；BM25 词频统计、图构建及 SparseCL 初召回都在各自场景内完成。不能把其他场景的记忆作为背景混入后继续使用原金标。

## 6. 已发布版本规模与具体例子

以下是固定文件的统计，不是任意重新生成数据都必须达到的数量。

| 版本 | 总旧记忆数 | 每场景最少 / 中位 / 最多 | 每场景平均 |
| --- | ---: | ---: | ---: |
| V1 base | 40,527 | 83 / 102 / 116 | 101.32 |
| V1 augmented | 44,622 | 91 / 112 / 128 | 111.56 |
| V2 base | 103,613 | 179 / 257 / 323 | 259.03 |
| V2 augmented | 107,708 | 190 / 267 / 333 | 269.27 |

V1 扩增版曾有 400 个官方、1,818 个合成、74 个历史补标目标，共 2,292 个。V2 共 2,218 个目标，只包含前两类。所有版本均按 400 个场景组织；400 个场景的构建数量与选定 60 个场景的实验数量不能混用。

Austin 场景 `7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f` 的位置是 `[13, 33]`。只从 S0–S32 抽历史，单独保留 S13 的 `M_old`：`I've been based in Seattle for the last few years.`。S33 的 `M_new` 是更新 Austin 新住处地址及办理当地服务；S33–S49 不进入背景抽取。

| Austin 组成 | V1 | V2 |
| --- | ---: | ---: |
| 普通历史摘句 | 96 | 266 |
| 官方旧句 | 1 | 1 |
| 合成相容 / 合成目标 | 3 / 4 | 3 / 4，原样继承 |
| 扩增旧库合计 | 104 | 274 |
| 已知目标 | 5 | 5 |

例如，V2 新收录的历史正文 `I've been impressed by the reusable beeswax wraps I recently purchased for food storage.` 对应 ID `stale_m_0138a28ba0bb09619c960394`，出处为 S10、发言 8。它的新增说明取消条数上限后保留了更多原句，不代表该句经过模型确认相容。[Austin V1 逐句溯源案例](../reports/stale_austin_extraction_walkthrough.zh.md) 保留旧版 104 条的详细对照，不能当作 V2 库大小。

下面是该场景 **V2 base** 的三条实际记录节选。`queries.jsonl` 和 `gold.jsonl` 各一条；`memories.jsonl` 除所示官方旧句外，还有 266 条历史。扩增版还会加入上述 7 条合成记忆，并把 4 个合成目标加入金标。

`queries.jsonl`：

```json
{"query_id":"stale_q_7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f","scenario_id":"7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f","content":"I just finished updating my address after settling into my new place in Austin, and I’m trying to get set up with the local utilities and services here.","source_ref":{"dataset":"STALE","record_type":"M_new","type":"T1","session_index":33}}
```

`memories.jsonl` 中的官方旧句：

```json
{"memory_id":"stale_m_14d051a4cc4e2760f4ccbe00","scenario_id":"7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f","content":"I've been based in Seattle for the last few years.","source_ref":{"dataset":"STALE","record_type":"M_old","session_index":13}}
```

`gold.jsonl`：

```json
{"query_id":"stale_q_7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f","conflict_old_memory_ids":["stale_m_14d051a4cc4e2760f4ccbe00"],"target_details":{"stale_m_14d051a4cc4e2760f4ccbe00":{"source":"stale_native","relation":"official_stale"}},"fully_reviewed":false,"human_reviewed":false}
```

## 7. 验收能证明什么

[audit_stale_v2.py](../scripts/audit_stale_v2.py) 从原始文件重建 V1 并核对记录，验证 V2 覆盖全部符合规则的历史句、V1 代表句与 ID 未变、出处及截止位置正确、合成数据和标签继承正确、历史目标已过滤，并逐场比较聚合文件与导出文件。构建前 V1 全目录哈希及旧缓存快照用于验证原文件未改动。

这些检查证明结构、来源、边界和继承符合约定；不会证明合成标签百分之百正确，也不会证明所有背景都与新记忆相容。新增背景不在金标和待定列表内，是当前数据政策，不是语义负例结论。检索指标须按[已标注目标口径](evaluation_protocol.zh.md)解释。

构建脚本本身不划分 Dev/Test，也不选“表现好的场景”。60 场景实验选择属于实验协议：V1 固定选择 30 T1+30 T2，V2 复用同样 ID/顺序与查询。V2 是同场景扩库复测，不是新增的独立测试集。具体结果和跨版本限制见[最新 V2 报告](../reports/stale_v2_60scene_bipolar_seed20261122_experiment.md)。
