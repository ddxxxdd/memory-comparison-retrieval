# 锚定四维失效/冲突探针 + 旧记忆库双极图扩散（`anchored_bipolar_diffusion`）核心方法总结

> **任务定位**：Agent 长期记忆**写入时（Write-time）**的一对多隐式冲突与状态失效检索——给定单用户历史记忆库 $\mathcal{M} = \{m_1, \dots, m_N\}$ 与新写入陈述 $q$，在**零离线 LLM 抽取**、**在线 LLM 仅见 $q$ 不可见旧库**的约束下，将库中与 $q$ 存在隐式逻辑矛盾或被 $q$ 隐式使之失效的一条或多条独立旧记忆召回至 Top-$K$，交由下游冲突判别器核查。

---

## 一、核心设计思想（通俗直觉版）

传统向量检索只看“话题像不像”，在写入时隐式冲突检索中同时面临三个痛点。本方法将三者拆解并逐个击破：

1. **解决“字面不挨边、搜不到”——锚定四维反向探针（画通缉令）**  
   直接拿新记忆原句去搜旧库，常因抽象陈述与具体行为词面零重合而召回失败。我们先让 LLM 仅读取新记忆，先写一句**新状态锚点**作为对照靶子（防止轻量模型偷懒复述新记忆），再从 4 个领域无关的纯逻辑维度反向生成 4 条肯定句“失效/冲突画像”，把抽象新事实转化为更贴近旧记忆表达的具体陈述。
2. **解决“一对多漏抓同伙”——旧记忆库邻域图正向扩散（顺藤摸瓜）**  
   同一个旧状态在长期记忆中往往留下多条彼此高度相关的记录，在旧库向量空间中天然抱团。离线阶段无需调用任何 LLM，仅凭冻结句向量为每条旧记忆连接最相似的 5 个邻居构成关系图；在线以探针命中的 Top-5 高分记忆为正种子，顺着图边扩散 1 步，把探针漏猜的同源失效记忆顺藤摸瓜拉进 Top-$K$。
3. **解决“顺藤摸瓜误抓相容干扰”——相容负源与邻域抑制（贴好人卡）**  
   如果某条旧记忆与“新记忆原句”的相似度高于与“所有反向探针”的最高相似度，说明它更偏向顺着新记忆说话的相容记忆。我们在扣除其自身分数的同时，顺着同一张图轻度抑制将其作为邻居的记忆，把同话题相容干扰簇成片压低。

---

## 二、形式化算法流程与锁定参数

### Step 1：离线建库与构图（零离线 LLM 抽取）

使用冻结编码器（`BAAI/bge-base-en-v1.5`，CLS + L2 归一化）将旧记忆库 $\mathcal{M}=\{m_1,\dots,m_N\}$ 编码为单位向量矩阵 $X\in\mathbb{R}^{N\times d}$。计算两两余弦相似度矩阵 $A = X X^\top$（对角线置 0），每个节点仅保留最大的 $k=5$ 条正相似有向边（按分数降序、`memory_id` 升序确定性打破平局），再按行归一化得到有向转移矩阵 $P\in\mathbb{R}^{N\times N}$（孤立行保持为 0）。

### Step 2：在线锚定四维探针生成（零领域词汇，在线 LLM 仅见 $q$）

将新记忆 $q$ 输入 LLM（`deepseek-v4.1-flash`，温度 0、JSON 模式），零样本生成：

- **`new_state_anchor`**（1 句，不超过 20 词）：客观概括 $q$ 确立的核心新事实或状态，**仅作生成期防复述对照锚点，绝不进入向量编码与检索**；
- **`violation_probes`**（4 条肯定陈述句，每条不超过 30 词，避免显式否定词，语种与 $q$ 一致），按以下 4 个纯逻辑维度各生成 1 条：
  1. **(a) Superseded or opposite state**：与锚点直接互斥或被其取代的旧状态/旧属性值；
  2. **(b) Rule–instance contradiction**：违反锚点普遍约束的具体行为，或被锚点具体事件打破的旧绝对原则；
  3. **(c) Invalidated precondition or dependency**：一旦锚点成立，其必要前提不再成立的持续状态、计划或依赖关系；
  4. **(d) Incompatible condition or context**：与锚点所隐含的物理、时间或环境条件无法共存的旧前提。

### Step 3：双极初始打分与正负源构造

使用同一冻结编码器将 $q$ 与 4 条探针编码为单位向量 $\mathbf{v}_q$ 和 $\{\mathbf{u}_j^-\}_{j=1}^4$，计算每条旧记忆的初始冲突分与原句相似度：

$$r_{0,i} = \max_{j \in \{1,\dots,4\}} \mathbf{x}_i^\top \mathbf{u}_j^-, \qquad s_i^+ = \mathbf{x}_i^\top \mathbf{v}_q$$

据此构造两极信号向量：

- **冲突正种子 $\mathbf{z}_{\text{pos}} \in \mathbb{R}^N$**：仅保留 $\mathbf{r}_0$ 排名前 `top_seeds = 5` 节点的原始分数，其余位置置 0；
- **相容负源 $\mathbf{z}_{\text{neg}} \in \mathbb{R}^N$**：
  $$z_{\text{neg}, i} = \max\!\big(0,\; s_i^+ - r_{0,i}\big)$$
  仅对“比反例探针更接近新记忆原句”的记忆非零，对已落在冲突侧的记忆（$r_{0,i} \ge s_i^+$）严格为 0。

### Step 4：一步双极图扩散与全库重排

使用锁定参数 $k=5,\ \text{top\_seeds}=5,\ \alpha=0.2,\ \beta=0.3$，计算每条旧记忆的最终得分：

$$\mathbf{r} = (1-\alpha)\,\mathbf{r}_0 + \alpha\,(P\,\mathbf{z}_{\text{pos}}) - \beta\,\big(\mathbf{z}_{\text{neg}} + \alpha\,(P\,\mathbf{z}_{\text{neg}})\big)$$

即：

$$\mathbf{r} = 0.8\,\mathbf{r}_0 + 0.2\,(P\,\mathbf{z}_{\text{pos}}) - 0.3\,\big(\mathbf{z}_{\text{neg}} + 0.2\,(P\,\mathbf{z}_{\text{neg}})\big)$$

按 $\mathbf{r}$ 对整个旧记忆库降序排序，输出 Top-$K$ 候选。

---

## 三、核心消融实验臂对照

主包模块 `memory_retrieval/bipolar_graph.py` 内置以下三个消融臂，共享同一批探针与 BGE 向量：

| 实验臂名称 | 计算公式 | 验证的科学问题 |
|---|---|---|
| `anchored_probes_only` | $\mathbf{r}_0$ | 仅靠四维锚定探针跨越语义鸿沟的基础召回能力 |
| `anchored_pos_diffusion` | $(1-\alpha)\mathbf{r}_0 + \alpha P\mathbf{z}_{\text{pos}}$ | 叠加正向邻域图扩散后，对一对多漏检目标的拉回收益 |
| `anchored_bipolar_diffusion` | $(1-\alpha)\mathbf{r}_0 + \alpha P\mathbf{z}_{\text{pos}} - \beta(\mathbf{z}_{\text{neg}} + \alpha P\mathbf{z}_{\text{neg}})$ | 完整方法：叠加相容负源与邻域抑制后，对相容干扰簇的压制增益 |
