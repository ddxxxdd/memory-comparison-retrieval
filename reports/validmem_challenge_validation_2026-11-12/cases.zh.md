# 可追溯案例

以下均为来源支持的受控反事实，按结果条件选作解释示例；不是自然发生频率估计，也不参与样本筛选。完整前五名、前景和配对结果见 case_evidence.json。

## identity_pressure

case `P-89fc22a15098c6608570`；anchor `TC-A-0152`；seed=17；N=200；G=1。

查询：John's favorite coffee shop is 'Espresso Haven' on Elm Street.

- 金标 `C-695d25e0c6b4ab97b063`：John's favorite coffee shop is 'The Daily Grind' on 2nd Avenue.

| 方法 | 金标排名 | 第一名 |
|---|---|---|
| bm25 | 16 | Emma's favorite coffee shop is 'Espresso Haven' on Elm Street. |
| dense | 9 | Emma's favorite coffee shop is 'Espresso Haven' on Elm Street. |
| hybrid | 14 | Emma's favorite coffee shop is 'Espresso Haven' on Elm Street. |
| sparsecl | 16 | Bob Garcia's favorite coffee shop is 'Espresso Haven' on Elm Street. |

| 配对条件 | d | N | G | BM25 名次 | Dense 名次 | Hybrid 名次 | SparseCL 名次 |
|---|---:|---:|---:|---|---|---|---|
| no_injection | 0 | 200 | 1 | 1 | 1 | 1 | 1 |
| other_entity_query_value | 1 | 200 | 1 | 2 | 1 | 1 | 2 |
| other_entity_old_value | 1 | 200 | 1 | 1 | 1 | 1 | 1 |
| other_entity_query_value | 5 | 200 | 1 | 6 | 3 | 5 | 6 |
| other_entity_old_value | 5 | 200 | 1 | 1 | 1 | 1 | 1 |
| other_entity_query_value | 15 | 200 | 1 | 16 | 9 | 14 | 16 |
| other_entity_old_value | 15 | 200 | 1 | 1 | 1 | 1 | 1 |

## identity_small_budget_success

case `P-da6a652c2d303cffc328`；anchor `TC-A-0060`；seed=17；N=200；G=1。

查询：Liam Chen's preferred grocery store is Whole Foods.

- 金标 `C-eecd9c503a9067d29587`：Liam Chen's preferred grocery store is Target.

| 方法 | 金标排名 | 第一名 |
|---|---|---|
| bm25 | 1 | Liam Chen's preferred grocery store is Target. |
| dense | 1 | Liam Chen's preferred grocery store is Target. |
| hybrid | 1 | Liam Chen's preferred grocery store is Target. |
| sparsecl | 1 | Liam Chen's preferred grocery store is Target. |

| 配对条件 | d | N | G | BM25 名次 | Dense 名次 | Hybrid 名次 | SparseCL 名次 |
|---|---:|---:|---:|---|---|---|---|
| no_injection | 0 | 200 | 1 | 1 | 1 | 1 | 1 |
| other_entity_query_value | 1 | 200 | 1 | 2 | 1 | 2 | 1 |
| other_entity_old_value | 1 | 200 | 1 | 1 | 1 | 1 | 1 |
| other_entity_query_value | 5 | 200 | 1 | 1 | 1 | 1 | 1 |
| other_entity_old_value | 5 | 200 | 1 | 1 | 1 | 1 | 1 |
| other_entity_query_value | 15 | 200 | 1 | 1 | 1 | 1 | 1 |
| other_entity_old_value | 15 | 200 | 1 | 1 | 1 | 1 | 1 |

## chinese_configuration_gap

case `P-ef325dba55579bba46b3`；anchor `TC-A-0161`；seed=17；N=200；G=1。

查询：Frank Hughes首选的咖啡品牌是Blue Bottle。

- 金标 `C-d350cfe721d32b96f1a4`：Frank Hughes's preferred coffee brand is Starbucks.

| 方法 | 金标排名 | 第一名 |
|---|---|---|
| bm25 | 4 | Liam Hughes's preferred coffee brand is Blue Bottle. |
| dense | 1 | Frank Hughes's preferred coffee brand is Starbucks. |
| hybrid | 2 | Liam Hughes's preferred coffee brand is Blue Bottle. |
| sparsecl | 18 | Sam Irwin上周买了两瓶牛奶，放在冰箱里。 |

| 配对条件 | d | N | G | BM25 名次 | Dense 名次 | Hybrid 名次 | SparseCL 名次 |
|---|---:|---:|---:|---|---|---|---|
| original | 5 | 200 | 1 | 2 | 1 | 1 | 2 |
| canonical | 5 | 200 | 1 | 2 | 1 | 1 | 2 |
| paraphrase | 5 | 200 | 1 | 2 | 2 | 2 | 2 |
| chinese | 5 | 200 | 1 | 4 | 1 | 2 | 18 |

## multiple_targets_without_capacity_deficit

case `P-f5c3a0a8c46be9ad1da3`；anchor `TC-A-0033`；seed=17；N=200；G=4。

查询：Peter Fisher's preferred grocery store is Trader Joe's.

- 金标 `C-64d68c06dc210985654c`：Peter Fisher's preferred grocery store is FreshFoods.
- 金标 `C-8e708ffa6e3e0d8c3be3`：Peter Fisher's preferred grocery store is GreenGrocers.
- 金标 `C-a5dc985416f90f19294e`：Peter Fisher's preferred grocery store is FreshMart.
- 金标 `C-d20c6ff9df2b244366be`：Peter Fisher's preferred grocery store is Costco.

| 方法 | 金标排名 | 第一名 |
|---|---|---|
| bm25 | 4, 5, 6, 7 | Yara Fisher's preferred grocery store is Trader Joe's. |
| dense | 2, 6, 4, 3 | Jack Fisher's preferred grocery store is Trader Joe's. |
| hybrid | 3, 7, 5, 4 | Jack Fisher's preferred grocery store is Trader Joe's. |
| sparsecl | 6, 5, 9, 4 | Jack Fisher's preferred grocery store is Trader Joe's. |

| 配对条件 | d | N | G | BM25 名次 | Dense 名次 | Hybrid 名次 | SparseCL 名次 |
|---|---:|---:|---:|---|---|---|---|
| distinct_observed_values | 5 | 200 | 1 | 3 | 2 | 3 | 4 |
| distinct_observed_values | 5 | 200 | 2 | 4, 5 | 4, 2 | 4, 3 | 5, 4 |
| distinct_observed_values | 5 | 200 | 4 | 4, 5, 6, 7 | 2, 6, 4, 3 | 3, 7, 5, 4 | 6, 5, 9, 4 |
| distinct_observed_values | 5 | 200 | 6 | 11, 6, 7, 8, 9, 10 | 3, 2, 8, 5, 7, 4 | 7, 3, 9, 6, 11, 8 | 3, 7, 6, 11, 8, 5 |

## scale_bm25_sensitivity

case `P-af44a46f5f5396a605dc`；anchor `TC-A-0150`；seed=17；N=1000；G=1。

查询：Chloe Garcia's preferred grocery store is Trader Joe's.

- 金标 `C-e46b12a3d16cc8188043`：Chloe Garcia's preferred grocery store is Costco.

| 方法 | 金标排名 | 第一名 |
|---|---|---|
| bm25 | 6 | Alice Garcia's preferred grocery store is Trader Joe's. |
| dense | 1 | Chloe Garcia's preferred grocery store is Costco. |
| hybrid | 3 | Alice Garcia's preferred grocery store is Trader Joe's. |
| sparsecl | 2 | Alice Garcia's preferred grocery store is Trader Joe's. |

| 配对条件 | d | N | G | BM25 名次 | Dense 名次 | Hybrid 名次 | SparseCL 名次 |
|---|---:|---:|---:|---|---|---|---|
| fixed_five_added_distractors | 5 | 50 | 1 | 3 | 1 | 2 | 2 |
| fixed_five_added_distractors | 5 | 200 | 1 | 3 | 1 | 2 | 2 |
| fixed_five_added_distractors | 5 | 1000 | 1 | 6 | 1 | 3 | 2 |

## restatement_sparsecl_success

case `P-8ee0adf607693997c5bd`；anchor `TC-A-0027`；seed=17；N=200；G=1。

查询：Sam Jensen's preferred grocery store is Trader Joe's.

- 金标 `C-8a857c1284d43c53a288`：Sam Jensen's preferred grocery store is Walmart.

| 方法 | 金标排名 | 第一名 |
|---|---|---|
| bm25 | 2 | Sam Jensen's preferred grocery store is Trader Joe's. |
| dense | 2 | Sam Jensen's preferred grocery store is Trader Joe's. |
| hybrid | 2 | Sam Jensen's preferred grocery store is Trader Joe's. |
| sparsecl | 1 | Sam Jensen's preferred grocery store is Walmart. |

| 配对条件 | d | N | G | BM25 名次 | Dense 名次 | Hybrid 名次 | SparseCL 名次 |
|---|---:|---:|---:|---|---|---|---|
| same_entity_mention | 1 | 200 | 1 | 1 | 1 | 1 | 2 |
| other_entity_mention | 1 | 200 | 1 | 1 | 1 | 1 | 1 |
| same_value_restatement | 1 | 200 | 1 | 2 | 2 | 2 | 1 |

## event_attribute

case `P-f7ffbdc520485fed729c`；anchor `AF-event_registration-05`；seed=17；N=200；G=1。

查询：The community event organized by Chloe Garcia is scheduled for day 20.

- 金标 `A-6894f74da2b054fa4c7a`：The community event organized by Chloe Garcia is scheduled for day 50.

| 方法 | 金标排名 | 第一名 |
|---|---|---|
| bm25 | 2 | Registration for the community event organized by Chloe Garcia closes on day 20. |
| dense | 2 | Registration for the community event organized by Chloe Garcia closes on day 20. |
| hybrid | 2 | Registration for the community event organized by Chloe Garcia closes on day 20. |
| sparsecl | 2 | Registration for the community event organized by Chloe Garcia closes on day 20. |

| 配对条件 | d | N | G | BM25 名次 | Dense 名次 | Hybrid 名次 | SparseCL 名次 |
|---|---:|---:|---:|---|---|---|---|
| no_injection | 0 | 200 | 1 | 1 | 1 | 1 | 1 |
| same_entity_other_attribute | 1 | 200 | 1 | 2 | 2 | 2 | 2 |
| other_entity_other_attribute | 1 | 200 | 1 | 1 | 1 | 1 | 1 |
| same_value_restatement | 1 | 200 | 1 | 2 | 2 | 2 | 1 |

## adult_child_scope

case `P-fac287bc4bf2ba3235dd`；anchor `AF-adult_child_fee-02`；seed=17；N=200；G=1。

查询：Adult entry to the park managed by Zack Fisher costs $5 per person.

- 金标 `A-70cf8fa619888720b69f`：Adult entry to the park managed by Zack Fisher costs $10 per person.

| 方法 | 金标排名 | 第一名 |
|---|---|---|
| bm25 | 1 | Adult entry to the park managed by Zack Fisher costs $10 per person. |
| dense | 1 | Adult entry to the park managed by Zack Fisher costs $10 per person. |
| hybrid | 1 | Adult entry to the park managed by Zack Fisher costs $10 per person. |
| sparsecl | 2 | Child entry to the park managed by Zack Fisher costs $5 per person. |

| 配对条件 | d | N | G | BM25 名次 | Dense 名次 | Hybrid 名次 | SparseCL 名次 |
|---|---:|---:|---:|---|---|---|---|
| no_injection | 0 | 200 | 1 | 1 | 1 | 1 | 1 |
| same_entity_other_attribute | 1 | 200 | 1 | 1 | 1 | 1 | 2 |
| other_entity_other_attribute | 1 | 200 | 1 | 1 | 1 | 1 | 1 |
| same_value_restatement | 1 | 200 | 1 | 2 | 2 | 2 | 1 |
