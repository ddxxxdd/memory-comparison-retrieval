# STALE V2 背景历史构建与验收

服务器项目：`/home/dzp/paper/memory-comparison-retrieval`。

本文提供操作命令及验收前提；完整字段映射、筛选阈值、合成流程和 V1/V2 例子见[数据集构建规范](stale_dataset_construction.zh.md)。截至 2026-11-23，完整 V2 `base/augmented` 及场景导出在服务器；本地只有 V2 base 和部分运行结果，不能据此认为本地 augmented 也已同步。

2026-11-23 按用户的新口径，历史句只通过规则抽取并作为旧库背景。构建不调用模型补标历史目标，也不新增历史待定项。金标只包含官方 `M_old` 和生成后经评审确认的合成冲突/过时句。

## 数据与标签口径

- `data/stale_scenarios/` 保留为 V1，所有文件不变。
- V2 基础版为 `data/stale_scenarios_v2/base/`，扩增版为 `data/stale_scenarios_v2/augmented/`。
- 两版沿用 `scenarios/T1/<scenario_id>/`、`scenarios/T2/<scenario_id>/`。每条查询只检索自己的场景旧库。
- V2 取消每次发言最多一句、每场会话最多三条的上限；其他筛选、规范化去重和写入前已完成会话的边界沿用 V1。重复句优先保留 V1 已选的内容、ID 和出处。
- 全部 V1 记忆和合成记忆原样继承，合成项不重新生成或评审。只从金标 ID 列表及 `target_details` 移除 `source=history_review` 的 74 项，其记忆内容仍保留。
- V1 原有 4,820 个待定标记及其他金标元数据原样继承；被移除的 74 个目标不会转成待定项。
- 以后新建 V1 扩增版时，生成器和评审器只处理合成候选。合成相容句需通过评审但不成为金标；历史句不调用模型。

“仅作背景”不等于经语义确认相容。待定列表不是所有未审背景的全集，新增历史也不因未列入该列表而成为已确认负例。构建和验收只验证规则及文件一致性，指标按[已标注目标口径](evaluation_protocol.zh.md)解读。

| 数量 | 预期值 |
| --- | ---: |
| 场景 | 400（T1 200，T2 200） |
| V1 继承记忆 | 44,622 |
| 新增历史背景 | 63,086 |
| V2 base 记忆 | 103,613 |
| V2 augmented 记忆 | 107,708 |
| 继承合成记忆 | 4,095 |
| 移除历史目标 | 74 |
| 官方目标 | 400 |
| 合成目标 | 1,818 |
| 最终目标 | 2,218 |
| `history_review` 目标 | 0 |
| 保留旧待定项 | 4,820 |
| 新增历史目标、待定项、合成记忆 | 均为 0 |
| V2 构建模型调用 | 0 |

## 测试、构建与导出

原始数据固定版本 SHA256：`5f3ec375179e20e2e94469e018189188f34e2e7e5f21cbecbd99fcfa648c1876`。文件位于 `data/stale_sources/T1_T2_400_FULL.json`。

以下命令针对现有服务器的冻结 V1 和既有构建快照。构建器拒绝覆盖已有输出；存在的 V2 会被核对，而不会因为目录存在就被认定正确。导出命令会按聚合文件重建 `scenarios/`。重新在其他位置构建时，显式传入新的 `--output`，最终审计也要相应传入 `--v1/--v2`；不要原地删除或覆盖已发布数据。

```bash
set -euo pipefail
cd /home/dzp/paper/memory-comparison-retrieval
py=.venv/bin/python
v1=data/stale_scenarios
v2=data/stale_scenarios_v2
src=data/stale_sources/T1_T2_400_FULL.json
"$py" -m pytest -q
# 已有经过全量核对的 V2 base 可直接复用。
if [[ ! -d "$v2/base" ]]; then
  "$py" scripts/convert_stale_scenarios.py --version v2 --source "$src" --output "$v2/base"
fi
"$py" scripts/augment_stale_scenarios.py --base "$v2/base" --inherit-v1 "$v1/augmented" \
  --output "$v2/augmented" --check-only
if [[ ! -d "$v2/augmented" ]]; then
  "$py" scripts/augment_stale_scenarios.py --base "$v2/base" --inherit-v1 "$v1/augmented" \
    --output "$v2/augmented"
fi
"$py" scripts/export_stale_scenarios.py --dataset "$v2/base" "$v2/augmented"
"$py" scripts/audit_stale_v2.py --source "$src" --snapshot "$v2/work/server_before.json" \
  --preserved-work-snapshot "$v2/work/history_policy_cache_before.json" --final \
  --report "runs/experiments/stale-v2-audit-$(date -u +%Y%m%dT%H%M%SZ).json"
```

服务器 `data/stale_scenarios_v2/work/run_v2.sh` 是已保存的构建辅助脚本，不是 Git 跟踪的通用入口。V2 离线继承不需要 `.env.review`，不读取评审配置或旧模型缓存。V1 合成候选评审仍可通过 `--review-env`、`--review-model` 配置模型；重新生成不能保证复现冻结 V1 的历史合成结果。

## 验收与交接

`work/server_before.json` 是构建前 V1 全目录 2,411 个文件的 SHA256 快照。`work/history_policy_cache_before.json` 记录补标任务停止后的缓存和检查点哈希。旧模型缓存及检查点留存，仅用于追溯，不参与本次 V2 标签构建。

本节的 `work/` 均相对于 `data/stale_scenarios_v2/`。这两份是已有服务器运行的证据文件，构建入口不会自动创建它们。V1 快照格式为 `{"v1_root":"绝对路径","v1_files_sha256":{"相对路径":"SHA256"}}`；旧缓存快照为 `{"work_root":"绝对路径","files_sha256":{"相对路径":"SHA256"}}`。在另一个环境运行时，需要在构建前为该环境生成对应快照；事后生成的快照不能作为“构建前未改动”的证据。`--final` 必须提供 V1 快照，旧缓存快照参数可选。

验收逐场检查原始历史的完整覆盖、用户发言出处、写入时间边界、V1 ID/内容/出处及合成标签继承。最终金标必须严格等于 V1 金标减去 74 项历史目标；不得新增待定项。两份场景索引各为 400 项，三种聚合 JSONL 与逐场文件一致。

运行日志：`work/background_build.log`；测试日志：`work/background_tests.log`；完整验收：`work/final_audit.json`。旧补标日志保存在 `work/review.log`、`work/build.log`，停止记录为 `work/history_policy_stop.json`。

## 运行当前方法与复测 60 场景

普通单场景或全量实验使用主包，`--dataset` 指向 `base/augmented` 或一个场景目录；不要指向上一级 `stale_scenarios_v2/` 或 `scenarios/`。三个 JSONL 必须直接位于传入目录。

```bash
cd /home/dzp/paper/memory-comparison-retrieval
.venv/bin/python -m memory_retrieval \
  --dataset data/stale_scenarios_v2/augmented/scenarios/T1/7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f \
  --arms all anchored_probes_only anchored_pos_diffusion
```

严格复测 V1 已选 60 个场景时，使用 [run_stale_v2_60scene.py](../scripts/run_stale_v2_60scene.py)。它依赖服务器保存的 V1 协议、选择、输入与排名，核对冻结主包/模型/输入哈希，并从 `.env` 检查与 V1 相同的模型配置。普通从 Git 克隆且没有这些历史产物的环境，不能直接执行该固定复测入口。

```bash
# 新建一个未存在的输出目录；会实际调用模型并重新编码。
.venv/bin/python scripts/run_stale_v2_60scene.py \
  --output "runs/experiments/stale-v2-60scene-repeat-$(date -u +%Y%m%dT%H%M%SZ)"
# 只恢复上一次准备完成的目录时，使用同一明确路径并加 --resume。
```

该脚本会把摘要写到固定的 `reports/stale_v2_60scene_bipolar_seed20261122_experiment.md`，不同复测可更新这份摘要；各次原始结果以自己的输出目录为准。已有完成实验无需为阅读文档而重跑。最新报告、数据版本和已实现指标见[评测口径](evaluation_protocol.zh.md)。
