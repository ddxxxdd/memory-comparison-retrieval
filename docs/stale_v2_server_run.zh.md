# STALE V2 背景历史构建与验收

服务器项目：`/home/dzp/paper/memory-comparison-retrieval`。

2026-11-23 按用户的新口径，历史句只通过规则抽取并作为旧库背景。构建不调用模型补标历史目标，也不新增历史待定项。金标只包含官方 `M_old` 和生成后经评审确认的合成冲突/过时句。

## 数据与标签口径

- `data/stale_scenarios/` 保留为 V1，所有文件不变。
- V2 基础版为 `data/stale_scenarios_v2/base/`，扩增版为 `data/stale_scenarios_v2/augmented/`。
- 两版沿用 `scenarios/T1/<scenario_id>/`、`scenarios/T2/<scenario_id>/`。每条查询只检索自己的场景旧库。
- V2 取消每次发言最多一句、每场会话最多三条的上限；其他筛选、规范化去重和写入前已完成会话的边界沿用 V1。重复句优先保留 V1 已选的内容、ID 和出处。
- 全部 V1 记忆和合成记忆原样继承，合成项不重新生成或评审。只从金标 ID 列表及 `target_details` 移除 `source=history_review` 的 74 项，其记忆内容仍保留。
- V1 原有 4,820 个待定标记及其他金标元数据原样继承；被移除的 74 个目标不会转成待定项。
- 以后新建 V1 扩增版时，生成器和评审器只处理合成候选。合成相容句需通过评审但不成为金标；历史句不调用模型。

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
  --report "$v2/work/final_audit.json"
```

服务器 `work/run_v2.sh` 已改为相同的离线构建流程。不需要 `.env.review`，V2 不读取评审配置或旧模型缓存。V1 合成候选评审仍可通过 `--review-env`、`--review-model` 配置模型。

## 验收与交接

`work/server_before.json` 是构建前 V1 全目录 2,411 个文件的 SHA256 快照。`work/history_policy_cache_before.json` 记录补标任务停止后的缓存和检查点哈希。旧模型缓存及检查点留存，仅用于追溯，不参与本次 V2 标签构建。

验收逐场检查原始历史的完整覆盖、用户发言出处、写入时间边界、V1 ID/内容/出处及合成标签继承。最终金标必须严格等于 V1 金标减去 74 项历史目标；不得新增待定项。两份场景索引各为 400 项，三种聚合 JSONL 与逐场文件一致。

运行日志：`work/background_build.log`；测试日志：`work/background_tests.log`；完整验收：`work/final_audit.json`。旧补标日志保存在 `work/review.log`、`work/build.log`，停止记录为 `work/history_policy_stop.json`。
