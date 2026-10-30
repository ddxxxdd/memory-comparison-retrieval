"""v7 实验的补充分析：异主体候选比例与未过滤 top-k 对照。

数据集 v7 已不区分个人/公共记忆（source_ref 无 scope/user_id），因此主体一致性
只能按正文人名判断：正文以"名 姓"两段式人名开头时取该人名，否则视为无主体。
本脚本只做统计，不参与检索。

运行：.venv/Scripts/python.exe scripts/analyze_v7_experiment.py
"""

"""v7 分析：主体信息只在正文里，按"正文人名"判定是否同一主体。

v7 的 source_ref 已不含 scope/user_id，所以：
- 混入 = 候选正文的人名与查询正文的人名不同；
- 分渠道统计不再可行，改为按人名统计。
"""

import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(r"C:\Users\dzp\project\paper\memory-comparison-retrieval")
RUN = BASE / "runs" / "experiments" / "v7-20261029"
DATA = BASE / "data" / "validmem_v7"

# v7 的人名都是"名 姓"两段式；单段首词多为公共事实的开头（Calculus/Recommended/History）。
NAME = re.compile(r"^\s*([A-Z][a-z]+ [A-Z][a-z]+)(?![a-z])")


def name_of(text):
    match = NAME.match(text)
    return match.group(1) if match else None


gold = {}
with (DATA / "gold.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            row = json.loads(line)
            gold[row["query_id"]] = row["conflict_old_memory_ids"]

query_name = {}
with (DATA / "queries.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            row = json.loads(line)
            query_name[row["query_id"]] = name_of(row["content"])

memory_name = {}
with (DATA / "memories.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            row = json.loads(line)
            memory_name[row["memory_id"]] = name_of(row["content"])


def load(arm):
    rows = {}
    with (RUN / "arms" / f"{arm}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[row["query_id"]] = row
    return rows


def measure(rows, k=None, unfiltered=False):
    matched = required = returned = cross = 0
    for query_id, row in rows.items():
        if query_id not in gold:
            continue
        targets = set(gold[query_id])
        if unfiltered:
            candidates = [c["memory_id"] for c in row["coarse_candidates"]][:k]
        else:
            candidates = [c["memory_id"] for c in row["candidates"]]
        got = set(candidates)
        matched += len(targets & got)
        required += len(targets)
        returned += len(candidates)
        mine = query_name.get(query_id)
        if mine:
            cross += sum(1 for m in candidates
                         if memory_name.get(m) and memory_name[m] != mine)
    return {"recall": matched / required if required else None,
            "purity": matched / returned if returned else None,
            "cand": returned / len(rows) if rows else None,
            "cross": cross, "returned": returned}


arms = ["proposed", "dense", "bm25", "hybrid"]
rows_by_arm = {arm: load(arm) for arm in arms}

print("== 过滤后 ==")
print(f"{'arm':10s} {'recall':>7s} {'purity':>7s} {'cand/q':>7s} {'异主体候选':>10s} {'占比':>7s}")
for arm in arms:
    s = measure(rows_by_arm[arm])
    rate = s["cross"] / s["returned"] if s["returned"] else 0
    print(f"{arm:10s} {s['recall']:7.4f} {s['purity']:7.4f} {s['cand']:7.2f} "
          f"{s['cross']:10d} {rate:7.3f}")

print("\n== 未做主体判断（未过滤粗候选前 10 条） ==")
print(f"{'arm':10s} {'recall@10':>9s} {'异主体候选':>10s} {'占比':>7s}")
for arm in arms:
    s = measure(rows_by_arm[arm], k=10, unfiltered=True)
    rate = s["cross"] / s["returned"] if s["returned"] else 0
    print(f"{arm:10s} {s['recall']:9.4f} {s['cross']:10d} {rate:7.3f}")

print("\n== gold 对的人名一致性 ==")
same = diff = unnamed = 0
for query_id, targets in gold.items():
    mine = query_name.get(query_id)
    for mid in targets:
        theirs = memory_name.get(mid)
        if mine and theirs:
            same += mine == theirs
            diff += mine != theirs
        else:
            unnamed += 1
print(f"同名 {same} / 不同名 {diff} / 至少一侧无人名 {unnamed}")

print("\n== 每条查询的异主体候选分布（过滤后） ==")
for arm in arms:
    counts = Counter()
    for query_id, row in rows_by_arm[arm].items():
        mine = query_name.get(query_id)
        n = sum(1 for c in row["candidates"]
                if mine and memory_name.get(c["memory_id"]) and memory_name[c["memory_id"]] != mine)
        counts[n] += 1
    print(f"{arm:10s} 有异主体候选的查询 {sum(v for k,v in counts.items() if k)}/{sum(counts.values())}，分布 {dict(sorted(counts.items()))}")
