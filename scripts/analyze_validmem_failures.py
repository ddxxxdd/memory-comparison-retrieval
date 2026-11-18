"""Read-only analysis of archived ValidMem v7 baseline rankings (no inference).

Run from the repository: .venv/Scripts/python.exe scripts/analyze_validmem_failures.py
Only this analysis's reports directory is written. Existing data/runs are hash checked.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/validmem_failure_analysis_2026-11-12"
DATA = ROOT / "data/validmem_v7"
MAIN = ROOT / "runs/experiments/validmem-v7-main-20261030"
SPARSE = ROOT / "runs/experiments/validmem-v7-sparsecl-exploratory-20261031-retry1"
METHODS = ("bm25", "dense", "hybrid", "sparsecl")
BUDGETS = (1, 2, 3, 5, 10, 20, 30, 50)
TYPE_NAMES = {
    "different_named_referent": "具名主体/所指人物不同",
    "different_object": "事项对象、学科或明示事件不同",
    "different_attribute": "属性不同或未断言待比较属性值",
    "same_value": "同主体同事项同值",
    "unresolved_scope_or_assertion": "实例范围或断言含义不明",
    "possible_missing_gold": "疑似金标遗漏",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path, key):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows), (path, "duplicate IDs")
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    memories = read_jsonl(DATA / "memories.jsonl", "memory_id")
    queries = read_jsonl(DATA / "queries.jsonl", "query_id")
    gold_rows = read_jsonl(DATA / "gold.jsonl", "query_id")
    gold = {qid: set(row["conflict_old_memory_ids"]) for qid, row in gold_rows.items()}
    assert set(queries) == set(gold)
    assert all(targets <= memories.keys() and targets for targets in gold.values())
    paths = [DATA / name for name in ("memories.jsonl", "queries.jsonl", "gold.jsonl",
             "manifest.json", "gold_audit.jsonl", "pair_label_rationales.jsonl", "provenance.jsonl")]
    for run in (MAIN, SPARSE):
        config = read_json(run / "run_config.json")
        paths.append(run / "run_config.json")
        for name in ("memories.jsonl", "queries.jsonl", "gold.jsonl"):
            assert sha256(DATA / name) == config["dataset"]["files_sha256"][name], (run, name)
        assert config["dataset"]["memory_count"] == len(memories)
        assert set(config["sample"]["query_ids"]) == set(queries)
        for name, key in (("queries.jsonl", "query_id"), ("gold.jsonl", "query_id")):
            snapshot_path = run / "input" / name
            snapshot = read_jsonl(snapshot_path, key)
            paths.append(snapshot_path)
            assert set(snapshot) == set(queries)
            for qid in queries:
                if name == "queries.jsonl":
                    assert snapshot[qid]["content"] == queries[qid]["content"]
                else:
                    assert set(snapshot[qid]["conflict_old_memory_ids"]) == gold[qid]
        assert not (run / "failure.json").exists()
    index_path = MAIN / "index/index.json"
    indexed_memories = {r["memory_id"]: r["content"] for r in read_json(index_path)["memories"]}
    assert indexed_memories == {mid: row["content"] for mid,row in memories.items()}
    sparse_input_path = SPARSE / "input/sparsecl_input.json"
    sparse_input = read_json(sparse_input_path)
    assert {r["memory_id"]:r["content"] for r in sparse_input["memories"]} == indexed_memories
    assert {r["query_id"]:r["content"] for r in sparse_input["queries"]} == {
        qid:row["content"] for qid,row in queries.items()}
    paths += [index_path, sparse_input_path, Path(__file__)]
    paths += [ROOT / p for p in ("memory_retrieval/baselines.py", "memory_retrieval/pipeline.py",
              "memory_retrieval/sparsecl.py", "scripts/run_experiment.py")]
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in paths}
    rankings, scores, raw = {}, {}, {}
    for method in METHODS:
        path = (SPARSE if method == "sparsecl" else MAIN) / "arms" / f"{method}.jsonl"
        source_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
        rows = read_jsonl(path, "query_id")
        raw[method] = rows
        assert set(rows) == set(queries), (method, "incomplete run")
        rankings[method], scores[method] = {}, {}
        for qid, row in rows.items():
            candidates = row["ranked_candidates" if method == "sparsecl" else "candidates"]
            ids = [c["memory_id"] for c in candidates]
            assert len(ids) == len(set(ids)) == (1000 if method == "sparsecl" else 50)
            assert set(ids) <= memories.keys()
            assert row["library_size"] == len(memories)
            assert row.get("status", "ok") == "ok"
            assert all(candidates[i]["score"] >= candidates[i + 1]["score"] for i in range(len(ids) - 1))
            if method == "sparsecl":
                assert row["candidates"] == candidates[:50]
                assert set(ids) == {c["memory_id"] for c in row["initial_candidates"]}
            else:
                assert row["content"] == queries[qid]["content"]
                assert all(c["content"] == memories[c["memory_id"]]["content"] for c in candidates)
            rankings[method][qid] = ids
            scores[method][qid] = {c["memory_id"]: c["score"] for c in candidates}
    ranks = {m: {q: {mid: i for i, mid in enumerate(ids, 1)} for q, ids in byq.items()}
             for m, byq in rankings.items()}
    pair_count = sum(map(len, gold.values()))
    metrics, per_query, common = [], [], []
    for k in BUDGETS:
        pair_misses, query_misses = {}, {}
        for method in METHODS:
            detail = []
            pair_misses[method], query_misses[method] = set(), set()
            for qid, targets in gold.items():
                candidates = set(rankings[method][qid][:k])
                hits = len(targets & candidates)
                missed = targets - candidates
                capacity = max(0, len(targets) - k)
                extra = min(len(targets), k) - hits
                assert len(missed) == capacity + extra
                item = dict(method=method, k=k, query_id=qid, gold_count=len(targets), hits=hits,
                            misses=len(missed), non_gold=k-hits, capacity_deficit=capacity,
                            extra_deficit=extra, full=int(not missed), zero_hit=int(hits == 0))
                detail.append(item)
                per_query.append(item)
                pair_misses[method].update((qid, mid) for mid in missed)
                if missed:
                    query_misses[method].add(qid)
            sums = {key: sum(d[key] for d in detail) for key in detail[0] if key not in ("method", "k", "query_id")}
            assert sums["gold_count"] == pair_count
            metrics.append(dict(method=method, k=k, hits=sums["hits"], misses=sums["misses"],
                                recall=sums["hits"]/pair_count, non_gold=sums["non_gold"],
                                purity=sums["hits"]/(k*len(gold)), full_queries=sums["full"],
                                failed_queries=len(gold)-sums["full"], zero_hit_queries=sums["zero_hit"],
                                capacity_deficit=sums["capacity_deficit"], extra_deficit=sums["extra_deficit"],
                                unavoidable_non_gold=k*len(gold)-sum(min(k,len(g)) for g in gold.values())))
        pair_hist = Counter(Counter(p for missing in pair_misses.values() for p in missing).values())
        query_hist = Counter(Counter(q for missing in query_misses.values() for q in missing).values())
        common.append(dict(k=k, pair_missed_by_n=dict(sorted(pair_hist.items())),
                           query_failed_by_n=dict(sorted(query_hist.items())),
                           all_four_queries=sorted(set.intersection(*query_misses.values())),
                           all_four_zero_hit_queries=sorted(q for q in queries if all(
                               not (gold[q] & set(rankings[m][q][:k])) for m in METHODS)),
                           any_failed_queries=sorted(set.union(*query_misses.values())),
                           all_four_pairs=[list(p) for p in sorted(set.intersection(*pair_misses.values()))]))
    # Independent check against both existing reports, without using their totals above.
    for report in ("validmem-v7-main-20261030-comparison.json", "validmem-v7-sparsecl-exploratory-20261031-retry1.json"):
        path = ROOT / "reports" / report
        source_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
        for point in read_json(path)["points"]:
            method, k = point["arm"], point.get("top_k")
            if method not in METHODS or k not in BUDGETS:
                continue
            if method == "sparsecl" and point["variant"] != "arguana_alpha_0.18":
                continue
            row = next(r for r in metrics if r["method"] == method and r["k"] == k)
            assert row["hits"] == point["retrieved_conflict_pairs"]
            assert abs(row["purity"] - point["candidate_purity"]) < 1e-12
    gold_ranks = [dict(query_id=q, memory_id=mid, gold_count=len(gold[q]),
                       **{m: ranks[m][q].get(mid, ">50" if m != "sparsecl" else ">1000") for m in METHODS},
                       query_text=queries[q]["content"], memory_text=memories[mid]["content"])
                  for q in sorted(gold) for mid in sorted(gold[q])]
    bounds = (1,2,3,5,10,20,30,50)
    histograms = {}
    for method in METHODS:
        hist = Counter()
        for row in gold_ranks:
            rank = row[method]
            label = next((str(b) for b in bounds if isinstance(rank,int) and rank <= b), ">50")
            hist[label] += 1
        assert sum(hist.values()) == pair_count
        histograms[method] = dict(hist)
    # Fixed, exhaustive semantic-audit cohorts: every top-1 non-gold pair, and every
    # top-10 non-gold pair from a method-query that fails to cover its gold at k=10.
    audit_pairs = {}
    for method in METHODS:
        for qid in queries:
            for k, cohort in ((1, "top1_non_gold"), (10, "top10_failed_query_non_gold")):
                ids = rankings[method][qid][:k]
                if k == 10 and gold[qid] <= set(ids):
                    continue
                for mid in set(ids) - gold[qid]:
                    audit_pairs.setdefault((qid,mid), {"cohorts": set(), "occurrences": []})
                    audit_pairs[qid,mid]["cohorts"].add(cohort)
                    audit_pairs[qid,mid]["occurrences"].append(dict(method=method,k=k,rank=ranks[method][qid][mid]))
    audit_packet = [dict(query_id=q, memory_id=mid, query_text=queries[q]["content"],
                         memory_text=memories[mid]["content"], cohorts=sorted(info["cohorts"]),
                         occurrences=info["occurrences"], ranks={m:ranks[m][q].get(mid,">50") for m in METHODS},
                         gold=[r for r in gold_ranks if r["query_id"]==q])
                    for (q,mid),info in sorted(audit_pairs.items())]
    review_qids = next(c for c in common if c["k"]==5)["any_failed_queries"]
    dossiers = []
    for qid in review_qids:
        candidate_ids = set().union(*(set(rankings[m][qid][:10]) for m in METHODS)) - gold[qid]
        dossiers.append(dict(query_id=qid, content=queries[qid]["content"],
                             gold=[r for r in gold_ranks if r["query_id"]==qid],
                             distractors=[dict(memory_id=mid, content=memories[mid]["content"],
                                               ranks={m:ranks[m][qid].get(mid,">50") for m in METHODS})
                                          for mid in sorted(candidate_ids)]))
    bge_ranks = {q:{c["memory_id"]:i for i,c in enumerate(row["cosine_ranked_candidates"],1)}
                for q,row in raw["sparsecl"].items()}
    bge_comparison = dict(initial_gold_hits=sum(len(gold[q]&bge_ranks[q].keys()) for q in gold),
                          gold_rank_change=dict(Counter("same" if bge_ranks[q][mid]==ranks["sparsecl"][q][mid]
                                                       else "up" if ranks["sparsecl"][q][mid]<bge_ranks[q][mid] else "down"
                                                       for q in gold for mid in gold[q])), budgets=[])
    for k in BUDGETS:
        changed, gained, lost, sets_changed = 0, 0, 0, 0
        for q in gold:
            before = set(list(bge_ranks[q])[:k])
            after = set(rankings["sparsecl"][q][:k])
            sets_changed += before != after
            changed += (before & gold[q]) != (after & gold[q])
            gained += len((after-before)&gold[q])
            lost += len((before-after)&gold[q])
        bge_comparison["budgets"].append(dict(k=k, gold_changed_queries=changed,gained=gained,lost=lost,candidate_sets_changed=sets_changed))
    summary = dict(memory_count=len(memories), query_count=len(queries), gold_pairs=pair_count,
                   gold_count_histogram=dict(sorted(Counter(map(len,gold.values())).items())),
                   human_reviewed=sum(r.get("human_reviewed",False) for r in gold_rows.values()),
                   fully_reviewed=sum(r.get("fully_reviewed",False) for r in gold_rows.values()),
                   metrics=metrics, common=common, rank_histograms=histograms, bge_comparison=bge_comparison,
                   audit_cohorts={cohort:dict(unique_pairs=sum(cohort in p["cohorts"] for p in audit_packet),
                                             method_pair_occurrences=sum(sum(o["k"]==k for o in p["occurrences"]) for p in audit_packet))
                                  for cohort,k in (("top1_non_gold",1),("top10_failed_query_non_gold",10))})
    summary["prefix_groups"] = {
        prefix: {m:dict(queries=sum(q.startswith(prefix) for q in queries),
                       gold_pairs=sum(len(gold[q]) for q in gold if q.startswith(prefix)),
                       hits_at_10=sum(len(gold[q]&set(rankings[m][q][:10])) for q in gold if q.startswith(prefix)))
                 for m in METHODS} for prefix in ("TC-A", "TW")}
    contains_han = lambda value: any("\u4e00" <= char <= "\u9fff" for char in value)
    summary["han_presence_mismatch_gold_pairs"] = [
        dict(query_id=r["query_id"], memory_id=r["memory_id"])
        for r in gold_ranks if contains_han(r["query_text"]) != contains_han(r["memory_text"])]
    top1_pairs = [p for p in audit_packet if "top1_non_gold" in p["cohorts"]]
    summary["top1_audit_concentration"] = dict(
        unique_queries=len({p["query_id"] for p in top1_pairs}),
        pairs_with_literal_grocery_in_query=sum("grocery" in p["query_text"].lower() for p in top1_pairs),
        candidate_memory_frequency=dict(Counter(p["memory_id"] for p in top1_pairs).most_common()))
    summary["extra_deficit_context_at_10"] = {
        context: {m:sum(r["extra_deficit"] for r in per_query
                        if r["method"]==m and r["k"]==10 and r["query_id"] in qids) for m in METHODS}
        for context,qids in (("cross_language_case", {"TW-0018"}),
                             ("paraphrase_and_scope_case", {"TW-0020"}),
                             ("cross_course_cases", {"TW-0002", "TW-0013"}))}
    duplicate_row = raw["sparsecl"]["TC-A-0123"]
    duplicate_cosine = next(c["score"] for c in duplicate_row["cosine_ranked_candidates"] if c["memory_id"]=="M-A-0312")
    duplicate_fused = scores["sparsecl"]["TC-A-0123"]["M-A-0312"]
    summary["duplicate_case_score_check"] = dict(
        query_id="TC-A-0123", memory_id="M-A-0312", cosine=duplicate_cosine,
        fused=duplicate_fused, inferred_hoyer=(duplicate_fused-duplicate_cosine)/duplicate_row["sparsecl_alpha"])
    annotation_path = OUT / "semantic_annotations.jsonl"
    if annotation_path.exists():
        annotations = [json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_pair = {(a["query_id"],a["memory_id"]):a for a in annotations}
        assert len(by_pair)==len(annotations) and set(by_pair)==set(audit_pairs), "annotation cohort mismatch"
        for row in audit_packet:
            row["annotation"] = by_pair[row["query_id"],row["memory_id"]]
        source_hashes[str(annotation_path.relative_to(ROOT)).replace("\\", "/")] = sha256(annotation_path)
        summary["semantic_counts"] = {}
        for cohort,k in (("top1_non_gold",1),("top10_failed_query_non_gold",10)):
            pairs = [p for p in audit_packet if cohort in p["cohorts"]]
            summary["semantic_counts"][cohort] = dict(
                unique_pairs=dict(Counter(p["annotation"]["primary_type"] for p in pairs)),
                by_method={m:dict(Counter(p["annotation"]["primary_type"] for p in pairs
                                          for o in p["occurrences"] if o["method"]==m and o["k"]==k)) for m in METHODS})
        lines = ["# 语义审阅台账", "", "逐条读取原文后的单助手审阅，不是独立人工标注。`text_supported_non_conflict` 指按本地 v7 定义、正文支持非冲突；不证明模型内部错误机制。`unresolved` 与 `possible_label_gap` 不计为已确认负例。排名顺序统一为 BM25 / Dense / Hybrid / SparseCL；`>50` 为截断，非第 51 名。", ""]
        for i,row in enumerate(audit_packet,1):
            note = row["annotation"]
            lines += [f"## {i}. {row['query_id']} × {row['memory_id']}", "",
                      f"- 新记忆：{row['query_text']}", f"- 历史记忆：{row['memory_text']}",
                      f"- 排名：{' / '.join(str(row['ranks'][m]) for m in METHODS)}",
                      f"- 审阅范围：{', '.join(row['cohorts'])}",
                      f"- 类型：{TYPE_NAMES[note['primary_type']]}；证据状态：`{note['evidence_status']}`。",
                      f"- 依据：{note['observation']}", ""]
        (OUT / "semantic_audit.md").write_text("\n".join(lines),encoding="utf-8")
    write_csv("metrics.csv", metrics)
    write_csv("per_query.csv", per_query)
    write_csv("gold_ranks.csv", gold_ranks)
    write_json("analysis.json", summary)
    write_json("audit_packet.json", audit_packet)
    write_json("failure_dossiers.json", dossiers)
    write_json("source_manifest.json", source_hashes)
    lines = ["# 可复算统计表", "", "由 `scripts/analyze_validmem_failures.py` 直接生成。非金标候选以现有标签计，不等于最终判定器的冲突误报。", "",
             "| k | 方法 | 命中 | 漏检 | micro召回 | 非金标候选 | 纯度 | 找全/181 | 零命中查询 | 容量缺口 | 额外缺口 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['k']} | {row['method']} | {row['hits']} | {row['misses']} | {row['recall']:.2%} | {row['non_gold']} | {row['purity']:.2%} | {row['full_queries']} | {row['zero_hit_queries']} | {row['capacity_deficit']} | {row['extra_deficit']} |")
    lines += ["", "## 金标排名分布", "", "计数单位为查询—金标对；区间不累计。", "",
              "| 方法 | 1 | 2 | 3 | 4–5 | 6–10 | 11–20 | 21–30 | 31–50 | >50 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        lines.append("| " + m + " | " + " | ".join(str(histograms[m].get(str(b),0)) for b in (*bounds,">50")) + " |")
    lines += ["", "## 共同失败", "", "| k | 至少一法未找全查询 | 四法均未找全查询 | 四法均零命中查询 | 至少一法漏掉的去重金标对 | 四法共同漏掉的金标对 |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in common:
        lines.append(f"| {row['k']} | {len(row['any_failed_queries'])} | {len(row['all_four_queries'])} | {len(row['all_four_zero_hit_queries'])} | {sum(row['pair_missed_by_n'].values())} | {len(row['all_four_pairs'])} |")
    if "semantic_counts" in summary:
        for cohort,counts in summary["semantic_counts"].items():
            lines += ["", f"## {cohort}", "", "方法列按方法—配对计数；去重列跨方法去重。", "",
                      "| 类型 | 去重配对数 | BM25 | Dense | Hybrid | SparseCL |", "|---|---:|---:|---:|---:|---:|"]
            for typ,label in TYPE_NAMES.items():
                lines.append(f"| {label} | {counts['unique_pairs'].get(typ,0)} | " + " | ".join(str(counts['by_method'][m].get(typ,0)) for m in METHODS) + " |")
    (OUT / "statistics.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # Verify every read-only source again after producing outputs.
    assert all(sha256(ROOT / p)==digest for p,digest in source_hashes.items())
    print(json.dumps(dict(queries=len(queries), gold_pairs=pair_count, audit_pairs=len(audit_packet),
                          cohorts=summary["audit_cohorts"], failed_queries_at_5=review_qids,
                          common_at_10=next(c for c in common if c["k"]==10)),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
