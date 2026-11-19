"""Validate retrieval labels and compute experiment indicators."""

from collections import Counter
from statistics import mean

from .data import memory_banks, nonempty_text
from .sparsecl import ndcg_at_10


LABELS = {"conflict", "no_conflict", "uncertain"}


def validate_gold(rows, memory_ids=None):
    """Validate gold rows in either of the two schemas this project uses.

    Two-layer schema (original): ``gold_compare_ids`` + ``conflict_labels``, where the
    comparable set is larger than the conflicting set.

    Conflict-only schema (stage-1 retrieval as currently defined): the only target is the
    conflicting old memories, given as ``conflict_old_memory_ids``. There is no comparable
    set, because the task is to find every conflicting memory and nothing else.
    """
    seen = set()
    for row in rows:
        query_id = nonempty_text(row.get("query_id"), "query_id")
        if query_id in seen:
            raise ValueError(f"金标 query_id 重复: {query_id}")
        seen.add(query_id)

        if "conflict_old_memory_ids" in row and "gold_compare_ids" not in row:
            targets = row["conflict_old_memory_ids"]
            if not isinstance(targets, list):
                raise ValueError("conflict_old_memory_ids 必须是数组")
            unjudged = row.get("unjudged_old_memory_ids", [])
            if not isinstance(unjudged, list):
                raise ValueError("unjudged_old_memory_ids 必须是数组")
            for mid in targets + unjudged:
                nonempty_text(mid, "memory_id")
            if len(set(targets + unjudged)) != len(targets) + len(unjudged):
                raise ValueError("冲突金标和待定旧记忆的 ID 必须唯一且互不重叠")
            if memory_ids is not None and set(targets + unjudged) - memory_ids:
                raise ValueError(f"{query_id}: 金标引用了不存在的旧记忆")
            details = row.get("target_details")
            if details is not None:
                if not isinstance(details, dict) or set(details) != set(targets):
                    raise ValueError(f"{query_id}: target_details 必须逐条对应冲突金标")
                for detail in details.values():
                    if not isinstance(detail, dict):
                        raise ValueError(f"{query_id}: target_details 条目必须是对象")
                    nonempty_text(detail.get("source"), "target_details.source")
                    nonempty_text(detail.get("relation"), "target_details.relation")
            for key in ("fully_reviewed", "human_reviewed"):
                if key in row and not isinstance(row[key], bool):
                    raise ValueError(f"{key} 必须是布尔值")
            continue

        gold = row.get("gold_compare_ids")
        uncertain = row.get("uncertain_compare_ids", [])
        if not isinstance(gold, list) or not isinstance(uncertain, list):
            raise ValueError("gold_compare_ids 和 uncertain_compare_ids 必须是数组")
        for mid in gold + uncertain:
            nonempty_text(mid, "memory_id")
        if len(set(gold)) != len(gold) or len(set(uncertain)) != len(uncertain) or set(gold) & set(uncertain):
            raise ValueError("金标 ID 重复或与不确定 ID 重叠")
        if memory_ids is not None and (set(gold) | set(uncertain)) - memory_ids:
            raise ValueError(f"{query_id}: 金标引用了不存在的旧记忆")
        labels = row.get("conflict_labels")
        if not isinstance(labels, dict) or set(labels) != set(gold):
            raise ValueError("conflict_labels 必须为每个 gold_compare_ids 提供逐对标签")
        if any(label not in LABELS for label in labels.values()):
            raise ValueError("非法 conflict_label")
    return rows


def validate_dataset(memories, queries, gold_rows=None):
    banks = memory_banks(memories)
    for query in queries:
        if query.get("scenario_id") not in banks:
            raise ValueError(f"{query['query_id']}: 查询未匹配到同场景旧记忆库")
    if gold_rows is None:
        return banks
    validate_gold(gold_rows, {memory.memory_id for memory in memories})
    if not queries:
        return banks
    query_scenarios = {query["query_id"]: query.get("scenario_id") for query in queries}
    if set(query_scenarios) != {row["query_id"] for row in gold_rows}:
        raise ValueError("查询与金标 query_id 不一致")
    if None not in banks:
        ids_by_scenario = {scenario: {m.memory_id for m in bank} for scenario, bank in banks.items()}
        for row in gold_rows:
            qid = row["query_id"]
            refs = (row.get("gold_compare_ids", row.get("conflict_old_memory_ids", []))
                    + row.get("uncertain_compare_ids", []) + row.get("unjudged_old_memory_ids", []))
            if set(refs) - ids_by_scenario[query_scenarios[qid]]:
                raise ValueError(f"{qid}: 金标或待定项引用了其他场景的旧记忆")
    return banks


def target_ids(row):
    """The old memories a write must retrieve: conflicting ones only."""
    if "conflict_old_memory_ids" in row and "gold_compare_ids" not in row:
        return list(row["conflict_old_memory_ids"])
    labels = row.get("conflict_labels") or {}
    return [mid for mid, label in labels.items() if label == "conflict"]


def ranking_metrics(rankings, gold, cutoffs):
    """Metrics against the labeled targets. Preserve missing ranks in a truncated pool."""
    targets = {row["query_id"]: set(target_ids(row)) for row in gold}
    if set(rankings) != set(targets) or not targets:
        raise ValueError("排名必须逐条覆盖所选查询")
    positions = {}
    for qid, ids in rankings.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"{qid}: 排名有重复的旧记忆")
        locations = {mid: rank for rank, mid in enumerate(ids, 1)}
        positions[qid] = {mid: locations.get(mid) for mid in sorted(targets[qid])}
    partial = any(row.get("fully_reviewed") is False or row.get("unjudged_old_memory_ids")
                  or row.get("uncertain_compare_ids") for row in gold)
    total = sum(map(len, targets.values()))
    at_k = {}
    for k in sorted(set(cutoffs)):
        hits = {q: len(set(ids[:k]) & targets[q]) for q, ids in rankings.items()}
        count = sum(hits.values())
        returned = sum(min(k, len(ids)) for ids in rankings.values())
        fraction = count / returned if returned else 0.0
        source_totals, source_hits = Counter(), Counter()
        for row in gold:
            qid = row["query_id"]
            for mid, detail in row.get("target_details", {}).items():
                source = detail["source"]
                source_totals[source] += 1
                pos = positions[qid][mid]
                source_hits[source] += int(pos is not None and pos <= k)
        recalls = [hits[q] / len(ids) for q, ids in targets.items() if ids]
        at_k[str(k)] = {"hits": count, "gold_pairs": total, "candidates": returned,
                        "micro_recall": count / total if total else None,
                        "macro_recall": mean(recalls) if recalls else None,
                        "fully_covered_queries": sum(hits[q] == len(ids) for q, ids in targets.items()),
                        "labeled_hit_fraction": fraction, "candidate_purity": None if partial else fraction,
                        "recall_by_source": {s: {"hits": source_hits[s], "gold_pairs": n,
                                                 "candidate_recall_micro": source_hits[s] / n}
                                             for s, n in source_totals.items()}}
    binary_gold = [{"query_id": q, "conflict_old_memory_ids": ids} for q, ids in targets.items()]
    return {"at_k": at_k, "ndcg_at_10": ndcg_at_10(binary_gold, rankings), "gold_ranks": positions}
