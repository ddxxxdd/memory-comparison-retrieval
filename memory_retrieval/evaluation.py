"""Compute the experiment indicators on the supplied gold rows."""

import numpy as np

from .data import canonical_json, nonempty_text


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
            for mid in targets:
                nonempty_text(mid, "memory_id")
            if len(set(targets)) != len(targets):
                raise ValueError("conflict_old_memory_ids 有重复")
            if memory_ids is not None and set(targets) - memory_ids:
                raise ValueError(f"{query_id}: 冲突金标引用了不存在的旧记忆")
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


def target_ids(row):
    """The old memories a write must retrieve: conflicting ones only."""
    if "conflict_old_memory_ids" in row and "gold_compare_ids" not in row:
        return list(row["conflict_old_memory_ids"])
    labels = row.get("conflict_labels") or {}
    return [mid for mid, label in labels.items() if label == "conflict"]


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _percentiles(values):
    return {"p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None}


def evaluate(gold_rows, results):
    validate_gold(gold_rows)
    if not gold_rows or not results:
        raise ValueError("金标与结果不能为空")
    result_by_id = {row["query_id"]: row for row in results}
    gold_by_id = {row["query_id"]: row for row in gold_rows}
    if len(result_by_id) != len(results) or set(result_by_id) != set(gold_by_id):
        raise ValueError("结果必须恰好覆盖全部金标查询，且不能重复或混入其他查询")
    signatures = {canonical_json({key: r.get(key) for key in (
        "index_id", "method", "rerank", "selection_policy", "similarity_threshold",
        "coarse_k", "final_k", "judged", "llm", "embedding", "prompt_hash", "library_size"
    )}) for r in results}
    if len(signatures) > 1:
        raise ValueError("一次评测不能混合不同索引、方法、模型、提示或预算")
    for result in results:
        coarse_ids = [r["memory_id"] for r in result["coarse_candidates"]]
        final_ids = [r["memory_id"] for r in result["candidates"]]
        policy = result.get("selection_policy", "top_k")
        invalid_selection = False
        if policy == "top_k":
            coarse_k, final_k = result.get("coarse_k"), result.get("final_k")
            invalid_selection = (not isinstance(coarse_k, int) or not isinstance(final_k, int)
                                 or len(coarse_ids) > coarse_k or len(final_ids) > final_k)
        elif policy == "similarity_threshold":
            threshold = result.get("similarity_threshold")
            invalid_selection = (isinstance(threshold, bool)
                                 or not isinstance(threshold, (int, float))
                                 or not np.isfinite(threshold)
                                 or not 0 <= threshold <= 1
                                 or any(not isinstance(row.get("coarse_score"), (int, float))
                                        or not np.isfinite(row["coarse_score"])
                                        or row["coarse_score"] < threshold
                                        for row in result["coarse_candidates"])
                                 or any(not isinstance(row.get("fine_score" if result["rerank"] else "coarse_score"),
                                                      (int, float))
                                        or not np.isfinite(row["fine_score" if result["rerank"] else "coarse_score"])
                                        or row["fine_score" if result["rerank"] else "coarse_score"] < threshold
                                        for row in result["candidates"]))
        else:
            invalid_selection = True
        if (len(coarse_ids) != len(set(coarse_ids)) or len(final_ids) != len(set(final_ids))
                or not set(final_ids) <= set(coarse_ids) or invalid_selection):
            raise ValueError("结果中的候选重复、超出预算或不属于粗候选集合")
        judgments = result["judgments"]
        judged_ids = [row["memory_id"] for row in judgments]
        if (len(judged_ids) != len(set(judged_ids))
                or (result["judged"] and set(judged_ids) != set(final_ids))
                or (not result["judged"] and judgments)
                or any(row["conflict_label"] not in LABELS for row in judgments)):
            raise ValueError("冲突判定必须完整、唯一地对应本次最终候选")

    eligible, excluded = [], []
    for gold in gold_rows:
        if gold.get("uncertain_compare_ids"):
            excluded.append(gold["query_id"])
        else:
            eligible.append((gold, result_by_id[gold["query_id"]]))

    matched = required = returned = false_candidates = 0
    query_recalls = []
    query_full_coverage = []
    candidate_counts = []
    for gold, result in eligible:
        # 冲突-only 金标里,检索目标就是冲突旧记忆;两层金标仍以应比较集合为目标。
        targets = set(gold["gold_compare_ids"]) if "gold_compare_ids" in gold else set(target_ids(gold))
        candidates = {row["memory_id"] for row in result["candidates"]}
        query_hits = len(targets & candidates)
        matched += query_hits
        required += len(targets)
        returned += len(candidates)
        false_candidates += len(candidates - targets)
        query_recalls.append(query_hits / len(targets) if targets else 1.0)
        query_full_coverage.append(int(targets <= candidates))
        candidate_counts.append(len(candidates))
    retrieval = {
        "evaluated_queries": len(eligible),
        "retrieved_required_memories": matched,
        "required_memories": required,
        "candidate_recall": _rate(matched, required),
        "candidate_recall_micro": _rate(matched, required),
        "candidate_recall_macro": float(np.mean(query_recalls)) if query_recalls else None,
        "full_coverage": float(np.mean(query_full_coverage)) if query_full_coverage else None,
        "returned_candidates": returned,
        "non_comparable_candidates": false_candidates,
        "candidate_purity": _rate(matched, returned),
        "candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else None,
        "candidate_count_p95": float(np.percentile(candidate_counts, 95)) if candidate_counts else None,
        "candidate_count_max": max(candidate_counts) if candidate_counts else None,
    }

    conflict = None
    if all(row["judged"] for row in results):
        tp = fp = fn = tn = 0
        for gold, result in eligible:
            if "conflict_labels" in gold:
                labels = gold["conflict_labels"]
                positive = {mid for mid, label in labels.items() if label == "conflict"}
                uncertain = {mid for mid, label in labels.items() if label == "uncertain"}
            else:
                positive = set(gold["conflict_old_memory_ids"])
                uncertain = set(gold.get("uncertain_compare_ids") or [])
            judgments = {row["memory_id"]: row["conflict_label"] for row in result["judgments"]}
            predicted = {mid for mid, label in judgments.items() if label == "conflict" and mid not in uncertain}
            tp += len(positive & predicted)
            fn += len(positive - predicted)
            fp += len(predicted - positive)
            known_non_conflicts = set(judgments) - positive - uncertain
            tn += len(known_non_conflicts - predicted)
        conflict = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1": _rate(2 * tp, 2 * tp + fp + fn),
            "false_alarm_rate": _rate(fp, fp + tn),
            "miss_rate": _rate(fn, tp + fn),
        }

    stages = {"extraction": [], "retrieval": [], "reranking": [], "llm_judgment": [], "total": []}
    for row in results:
        timing = row["timing"]
        stages["extraction"].append(1000 * timing.get("description", 0.0))
        stages["retrieval"].append(1000 * timing.get("coarse_retrieval", timing.get("retrieval", 0.0)))
        stages["reranking"].append(1000 * (timing.get("hypotheses", 0.0) + timing.get("fine_ranking", 0.0)))
        if row["judged"]:
            stages["llm_judgment"].append(1000 * timing.get("conflict_judgment", 0.0))
        stages["total"].append(1000 * timing["total_seconds"])
    return {
        "total_queries": len(results),
        "library_size": results[0].get("library_size"),
        "candidate_retrieval": retrieval,
        "conflict_classification": conflict,
        "write_latency_ms": {stage: _percentiles(values) for stage, values in stages.items()},
        "excluded_queries": excluded,
    }
