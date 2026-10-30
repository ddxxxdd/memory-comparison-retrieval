"""Offline budget comparison for a saved ValidMem v7 retrieval run.

No model or embedding calls are made. BM25, Dense and Hybrid use their saved native
rankings at several top-k budgets; proposed is reported at its saved dynamic-threshold
working point. The query set and gold must match the run's frozen inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs" / "experiments" / "v7-20261029"
DEFAULT_DATASET = ROOT / "data" / "validmem_v7"
DEFAULT_OUTPUT = ROOT / "reports" / "v7_experiment_2026-10-29.md"
DEFAULT_KS = (1, 2, 3, 5, 10, 20, 30, 50)
BASELINES = ("bm25", "dense", "hybrid")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def metrics(gold_rows, candidates_by_query):
    hits_total = targets_total = returned_total = 0
    query_recalls = []
    full_coverage = []
    candidate_counts = []
    for gold in gold_rows:
        query_id = gold["query_id"]
        targets = set(gold["conflict_old_memory_ids"])
        candidate_ids = list(candidates_by_query[query_id])
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"{query_id}: 候选排序中 memory_id 重复")
        if not targets:
            raise ValueError(f"{query_id}: v7 冲突-only 金标为空")
        found = len(targets & set(candidate_ids))
        hits_total += found
        targets_total += len(targets)
        returned_total += len(candidate_ids)
        query_recalls.append(found / len(targets))
        full_coverage.append(int(targets <= set(candidate_ids)))
        candidate_counts.append(len(candidate_ids))

    return {
        "evaluated_queries": len(gold_rows),
        "conflict_pairs": targets_total,
        "retrieved_conflict_pairs": hits_total,
        "recall_micro": hits_total / targets_total if targets_total else None,
        "recall_macro": statistics.mean(query_recalls) if query_recalls else None,
        "full_coverage": statistics.mean(full_coverage) if full_coverage else None,
        "candidate_purity": hits_total / returned_total if returned_total else None,
        "candidate_count_mean": statistics.mean(candidate_counts) if candidate_counts else None,
        "candidate_count_p95": percentile(candidate_counts, 95),
        "candidate_count_max": max(candidate_counts, default=0),
        "returned_candidates": returned_total,
    }


def grouped_bootstrap(gold_rows, candidates_by_query, query_groups, samples=2000, seed=20261030):
    query_ids = {row["query_id"] for row in gold_rows}
    if set(query_groups) != query_ids:
        raise ValueError("query_groups.json 必须恰好覆盖本次运行中的所有查询")
    group_to_queries = {}
    for query_id, group_id in query_groups.items():
        group_to_queries.setdefault(group_id, []).append(query_id)
    groups = sorted(group_to_queries)
    if len(groups) < 2:
        return None

    gold_by_id = {row["query_id"]: row for row in gold_rows}
    per_query = {}
    for query_id in query_ids:
        targets = set(gold_by_id[query_id]["conflict_old_memory_ids"])
        candidates = set(candidates_by_query[query_id])
        per_query[query_id] = {
            "hits": len(targets & candidates),
            "targets": len(targets),
            "returned": len(candidates),
            "full": int(targets <= candidates),
            "recall": len(targets & candidates) / len(targets) if targets else 1.0,
        }

    rng = random.Random(seed)
    draws = {name: [] for name in ("recall_micro", "recall_macro", "full_coverage",
                                   "candidate_purity", "candidate_count_mean")}
    for _ in range(samples):
        sampled_groups = [rng.choice(groups) for _ in groups]
        sampled_queries = [query_id for group_id in sampled_groups
                           for query_id in group_to_queries[group_id]]
        rows = [per_query[query_id] for query_id in sampled_queries]
        hits = sum(row["hits"] for row in rows)
        targets = sum(row["targets"] for row in rows)
        returned = sum(row["returned"] for row in rows)
        draws["recall_micro"].append(hits / targets if targets else 0.0)
        draws["recall_macro"].append(statistics.mean(row["recall"] for row in rows))
        draws["full_coverage"].append(statistics.mean(row["full"] for row in rows))
        draws["candidate_purity"].append(hits / returned if returned else 0.0)
        draws["candidate_count_mean"].append(returned / len(rows) if rows else 0.0)
    return {name: {"low": percentile(values, 2.5), "high": percentile(values, 97.5)}
            for name, values in draws.items()}


def candidate_points(run_dir, gold_rows, ks):
    query_ids = {row["query_id"] for row in gold_rows}
    points = []
    details = {}
    arms_dir = run_dir / "arms"

    for arm in ("proposed", *BASELINES):
        path = arms_dir / f"{arm}.jsonl"
        if not path.is_file():
            raise ValueError(f"运行结果缺少方法臂：{path}")
        rows = read_jsonl(path)
        by_id = {row["query_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != query_ids:
            raise ValueError(f"{arm}: 结果必须恰好覆盖冻结输入中的每条查询一次")
        details[arm] = by_id

    proposed = details["proposed"]
    proposed_candidates = {qid: [row["memory_id"] for row in result["candidates"]]
                           for qid, result in proposed.items()}
    proposed_thresholds = {result.get("similarity_threshold") for result in proposed.values()}
    if len(proposed_thresholds) != 1:
        raise ValueError("proposed 的动态阈值在同一方法臂中不一致")
    points.append({"arm": "proposed", "budget": "dynamic", "threshold": next(iter(proposed_thresholds)),
                   **metrics(gold_rows, proposed_candidates)})
    candidate_sets = {"proposed:dynamic": proposed_candidates}

    max_k = max(ks)
    for arm in BASELINES:
        by_id = details[arm]
        for qid, result in by_id.items():
            ranking = result.get("coarse_candidates") or []
            if len(ranking) < min(max_k, result.get("library_size", max_k)):
                raise ValueError(
                    f"{arm}/{qid}: 保存的排序深度 {len(ranking)} 不足以评估 top-{max_k}；"
                    "请以 --coarse-k 50 重跑后再做预算分析。"
                )
            ranked_ids = [row["memory_id"] for row in ranking]
            if len(ranked_ids) != len(set(ranked_ids)):
                raise ValueError(f"{arm}/{qid}: 保存排序中 memory_id 重复")
            expected_final = ranked_ids[:len(result.get("candidates") or [])]
            actual_final = [row["memory_id"] for row in result.get("candidates") or []]
            if actual_final != expected_final:
                raise ValueError(f"{arm}/{qid}: top-k 候选与保存的原生排序不一致")

        for k in ks:
            selected = {qid: [row["memory_id"] for row in result["coarse_candidates"][:k]]
                        for qid, result in by_id.items()}
            points.append({"arm": arm, "budget": f"top-{k}", "threshold": None,
                           "top_k": k, **metrics(gold_rows, selected)})
            candidate_sets[f"{arm}:top-{k}"] = selected
    return points, details, candidate_sets


def costs_by_arm(details, run_config):
    rows = []
    index_info = run_config.get("index") or {}
    has_isolated_cache = run_config.get("cache_policy") == "isolated"
    bm25_index_recorded = isinstance(index_info.get("bm25_build_seconds"), (int, float))
    for arm, by_id in details.items():
        results = list(by_id.values())
        usages = [row.get("usage") or {} for row in results]
        latencies = [1000 * (row.get("timing") or {}).get("total_seconds", 0) for row in results]
        retrieval_latencies = [1000 * (row.get("timing") or {}).get(
            "retrieval", (row.get("timing") or {}).get("coarse_retrieval", 0)) for row in results]
        calls = {"llm_attempts": 0, "embedding_attempts": 0,
                 "llm_cache_hits": 0, "embedding_cache_hits": 0,
                 "llm_prompt_tokens": 0, "llm_completion_tokens": 0}
        for result in results:
            for event in result.get("api_events") or []:
                stage = str(event.get("stage") or "")
                bucket = "embedding" if "embedding" in stage else "llm"
                if event.get("cached") is True:
                    calls[f"{bucket}_cache_hits"] += 1
                elif event.get("cached") is False:
                    # APIClient logs one event per actual network attempt, including retries.
                    calls[f"{bucket}_attempts"] += 1
                else:
                    # invalid_output is a local validation event after a recorded API response.
                    continue
                if event.get("cached") is False and bucket == "llm":
                    event_usage = event.get("usage") or {}
                    calls["llm_prompt_tokens"] += event_usage.get("prompt_tokens", 0) or 0
                    calls["llm_completion_tokens"] += event_usage.get("completion_tokens", 0) or 0
        comparable = has_isolated_cache and (arm not in {"bm25", "hybrid"} or bm25_index_recorded)
        rows.append({
            "arm": arm,
            "api_attempts": sum(row.get("api_attempts", 0) for row in usages),
            "cache_hits": sum(row.get("cache_hits", 0) for row in usages),
            "known_prompt_tokens": sum(row.get("known_prompt_tokens", 0) for row in usages),
            "known_completion_tokens": sum(row.get("known_completion_tokens", 0) for row in usages),
            "known_total_tokens": sum(row.get("known_total_tokens", 0) for row in usages),
            **calls,
            "query_total_p50_ms": percentile(latencies, 50),
            "query_total_p95_ms": percentile(latencies, 95),
            "retrieval_p50_ms": percentile(retrieval_latencies, 50),
            "retrieval_p95_ms": percentile(retrieval_latencies, 95),
            "latency_comparable": comparable,
        })
    return rows


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(run_dir, dataset_dir, run_config, points, costs, gold_rows, output_path, ks,
                  group_count, bootstrap_samples):
    settings = run_config.get("settings") or {}
    split = run_config.get("evaluation_split", "exploratory (legacy run; split not recorded)")
    sample = run_config.get("sample") or {}
    index = run_config.get("index") or {}
    depth = settings.get("baseline_budget_depth", settings.get("coarse_k", "unknown"))
    lines = [
        "# ValidMem v7 原生 baseline 预算比较",
        "",
        f"- 运行：`{run_dir.name}`（评估标记：`{split}`）",
        f"- 数据：`{dataset_dir.name}`；{len(sample.get('query_ids') or [])} 条查询、"
        f"{sum(len(row['conflict_old_memory_ids']) for row in gold_rows)} 个冲突金标；"
        f"旧记忆 {run_config['dataset']['memory_count']} 条。",
        f"- baseline 保存排序深度：top-{depth}；预算网格：{', '.join(map(str, ks))}。",
        f"- proposed 动态阈值：{fmt(points[0]['threshold'])}。",
        "",
        "proposed 保留动态阈值候选；BM25、Dense、Hybrid 保留原生 top-k 排序，"
        "没有给 baseline 添加主体/事项筛选模块。各行使用同一查询、旧库与冲突金标。",
        "",
        "## 召回—候选负担曲线数据",
        "",
        f"baseline 各预算点从本次保存的同一条 top-{depth} 排序截取得到，不额外调用模型。"
        " recall_micro 按冲突配对汇总，recall_macro 对每条查询等权；完整找全率是所有冲突金标均被找回的查询占比。",
        "",
        "| 方法 | 设置 | micro 召回 | macro 召回 | 完整找全率 | 候选纯度 | 候选均值 | p95 | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in points:
        budget = f"阈值 {fmt(row['threshold'])}" if row["arm"] == "proposed" else row["budget"]
        lines.append(
            f"| {row['arm']} | {budget} | {fmt(row['recall_micro'])} | {fmt(row['recall_macro'])} | "
            f"{fmt(row['full_coverage'])} | {fmt(row['candidate_purity'])} | "
            f"{fmt(row['candidate_count_mean'], 2)} | {fmt(row['candidate_count_p95'], 1)} | "
            f"{row['candidate_count_max']} |"
        )

    lines.extend(["", "## 按场景/模板族分组 Bootstrap 95% 区间", ""])
    if group_count is None:
        lines.append("未计算：运行输入没有经核验的 `query_groups.json`。查询可能共享场景或模板，不能假设每条查询相互独立。")
    elif group_count < 2:
        lines.append("未计算：有效分组少于 2 组，无法估计组间不确定性。")
    else:
        lines.extend([
            f"按 {group_count} 个场景/模板族整组有放回抽样 {bootstrap_samples:,} 次。"
            "区间描述当前样本的组间波动，不替代独立测试集。",
            "",
            "| 方法 | 设置 | micro 召回 95% CI | 候选纯度 95% CI | 候选均值 95% CI |",
            "|---|---|---:|---:|---:|",
        ])
        for row in points:
            ci = row.get("group_bootstrap_95ci")
            if not ci:
                continue
            budget = f"阈值 {fmt(row['threshold'])}" if row["arm"] == "proposed" else row["budget"]
            rec = ci["recall_micro"]
            purity = ci["candidate_purity"]
            count = ci["candidate_count_mean"]
            lines.append(f"| {row['arm']} | {budget} | [{fmt(rec['low'])}, {fmt(rec['high'])}] | "
                         f"[{fmt(purity['low'])}, {fmt(purity['high'])}] | "
                         f"[{fmt(count['low'], 2)}, {fmt(count['high'], 2)}] |")

    lines.extend([
        "",
        "## 运行成本记录",
        "",
        "| 方法 | LLM 网络尝试 | 嵌入网络尝试 | 缓存命中 | LLM prompt/completion token | 已知总 token | 查询 p50/p95 ms | 可作方法间时延比较 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in costs:
        latency_note = "可作诊断" if row["latency_comparable"] else "否：旧运行配置不足以控制缓存/建索引成本"
        lines.append(
            f"| {row['arm']} | {row['llm_attempts']} | {row['embedding_attempts']} | "
            f"{row['llm_cache_hits'] + row['embedding_cache_hits']} | "
            f"{row['llm_prompt_tokens']} / {row['llm_completion_tokens']} | {row['known_total_tokens']} | "
            f"{fmt(row['query_total_p50_ms'], 1)} / {fmt(row['query_total_p95_ms'], 1)} | {latency_note} |"
        )

    build_usage = index.get("build_usage") or {}
    lines.extend([
        "",
        "## 离线索引构建成本",
        "",
        f"- 共享旧记忆索引（内容向量与描述目录）：{fmt(index.get('build_seconds'), 1)} 秒；"
        f"API 网络尝试 {build_usage.get('api_attempts', '未记录')} 次，"
        f"缓存命中 {build_usage.get('cache_hits', '未记录')} 次，"
        f"已知 token {build_usage.get('known_total_tokens', '未记录')}。",
        f"- BM25 索引：{fmt(index.get('bm25_build_seconds'), 4)} 秒，"
        f"{index.get('bm25_document_count', '未记录')} 条记忆；不调用 API。",
    ])

    caveats = [
        "这 181 条查询已用于数据修复、提示词和流程诊断，结果属于探索性分析；不能把其中的最优 top-k 或阈值当作独立测试结果。",
        "正式测试应使用新建且人工核验的查询/金标，按场景或模板族整组留出；在开发集选定 top-k 与动态阈值后冻结设置。测试集需与当前 v7 查询 ID 和正文完全不重叠。",
        "本报告的多 top-k 指标是对已存排序的离线截断，适合比较召回与候选数；每个预算点不是一次独立运行的时延测量。",
        "时延来自单次运行，未做重复运行或置信区间估计；即使新运行缓存隔离，也只作成本诊断。",
    ]
    if not run_config.get("cache_policy"):
        caveats.append("该历史运行没有记录按方法隔离的缓存策略；各方法的缓存命中状态不同，时延不可作公平胜负结论。")
    if not isinstance(index.get("bm25_build_seconds"), (int, float)):
        caveats.append("该历史运行没有单独记录 BM25 索引构建；旧代码在每条查询新建 Retriever 时懒构建 BM25，因此 BM25/Hybrid 的检索计时混入了重复建索引成本。修复后的运行会把 BM25 索引只构建一次并单独记录。")
    if not index.get("build_seconds"):
        caveats.append("该运行复用了既有描述索引，未测量本轮离线建库时间；建库成本不能从本报告推断。")
    lines.extend(["", "## 解释边界", "", *[f"- {note}" for note in caveats],
                  "", "逐点机器可读数据见同名 `.json` 和 `.csv` 文件。"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(run_dir, dataset_dir, output_path, ks, bootstrap_samples=2000, bootstrap_seed=20261030):
    run_config = read_json(run_dir / "run_config.json")
    input_gold = read_jsonl(run_dir / "input" / "gold.jsonl")
    input_queries = read_jsonl(run_dir / "input" / "queries.jsonl")
    dataset_gold = {row["query_id"]: row for row in read_jsonl(dataset_dir / "gold.jsonl")}
    dataset_queries = {row["query_id"]: row for row in read_jsonl(dataset_dir / "queries.jsonl")}
    files = run_config.get("dataset", {}).get("files_sha256", {})
    for name in ("memories.jsonl", "queries.jsonl", "gold.jsonl"):
        if name not in files or sha256(dataset_dir / name) != files[name]:
            raise ValueError(f"当前数据文件与运行冻结的输入哈希不一致：{name}")
    selected_ids = {row["query_id"] for row in input_gold}
    if selected_ids != {row["query_id"] for row in input_queries}:
        raise ValueError("运行冻结的 queries.jsonl 与 gold.jsonl 查询不匹配")
    if not selected_ids <= set(dataset_gold) or not selected_ids <= set(dataset_queries):
        raise ValueError("运行输入中的查询不属于给定数据集")
    for row in input_gold:
        if row["conflict_old_memory_ids"] != dataset_gold[row["query_id"]]["conflict_old_memory_ids"]:
            raise ValueError(f"{row['query_id']}: 运行金标与当前数据集不一致")

    points, details, candidate_sets = candidate_points(run_dir, input_gold, ks)
    groups_path = run_dir / "input" / "query_groups.json"
    query_groups = read_json(groups_path) if groups_path.is_file() else None
    group_record = run_config.get("query_groups") or {}
    if group_record.get("provided") and query_groups is None:
        raise ValueError("运行记录声明提供 query_groups.json，但冻结输入中缺少该文件")
    if query_groups is not None and group_record.get("sha256") != sha256(groups_path):
        raise ValueError("query_groups.json 与运行配置记录的冻结哈希不一致")
    if query_groups is not None and (
            not isinstance(query_groups, dict)
            or any(not isinstance(group, str) or not group.strip()
                   for group in query_groups.values())):
        raise ValueError("query_groups.json 必须是 query_id 到非空分组字符串的 JSON 对象")
    group_count = len(set(query_groups.values())) if query_groups is not None else None
    if query_groups is not None:
        for point in points:
            key = f"{point['arm']}:{point['budget']}"
            point["group_bootstrap_95ci"] = grouped_bootstrap(
                input_gold, candidate_sets[key], query_groups,
                samples=bootstrap_samples, seed=bootstrap_seed)
    costs = costs_by_arm(details, run_config)
    output = {
        "run_id": run_config.get("run_id", run_dir.name),
        "evaluation_split": run_config.get("evaluation_split", "exploratory_legacy_unrecorded"),
        "dataset_hashes": files,
        "query_count": len(input_gold),
        "conflict_pair_count": sum(len(row["conflict_old_memory_ids"]) for row in input_gold),
        "group_count": group_count,
        "bootstrap_samples": bootstrap_samples if group_count is not None else 0,
        "baseline_rank_depth": run_config.get("settings", {}).get("baseline_budget_depth",
                                  run_config.get("settings", {}).get("coarse_k")),
        "points": points,
        "costs": costs,
        "index": run_config.get("index", {}),
    }
    render_report(run_dir, dataset_dir, run_config, points, costs, input_gold,
                  output_path, ks, group_count, bootstrap_samples)
    output_path.with_suffix(".json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ("arm", "budget", "top_k", "threshold", "recall_micro", "recall_macro",
                   "full_coverage", "candidate_purity", "candidate_count_mean",
                   "candidate_count_p95", "candidate_count_max", "conflict_pairs",
                   "retrieved_conflict_pairs", "recall_micro_ci_low", "recall_micro_ci_high",
                   "candidate_purity_ci_low", "candidate_purity_ci_high")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for point in points:
            row = dict(point)
            ci = point.get("group_bootstrap_95ci") or {}
            row["recall_micro_ci_low"] = ci.get("recall_micro", {}).get("low")
            row["recall_micro_ci_high"] = ci.get("recall_micro", {}).get("high")
            row["candidate_purity_ci_low"] = ci.get("candidate_purity", {}).get("low")
            row["candidate_purity_ci_high"] = ci.get("candidate_purity", {}).get("high")
            writer.writerow(row)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ks", type=lambda value: tuple(sorted({int(item) for item in value.split(",")})),
                        default=DEFAULT_KS, help="预算逗号列表；默认 1,2,3,5,10,20,30,50")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261030)
    args = parser.parse_args(argv)
    if not args.ks or any(k < 1 for k in args.ks):
        raise SystemExit("--ks 只接受正整数")
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples 必须是正整数")
    result = analyze(args.run_dir.resolve(), args.dataset.resolve(), args.output.resolve(),
                     args.ks, args.bootstrap_samples, args.bootstrap_seed)
    print(f"离线预算比较已写入：{args.output.resolve()}")
    print(f"查询 {result['query_count']} 条；冲突对 {result['conflict_pair_count']} 个；未调用模型或向量服务。")


if __name__ == "__main__":
    main()
