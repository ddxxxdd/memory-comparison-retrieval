"""Rank approved synthetic conflict cases with the existing BM25/Dense/Hybrid methods."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_retrieval.api import APIClient, APIError, usage_summary
from memory_retrieval.baselines import BM25
from memory_retrieval.config import Settings
from memory_retrieval.data import (RETRIEVAL_TEXT_VERSION, Memory, canonical_json,
                                   fingerprint, read_jsonl, write_json)
from memory_retrieval.index import MemoryIndex
from memory_retrieval.pipeline import Retriever


DEFAULT_DATASET = WORKSPACE / "research_notes" / "synthetic_conflict_probe_v1"
DATA_FILES = ("memories.jsonl", "queries.jsonl", "banks.jsonl", "cases.jsonl", "gold.jsonl")
TOP_KS = (1, 3, 5, 10, 20)
METHODS = ("bm25", "dense", "hybrid")
# 规模由数据目录的 manifest.json["scale"] 决定；缺失时回退到 v1 协议。
DEFAULT_SCALE = {
    "bank_size": 200,
    "distractor_levels": [0, 5, 15],
    "expected_counts": {"memories": 2580, "queries": 24, "banks": 36, "cases": 72, "gold": 72},
    "split_case_counts": {"dev": 24, "evaluation": 48},
}


def load_scale(directory):
    """Read the dataset's own scale block; fall back to the v1 protocol."""
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        return dict(DEFAULT_SCALE)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scale = manifest.get("scale")
    if not isinstance(scale, dict):
        return dict(DEFAULT_SCALE)
    merged = dict(DEFAULT_SCALE)
    merged.update({key: scale[key] for key in DEFAULT_SCALE if key in scale})
    merged["distractor_levels"] = sorted(merged["distractor_levels"])
    return merged


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_thresholds(value):
    try:
        thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("阈值须为逗号分隔的数字") from None
    if not thresholds or any(not math.isfinite(x) or not -1 <= x <= 1 for x in thresholds):
        raise argparse.ArgumentTypeError("Dense cosine 阈值须为 -1 到 1 之间的有限数值")
    if len(set(thresholds)) != len(thresholds):
        raise argparse.ArgumentTypeError("Dense 阈值不能重复")
    return thresholds


def load_dataset(directory):
    directory = Path(directory).resolve()
    scale = load_scale(directory)
    rows = {Path(name).stem: read_jsonl(directory / name) for name in DATA_FILES}
    expected_fields = {
        "memories": {"memory_id", "content"},
        "queries": {"query_id", "content"},
        "banks": {"bank_id", "scenario_id", "distractor_level", "memory_ids"},
        "cases": {"case_id", "scenario_id", "split", "query_id", "bank_id", "wording_variant"},
        "gold": {"case_id", "conflict_old_memory_ids", "expected_conflict_count", "review_status"},
    }
    expected_counts = scale["expected_counts"]
    for name, expected in expected_counts.items():
        if len(rows[name]) != expected:
            raise ValueError(f"{name}.jsonl 应有 {expected} 行，实际 {len(rows[name])} 行")
        if any(set(row) != expected_fields[name] for row in rows[name]):
            raise ValueError(f"{name}.jsonl 字段与已审核数据格式不一致")

    memories = {row["memory_id"]: row for row in rows["memories"]}
    queries = {row["query_id"]: row for row in rows["queries"]}
    banks = {row["bank_id"]: row for row in rows["banks"]}
    cases = {row["case_id"]: row for row in rows["cases"]}
    gold = {row["case_id"]: row for row in rows["gold"]}
    for name, mapping, source_rows in (
            ("memory_id", memories, rows["memories"]),
            ("query_id", queries, rows["queries"]),
            ("bank_id", banks, rows["banks"]),
            ("case_id", cases, rows["cases"]),
            ("gold.case_id", gold, rows["gold"])):
        if len(mapping) != len(source_rows):
            raise ValueError(f"{name} 有重复值")
    if set(cases) != set(gold):
        raise ValueError("cases.jsonl 与 gold.jsonl 的 case_id 必须完全一致")

    for bank in rows["banks"]:
        ids = bank["memory_ids"]
        if len(ids) != scale["bank_size"] or len(set(ids)) != scale["bank_size"]:
            raise ValueError(f"{bank['bank_id']} 必须恰好包含 {scale['bank_size']} 条不同记忆")
        if bank["distractor_level"] not in scale["distractor_levels"]:
            raise ValueError(f"{bank['bank_id']} 的干扰数量档位非法")
        if set(ids) - memories.keys():
            raise ValueError(f"{bank['bank_id']} 引用了不存在的记忆")

    variants_per_bank = {}
    scenario_splits = {}
    for case in rows["cases"]:
        if case["query_id"] not in queries or case["bank_id"] not in banks:
            raise ValueError(f"{case['case_id']} 缺少 query 或 bank 引用")
        if case["wording_variant"] not in {"D", "P"}:
            raise ValueError(f"{case['case_id']} 的表述版本非法")
        bank = banks[case["bank_id"]]
        if case["scenario_id"] != bank["scenario_id"]:
            raise ValueError(f"{case['case_id']} 与 bank 的 scenario_id 不一致")
        if case["split"] not in {"dev", "evaluation"}:
            raise ValueError(f"{case['case_id']} 的 split 非法")
        prior = scenario_splits.setdefault(case["scenario_id"], case["split"])
        if prior != case["split"]:
            raise ValueError(f"{case['scenario_id']} 被拆分到多个 split")
        bank_variants = variants_per_bank.setdefault(case["bank_id"], {"D": 0, "P": 0})
        bank_variants[case["wording_variant"]] += 1
        target_ids = gold[case["case_id"]]["conflict_old_memory_ids"]
        expected_count = gold[case["case_id"]]["expected_conflict_count"]
        if (not isinstance(target_ids, list)
                or any(not isinstance(memory_id, str) for memory_id in target_ids)
                or len(target_ids) != len(set(target_ids))
                or isinstance(expected_count, bool) or not isinstance(expected_count, int)
                or len(target_ids) != gold[case["case_id"]]["expected_conflict_count"]
                or set(target_ids) - set(bank["memory_ids"])):
            raise ValueError(f"{case['case_id']} 的 gold 数量或 bank 引用不一致")

    if any(variants != {"D": 1, "P": 1} for variants in variants_per_bank.values()):
        raise ValueError("每个 bank 应恰好对应 D/P 各一个 case")
    split_counts = {split: sum(row["split"] == split for row in rows["cases"])
                    for split in ("dev", "evaluation")}
    if split_counts != scale["split_case_counts"]:
        raise ValueError(f"case split 数量不符合协议：{split_counts}，期望 {scale['split_case_counts']}")

    hashes = {name: file_hash(directory / name) for name in DATA_FILES}
    return {
        "directory": directory,
        "scale": scale,
        "rows": rows,
        "memories": memories,
        "queries": queries,
        "banks": banks,
        "cases": cases,
        "gold": gold,
        "scenario_splits": scenario_splits,
        "split_counts": split_counts,
        "file_hashes": hashes,
        "signature": fingerprint(hashes),
    }


def make_bank_retriever(bank, memory_map, content_vectors_by_text, client, settings):
    # The index contains only this case's bank. No labels or internal worlds enter it.
    memories = [Memory(memory_id, memory_map[memory_id]["content"], {})
                for memory_id in bank["memory_ids"]]
    texts = [memory.retrieval_text for memory in memories]
    content_vectors = np.stack([content_vectors_by_text[text] for text in texts])
    embedding = settings.embedding_identity()
    index_id = fingerprint({"bank_id": bank["bank_id"], "memory_ids": bank["memory_ids"],
                            "embedding": embedding,
                            "retrieval_text_version": RETRIEVAL_TEXT_VERSION})
    index = MemoryIndex(
        memories=memories,
        catalog=[],
        description_vectors=np.empty((0, content_vectors.shape[1]), dtype=np.float32),
        content_vectors=content_vectors,
        manifest={"index_id": index_id, "embedding": embedding,
                  "retrieval_text_version": RETRIEVAL_TEXT_VERSION},
    )
    return Retriever(index, client, bm25=BM25(texts))


def retrieve_case(case, query, bank, memory_map, retriever, method):
    # This function receives no gold row; retrieval is limited to the supplied bank.
    bank_size = len(bank["memory_ids"])
    result = retriever.query(
        case["case_id"], query["content"], retrieval_text=query["content"],
        method=method, rerank=False, judge=False, coarse_k=bank_size, final_k=bank_size,
    )
    ranked = result["coarse_candidates"]
    ranked_ids = [row["memory_id"] for row in ranked]
    if len(ranked) != bank_size or set(ranked_ids) != set(bank["memory_ids"]):
        raise ValueError(f"{case['case_id']}/{method} 未返回该 bank 的完整 {bank_size} 条排序")
    if [row["memory_id"] for row in result["candidates"]] != ranked_ids:
        raise ValueError(f"{case['case_id']}/{method} 的原生排序被额外过滤")
    if any(not math.isfinite(float(row["score"])) for row in ranked):
        raise ValueError(f"{case['case_id']}/{method} 出现非有限分数")
    return {
        "case_id": case["case_id"],
        "scenario_id": case["scenario_id"],
        "split": case["split"],
        "query_id": case["query_id"],
        "query_content": query["content"],
        "wording_variant": case["wording_variant"],
        "bank_id": bank["bank_id"],
        "distractor_level": bank["distractor_level"],
        "method": method,
        "score_kind": {"bm25": "bm25_score", "dense": "cosine_similarity",
                       "hybrid": "rrf_score"}[method],
        "timing": result["timing"],
        "usage": result["usage"],
        "ranked_candidates": [
            {"rank": position, "memory_id": row["memory_id"],
             "content": memory_map[row["memory_id"]]["content"],
             "score": float(row["score"])}
            for position, row in enumerate(ranked, 1)
        ],
    }


def _mean(values):
    return statistics.mean(values) if values else None


def _p95(values):
    return float(np.percentile(values, 95)) if values else None


def aggregate(case_rows):
    hits = sum(len(set(row["gold_ids"]) & set(row["candidate_ids"])) for row in case_rows)
    target_count = sum(len(row["gold_ids"]) for row in case_rows)
    candidate_count = sum(len(row["candidate_ids"]) for row in case_rows)
    positive = [row for row in case_rows if row["gold_ids"]]
    no_conflict = [row for row in case_rows if not row["gold_ids"]]
    positive_recalls = [
        len(set(row["gold_ids"]) & set(row["candidate_ids"])) / len(row["gold_ids"])
        for row in positive
    ]
    positive_full = [set(row["gold_ids"]) <= set(row["candidate_ids"]) for row in positive]
    no_conflict_counts = [len(row["candidate_ids"]) for row in no_conflict]
    all_counts = [len(row["candidate_ids"]) for row in case_rows]
    timing_rows = [row.get("timing", {}) for row in case_rows]
    timing_stages = sorted({stage for timing in timing_rows for stage in timing
                            if stage != "total_seconds"})
    timing_summary = {
        "total_seconds_mean": _mean([timing["total_seconds"] for timing in timing_rows
                                      if "total_seconds" in timing]),
        "total_seconds_p95": _p95([timing["total_seconds"] for timing in timing_rows
                                    if "total_seconds" in timing]),
        "stages": {
            stage: {
                "mean_seconds": _mean([timing.get(stage, 0.0) for timing in timing_rows]),
                "p95_seconds": _p95([timing.get(stage, 0.0) for timing in timing_rows]),
            }
            for stage in timing_stages
        },
    }
    usage_rows = [row.get("usage", {}) for row in case_rows]
    usage_fields = ("api_attempts", "cache_hits", "invalid_outputs", "known_prompt_tokens",
                    "known_completion_tokens", "known_total_tokens", "responses_without_usage")
    usage_totals = {field: sum(usage.get(field, 0) for usage in usage_rows)
                    for field in usage_fields}
    return {
        "case_count": len(case_rows),
        "positive_query_count": len(positive),
        "conflict_memory_count": target_count,
        "retrieved_conflict_memory_count": hits,
        "conflict_recall_micro": hits / target_count if target_count else None,
        "positive_query_recall_macro": _mean(positive_recalls),
        "positive_query_full_coverage": _mean([int(value) for value in positive_full]),
        "candidate_purity": hits / candidate_count if candidate_count else None,
        "candidate_count_mean": _mean(all_counts),
        "candidate_count_p95": _p95(all_counts),
        "candidate_count_max": max(all_counts, default=0),
        "no_conflict_query_count": len(no_conflict),
        "no_conflict_return_rate": (_mean([int(count > 0) for count in no_conflict_counts])
                                     if no_conflict else None),
        "no_conflict_candidate_count_mean": _mean(no_conflict_counts),
        "no_conflict_candidate_count_p95": _p95(no_conflict_counts),
        "no_conflict_candidate_count_max": max(no_conflict_counts, default=0),
        "timing": timing_summary,
        "usage_totals": usage_totals,
    }


def grouped_metrics(rows):
    return {
        "overall": aggregate(rows),
        "by_wording_variant": {
            variant: aggregate([row for row in rows if row["wording_variant"] == variant])
            for variant in ("D", "P")
        },
        "by_distractor_level": {
            str(level): aggregate([row for row in rows if row["distractor_level"] == level])
            for level in sorted({row["distractor_level"] for row in rows})
        },
        "by_case_type": {
            kind: aggregate([row for row in rows
                             if bool(row["gold_ids"]) == (kind == "positive")])
            for kind in ("positive", "no_conflict")
        },
        "by_scenario": {
            scenario: aggregate([row for row in rows if row["scenario_id"] == scenario])
            for scenario in sorted({row["scenario_id"] for row in rows})
        },
    }


def summarize_rankings(rankings, cases, gold, thresholds):
    case_rows = []
    for ranking in rankings:
        target_ids = gold[ranking["case_id"]]["conflict_old_memory_ids"]
        for k in TOP_KS:
            case_rows.append({**ranking, "selection": f"top_{k}", "parameter": k,
                              "gold_ids": target_ids,
                              "candidate_ids": [row["memory_id"]
                                                for row in ranking["ranked_candidates"][:k]]})
        if ranking["method"] == "dense":
            for threshold in thresholds:
                case_rows.append({**ranking, "selection": "dense_threshold",
                                  "parameter": threshold, "gold_ids": target_ids,
                                  "candidate_ids": [row["memory_id"]
                                                    for row in ranking["ranked_candidates"]
                                                    if row["score"] >= threshold]})

    policies = []
    keys = sorted({(row["method"], row["selection"], row["parameter"]) for row in case_rows},
                  key=lambda item: (METHODS.index(item[0]), item[1], item[2]))
    for method, selection, parameter in keys:
        selected = [row for row in case_rows
                    if (row["method"], row["selection"], row["parameter"])
                    == (method, selection, parameter)]
        policies.append({"method": method, "selection": selection,
                         "parameter": parameter, "metrics": grouped_metrics(selected)})
    return policies


def write_rankings(path, rows):
    Path(path).write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def run_baselines(args):
    dataset = load_dataset(args.dataset)
    frozen_embedding = None
    if args.split == "dev":
        if args.frozen_config:
            raise ValueError("dev 运行不能读取冻结评估配置")
        if not args.dense_thresholds:
            raise ValueError("dev 运行须显式提供 --dense-thresholds 以生成阈值曲线")
        thresholds = args.dense_thresholds
        threshold_source = "dev_grid"
        frozen_config_hash = None
    else:
        if args.dense_thresholds:
            raise ValueError("evaluation 不接受阈值列表；请使用 dev 冻结的 --frozen-config")
        if not args.frozen_config:
            raise ValueError("evaluation 必须提供由 dev 运行生成的 --frozen-config")
        frozen_path = Path(args.frozen_config).resolve()
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if (frozen.get("source_split") != "dev"
                or frozen.get("dataset_signature") != dataset["signature"]):
            raise ValueError("冻结配置必须来自同一数据版本的 dev 运行")
        frozen_embedding = frozen.get("embedding")
        if not isinstance(frozen_embedding, dict):
            raise ValueError("冻结配置缺少 dev embedding 身份")
        threshold = frozen.get("dense_threshold")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not -1 <= threshold <= 1):
            raise ValueError("冻结配置中的 Dense 阈值非法")
        thresholds = [float(threshold)]
        threshold_source = "frozen_dev_config"
        frozen_config_hash = file_hash(frozen_path)

    selected_cases = [case for case in dataset["rows"]["cases"] if case["split"] == args.split]
    if not selected_cases:
        raise ValueError(f"没有 {args.split} cases")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("输出目录已存在；请使用新路径")
    settings = Settings.from_env(args.env)
    settings.require(embedding=True)
    if frozen_embedding is not None and frozen_embedding != settings.embedding_identity():
        raise ValueError("evaluation embedding 配置必须与冻结阈值的 dev run 一致")
    output.mkdir(parents=True)
    cache_directory = output / "embedding_cache"
    bank_embedding_client = APIClient(replace(
        settings, cache_path=cache_directory / "bank_embeddings.sqlite3"))
    method_clients = {
        method: APIClient(replace(settings, cache_path=cache_directory / f"{method}.sqlite3"))
        for method in METHODS
    }

    memory_map = dataset["memories"]
    bank_map = dataset["banks"]
    query_map = dataset["queries"]
    needed_bank_ids = {case["bank_id"] for case in selected_cases}
    needed_banks = [bank_map[bank_id] for bank_id in sorted(needed_bank_ids)]
    unique_bank_texts = list(dict.fromkeys(
        memory_map[memory_id]["content"]
        for bank in needed_banks for memory_id in bank["memory_ids"]))
    unique_bank_vectors = bank_embedding_client.embed(
        unique_bank_texts, "synthetic_conflict_bank_embedding")
    content_vectors_by_text = dict(zip(unique_bank_texts, unique_bank_vectors))
    retrievers = {
        (bank["bank_id"], method): make_bank_retriever(
            bank, memory_map, content_vectors_by_text,
            method_clients[method], method_clients[method].settings)
        for bank in needed_banks for method in METHODS
    }
    rankings = []
    for case in selected_cases:
        query = query_map[case["query_id"]]
        bank = bank_map[case["bank_id"]]
        for method in METHODS:
            retriever = retrievers[(bank["bank_id"], method)]
            rankings.append(retrieve_case(case, query, bank, memory_map, retriever, method))

    policies = summarize_rankings(rankings, dataset["cases"], dataset["gold"], thresholds)
    all_api_events = [event for client in [bank_embedding_client, *method_clients.values()]
                      for event in client.events]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_record = {
        "run_id": run_id,
        "dataset": str(dataset["directory"]),
        "dataset_signature": dataset["signature"],
        "dataset_file_sha256": dataset["file_hashes"],
        "split": args.split,
        "case_count": len(selected_cases),
        "case_ids": [case["case_id"] for case in selected_cases],
        "methods": list(METHODS),
        "top_k_values": list(TOP_KS),
        "dense_thresholds": thresholds,
        "dense_threshold_source": threshold_source,
        "frozen_config_sha256": frozen_config_hash,
        "embedding": settings.embedding_identity(),
        "llm_called": False,
        "reranker_or_judge_called": False,
        "worlds_file_read": False,
        "bank_size_per_case": dataset["scale"]["bank_size"],
        "bank_embedding_inputs": {
            "bank_memory_entries": sum(len(bank["memory_ids"]) for bank in needed_banks),
            "unique_texts_encoded": len(unique_bank_texts),
        },
        "api_events": {
            "actual_attempts": sum(event.get("cached") is False for event in all_api_events),
            "cache_hits": sum(event.get("cached") is True for event in all_api_events),
            "stages": sorted({event.get("stage") for event in all_api_events}),
            "usage": usage_summary(all_api_events),
        },
    }
    summary = {
        "dataset_signature": dataset["signature"],
        "split": args.split,
        "case_count": len(selected_cases),
        "metrics_note": {
            "positive_query_full_coverage": "仅在gold非空的case上计算。",
            "candidate_purity": "冲突gold命中数 / 返回候选数；无冲突case的所有候选计入非冲突候选。",
            "dense_threshold": "使用原生Dense cosine分数，按 score >= threshold 选候选。",
        },
        "policies": policies,
    }
    write_rankings(output / "rankings.jsonl", rankings)
    write_json(output / "summary.json", summary)
    write_json(output / "run.json", run_record)
    print(json.dumps({"run": str(output), "split": args.split,
                      "cases": len(selected_cases), "ranking_rows": len(rankings),
                      "api_actual_attempts": run_record["api_events"]["actual_attempts"],
                      "llm_called": False}, ensure_ascii=False, indent=2))


def freeze_dev_threshold(args):
    dev_dir = Path(args.dev_run).resolve()
    run_path = dev_dir / "run.json"
    summary_path = dev_dir / "summary.json"
    if not run_path.is_file() or not summary_path.is_file():
        raise ValueError("dev run 缺少 run.json 或 summary.json")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("split") != "dev":
        raise ValueError("只能从 dev run 冻结 Dense 阈值")
    if any(abs(float(value) - args.threshold) < 1e-12
           for value in run.get("dense_thresholds", [])) is False:
        raise ValueError("所选阈值不在该 dev run 计算过的 Dense 阈值列表中")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("冻结配置已存在；请使用新路径")
    frozen = {
        "dataset_signature": run["dataset_signature"],
        "source_split": "dev",
        "source_run_id": run["run_id"],
        "source_summary_sha256": file_hash(summary_path),
        "embedding": run["embedding"],
        "dense_threshold": float(args.threshold),
        "selection_note": "由使用者依据 dev 汇总显式指定；evaluation 只读取该冻结值。",
    }
    write_json(output, frozen)
    print(json.dumps({"frozen_config": str(output),
                      "source_run_id": run["run_id"],
                      "dense_threshold": args.threshold}, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="离线检查数据映射与case覆盖；不调用API")
    validate.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)

    run = subparsers.add_parser("run", help="运行 BM25、Dense、Hybrid 原生排序并汇总")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--split", choices=("dev", "evaluation"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--env", type=Path, help="项目 .env 路径；只需要 embedding 配置")
    run.add_argument("--dense-thresholds", type=parse_thresholds,
                     help="仅dev使用，逗号分隔的Dense cosine阈值列表")
    run.add_argument("--frozen-config", type=Path,
                     help="evaluation必填：由dev run冻结的Dense阈值配置")

    freeze = subparsers.add_parser("freeze", help="将使用者选定的dev Dense阈值写入冻结配置")
    freeze.add_argument("--dev-run", type=Path, required=True)
    freeze.add_argument("--threshold", type=float, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            dataset = load_dataset(args.dataset)
            print(json.dumps({
                "status": "valid",
                "dataset": str(dataset["directory"]),
                "counts": dataset["scale"]["expected_counts"],
                "case_counts_by_split": dataset["split_counts"],
                "case_ids_match_gold": True,
                "bank_size_per_case": dataset["scale"]["bank_size"],
                "worlds_file_read": False,
                "api_calls": 0,
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command == "freeze":
            if not math.isfinite(args.threshold) or not -1 <= args.threshold <= 1:
                raise ValueError("Dense cosine 阈值须为 -1 到 1 之间的有限数值")
            freeze_dev_threshold(args)
            return 0
        run_baselines(args)
        return 0
    except (APIError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
