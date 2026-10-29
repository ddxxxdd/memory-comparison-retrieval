"""Run retrieval methods over one shared memory bank and a batch of queries."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_retrieval.adapters import load_graph, load_text
from memory_retrieval.api import APIClient, APIError
from memory_retrieval.config import (DESCRIPTION_DUPLICATE_THRESHOLD, MEMORY_TEMPLATE_THRESHOLD,
                                     SIMILARITY_THRESHOLD, Settings)
from memory_retrieval.data import build_retrieval_text, canonical_json, load_queries, read_jsonl, write_json
from memory_retrieval.evaluation import evaluate, validate_gold
from memory_retrieval.index import MemoryIndex, build_index
from memory_retrieval.pipeline import Retriever
from memory_retrieval.prompts import INDEX_PROMPT_HASH


ARM_SPECS = {
    "proposed": {"method": "proposed", "rerank": True, "kind": "main_method"},
    "independent": {"method": "independent", "rerank": True, "kind": "ablation"},
    "no-rerank": {"method": "proposed", "rerank": False, "kind": "ablation"},
    "dense": {"method": "dense", "rerank": True, "kind": "retrieval_diagnostic"},
    "bm25": {"method": "bm25", "rerank": True, "kind": "retrieval_diagnostic"},
    "hybrid": {"method": "hybrid", "rerank": True, "kind": "retrieval_diagnostic"},
}


def write_jsonl(path, rows):
    Path(path).write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_queries(queries, limit, seed, query_ids=None):
    """Pick the sampled writes. Explicit ids keep a scale comparison controlled.

    Comparing two bank sizes only isolates the library if both runs process the same
    writes, so --query-ids overrides the random sample.
    """
    if query_ids:
        wanted = list(dict.fromkeys(query_ids))
        by_id = {row["query_id"]: row for row in queries}
        missing = [qid for qid in wanted if qid not in by_id]
        if missing:
            raise ValueError(f"--query-ids 中有查询不在数据集中:{missing}")
        return [by_id[qid] for qid in wanted]
    if limit is None or limit >= len(queries):
        return list(queries)
    if limit < 1:
        raise ValueError("--limit 必须大于 0")
    return random.Random(seed).sample(queries, limit)


def usage_totals(rows):
    keys = ("api_attempts", "cache_hits", "invalid_outputs", "known_prompt_tokens",
            "known_completion_tokens", "known_total_tokens", "responses_without_usage")
    return {key: sum(row["usage"].get(key, 0) for row in rows) for key in keys}


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def write_summary(path, report):
    lines = [
        "# 来源数据小规模实验",
        "",
        "> 指标按数据集提供的金标对全部选中查询计算；标签核对状态由数据维护者在数据集中管理。",
        "> 本次只使用一个共同旧库，不划分 dev/test；库容曲线需另行测量。",
        "",
        f"- 旧记忆：{report['dataset']['memory_count']} 条",
        f"- 查询：{report['sample']['query_count']} 条",
        f"- 固定阈值：{SIMILARITY_THRESHOLD}；并行线程：{report['settings']['workers']}",
        f"- 冲突判定方法臂：{', '.join(arm for arm, data in report['arms'].items() if data['judged']) or '未运行'}",
        "",
        "| 方法臂 | 候选召回率 | 候选纯度 | 冲突 F1 | 误报率 | 漏报率 | 写入 p50 ms | 写入 p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, data in report["arms"].items():
        metrics = data["metrics"]
        retrieval = metrics["candidate_retrieval"]
        conflict = metrics["conflict_classification"] or {}
        latency = metrics["write_latency_ms"]["total"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            arm, fmt(retrieval["candidate_recall"]), fmt(retrieval["candidate_purity"]),
            fmt(conflict.get("f1")), fmt(conflict.get("false_alarm_rate")),
            fmt(conflict.get("miss_rate")), fmt(latency["p50"], 1), fmt(latency["p95"], 1)))
    lines.extend([
        "",
        "逐阶段时延和指标见 `report.json`；逐查询原始结果保存在 `arms/`。",
        "",
        "`dense` 和 `bm25` 是检索流程诊断，不代表 Mem0、MemLACE 或 SparseCL 的复现。",
        "",
        "时延受 API 缓存、服务响应和并发影响。单个库容点不能支持规模复杂度结论。",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=ROOT / "data" / "dialogue_nli_mixed_v1")
    result.add_argument("--output-root", type=Path, default=ROOT / "runs" / "experiments")
    result.add_argument("--run-name", help="输出目录名；默认使用 UTC 时间戳")
    result.add_argument("--env", type=Path, help="环境文件；默认项目根目录 .env")
    result.add_argument("--index-dir", type=Path, help="可选复用索引；必须与当前完整旧库及模型一致")
    result.add_argument("--build-batch-size", type=int, default=16,
                        help="离线建库每批处理的记忆条数（描述模式固定为批式复用）")
    result.add_argument("--build-consolidate", action="store_true",
                        help="建库时开启目录归并（M2；默认关闭）")
    result.add_argument("--build-no-repair", action="store_true",
                        help="建库时关掉抽取校验后的重写（M1 消融）")
    result.add_argument("--adapter", choices=("text", "graph"), default="text")
    result.add_argument("--limit", type=int, default=12, help="随机抽取的查询数；默认 12，设为 0 运行全部")
    result.add_argument("--seed", type=int, default=20261014)
    result.add_argument("--query-ids", type=lambda s: [x for x in s.split(",") if x],
                        help="显式指定 query_id(逗号分隔);库容对比时让两次运行处理同一批写入")
    result.add_argument("--workers", type=int, default=4, help="每个方法臂的并行查询数")
    result.add_argument("--arms", nargs="+", choices=tuple(ARM_SPECS), default=("proposed",),
                        help="运行的方法；默认只运行主方法，其他代码路径尚不是已确认的正式对照")
    result.add_argument("--judge-arms", nargs="*", choices=tuple(ARM_SPECS), default=(),
                        help="运行逐对冲突判定的臂；默认不调用判定器")
    result.add_argument("--coarse-k", type=int)
    result.add_argument("--final-k", type=int)
    return result


def _run_one(index, settings, query, spec, judge, coarse_k, final_k):
    retriever = Retriever(index, APIClient(settings))
    # 全库检索：不按 user_id/scope 预过滤，主体信息由检索文本提供给三种方法。
    return retriever.query(
        query["query_id"], query["content"],
        retrieval_text=build_retrieval_text(query["content"], query.get("source_ref")),
        query_source_ref=query.get("source_ref"),
        method=spec["method"], rerank=spec["rerank"], judge=judge,
        coarse_k=coarse_k, final_k=final_k)


def run_arm(index, settings, queries, spec, arm, judge, workers, coarse_k, final_k,
            results_path, failure_path):
    results = []
    errors = []
    sample_indices = {query["query_id"]: i for i, query in enumerate(queries)}
    with Path(results_path).open("x", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one, index, settings, query, spec, judge, coarse_k, final_k): query
                for query in queries
            }
            for future in as_completed(futures):
                query = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    errors.append({"query_id": query["query_id"],
                                   "error": f"{type(exc).__name__}: {exc}"})
                    continue
                row["sample_index"] = sample_indices[query["query_id"]]
                stream.write(canonical_json(row) + "\n")
                stream.flush()
                results.append(row)
                print(f"[{arm}] 完成 {len(results)}/{len(queries)}: {query['query_id']}", flush=True)
    if errors:
        write_json(failure_path, {"stage": "query", "arm": arm,
                                  "completed_queries": len(results), "errors": errors})
        raise APIError(f"{arm}: {len(errors)} 条查询失败；成功结果已保留")
    return results


def main(argv=None):
    args = parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if args.limit < 0 or args.workers < 1:
        raise SystemExit("--limit 不能小于 0，--workers 必须大于 0")
    if not set(args.judge_arms) <= set(args.arms):
        raise SystemExit("--judge-arms 中的每个方法都必须包含在 --arms 中")

    dataset_dir = args.dataset.resolve()
    files = {name: dataset_dir / name for name in ("memories.jsonl", "queries.jsonl", "gold.jsonl")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit("数据集文件缺失：" + ", ".join(missing))

    load_memory_rows = load_text if args.adapter == "text" else load_graph
    memories = load_memory_rows(files["memories.jsonl"])
    all_queries = load_queries(files["queries.jsonl"])
    all_gold = read_jsonl(files["gold.jsonl"])
    validate_gold(all_gold, {memory.memory_id for memory in memories})
    if {row["query_id"] for row in all_queries} != {row["query_id"] for row in all_gold}:
        raise SystemExit("查询与金标 query_id 不一致")
    queries = select_queries(all_queries, args.limit or None, args.seed, args.query_ids)
    selected_ids = {row["query_id"] for row in queries}
    gold = [row for row in all_gold if row["query_id"] in selected_ids]

    run_name = args.run_name or datetime.now(timezone.utc).strftime("source-%Y%m%dT%H%M%S-%fZ")
    run_dir = (args.output_root / run_name).resolve()
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"输出目录已存在，不覆盖已有实验：{run_dir}") from None
    input_dir, index_output, arm_output = run_dir / "input", run_dir / "index", run_dir / "arms"
    input_dir.mkdir()
    arm_output.mkdir()
    write_jsonl(input_dir / "queries.jsonl", queries)
    write_jsonl(input_dir / "gold.jsonl", gold)

    settings = Settings.from_env(args.env)
    settings.require(llm=True, embedding=True)
    index_started = time.perf_counter()
    try:
        if args.index_dir:
            index = MemoryIndex.load(args.index_dir.resolve())
            # 复用索引按 memory_id + 检索文本比对：scope/user_id 已经影响索引文本，
            # 正文不变而主体被修正时，旧索引的向量与描述都已过期，不能继续复用。
            indexed = {(row.memory_id, row.retrieval_text) for row in index.memories}
            current = {(row.memory_id, row.retrieval_text) for row in memories}
            if indexed != current:
                missing = sorted(indexed - current)[:3]
                extra = sorted(current - indexed)[:3]
                raise ValueError(
                    "复用索引中的旧库与当前数据集不一致(按 memory_id+retrieval_text):"
                    f"索引多出 {missing},数据集多出 {extra}")
            index.check_embedding(settings)
            if index.manifest.get("prompt_hash") != INDEX_PROMPT_HASH and any(
                    ARM_SPECS[arm]["method"] in {"proposed", "independent"} for arm in args.arms):
                raise ValueError("复用索引的描述提示版本与当前代码不一致")
            build_seconds = 0.0
            index_origin = str(args.index_dir.resolve())
            build_usage = {}
        else:
            index = build_index(memories, APIClient(settings), index_output,
                                mode="batch_reuse", batch_size=args.build_batch_size,
                                description_threshold=(DESCRIPTION_DUPLICATE_THRESHOLD
                                                       if args.build_consolidate else None),
                                template_threshold=(MEMORY_TEMPLATE_THRESHOLD
                                                    if args.build_consolidate else None),
                                repair_rounds=0 if args.build_no_repair else 1)
            build_seconds = time.perf_counter() - index_started
            index_origin = str(index_output)
            build_usage = index.manifest.get("build", {}).get("usage", {})
    except (APIError, OSError, ValueError) as exc:
        write_json(run_dir / "failure.json", {"stage": "index_setup", "error": str(exc)})
        raise

    if args.coarse_k is not None or args.final_k is not None:
        coarse_k = args.coarse_k if args.coarse_k is not None else settings.coarse_k
        final_k = args.final_k if args.final_k is not None else settings.final_k
        settings.validate_budget(coarse_k, final_k)
    else:
        coarse_k, final_k = settings.coarse_k, settings.final_k

    manifest = {
        "run_id": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "shared_memory_bank_source_pilot",
        "dataset": {"path": str(dataset_dir), "files_sha256": {name: sha256(path) for name, path in files.items()},
                    "memory_count": len(memories), "query_count": len(all_queries), "gold_count": len(all_gold)},
        "sample": {"query_count": len(queries), "query_ids": [row["query_id"] for row in queries],
                   "limit": args.limit, "seed": args.seed,
                   "explicit_query_ids": bool(args.query_ids)},
        "settings": {"similarity_threshold": SIMILARITY_THRESHOLD, "coarse_k": coarse_k, "final_k": final_k,
                     "workers": args.workers, "llm": settings.llm_identity(),
                     "embedding": settings.embedding_identity(), "llm_temperature": settings.llm_temperature,
                     "llm_json_mode": settings.llm_json_mode, "api_timeout_seconds": settings.timeout},
        "index": {"index_id": index.manifest["index_id"], "origin": index_origin,
                  "description_mode": index.manifest["description_mode"],
                  "batch_size": index.manifest.get("batch_size"),
                  "build_seconds": build_seconds, "build_usage": build_usage},
        "arms": {arm: {**ARM_SPECS[arm], "judged": arm in args.judge_arms} for arm in args.arms},
    }
    write_json(run_dir / "run_config.json", manifest)
    report = {"run": manifest, "dataset": {"memory_count": len(memories)},
              "sample": manifest["sample"], "settings": manifest["settings"],
              "index": manifest["index"], "arms": {},
              "scale_curve": {"status": "not_measured", "library_sizes": [len(memories)]}}

    for arm in args.arms:
        spec = ARM_SPECS[arm]
        results_path = arm_output / f"{arm}.jsonl"
        print(f"[{arm}] 并行运行 {len(queries)} 条查询，workers={args.workers}", flush=True)
        try:
            results = run_arm(index, settings, queries, spec, arm, arm in args.judge_arms,
                              args.workers, coarse_k, final_k, results_path, run_dir / "failure.json")
            metrics = evaluate(gold, results)
        except (APIError, OSError, ValueError, KeyError, TypeError) as exc:
            if not (run_dir / "failure.json").exists():
                write_json(run_dir / "failure.json", {"stage": "query", "arm": arm, "error": str(exc)})
            raise
        report["arms"][arm] = {
            "kind": spec["kind"], "judged": arm in args.judge_arms,
            "result_file": str(results_path), "metrics": metrics,
            "usage": usage_totals(results),
        }
        write_json(run_dir / "report.json", report)

    write_summary(run_dir / "summary.md", report)
    print(f"实验完成：{run_dir}", flush=True)
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return run_dir


if __name__ == "__main__":
    try:
        main()
    except (APIError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f"实验失败：{exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
