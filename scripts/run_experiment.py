"""Run retrieval methods over one shared memory bank and a batch of queries."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_retrieval.adapters import load_graph, load_text
from memory_retrieval.api import APIClient, APIError
from memory_retrieval.baselines import BM25
from memory_retrieval.config import (DESCRIPTION_DUPLICATE_THRESHOLD, MEMORY_TEMPLATE_THRESHOLD,
                                     SIMILARITY_THRESHOLD, Settings)
from memory_retrieval.data import (build_retrieval_text, canonical_json, load_queries, read_jsonl,
                                   write_json)
from memory_retrieval.evaluation import evaluate, validate_gold
from memory_retrieval.index import MemoryIndex, build_index
from memory_retrieval.pipeline import Retriever
from memory_retrieval.prompts import INDEX_PROMPT_HASH, PROMPT_HASH, PROMPT_VERSION


ARM_SPECS = {
    "proposed": {"method": "proposed", "rerank": True, "kind": "main_method"},
    "independent": {"method": "independent", "rerank": True, "kind": "ablation"},
    "no-rerank": {"method": "proposed", "rerank": False, "kind": "ablation"},
    "dense": {"method": "dense", "rerank": True, "kind": "baseline"},
    "bm25": {"method": "bm25", "rerank": True, "kind": "baseline"},
    "hybrid": {"method": "hybrid", "rerank": True, "kind": "baseline"},
    "sparsecl": {"method": "sparsecl", "rerank": True, "kind": "baseline"},
}

SPARSECL_ROOT = ROOT.parent / "baselines" / "SparseCL"
SPARSECL_WORKER = ROOT / "scripts" / "sparsecl_worker.py"
SPARSECL_WSL_DISTRO = "Ubuntu-22.04"
SPARSECL_WSL_USER = "dzp"
SPARSECL_WSL_PYTHON = "/home/dzp/.venvs/sparsecl/bin/python"
SPARSECL_BATCH_SIZE = 4
SPARSECL_CPU_THREADS = 4


def write_jsonl(path, rows):
    Path(path).write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def call_totals(rows):
    totals = {"llm_attempts": 0, "embedding_attempts": 0,
              "llm_cache_hits": 0, "embedding_cache_hits": 0,
              "llm_prompt_tokens": 0, "llm_completion_tokens": 0}
    for row in rows:
        for event in row.get("api_events") or []:
            bucket = "embedding" if "embedding" in str(event.get("stage") or "") else "llm"
            if event.get("cached") is True:
                totals[f"{bucket}_cache_hits"] += 1
            elif event.get("cached") is False:
                # APIClient logs one event per actual network attempt, including retries.
                totals[f"{bucket}_attempts"] += 1
            else:
                # invalid_output is a local validation event after a recorded API response.
                continue
            if event.get("cached") is False and bucket == "llm":
                usage = event.get("usage") or {}
                totals["llm_prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
                totals["llm_completion_tokens"] += usage.get("completion_tokens", 0) or 0
    return totals


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def write_summary(path, report):
    split = report["run"].get("evaluation_split", "exploratory")
    active_arms = set(report["run"].get("arms") or {})
    uses_project_methods = bool(active_arms - {"sparsecl"})
    lines = [
        "# 检索实验结果",
        "",
        "> 指标按数据集提供的金标对全部选中查询计算；标签核对状态由数据维护者在数据集中管理。",
        f"> 评估集标记：`{split}`。`exploratory` 结果用于诊断，不能作为独立测试结论。",
        "",
        f"- 旧记忆：{report['dataset']['memory_count']} 条",
        f"- 查询：{report['sample']['query_count']} 条",
        (f"- 动态阈值：{report['settings']['similarity_threshold']}；并行线程：{report['settings']['workers']}"
         if uses_project_methods else "- 本次仅运行 SparseCL baseline。"),
        ("- API 缓存策略：未调用项目 API（本次仅运行 SparseCL）"
         if set(report["run"].get("arms") or {}) == {"sparsecl"}
         else f"- API 缓存策略：{report['run'].get('cache_policy', 'legacy_shared')}"),
        "",
        "| 方法臂 | micro 召回 | macro 召回 | 完整找全率 | 候选纯度 | 候选均值 | 候选 p95 / max | 查询时延口径 | p50 / p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for arm, data in report["arms"].items():
        metrics = data["metrics"]
        retrieval = metrics["candidate_retrieval"]
        latency = metrics["write_latency_ms"]["total"]
        latency_basis = ("批次耗时均摊" if data.get("latency_basis") == "batch_amortized_per_query"
                         else "逐查询计时")
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            arm, fmt(retrieval["candidate_recall_micro"]), fmt(retrieval["candidate_recall_macro"]),
            fmt(retrieval["full_coverage"]), fmt(retrieval["candidate_purity"]),
            fmt(retrieval["candidate_count_mean"], 1),
            f"{fmt(retrieval['candidate_count_p95'], 1)} / {retrieval['candidate_count_max']}",
            latency_basis, f"{fmt(latency['p50'], 1)} / {fmt(latency['p95'], 1)}"))
    lines.extend([
        "",
        "逐阶段时延、API 调用/token 与指标见 `report.json`；逐查询原始结果保存在 `arms/`。",
        "SparseCL 的查询时延按编码批次耗时均摊到批内查询，仅表示摊销成本，不代表单条查询实际响应时延。",
        "",
        ("BM25、Dense、Hybrid 保留原生 top-k 策略；预算曲线由 `scripts/analyze_v7_experiment.py` 从保存的排序离线计算。"
         if active_arms & {"bm25", "dense", "hybrid"}
         else "预算曲线由 `scripts/analyze_v7_experiment.py` 从保存的排序离线计算。"),
        "",
        ("各臂使用独立缓存时，跨方法不会互相预热；单次时延仍受网络波动影响。索引建库时间单独报告，单一库容点不能支持复杂度结论。"
         if uses_project_methods
         else "SparseCL 旧库双编码与 HNSW 建库成本单独报告；查询时延为批次摊销值。"),
    ])
    sparsecl = report["run"].get("sparsecl")
    if sparsecl:
        lines.insert(9, f"- SparseCL alpha：{sparsecl['alpha']}（{sparsecl['alpha_source']}）；"
                         f"BGE 初召回深度：{sparsecl['initial_retrieval_depth']}；"
                         f"dev NDCG@10：{fmt(sparsecl.get('dev_ndcg_at_10'))}。")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=ROOT / "data" / "validmem_v7")
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
    result.add_argument("--evaluation-split", choices=("exploratory", "dev", "test"),
                        default="exploratory",
                        help="记录评估集用途；test 必须使用与 v7 已查看查询不重叠的人工核验数据")
    result.add_argument("--dev-run", type=Path,
                        help="test 运行必填：开发集运行目录，用于核对冻结阈值和查询不重叠")
    result.add_argument("--query-groups", type=Path,
                        help="JSON 对象：query_id 到场景/模板族；用于整组留出和分组 bootstrap")
    result.add_argument("--seed", type=int, default=20261014)
    result.add_argument("--query-ids", type=lambda s: [x for x in s.split(",") if x],
                        help="显式指定 query_id(逗号分隔);库容对比时让两次运行处理同一批写入")
    result.add_argument("--workers", type=int, default=4, help="每个方法臂的并行查询数")
    result.add_argument("--arms", nargs="+", choices=tuple(ARM_SPECS), default=("proposed",),
                        help="运行的方法；主比较请显式指定 proposed bm25 dense hybrid")
    result.add_argument("--judge-arms", nargs="*", choices=tuple(ARM_SPECS), default=(),
                        help="运行逐对冲突判定的臂；默认不调用判定器")
    result.add_argument("--coarse-k", type=int)
    result.add_argument("--final-k", type=int)
    result.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help="proposed/independent 的开发集固定阈值；baseline 忽略此项")
    result.add_argument("--cache-policy", choices=("isolated", "shared"), default="isolated",
                        help="isolated 为每个方法臂使用独立临时 API 缓存，避免跨臂预热")
    result.add_argument("--sparsecl-root", type=Path, default=SPARSECL_ROOT,
                        help="SparseCL 复现仓库目录；默认使用工作区 baselines/SparseCL")
    result.add_argument("--sparsecl-wsl-distro", default=SPARSECL_WSL_DISTRO)
    result.add_argument("--sparsecl-wsl-user", default=SPARSECL_WSL_USER)
    result.add_argument("--sparsecl-wsl-python", default=SPARSECL_WSL_PYTHON)
    result.add_argument("--sparsecl-batch-size", type=int, default=SPARSECL_BATCH_SIZE)
    result.add_argument("--sparsecl-cpu-threads", type=int, default=SPARSECL_CPU_THREADS)
    return result


def sparsecl_launcher(args):
    root = args.sparsecl_root.resolve()
    if not (root / "sparsecl" / "models.py").is_file():
        raise ValueError(f"找不到已复现的 SparseCL 模型代码：{root}")
    if not SPARSECL_WORKER.is_file():
        raise ValueError(f"SparseCL WSL worker 不存在：{SPARSECL_WORKER}")
    if args.sparsecl_batch_size < 1 or args.sparsecl_cpu_threads < 1:
        raise ValueError("SparseCL batch size 与 CPU threads 必须为正整数")
    return {"distro": args.sparsecl_wsl_distro, "user": args.sparsecl_wsl_user,
            "python": args.sparsecl_wsl_python, "baseline_root": str(root),
            "batch_size": args.sparsecl_batch_size,
            "cpu_threads": args.sparsecl_cpu_threads}


def sparsecl_memory_bank_sha256(memories):
    payload = [{"memory_id": memory.memory_id, "content": memory.content.strip()}
               for memory in memories]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def wsl_path(path, launcher):
    shell_command = f"wslpath -a {shlex.quote(str(Path(path).resolve()))}"
    command = ["wsl.exe", "-d", launcher["distro"], "-u", launcher["user"], "--",
               "bash", "-lc", shell_command]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
    if completed.returncode:
        raise ValueError(f"无法将路径转换为 WSL 路径：{path}\n{(completed.stderr or '').strip()}")
    return completed.stdout.strip()


def run_sparsecl_arm(args, memories, queries, gold, final_k, run_dir, cache_root,
                     expected_signature=None, expected_cache_artifacts=None, alpha=None):
    launcher = sparsecl_launcher(args)
    input_path = run_dir / "input" / "sparsecl_input.json"
    output_path = run_dir / "arms" / "sparsecl.jsonl"
    metadata_path = run_dir / "sparsecl_run.json"
    payload = {
        "mode": args.evaluation_split,
        "final_k": final_k,
        "expected_signature": expected_signature,
        "expected_cache_artifacts": expected_cache_artifacts,
        "alpha": alpha,
        "memories": [{"memory_id": memory.memory_id, "content": memory.content}
                     for memory in memories],
        "queries": [{"query_id": query["query_id"], "content": query["content"]}
                    for query in queries],
    }
    if args.evaluation_split == "dev":
        payload["qrels"] = [{"query_id": row["query_id"],
                             "conflict_old_memory_ids": row["conflict_old_memory_ids"]}
                            for row in gold]
    write_json(input_path, payload)
    worker = wsl_path(SPARSECL_WORKER, launcher)
    baseline_root = wsl_path(launcher["baseline_root"], launcher)
    input_wsl = wsl_path(input_path, launcher)
    output_wsl = wsl_path(output_path, launcher)
    metadata_wsl = wsl_path(metadata_path, launcher)
    cache_wsl = wsl_path(cache_root, launcher)
    command = ["wsl.exe", "-d", launcher["distro"], "-u", launcher["user"], "--",
               launcher["python"], worker, "--input", input_wsl, "--output", output_wsl,
               "--metadata", metadata_wsl, "--baseline-root", baseline_root,
               "--cache-root", cache_wsl, "--batch-size", str(launcher["batch_size"]),
               "--cpu-threads", str(launcher["cpu_threads"])]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise ValueError(f"SparseCL WSL worker 退出，状态码 {completed.returncode}")
    if not output_path.is_file() or not metadata_path.is_file():
        raise ValueError("SparseCL worker 未生成完整的结果和运行元数据")
    results = read_jsonl(output_path)
    result_ids = [row.get("query_id") for row in results]
    expected_ids = [row["query_id"] for row in queries]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(expected_ids):
        raise ValueError("SparseCL 输出必须恰好覆盖输入中的每条查询一次")
    metadata = read_json(metadata_path)
    if metadata.get("signature", {}).get("memory_bank_sha256") != sparsecl_memory_bank_sha256(memories):
        raise ValueError("SparseCL worker 返回的冻结旧库签名不匹配")
    if expected_signature is not None and metadata.get("signature") != expected_signature:
        raise ValueError("SparseCL worker 返回的模型/代码/索引签名与 dev 冻结设置不匹配")
    cache_artifacts = metadata.get("cache_artifacts")
    if (not isinstance(cache_artifacts, dict)
            or set(cache_artifacts) != {"vectors_sha256", "index_sha256"}
            or any(not isinstance(value, str) or len(value) != 64
                   for value in cache_artifacts.values())):
        raise ValueError("SparseCL worker 未返回有效的旧库向量与 HNSW 文件哈希")
    if expected_cache_artifacts is not None and cache_artifacts != expected_cache_artifacts:
        raise ValueError("SparseCL worker 返回的缓存文件哈希与 dev 冻结索引不一致")
    if alpha is not None and metadata.get("alpha") != alpha:
        raise ValueError("SparseCL worker 返回的 alpha 与 dev 冻结值不匹配")
    selected_alpha = metadata.get("alpha")
    if (isinstance(selected_alpha, bool) or not isinstance(selected_alpha, (int, float))
            or not 0 <= selected_alpha < 10):
        raise ValueError("SparseCL worker 返回了无效的 alpha")
    expected_source = {"dev": "dev_ndcg_at_10", "test": "dev_frozen",
                       "exploratory": "arguana_fixed_0.18"}[args.evaluation_split]
    expected_depth = min(1000, len(memories))
    expected_memory_ids = {memory.memory_id for memory in memories}
    if (metadata.get("alpha_source") != expected_source
            or metadata.get("initial_retrieval_depth") != expected_depth):
        raise ValueError("SparseCL alpha 来源或 Top-1000 初召回深度与评估协议不一致")
    for row in results:
        initial = row.get("initial_candidates") or []
        coarse = row.get("coarse_candidates") or []
        fused = row.get("ranked_candidates") or []
        initial_ids = [candidate.get("memory_id") for candidate in initial]
        coarse_ids = [candidate.get("memory_id") for candidate in coarse]
        fused_ids = [candidate.get("memory_id") for candidate in fused]
        final_ids = [candidate.get("memory_id") for candidate in row.get("candidates") or []]
        cosine_ids = [candidate.get("memory_id")
                      for candidate in row.get("cosine_ranked_candidates") or []]
        fixed_rank = row.get("arguana_alpha_ranked_candidates")
        fixed_ids = ([candidate.get("memory_id") for candidate in fixed_rank]
                     if fixed_rank is not None else None)
        expected_final = fused_ids[:len(final_ids)]
        fixed_rank_required = abs(selected_alpha - 0.18) >= 1e-12
        if (row.get("sparsecl_alpha") != metadata.get("alpha")
                or len(initial) != expected_depth or len(coarse) != expected_depth
                or len(set(initial_ids)) != expected_depth
                or not set(initial_ids) <= expected_memory_ids
                or len(fused) != expected_depth or len(set(fused_ids)) != expected_depth
                or coarse_ids != initial_ids or cosine_ids != initial_ids
                or set(fused_ids) != set(initial_ids)
                or len(final_ids) != min(final_k, expected_depth)
                or (fixed_rank_required and fixed_ids is None)
                or (fixed_ids is not None and (len(fixed_ids) != expected_depth
                                                or set(fixed_ids) != set(initial_ids)))
                or final_ids != expected_final):
            raise ValueError(f"SparseCL/{row.get('query_id')}: 初召回、融合排序或最终候选不一致")
    metadata["input_sha256"] = sha256(input_path)
    metadata["result_sha256"] = sha256(output_path)
    metadata["worker_metadata_sha256"] = sha256(metadata_path)
    return results, metadata, launcher


def _run_one(index, settings, query, spec, judge, coarse_k, final_k,
             similarity_threshold, bm25_index):
    retriever = Retriever(index, APIClient(settings), bm25=bm25_index)
    # v7 全库检索仅按正文工作，不使用 personal/world 或 owner 元数据预过滤。
    return retriever.query(
        query["query_id"], query["content"],
        retrieval_text=build_retrieval_text(query["content"], query.get("source_ref")),
        query_source_ref=query.get("source_ref"),
        method=spec["method"], rerank=spec["rerank"], judge=judge,
        coarse_k=coarse_k, final_k=final_k,
        similarity_threshold=similarity_threshold)


def run_arm(index, settings, queries, spec, arm, judge, workers, coarse_k, final_k,
            similarity_threshold, bm25_index, results_path, failure_path):
    results = []
    errors = []
    sample_indices = {query["query_id"]: i for i, query in enumerate(queries)}
    with Path(results_path).open("x", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one, index, settings, query, spec, judge, coarse_k, final_k,
                            similarity_threshold, bm25_index): query
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


def validate_completed_run(run_dir, run_config, queries):
    if (run_dir / "failure.json").exists():
        raise ValueError("dev 运行存在 failure.json，不能作为正式 test 的冻结依据")
    completion_path = run_dir / "completion.json"
    report_path = run_dir / "report.json"
    if not completion_path.is_file() or not report_path.is_file():
        raise ValueError("dev 运行缺少完成标记或最终 report.json")
    completion = read_json(completion_path)
    configured_arms = run_config.get("arms")
    recorded_arms = completion.get("arms")
    if not isinstance(configured_arms, dict) or not configured_arms:
        raise ValueError("dev 运行配置未记录有效的方法臂")
    expected_arms = set(configured_arms)
    if not isinstance(recorded_arms, dict):
        raise ValueError("dev 完成标记未记录方法臂结果")
    if (completion.get("schema_version") != 1
            or completion.get("status") != "complete"
            or completion.get("run_id") != run_config.get("run_id")
            or completion.get("query_count") != len(queries)
            or set(recorded_arms) != expected_arms):
        raise ValueError("dev 完成标记与运行配置或查询数不一致")

    report = read_json(report_path)
    report_run = report.get("run")
    report_arms = report.get("arms")
    if (not isinstance(report_run, dict) or not isinstance(report_arms, dict)
            or report_run.get("run_id") != run_config.get("run_id")
            or set(report_arms) != expected_arms):
        raise ValueError("dev report.json 未完整记录配置中的所有方法臂")

    expected_query_ids = {row["query_id"] for row in queries}
    for arm in expected_arms:
        if arm not in ARM_SPECS:
            raise ValueError(f"dev 运行配置包含未知方法臂：{arm}")
        result_path = run_dir / "arms" / f"{arm}.jsonl"
        record = recorded_arms.get(arm)
        if not isinstance(record, dict):
            raise ValueError(f"dev 方法臂 {arm} 的完成记录无效")
        if (not result_path.is_file()
                or record.get("query_count") != len(queries)
                or record.get("result_sha256") != sha256(result_path)):
            raise ValueError(f"dev 方法臂 {arm} 的结果缺失、不完整或哈希不匹配")
        results = read_jsonl(result_path)
        result_ids = [row.get("query_id") for row in results]
        if len(result_ids) != len(expected_query_ids) or set(result_ids) != expected_query_ids:
            raise ValueError(f"dev 方法臂 {arm} 的结果未恰好覆盖全部开发查询")


def load_query_groups(path, queries):
    if path is None:
        return None
    groups = read_json(path)
    query_ids = {row["query_id"] for row in queries}
    if not isinstance(groups, dict) or set(groups) != query_ids:
        raise ValueError("--query-groups 必须为数据集每条查询提供且只提供一个分组")
    if any(not isinstance(group, str) or not group.strip() for group in groups.values()):
        raise ValueError("--query-groups 的每个分组名都必须是非空字符串")
    return groups


def validate_evaluation_split(dataset_dir, queries, gold, split, limit, query_ids,
                              similarity_threshold, dev_run_path, query_groups):
    if split != "test":
        if dev_run_path is not None:
            raise ValueError("--dev-run 仅用于 --evaluation-split test")
        return
    if limit != 0 or query_ids:
        raise ValueError("test 必须完整评估独立数据集中的全部查询；请使用 --limit 0 且不要传 --query-ids")
    if dev_run_path is None:
        raise ValueError("test 必须通过 --dev-run 指定已经冻结参数的开发集运行")
    if query_groups is None:
        raise ValueError("test 必须通过 --query-groups 提供场景/模板族分组")

    dev_run_path = dev_run_path.resolve()
    dev_config_path = dev_run_path / "run_config.json"
    dev_queries_path = dev_run_path / "input" / "queries.jsonl"
    if not dev_config_path.is_file() or not dev_queries_path.is_file():
        raise ValueError(f"--dev-run 缺少 run_config.json 或 input/queries.jsonl：{dev_run_path}")
    dev_config = read_json(dev_config_path)
    dev_split = dev_config.get("evaluation_split", "exploratory")
    if dev_split != "dev":
        raise ValueError("正式 test 必须引用一个明确标记为 dev 的开发集运行")
    dev_sample = dev_config.get("sample") or {}
    dev_dataset = dev_config.get("dataset") or {}
    if dev_sample.get("query_count") != dev_dataset.get("query_count"):
        raise ValueError("dev 运行必须完整包含其开发数据集中的所有查询")
    dev_memory_hash = dev_config.get("dataset", {}).get("files_sha256", {}).get("memories.jsonl")
    current_memory_hash = sha256(dataset_dir / "memories.jsonl")
    if dev_memory_hash != current_memory_hash:
        raise ValueError("dev 与 test 必须使用哈希完全相同的冻结旧记忆库")
    dev_threshold = dev_config.get("settings", {}).get("similarity_threshold")
    if dev_threshold is None or abs(float(dev_threshold) - similarity_threshold) > 1e-12:
        raise ValueError(
            f"test 动态阈值 {similarity_threshold} 必须与 dev 冻结值 {dev_threshold} 一致"
        )
    dev_groups_path = dev_run_path / "input" / "query_groups.json"
    if not dev_groups_path.is_file():
        raise ValueError("dev 运行缺少 input/query_groups.json，无法确认按组留出")
    dev_group_digest = (dev_config.get("query_groups") or {}).get("sha256")
    if dev_group_digest is None or sha256(dev_groups_path) != dev_group_digest:
        raise ValueError("dev 的 query_groups.json 与运行时冻结的分组映射不一致")
    dev_queries = read_jsonl(dev_queries_path)
    validate_completed_run(dev_run_path, dev_config, dev_queries)
    dev_groups = load_query_groups(dev_groups_path, dev_queries)
    overlapping_groups = sorted(set(query_groups.values()) & set(dev_groups.values()))
    if overlapping_groups:
        raise ValueError(f"dev 与 test 必须按场景/模板族整组隔离；重叠分组={overlapping_groups[:10]}")

    known_dataset = ROOT / "data" / "validmem_v7"
    known_queries_path = known_dataset / "queries.jsonl"
    known_queries = read_jsonl(known_queries_path)
    known_ids = {row["query_id"] for row in known_queries}
    normalize = lambda text: " ".join(text.casefold().split())
    known_texts = {normalize(row["content"]) for row in known_queries}
    id_overlap = sorted(known_ids & {row["query_id"] for row in queries})
    text_overlap = sorted({row["query_id"] for row in queries
                           if normalize(row["content"]) in known_texts})
    dev_ids = {row["query_id"] for row in dev_queries}
    dev_texts = {normalize(row["content"]) for row in dev_queries}
    dev_id_overlap = sorted(dev_ids & {row["query_id"] for row in queries})
    dev_text_overlap = sorted({row["query_id"] for row in queries
                               if normalize(row["content"]) in dev_texts})
    if id_overlap or text_overlap or dev_id_overlap or dev_text_overlap:
        raise ValueError(
            "test 查询与已查看的 v7 或开发集查询重叠；请按场景/模板族整组留出新查询。"
            f" v7 query_id/正文={id_overlap[:5]}/{text_overlap[:5]}，"
            f"dev query_id/正文={dev_id_overlap[:5]}/{dev_text_overlap[:5]}"
        )
    unreviewed = [row["query_id"] for row in gold
                  if row.get("fully_reviewed") is not True or row.get("human_reviewed") is not True]
    if unreviewed:
        raise ValueError(
            "test 集的金标必须逐查询设置 fully_reviewed=true 和 human_reviewed=true；"
            f"未核验示例={unreviewed[:5]}"
        )
    return {"dev_run_id": dev_config.get("run_id", dev_run_path.name),
            "dev_run_path": str(dev_run_path), "dev_split": dev_split,
            "memory_bank_sha256": current_memory_hash,
            "query_ids_and_texts_disjoint": True,
            "dev_group_count": len(set(dev_groups.values())),
            "test_group_count": len(set(query_groups.values())),
            "group_disjoint": True}


def validate_test_run_configuration(dev_reference, args, settings, coarse_k, final_k):
    if dev_reference is None:
        return
    expected_arms = {"proposed", "bm25", "dense", "hybrid"}
    if "sparsecl" in args.arms:
        expected_arms.add("sparsecl")
    if set(args.arms) != expected_arms or args.judge_arms:
        raise ValueError("正式 test 只运行完整主比较方法集合，且不启用逐对判定器")
    if args.cache_policy != "isolated":
        raise ValueError("正式 test 必须使用 --cache-policy isolated")
    if coarse_k < 50:
        raise ValueError("正式 test 至少要保存 top-50 排序，请设置 --coarse-k 50")

    dev_config = read_json(Path(dev_reference["dev_run_path"]) / "run_config.json")
    dev_settings = dev_config.get("settings") or {}
    if dev_settings.get("llm") != settings.llm_identity():
        raise ValueError("test 与 dev 的 LLM 服务/模型配置不一致")
    if dev_settings.get("embedding") != settings.embedding_identity():
        raise ValueError("test 与 dev 的嵌入服务/模型配置不一致")
    if dev_settings.get("prompt_hash") != PROMPT_HASH or dev_settings.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("test 与 dev 的查询提示版本不一致")
    if dev_settings.get("index_prompt_hash") != INDEX_PROMPT_HASH:
        raise ValueError("test 与 dev 的描述索引提示版本不一致")
    if dev_settings.get("similarity_threshold") != args.similarity_threshold:
        raise ValueError("test 的动态阈值必须与 dev 记录的阈值完全一致")
    for key, current in (("coarse_k", coarse_k), ("final_k", final_k), ("workers", args.workers)):
        if dev_settings.get(key) != current:
            raise ValueError(f"test 与 dev 的 {key} 设置不一致")
    operational_settings = (
        ("api_timeout_seconds", settings.timeout),
        ("api_max_retries", settings.max_retries),
        ("api_rate_limit_backoff_seconds", settings.rate_limit_backoff),
        ("embedding_batch_size", settings.embedding_batch_size),
    )
    for key, current in operational_settings:
        if dev_settings.get(key) != current:
            raise ValueError(f"test 与 dev 的 {key} 设置不一致")
    if dev_config.get("cache_policy") != args.cache_policy:
        raise ValueError("test 与 dev 的 API 缓存策略不一致")
    if set(dev_config.get("arms") or {}) != expected_arms:
        raise ValueError("dev 必须包含与 test 完全相同的主比较方法集合")
    if any((spec or {}).get("judged") for spec in dev_config.get("arms", {}).values()):
        raise ValueError("dev 主比较不能混入逐对冲突判定器")
    if args.index_dir is None:
        raise ValueError("正式 test 必须通过 --index-dir 复用 dev 冻结的同一个描述索引")
    if "sparsecl" in args.arms:
        dev_sparsecl = dev_config.get("sparsecl") or {}
        reference = dev_reference.get("sparsecl_alpha") or {}
        if not dev_sparsecl.get("signature") or reference.get("signature") != dev_sparsecl["signature"]:
            raise ValueError("SparseCL 开发集 alpha 签名缺失或与开发运行配置不一致")
        if dev_sparsecl.get("launcher") != sparsecl_launcher(args):
            raise ValueError("test 与 dev 的 SparseCL WSL 环境或推理资源设置不一致")
        if dev_sparsecl.get("cache_root") != str((args.index_dir.resolve() / "sparsecl").resolve()):
            raise ValueError("SparseCL test 必须复用 dev 冻结的双向量与 HNSW 索引缓存目录")


def load_sparsecl_alpha_reference(dev_reference, memories):
    dev_run_path = Path(dev_reference["dev_run_path"])
    dev_config = read_json(dev_run_path / "run_config.json")
    dev_sparsecl = dev_config.get("sparsecl") or {}
    signature = dev_sparsecl.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("SparseCL test 必须引用包含完整签名的 dev 运行")
    if signature.get("memory_bank_sha256") != sparsecl_memory_bank_sha256(memories):
        raise ValueError("SparseCL dev 与 test 的旧记忆正文或 ID 不一致")
    record_path = dev_run_path / "sparsecl_alpha.json"
    if not record_path.is_file():
        raise ValueError("SparseCL test 必须使用 dev 运行生成的 sparsecl_alpha.json")
    record = read_json(record_path)
    if (record.get("schema_version") != 2
            or record.get("split") != "dev" or record.get("selected_by") != "dev_ndcg_at_10"
            or record.get("signature") != signature
            or record.get("dev_run_id") != dev_config.get("run_id")):
        raise ValueError("SparseCL alpha 文件不是当前完整 dev 运行的冻结调参记录")
    if (record.get("alpha") != dev_sparsecl.get("alpha")
            or record.get("dev_ndcg_at_10") != dev_sparsecl.get("dev_ndcg_at_10")
            or record.get("cache_artifacts") != dev_sparsecl.get("cache_artifacts")
            or dev_sparsecl.get("alpha_record_sha256") != sha256(record_path)):
        raise ValueError("SparseCL alpha 文件与 dev 运行记录的校验值不一致")
    cache_artifacts = record.get("cache_artifacts")
    if (not isinstance(cache_artifacts, dict)
            or set(cache_artifacts) != {"vectors_sha256", "index_sha256"}
            or any(not isinstance(value, str) or len(value) != 64
                   for value in cache_artifacts.values())):
        raise ValueError("SparseCL dev 冻结记录缺少实际向量或 HNSW 文件哈希")
    query_path = dev_run_path / "input" / "queries.jsonl"
    gold_path = dev_run_path / "input" / "gold.jsonl"
    groups_path = dev_run_path / "input" / "query_groups.json"
    if not query_path.is_file() or not gold_path.is_file() or not groups_path.is_file():
        raise ValueError("SparseCL dev 运行缺少冻结的查询、金标或分组文件")
    if (record.get("dev_queries_sha256") != sha256(query_path)
            or record.get("dev_gold_sha256") != sha256(gold_path)
            or record.get("dev_query_groups_sha256") != sha256(groups_path)):
        raise ValueError("SparseCL alpha 文件与 dev 冻结查询、金标或分组文件不匹配")
    alpha = record.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha < 10:
        raise ValueError("SparseCL dev 冻结 alpha 无效")
    return {"alpha": float(alpha), "signature": signature,
            "cache_artifacts": cache_artifacts,
            "record_sha256": sha256(record_path), "record": record}


def main(argv=None):
    args = parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if args.limit < 0 or args.workers < 1:
        raise SystemExit("--limit 不能小于 0，--workers 必须大于 0")
    if not 0 <= args.similarity_threshold <= 1:
        raise SystemExit("--similarity-threshold 必须在 0 到 1 之间")
    if not set(args.judge_arms) <= set(args.arms):
        raise SystemExit("--judge-arms 中的每个方法都必须包含在 --arms 中")
    if "sparsecl" in args.judge_arms:
        raise SystemExit("SparseCL baseline 不运行逐对冲突判定器")
    if "sparsecl" in args.arms:
        try:
            sparsecl_launcher(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        if args.evaluation_split == "dev" and (args.limit != 0 or args.query_ids):
            raise SystemExit("SparseCL dev 必须用 --limit 0 完整运行，不能传 --query-ids")
        if args.evaluation_split == "dev" and args.query_groups is None:
            raise SystemExit("SparseCL dev 必须冻结 query-groups，供正式 test 按组隔离")
        if args.evaluation_split == "dev" and args.cache_policy != "isolated":
            raise SystemExit("SparseCL dev 必须使用 --cache-policy isolated，供 test 沿用冻结设置")

    dataset_dir = args.dataset.resolve()
    files = {name: dataset_dir / name for name in ("memories.jsonl", "queries.jsonl", "gold.jsonl")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit("数据集文件缺失：" + ", ".join(missing))

    load_memory_rows = load_text if args.adapter == "text" else load_graph
    memories = load_memory_rows(files["memories.jsonl"])
    all_queries = load_queries(files["queries.jsonl"])
    all_gold = read_jsonl(files["gold.jsonl"])
    try:
        query_groups = load_query_groups(args.query_groups, all_queries)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"查询分组文件无效：{exc}") from None
    validate_gold(all_gold, {memory.memory_id for memory in memories})
    if {row["query_id"] for row in all_queries} != {row["query_id"] for row in all_gold}:
        raise SystemExit("查询与金标 query_id 不一致")
    try:
        evaluation_reference = validate_evaluation_split(
            dataset_dir, all_queries, all_gold, args.evaluation_split,
            args.limit, args.query_ids, args.similarity_threshold, args.dev_run,
            query_groups)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
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
    selected_query_groups = ({qid: query_groups[qid] for qid in selected_ids}
                             if query_groups is not None else None)
    if selected_query_groups is not None:
        write_json(input_dir / "query_groups.json", selected_query_groups)

    settings = Settings.from_env(args.env)
    needs_project_index = any(arm != "sparsecl" for arm in args.arms)
    if needs_project_index:
        settings.require(llm=True, embedding=True)
    if args.coarse_k is not None or args.final_k is not None:
        coarse_k = args.coarse_k if args.coarse_k is not None else settings.coarse_k
        final_k = args.final_k if args.final_k is not None else settings.final_k
        if needs_project_index:
            settings.validate_budget(coarse_k, final_k)
        elif final_k < 1:
            raise SystemExit("--final-k 必须大于 0")
    else:
        coarse_k, final_k = settings.coarse_k, settings.final_k
    try:
        sparsecl_reference = None
        if "sparsecl" in args.arms and args.evaluation_split == "test":
            sparsecl_reference = load_sparsecl_alpha_reference(evaluation_reference, memories)
            evaluation_reference["sparsecl_alpha"] = sparsecl_reference
        validate_test_run_configuration(evaluation_reference, args, settings, coarse_k, final_k)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    sparsecl_cache_root = None
    if "sparsecl" in args.arms:
        base_index_dir = args.index_dir.resolve() if args.index_dir else index_output.resolve()
        sparsecl_cache_root = base_index_dir / "sparsecl"

    index = None
    build_seconds = 0.0
    index_origin = None
    build_usage = {}
    if needs_project_index:
        index_started = time.perf_counter()
        try:
            if args.index_dir:
                index = MemoryIndex.load(args.index_dir.resolve())
                if evaluation_reference is not None:
                    dev_config = read_json(Path(evaluation_reference["dev_run_path"]) / "run_config.json")
                    if index.manifest.get("index_id") != dev_config.get("index", {}).get("index_id"):
                        raise ValueError("test 必须复用 dev 中冻结的同一索引 index_id")
                # 复用索引按 memory_id + 检索文本逐条核对；正文或主体写法改变都会使
                # 已冻结的描述与向量失效，不能继续复用旧索引。
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
                index_origin = str(args.index_dir.resolve())
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

    bm25_index = None
    bm25_build_seconds = None
    if any(ARM_SPECS[arm]["method"] in {"bm25", "hybrid"} for arm in args.arms):
        bm25_started = time.perf_counter()
        bm25_index = BM25([memory.retrieval_text for memory in index.memories])
        bm25_build_seconds = time.perf_counter() - bm25_started

    manifest = {
        "run_id": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "shared_bank_retrieval_comparison",
        "evaluation_split": args.evaluation_split,
        "evaluation": {"split": args.evaluation_split,
                       "development_reference": evaluation_reference},
        "query_groups": {"provided": selected_query_groups is not None,
                         "group_count": len(set(selected_query_groups.values()))
                         if selected_query_groups is not None else None,
                         "sha256": sha256(input_dir / "query_groups.json")
                         if selected_query_groups is not None else None},
        "cache_policy": args.cache_policy,
        "dataset": {"path": str(dataset_dir), "files_sha256": {name: sha256(path) for name, path in files.items()},
                    "memory_count": len(memories), "query_count": len(all_queries), "gold_count": len(all_gold)},
        "sample": {"query_count": len(queries), "query_ids": [row["query_id"] for row in queries],
                   "limit": args.limit, "seed": args.seed,
                   "explicit_query_ids": bool(args.query_ids)},
        "settings": {"similarity_threshold": args.similarity_threshold,
                     "baseline_budget_depth": coarse_k if needs_project_index else None,
                     "coarse_k": coarse_k if needs_project_index else None,
                     "final_k": final_k,
                     "prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH,
                     "index_prompt_hash": INDEX_PROMPT_HASH,
                     "workers": args.workers,
                     "llm": settings.llm_identity() if needs_project_index else None,
                     "embedding": settings.embedding_identity() if needs_project_index else None,
                     "llm_temperature": settings.llm_temperature if needs_project_index else None,
                     "llm_json_mode": settings.llm_json_mode if needs_project_index else None,
                     "llm_reasoning_effort": settings.llm_reasoning_effort if needs_project_index else None,
                     "api_timeout_seconds": settings.timeout,
                     "api_max_retries": settings.max_retries,
                     "api_rate_limit_backoff_seconds": settings.rate_limit_backoff,
                     "embedding_batch_size": settings.embedding_batch_size},
        "index": {"index_id": index.manifest["index_id"] if index else None,
                  "origin": index_origin,
                  "description_mode": index.manifest["description_mode"] if index else None,
                  "batch_size": index.manifest.get("batch_size") if index else None,
                  "build_seconds": build_seconds, "build_usage": build_usage,
                  "bm25_build_seconds": bm25_build_seconds,
                  "bm25_document_count": len(index.memories) if bm25_index is not None else 0,
                  "sparsecl_cache_root": str(sparsecl_cache_root.resolve())
                  if sparsecl_cache_root else None},
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
            if arm == "sparsecl":
                alpha = sparsecl_reference["alpha"] if sparsecl_reference else None
                expected_signature = sparsecl_reference["signature"] if sparsecl_reference else None
                results, sparsecl_metadata, launcher = run_sparsecl_arm(
                    args, memories, queries, gold, final_k, run_dir, sparsecl_cache_root,
                    expected_signature=expected_signature,
                    expected_cache_artifacts=(sparsecl_reference["cache_artifacts"]
                                              if sparsecl_reference else None),
                    alpha=alpha)
                metadata = {**sparsecl_metadata, "launcher": launcher,
                            "cache_root": str(sparsecl_cache_root.resolve())}
                if args.evaluation_split == "dev":
                    alpha_record = {
                        "schema_version": 2,
                        "split": "dev",
                        "selected_by": "dev_ndcg_at_10",
                        "alpha": sparsecl_metadata["alpha"],
                        "dev_ndcg_at_10": sparsecl_metadata["dev_ndcg_at_10"],
                        "signature": sparsecl_metadata["signature"],
                        "cache_artifacts": sparsecl_metadata["cache_artifacts"],
                        "dev_run_id": run_name,
                        "dev_queries_sha256": sha256(input_dir / "queries.jsonl"),
                        "dev_gold_sha256": sha256(input_dir / "gold.jsonl"),
                        "dev_query_groups_sha256": sha256(input_dir / "query_groups.json"),
                    }
                    alpha_path = run_dir / "sparsecl_alpha.json"
                    write_json(alpha_path, alpha_record)
                    metadata["alpha_record_sha256"] = sha256(alpha_path)
                    metadata["alpha_record_path"] = str(alpha_path)
                manifest["sparsecl"] = metadata
                write_json(run_dir / "run_config.json", manifest)
                metrics = evaluate(gold, results)
                report["arms"][arm] = {
                    "kind": spec["kind"], "judged": False,
                    "result_file": str(results_path), "metrics": metrics,
                    "usage": usage_totals(results), "call_breakdown": call_totals(results),
                    "latency_basis": sparsecl_metadata["costs"]["query_latency_basis"],
                    "sparsecl": {key: metadata[key] for key in (
                        "alpha", "alpha_source", "dev_ndcg_at_10",
                        "initial_retrieval_depth", "costs")},
                }
                write_json(run_dir / "report.json", report)
                continue

            def execute(active_settings):
                return run_arm(index, active_settings, queries, spec, arm,
                               arm in args.judge_arms, args.workers, coarse_k, final_k,
                               args.similarity_threshold, bm25_index,
                               results_path, run_dir / "failure.json")

            if args.cache_policy == "isolated":
                with tempfile.TemporaryDirectory(prefix=f"retrieval-{arm}-") as cache_dir:
                    arm_settings = replace(settings, cache_path=Path(cache_dir) / "api_cache.sqlite3")
                    results = execute(arm_settings)
            else:
                results = execute(settings)
            metrics = evaluate(gold, results)
        except (APIError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            if not (run_dir / "failure.json").exists():
                write_json(run_dir / "failure.json", {"stage": "query", "arm": arm, "error": str(exc)})
            raise
        report["arms"][arm] = {
            "kind": spec["kind"], "judged": arm in args.judge_arms,
            "result_file": str(results_path), "metrics": metrics,
            "usage": usage_totals(results),
            "call_breakdown": call_totals(results),
            "latency_basis": "per_query_elapsed",
        }
        write_json(run_dir / "report.json", report)

    write_summary(run_dir / "summary.md", report)
    completion = {
        "schema_version": 1,
        "run_id": run_name,
        "status": "complete",
        "query_count": len(queries),
        "arms": {
            arm: {"query_count": len(queries),
                  "result_sha256": sha256(arm_output / f"{arm}.jsonl")}
            for arm in args.arms
        },
    }
    write_json(run_dir / "completion.json", completion)
    print(f"实验完成：{run_dir}", flush=True)
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return run_dir


if __name__ == "__main__":
    try:
        main()
    except (APIError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f"实验失败：{exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
