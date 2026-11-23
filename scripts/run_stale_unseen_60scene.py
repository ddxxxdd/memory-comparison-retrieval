"""Run the frozen 60-scene STALE comparison with resumable artifacts, without editing the main package."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from memory_retrieval.api import APIClient, usage_summary
from memory_retrieval.bipolar_graph import (ANCHORED_PROMPT_SHA256, audit_anchored_probes,
    build_knn_transition, compute_bipolar_diffusion_scores, extract_anchored_probes,
    validate_anchored_probe_structure)
from memory_retrieval.config import Settings
from memory_retrieval.data import sha256_file, write_json, write_jsonl
from memory_retrieval.dataset import load_dataset
from memory_retrieval.encoders import FrozenEncoder, MODEL_ROOT, MODEL_LOCK
from memory_retrieval.evaluation import grouped_ranking_metrics, target_ids
from memory_retrieval.experiment import score_original_text, sparsecl_pools, ranked_row
from memory_retrieval.sparsecl import score_candidates, fused_order

DEFAULT_OUTPUT = ROOT / "runs/experiments/stale-unseen-60scene-bipolar-seed20261122"
PROMPT_HASH = "a38b198018c62846780914f1ef90fe2461ee1554c421d42996f54525f93217bf"
BASELINES = ["bm25", "dense", "hybrid", "sparsecl_arguana", "sparsecl_msmarco", "sparsecl_hotpotqa"]
OUR = ["anchored_probes_only", "anchored_pos_diffusion", "anchored_bipolar_diffusion"]
NAMES = {"bm25": "BM25", "dense": "BGE Dense", "hybrid": "Hybrid", "sparsecl_arguana": "SparseCL ArguAna",
         "sparsecl_msmarco": "SparseCL MS MARCO", "sparsecl_hotpotqa": "SparseCL HotpotQA",
         "anchored_bipolar_diffusion": "完整方法", "anchored_probes_only": "仅探针", "anchored_pos_diffusion": "仅正扩散"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def verify_protocol(output):
    protocol = read(output / "protocol.json")
    assert ANCHORED_PROMPT_SHA256 == protocol["prompt_sha256"] == PROMPT_HASH
    assert protocol["graph"] == {"k": 5, "top_seeds": 5, "alpha": .2, "beta": .3}
    assert protocol["probe_workers"] == 16
    for name, digest in protocol["code_sha256"].items():
        assert sha256_file(ROOT / "memory_retrieval" / name) == digest, name
    assert sha256_file(ROOT / "memory_retrieval/model_lock.json") == protocol["model_lock_sha256"]
    for name, digest in protocol["input_sha256"].items():
        assert sha256_file(output / "inputs" / name) == digest, name
    for name in ("selection", "exclusions"):
        assert sha256_file(output / (name + ".json")) == protocol[name + "_sha256"]
    return protocol


def encode_worker(output, name):
    started = time.perf_counter()
    verify_protocol(output)
    dataset = load_dataset(output / "inputs")
    folder = output / "encoders" / name
    folder.mkdir(parents=True, exist_ok=True)
    info_path = folder / "identity.json"
    info = read(info_path) if info_path.exists() else {"timing_seconds": {}}
    encoder = None
    groups = {"memories": [m.content.strip() for m in dataset.memories],
              "queries": [q["content"].strip() for q in dataset.queries]}
    if name == "bge":
        groups["anchored_violation_probes"] = None
    try:
        for key, texts in groups.items():
            target = folder / (key + ".npy")
            if target.exists():
                assert sha256_file(target) == info["vector_sha256"][key]
                continue
            if texts is None:
                atomic_json(folder / "status.json", {"status": "waiting_for_probes"})
                while not (output / "anchored_probes.json").exists():
                    if (output / "probe_failures.json").exists():
                        atomic_json(folder / "status.json", {"status": "base_complete_probes_failed"})
                        return
                    time.sleep(2)
                records = read(output / "anchored_probes.json")
                assert [r["query_id"] for r in records] == [q["query_id"] for q in dataset.queries]
                texts = [text for row in records for text in row["violation_probes"]]
            if encoder is None:
                begin = time.perf_counter()
                encoder = FrozenEncoder(name, MODEL_ROOT)
                info["load_verify_seconds"] = time.perf_counter() - begin
            atomic_json(folder / "status.json", {"status": "encoding", "group": key, "texts": len(texts)})
            vectors, identity, seconds = encoder.encode_groups({key: texts})
            with (folder / (key + ".partial")).open("wb") as stream:
                np.save(stream, vectors[key])
            (folder / (key + ".partial")).replace(target)
            previous_audit = info.get("model", {}).get("token_audit", {})
            previous_audit.update(identity["token_audit"])
            info["model"] = {**identity, "token_audit": previous_audit}
            info["timing_seconds"].update(seconds)
            info.setdefault("vector_sha256", {})[key] = sha256_file(target)
            atomic_json(info_path, info)
        info["worker_wall_seconds"] = time.perf_counter() - started
        atomic_json(info_path, info)
        atomic_json(folder / "status.json", {"status": "complete"})
    except BaseException as exc:
        atomic_json(folder / "status.json", {"status": "failed", "error": str(exc)})
        raise


def collect_probes(output, client, queries):
    started = time.perf_counter()
    records, failures = {}, {}
    folder = output / "probe_records"
    folder.mkdir(exist_ok=True)

    def one(query):
        path = folder / (query["query_id"] + ".json")
        if path.exists():
            record = read(path)
            assert record["query_content"] == query["content"]
            validate_anchored_probe_structure({"violation_probes": record["violation_probes"]})
            return record
        # Exactly the main-package request and validator. Only APIClient transport
        # retries apply. Invalid output is preserved, never regenerated or repaired.
        values, _ = extract_anchored_probes(client, [query], max_workers=1)
        atomic_json(path, values[0])
        return values[0]

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(one, q): q["query_id"] for q in queries}
        for future in as_completed(futures):
            qid = futures[future]
            try:
                records[qid] = future.result()
            except Exception as exc:
                failures[qid] = {"type": type(exc).__name__, "error": str(exc)}
    if failures:
        atomic_json(output / "probe_failures.json", failures)
        raise RuntimeError("Probe generation failed; preserved raw outputs: " + ", ".join(failures))
    ordered = [records[q["query_id"]] for q in queries]
    atomic_json(output / "anchored_probes.json", ordered)
    return ordered, time.perf_counter() - started


def load_vectors(output, name):
    folder = output / "encoders" / name
    info = read(folder / "identity.json")
    values = {}
    for key, digest in info["vector_sha256"].items():
        path = folder / (key + ".npy")
        assert sha256_file(path) == digest
        values[key] = np.load(path, mmap_mode="r")
    return values, info


def score_all(output, dataset):
    vectors, _ = load_vectors(output, "bge")
    vectors["anchored_violation_probes"] = vectors["anchored_violation_probes"].reshape(len(dataset.queries), 4, 768)
    rows, _ = score_original_text(dataset, BASELINES[:3], vectors, [], .65)
    pools = sparsecl_pools(dataset, vectors)
    for name in ("arguana", "msmarco", "hotpotqa"):
        sparse, _ = load_vectors(output, name)
        arm = "sparsecl_" + name
        rows[arm] = {}
        alpha = MODEL_LOCK[name]["alpha"]
        for i, query in enumerate(dataset.queries):
            size, indices = pools[i]
            cos, hoyer = score_candidates(vectors["queries"][i], vectors["memories"][indices],
                                         sparse["queries"][i], sparse["memories"][indices])
            order, scores = fused_order(cos, hoyer, alpha)
            rows[arm][query["query_id"]] = {"query_id": query["query_id"], "scenario_id": query["scenario_id"],
                "method": arm, "library_size": size, "alpha": alpha, "coarse_k": len(indices),
                "ranked_candidates": [{"memory_id": dataset.memories[indices[j]].memory_id,
                    "score": float(scores[j]), "cosine": float(cos[j]), "hoyer": float(hoyer[j])} for j in order]}
    rows.update({method: {} for method in OUR})
    graph_folder = output / "graphs"
    graph_folder.mkdir(exist_ok=True)
    for sid, (memory_indices, query_indices) in dataset.groups().items():
        docs = [dataset.memories[i] for i in memory_indices]
        bank = vectors["memories"][memory_indices]
        path = graph_folder / (sid + ".npy")
        if path.exists():
            transition = np.load(path)
        else:
            transition = build_knn_transition(bank, docs, k=5)
            np.save(path, transition)
        for i in query_indices:
            scores = compute_bipolar_diffusion_scores(vectors["queries"][i], vectors["anchored_violation_probes"][i],
                        bank, transition, docs, top_seeds=5, alpha=.2, beta=.3)
            for method in OUR:
                rows[method][dataset.queries[i]["query_id"]] = ranked_row(dataset.queries[i], docs, method, scores[method])
    for method, values in rows.items():
        write_jsonl(output / "arms" / (method + ".jsonl"), [values[q["query_id"]] for q in dataset.queries])
    return rows, vectors


def evaluate(dataset, rows):
    import pytrec_eval
    rankings = {method: {qid: [r["memory_id"] for r in row["ranked_candidates"]] for qid, row in values.items()}
                for method, values in rows.items()}
    groups = grouped_ranking_metrics(rankings, dataset.queries, dataset.gold)
    qrels = {g["query_id"]: {mid: 1 for mid in target_ids(g)} for g in dataset.gold}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
    per_query, max_difference = {}, 0.0
    for method, values in rankings.items():
        for memory_indices, query_indices in dataset.groups().values():
            expected = {dataset.memories[i].memory_id for i in memory_indices}
            for i in query_indices:
                assert set(values[dataset.queries[i]["query_id"]]) == expected
        # Strict rank scores preserve the established tie order in every method.
        result = evaluator.evaluate({qid: {mid: float(len(ids) - i) for i, mid in enumerate(ids)} for qid, ids in values.items()})
        per_query[method] = {qid: row["ndcg_cut_10"] for qid, row in result.items()}
        for group, summary in groups.items():
            ids = [q["query_id"] for q in dataset.queries if group == "all" or q["source_ref"]["type"] == group]
            measured = float(np.mean([per_query[method][qid] for qid in ids]))
            difference = abs(measured - summary["methods"][method]["ndcg_at_10"])
            max_difference = max(max_difference, difference)
            assert difference < 1e-12
    return groups, per_query, {"pytrec_eval_verified": True, "max_ndcg_difference": max_difference,
                             "all_rankings_cover_exact_scene_bank": True, "method_group_rows": len(groups) * len(rows)}


def paired_statistics(queries, per_query):
    from scipy.stats import wilcoxon
    results = {}
    for group in ("T1", "T2", "all"):
        selected = [q for q in queries if group == "all" or q["source_ref"]["type"] == group]
        qids = [q["query_id"] for q in selected]
        indices = {kind: np.array([i for i, q in enumerate(selected) if q["source_ref"]["type"] == kind]) for kind in ("T1", "T2")}
        results[group] = {}
        for baseline in ("dense", "sparsecl_msmarco"):
            difference = np.array([per_query["anchored_bipolar_diffusion"][qid] - per_query[baseline][qid] for qid in qids])
            rounded = np.round(difference, 12)
            test = wilcoxon(rounded, zero_method="pratt", alternative="two-sided", method="approx", correction=False) if np.any(rounded) else None
            rng = np.random.default_rng(20261122)
            if group == "all":
                draws = np.concatenate([rng.choice(idx, size=(20000, len(idx)), replace=True) for idx in indices.values()], axis=1)
            else:
                draws = rng.integers(0, len(qids), size=(20000, len(qids)))
            lower, upper = np.quantile(difference[draws].mean(axis=1), [.025, .975])
            results[group][baseline] = {"scenarios": len(qids), "mean_ndcg_difference": float(difference.mean()),
                "wilcoxon_statistic": float(test.statistic) if test else 0., "wilcoxon_p_two_sided": float(test.pvalue) if test else 1.,
                "bootstrap_ci95": [float(lower), float(upper)], "bootstrap_replicates": 20000,
                "wins": int((rounded > 0).sum()), "ties": int((rounded == 0).sum()), "losses": int((rounded < 0).sum()),
                "per_scenario_difference": dict(zip(qids, map(float, difference))),
                "p_adjustment": "none; overall comparisons primary, T1/T2 exploratory"}
    return results


def table(group, methods):
    lines = ["| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 原生 M_old@10 |", "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        row = group["methods"][method]
        counts = [f"{row['at_k'][str(k)]['hits']}/{group['gold_pairs']}（{100*row['at_k'][str(k)]['micro_recall']:.1f}%）" for k in (3, 5, 10)]
        native = row["at_k"]["10"]["recall_by_source"]["stale_native"]
        lines.append(f"| {NAMES[method]} | " + " | ".join(counts) + f" | {row['ndcg_at_10']:.4f} | {native['hits']}/{native['gold_pairs']} |")
    return lines


def write_report(output, comparison, *, introduction=None, appendix=(), report_path=None, show_ablations=False):
    groups, sig, timing, usage = [comparison[k] for k in ("groups", "significance", "timing", "usage")]
    selection = read(output / "selection.json")
    lines = list(introduction) if introduction is not None else ["# STALE 未见 60 场景对照实验", "", "2026-11-20，在服务器执行；随机种子 20261122 是抽样标识。", "",
        f"从 400 场景池排除 {selection['counts']['excluded_total']} 个历史使用、提及或原生正文重复场景，从剩余 {selection['counts']['eligible_total']} 个中固定抽取 30 T1＋30 T2。样本 scenario_id 和规范化原生 M_old 正文与排除集均无交集，样本内部原生正文也互不重复。",
        f"本轮 {comparison['counts']['memories']} 条旧记忆，{groups['all']['gold_pairs']} 个已标注目标，含 60 个官方原生目标。其余目标来自既有扩增与模型复核；不是全量人工金标，未改写正文或标签。每个方法只在对应场景的旧库内排序。", ""]
    lines += [
        "主包原样调用：query-only 四维探针，提示词 SHA256 `" + PROMPT_HASH + "`；k=5、Top-5 种子、α=0.2、β=0.3。LLM 为当前配置的 deepseek-v4.1-flash，temperature=0、JSON、reasoning_effort=none、16 并发；仅传输错误按 APIClient 退避重试，所有响应原样保存，无筛选、改写或重抽。",
        "BGE 为冻结 BAAI/bge-base-en-v1.5（revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a），CLS/L2、FP32、max_length=512、batch=4、每进程 CPU threads=4、seed=0。三个 SparseCL 检查点及 α=0.18/2.19/4.82 沿用主包锁定配置；BM25 k1=1.2、b=0.75；Hybrid 等权 RRF=60。四个编码器任务与探针请求重叠运行，BGE 三个新臂共用同一批向量和图。", ""]
    for name in ("T1", "T2", "all"):
        group = groups[name]
        lines += [f"## {'总体' if name == 'all' else name}：{group['queries']} 场景、{group['gold_pairs']} 目标", ""] + table(group, BASELINES + (OUR if show_ablations else [OUR[-1]])) + [""]
    lines += ["## 逐场景配对显著性", "", "差值为完整方法减基线的平均 NDCG@10。Wilcoxon 双侧、Pratt 处理零差、正态近似，差值仅在检验秩时舍入至 12 位。Bootstrap 按场景成对重采样 20,000 次，种子 20261122；总体保持 T1/T2 各 30 个，取百分位 95% CI。配置在推理前冻结。p 值未做多重比较校正；总体为主要比较，T1/T2 为探索性分组。", "",
              "| 分组 | 对照 | 平均 ΔNDCG@10 | Wilcoxon p | Bootstrap 95% CI | 胜/平/负 |", "|---|---|---:|---:|---|---|" ]
    for group in ("T1", "T2", "all"):
        for base, row in sig[group].items():
            low, high = row["bootstrap_ci95"]
            lines.append(f"| {group} | {NAMES[base]} | {row['mean_ndcg_difference']:+.4f} | {row['wilcoxon_p_two_sided']:.6g} | [{low:+.4f}, {high:+.4f}] | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines += ["", "两个消融臂的完整排名和指标均保存在 comparison.json；与主方法使用相同探针和图，无额外 LLM 调用。", "",
              f"总实际运行 {timing['total_wall_seconds']:.2f} 秒，探针生成 {timing['probe_generation_wall_seconds']:.2f} 秒，评分与评测 {timing['scoring_evaluation_seconds']:.2f} 秒；并行阶段不能相加当作总耗时。",
              "Token / API：`" + json.dumps(usage, ensure_ascii=False) + "`。",
              "重试统计：`" + json.dumps(comparison["api_retry_stats"], ensure_ascii=False) + "`。",
              "探针基础审计（机械标记，非人工语义判定）：`" + json.dumps(comparison["probe_audit_summary"], ensure_ascii=False) + "`。", "",
              "全部 27 个方法×分组的 NDCG@10 由 pytrec_eval 从完整排名独立复算，误差 <1e-12；核验所有候选均属于对应场景。排名分数严格递减，仅用于保留原方法的平局顺序。主包、数据池和历史实验文件哈希均保持不变。", "",
              "原始目录：`" + str(output) + "`。`selection.json` / `exclusions.json` 保存抽样与排除证据；`protocol.json` 冻结参数和统计方案；`comparison.json`、`per_query_ndcg.json`、`arms/` 保存指标及完整排名；`probe_cache.sqlite3`、`probe_records/`、`api_events/` 保存原始响应及调用记录；`encoders/`、`graphs/` 支持断点恢复。", ""]
    text = "\n".join([*lines, *appendix])
    (output / "report.zh.md").write_text(text, encoding="utf-8")
    report_path = report_path or ROOT / "reports/stale_unseen_60scene_bipolar_seed20261122_experiment.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main(argv=None, *, report_writer=write_report):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=("bge", "arguana", "msmarco", "hotpotqa"))
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.worker:
        encode_worker(output, args.worker)
        return
    import fcntl
    lock_file = (output / "execution.lock").open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = verify_protocol(output)
    if (output / "completion.json").exists():
        raise ValueError("Experiment is already complete")
    if not args.resume and (output / "probe_cache.sqlite3").exists():
        raise ValueError("New experiment requires a new empty cache; use --resume for interruption")
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.resume and (output / "probe_failures.json").exists():
        archive = output / "resume_history" / attempt
        archive.mkdir(parents=True)
        (output / "probe_failures.json").replace(archive / "probe_failures.json")
    events_path = output / "api_events" / (attempt + ".json")
    started = time.perf_counter()
    dataset = load_dataset(output / "inputs")
    assert len(dataset.queries) == 60
    settings = replace(Settings.from_env(), cache_path=output / "probe_cache.sqlite3", llm_model="deepseek-v4.1-flash",
                       llm_temperature=0., llm_json_mode=True, llm_reasoning_effort="none")
    assert settings.llm_identity() == protocol["llm"]
    client = APIClient(settings)
    children = []
    status = {"status": "running", "attempt": attempt, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "methods": BASELINES + OUR, "probe_workers": 16, "probe_cache_initial_responses": 0 if not args.resume else None}
    def progress(stage, **extra):
        status.update(status=stage, elapsed_seconds=time.perf_counter() - started, **extra)
        atomic_json(output / "status.json", status)
        atomic_json(events_path, list(client.events))
        print(json.dumps({"stage": stage, **extra}), flush=True)
    try:
        progress("probes_and_encoders")
        shutil.copy2(Path(__file__), output / ("runner_" + attempt + ".py"))
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(collect_probes, output, client, dataset.queries)
            for name in ("bge", "arguana", "msmarco", "hotpotqa"):
                folder = output / "encoders" / name
                folder.mkdir(parents=True, exist_ok=True)
                if (folder / "status.json").exists() and read(folder / "status.json")["status"] == "complete":
                    continue
                log = (folder / ("execution_" + attempt + ".log")).open("w")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--worker", name],
                                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                log.close()
                children.append((name, process))
            while not future.done() or any(process.poll() is None for _, process in children):
                progress("probes_and_encoders", completed_probes=len(list((output / "probe_records").glob("*.json"))),
                         encoders={name: read(output / "encoders" / name / "status.json") if (output / "encoders" / name / "status.json").exists() else {"status": "starting"}
                                   for name in ("bge", "arguana", "msmarco", "hotpotqa")})
                time.sleep(15)
            records, probe_seconds = future.result()
        assert all(process.returncode == 0 for _, process in children), [(name, process.returncode) for name, process in children]
        atomic_json(events_path, list(client.events))
        progress("scoring_and_evaluation")
        begin = time.perf_counter()
        rows, vectors = score_all(output, dataset)
        groups, per_query, verification = evaluate(dataset, rows)
        statistics = paired_statistics(dataset.queries, per_query)
        audit = audit_anchored_probes(dataset.queries, records, vectors["queries"], vectors["anchored_violation_probes"])
        audit["token_audit"] = read(output / "encoders/bge/identity.json")["model"]["token_audit"]["anchored_violation_probes"]
        atomic_json(output / "anchored_probe_audit.json", audit)
        events = [event for p in sorted((output / "api_events").glob("*.json")) for event in read(p)]
        prior_attempts = [read(p) for p in (output / "attempts").glob("*.json")] if (output / "attempts").exists() else []
        timing = {"current_wall_seconds": time.perf_counter() - started,
                  "total_wall_seconds": time.perf_counter() - started + sum(r["wall_seconds"] for r in prior_attempts),
                  "probe_generation_wall_seconds": probe_seconds + sum(r.get("probe_generation_wall_seconds", 0) for r in prior_attempts),
                  "scoring_evaluation_seconds": time.perf_counter() - begin,
                  "encoders": {name: read(output / "encoders" / name / "identity.json") for name in ("bge", "arguana", "msmarco", "hotpotqa")}}
        comparison = {"groups": groups, "per_query_ndcg": per_query, "significance": statistics,
                      "counts": {"queries": len(dataset.queries), "memories": len(dataset.memories)}, "timing": timing,
                      "usage": usage_summary(events), "probe_audit_summary": {k: v for k, v in audit.items() if k != "details"},
                      "api_retry_stats": {"status_counts": dict(Counter(e["status"] for e in events)),
                          "http_status_counts": dict(Counter(str(e.get("http_status")) for e in events if e["status"] == "http_error")),
                          "successful_requests_after_retry": sum(e["status"] == "ok" and e.get("attempt", 1) > 1 for e in events),
                          "max_recorded_attempt": max((e.get("attempt", 1) for e in events), default=0)},
                      "verification": verification, "protocol": protocol}
        for name, value in (("comparison.json", comparison), ("per_query_ndcg.json", per_query), ("verification.json", verification)):
            atomic_json(output / name, value)
        verify_protocol(output)
        for relative, digest in read(output / "source_manifest.json").items():
            assert sha256_file(ROOT / relative) == digest, relative
        report_path = report_writer(output, comparison)
        atomic_json(output / "attempts" / (attempt + ".json"), {"status": "complete", "wall_seconds": time.perf_counter() - started,
                                                              "probe_generation_wall_seconds": probe_seconds})
        hashes = {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()
                  and p.name not in ("status.json", "probe_cache.sqlite3") and not p.name.endswith((".log", ".tmp"))}
        atomic_json(output / "completion.json", {"status": "complete", "artifacts_sha256": hashes,
                    "report_sha256": sha256_file(report_path)})
        progress("complete", completed_probes=60)
    except BaseException as exc:
        atomic_json(output / "attempts" / (attempt + ".json"), {"status": "failed", "wall_seconds": time.perf_counter() - started})
        progress("failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
