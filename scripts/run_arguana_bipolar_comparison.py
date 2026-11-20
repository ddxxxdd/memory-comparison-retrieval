"""Run the locked full method on official ArguAna; reuse verified SparseCL results."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sqlite3
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from memory_retrieval.api import APIClient, APIError, usage_summary
from memory_retrieval.baselines import rank
from memory_retrieval.bipolar_graph import (
    ANCHORED_FOUR_DIM_PROMPT, ANCHORED_PROMPT_SHA256, audit_anchored_probes,
    build_knn_transition, compute_bipolar_diffusion_scores, extract_anchored_probes,
    validate_anchored_probe_structure,
)
from memory_retrieval.config import Settings
from memory_retrieval.data import Memory, sha256_file, write_json
from memory_retrieval import encoders

LOCKED_PROMPT_SHA256 = "a38b198018c62846780914f1ef90fe2461ee1554c421d42996f54525f93217bf"
METHOD = "anchored_bipolar_diffusion"
GRAPH = {"k": 5, "top_seeds": 5, "alpha": 0.2, "beta": 0.3}


def load_official_arguana(model_root):
    """Preserve train/dev/test corpus merge order and all test qrels, including missing targets."""
    corpus, inputs = {}, {}
    for split in ("train", "dev", "test"):
        path = model_root / "data" / f"arguana_{split}_retrieval_final.pkl"
        inputs[split] = {"path": str(path), "sha256": sha256_file(path)}
        with path.open("rb") as stream:
            split_corpus, queries, qrels = pickle.load(stream), pickle.load(stream), pickle.load(stream)
        corpus.update(split_corpus)
    if set(queries) != set(qrels):
        raise ValueError("Official test query/qrels IDs differ")
    docs = [Memory(mid, row.get("text", "").strip(), "SparseCL official ArguAna") for mid, row in corpus.items()]
    queries = [{"query_id": qid, "content": text} for qid, text in queries.items()]
    missing = [{"query_id": qid, "memory_id": mid} for qid, rels in qrels.items()
               for mid, relevance in rels.items() if relevance > 0 and mid not in corpus]
    return docs, queries, qrels, inputs, missing


def historical_sparsecl(model_root):
    """Read original full-test outputs; never rerun or alter SparseCL."""
    history = model_root / "test_results/repro_arguana"
    log = history / "test/arguana_parallel_output"
    alpha_file = history / "dev/arguana_alpha.json"
    status_file = history / "formal_run_status.json"
    alpha = json.loads(alpha_file.read_text(encoding="utf-8-sig"))
    status = json.loads(status_file.read_text(encoding="utf-8-sig"))
    if alpha["split"] != "dev" or alpha["smoke_test"] is not False or alpha["alpha"] != 0.18:
        raise ValueError("Expected the verified full-dev alpha=0.18 record")
    if status["state"] != "completed" or status["exit_code"] != 0:
        raise ValueError("Historical SparseCL run did not complete")
    settings = alpha["settings"]
    if (settings["max_seq_length"] != 512 or settings["pooler_type"] != "avg"
            or settings["metric"] != "both_sum"):
        raise ValueError("Historical SparseCL protocol mismatch")
    asset_lock = model_root / "repro/assets.lock.json"
    if sha256_file(asset_lock) != settings["assets_lock_sha256"]:
        raise ValueError("Historical asset lock changed")
    lines = log.read_text().splitlines()
    metric_line = lines[lines.index("sparsity contradiction retrieval results") + 1]
    ndcg, recall = [ast.literal_eval(value) for value in re.findall(r"\{[^}]+\}", metric_line)]
    if ndcg["NDCG@10"] != 0.70406 or recall["Recall@10"] != 0.91038:
        raise ValueError("Historical reference differs from the audited result")
    return {"method": "BGE cosine + SparseCL ArguAna Hoyer", "alpha": alpha["alpha"],
            "metrics": {**ndcg, **recall}, "source": "historical full test; not rerun",
            "settings": settings, "files_sha256": {str(p): sha256_file(p)
            for p in (log, alpha_file, status_file, asset_lock)}}


def reconstruct_frozen_bank(model_root, docs):
    """Recover exact stored BGE vectors and verify their association with corpus IDs."""
    import faiss

    indices = sorted((model_root / "test_results/repro_arguana/indices").glob("arguana-bge-*.faiss"))
    if len(indices) != 1:
        raise ValueError("Expected one verified historical BGE index")
    path = indices[0]
    index = faiss.read_index(str(path))
    if index.ntotal != len(docs) or index.d != 768:
        raise ValueError("Historical BGE index dimensions/corpus size mismatch")
    bank = index.reconstruct_n(0, index.ntotal)
    digest = hashlib.sha256(bank.tobytes() + json.dumps([doc.memory_id for doc in docs]).encode()).hexdigest()[:16]
    if not path.name.endswith(f"-{digest}.faiss"):
        raise ValueError("Historical vector/ordered-ID checksum mismatch")
    if not np.isfinite(bank).all() or not np.allclose(np.linalg.norm(bank, axis=1), 1, atol=1e-6):
        raise ValueError("Historical BGE vectors are not finite unit vectors")
    return bank, {"path": str(path), "sha256": sha256_file(path),
                  "ordered_vector_digest": digest, "rows": len(docs), "dimension": 768}


def top10_for_official_evaluator(query_id, scores, docs):
    # The historical script truncates to 10 first; BEIR then drops the same-ID item
    # WITHOUT filling the list from rank 11. Preserve that exact evaluation order.
    order = rank(scores, docs)[:10]
    return {docs[i].memory_id: float(scores[i]) for i in order}


def official_evaluation(qrels, raw_top10):
    from beir.retrieval.evaluation import EvaluateRetrieval
    import pytrec_eval

    results = {qid: dict(values) for qid, values in raw_top10.items()}
    removed = sum(qid in values for qid, values in results.items())
    ndcg, _, recall, _ = EvaluateRetrieval.evaluate(qrels, results, [1, 10], ignore_identical_ids=True)
    per_query = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_1", "ndcg_cut_10", "recall_1", "recall_10"}).evaluate(results)
    if set(per_query) != set(qrels):
        raise ValueError("Evaluator omitted queries")
    mapping = {"NDCG@1": "ndcg_cut_1", "NDCG@10": "ndcg_cut_10",
               "Recall@1": "recall_1", "Recall@10": "recall_10"}
    exact = {name: sum(row[key] for row in per_query.values()) / len(per_query)
             for name, key in mapping.items()}
    metrics = {**ndcg, **recall}
    for key, value in exact.items():
        if abs(metrics[key] - round(value, 5)) > 1e-12:
            raise AssertionError("BEIR / pytrec_eval discrepancy: " + key)
    return {"metrics": metrics, "unrounded_metrics": exact,
            "queries_evaluated": len(per_query), "self_ids_removed_after_top10": removed,
            "pytrec_eval_verified": True}, per_query


def encoding_preflight(encoder, bank, docs, output):
    """Check threads on a fixed, unlabeled text batch. No retrieval or method tuning."""
    import torch

    texts = [doc.content for doc in docs[:12]]
    rows, values = [], {}
    # Warm up the tokenizer/model before timed checks.
    torch.set_num_threads(4)
    encoders.encode(texts[:4], encoder.tokenizer, encoder.model, 4, 512, False, "warmup")
    for threads in (4, 16):
        torch.set_num_threads(threads)
        vectors, seconds = encoders.encode(texts, encoder.tokenizer, encoder.model, 4, 512, False, "cpu_check")
        values[threads] = vectors
        rows.append({"cpu_threads": threads, "batch_size": 4, "texts": len(texts), "seconds": seconds})
    thread_error = float(np.max(np.abs(values[4] - values[16])))
    bank_error = float(np.max(np.abs(values[4] - bank[:len(texts)])))
    if thread_error > 1e-6 or bank_error > 2e-6:
        raise ValueError("Encoding equivalence check failed")
    chosen = 16 if rows[1]["seconds"] < rows[0]["seconds"] else 4
    torch.set_num_threads(chosen)
    encoder.identity["cpu_threads"] = chosen
    audit = {"checks": rows, "chosen_cpu_threads": chosen, "max_thread_difference": thread_error,
             "max_historical_bank_difference": bank_error, "frozen_batch_size": 4,
             "reason": "Hardware throughput only; no gold labels or retrieval metrics used"}
    write_json(output / "encoding_preflight.json", audit)
    return audit


def recover_probe_cache(previous_run, output, protocol):
    """Reuse exact successful requests after an interrupted run; preserve every original artifact."""
    previous_run = previous_run.resolve()
    previous = json.loads((previous_run / "status.json").read_text())
    if previous["status"] != "failed":
        raise ValueError("Only a finished failed run may supply a recovery cache")
    for key in ("prompt_sha256", "graph", "probe_workers", "llm", "dataset_inputs", "source_sha256"):
        if previous[key] != protocol[key]:
            raise ValueError("Recovery protocol differs: " + key)
    source = previous_run / "probe_cache.sqlite3"
    rejected, kept = [], 0
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as old:
        rows = old.execute("SELECT key, request, response FROM responses").fetchall()
    with sqlite3.connect(output / "probe_cache.sqlite3") as new:
        new.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, request TEXT, response TEXT)")
        for key, request, raw in rows:
            response = json.loads(raw)
            try:
                choice = response["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("LLM output was truncated")
                validate_anchored_probe_structure(json.loads(choice["message"]["content"]))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                rejected.append({"cache_key": key, "reason": type(exc).__name__,
                                 "finish_reason": response.get("choices", [{}])[0].get("finish_reason")})
            else:
                new.execute("INSERT INTO responses VALUES (?, ?, ?)", (key, request, raw))
                kept += 1
    events_path = previous_run / "api_events_all_attempts.json"
    if not events_path.exists():
        events_path = previous_run / "api_events.json"
    previous_events = json.loads(events_path.read_text())
    previous_wall = previous.get("total_attempt_wall_seconds", previous["wall_seconds"])
    if (previous_run / "cache_recovery.json").exists():
        previous_wall = previous["wall_seconds"] + json.loads(
            (previous_run / "cache_recovery.json").read_text())["previous_attempt_wall_seconds"]
    recovery = {"source_run": str(previous_run), "source_cache_sha256": sha256_file(source),
                "cached_successes_reused": kept, "invalid_responses_to_retry_unchanged": rejected,
                "previous_attempt_wall_seconds": previous_wall,
                "policy": "Retry only main-validator failures with identical requests; no semantic selection"}
    write_json(output / "cache_recovery.json", recovery)
    return previous_events, previous_wall


def resume_artifacts(output, previous, protocol):
    """Resume in place, backing up metadata/cache and reusing frozen vectors verbatim."""
    for key in ("prompt_sha256", "graph", "llm", "dataset_inputs", "source_sha256", "historical_bank"):
        if previous[key] != protocol[key]:
            raise ValueError("Resume protocol differs: " + key)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = output / "resume_history" / stamp
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in ("status.json", "protocol.json", "api_events.json", "api_events_all_attempts.json",
                 "bge_identity.json", "resume_info.json", Path(__file__).name):
        if (output / name).exists():
            shutil.copy2(output / name, snapshot / name)
    recovery = json.loads((output / "cache_recovery.json").read_text()) if (output / "cache_recovery.json").exists() else {}
    source = Path(recovery["source_run"]) if recovery else output
    if (output / "api_events_all_attempts.json").exists():
        events = json.loads((output / "api_events_all_attempts.json").read_text())
        base_wall = json.loads((output / "resume_info.json").read_text())["previous_attempt_wall_seconds"] if (output / "resume_info.json").exists() else recovery.get("previous_attempt_wall_seconds", 0)
    else:
        prior_events = source / "api_events_all_attempts.json"
        if not prior_events.exists():
            prior_events = source / "api_events.json"
        events = json.loads(prior_events.read_text()) if prior_events.exists() else []
        base_wall = recovery.get("previous_attempt_wall_seconds", 0)
        with sqlite3.connect((source / "probe_cache.sqlite3").as_uri() + "?mode=ro", uri=True) as old:
            old_responses = dict(old.execute("SELECT key, response FROM responses")) if source != output else {}
        with sqlite3.connect(output / "probe_cache.sqlite3") as db:
            for key, raw in db.execute("SELECT key, response FROM responses"):
                if old_responses.get(key) != raw:
                    response = json.loads(raw)
                    events.append({"stage": "anchored_four_dimension", "status": "ok", "cached": False,
                                   "seconds": None, "usage": response.get("usage"),
                                   "recovered_from_cache": True, "latency_unavailable": True})
    rejected = []
    with sqlite3.connect(output / "probe_cache.sqlite3") as db:
        with sqlite3.connect(snapshot / "probe_cache.sqlite3") as backup:
            db.backup(backup)
        for key, raw in db.execute("SELECT key, response FROM responses").fetchall():
            try:
                choice = json.loads(raw)["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("Truncated response")
                validate_anchored_probe_structure(json.loads(choice["message"]["content"]))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                rejected.append({"cache_key": key, "reason": type(exc).__name__})
                db.execute("DELETE FROM responses WHERE key=?", (key,))
        valid = db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    artifacts = {}
    for name in ("bank.npy", "queries.npy", "transition.npy"):
        path = output / name
        origin = path if path.exists() else source / name
        digest = sha256_file(origin)
        if origin != path:
            shutil.copy2(origin, path)
        assert sha256_file(path) == digest
        artifacts[name] = {"source": str(origin), "sha256": digest}
    info = {"snapshot": str(snapshot), "valid_cached_responses": valid, "invalid_responses_to_retry": rejected,
            "previous_probe_workers": previous["probe_workers"], "probe_workers": protocol["probe_workers"],
            "previous_attempt_wall_seconds": base_wall + previous["wall_seconds"], "artifacts_reused": artifacts,
            "accounting_note": "Interrupted in-flight requests without responses have unknown usage; recovered cache responses lack latency"}
    write_json(output / "resume_info.json", info)
    write_json(output / "api_events_all_attempts.json", events)
    return events, info["previous_attempt_wall_seconds"]


def extract_remaining_probes(client, queries, output, max_workers):
    """Retry malformed responses verbatim, preserving paid successes and every rejected response."""
    started, event_start = time.perf_counter(), len(client.events)
    retry_counts = {}
    lock = threading.Lock()
    records, failures = {}, {}

    def generate(query):
        while True:
            try:
                values, _ = extract_anchored_probes(client, [query], max_workers=1)
                record = values[0]
            except APIError:
                invalid = []
                with lock, sqlite3.connect(client.settings.cache_path) as db:
                    for key, request, raw in db.execute("SELECT key, request, response FROM responses").fetchall():
                        request_body = json.loads(request).get("body", {})
                        user_message = request_body.get("messages", [{}, {}])[1].get("content", "{}")
                        if json.loads(user_message).get("query") != query["content"]:
                            continue
                        try:
                            choice = json.loads(raw)["choices"][0]
                            if choice.get("finish_reason") == "length":
                                raise ValueError("Truncated response")
                            validate_anchored_probe_structure(json.loads(choice["message"]["content"]))
                        except (KeyError, IndexError, TypeError, ValueError) as exc:
                            invalid.append((key, request, raw, type(exc).__name__))
                    if not invalid or any(retry_counts.get(key, 0) >= 2 for key, *_ in invalid):
                        raise
                    with (output / "invalid_response_attempts.jsonl").open("a", encoding="utf-8") as archive:
                        for key, request, raw, reason in invalid:
                            archive.write(json.dumps({"key": key, "request": json.loads(request),
                                                      "response": json.loads(raw), "reason": reason,
                                                      "recorded_at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                            archive.flush()
                            db.execute("DELETE FROM responses WHERE key=?", (key,))
                            retry_counts[key] = retry_counts.get(key, 0) + 1
                print(json.dumps({"stage": "retrying_invalid_response_unchanged", "query_id": query["query_id"]}), flush=True)
            else:
                with lock, (output / "probe_progress.jsonl").open("a", encoding="utf-8") as journal:
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                return record

    # A failed query never cancels queued unrelated queries. The main extractor
    # still constructs, validates and caches every unmodified request.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate, query): query["query_id"] for query in queries}
        for future in as_completed(futures):
            query_id = futures[future]
            try:
                records[query_id] = future.result()
            except APIError as exc:
                failures[query_id] = str(exc)
    if failures:
        write_json(output / "failed_queries.json", failures)
        raise APIError(f"{len(failures)} queries failed; successful responses and vectors remain resumable")
    ordered = [records[query["query_id"]] for query in queries]
    audit = audit_anchored_probes(queries, ordered)
    audit["generation"] = {"wall_seconds": time.perf_counter() - started, "workers": max_workers,
                           "usage": usage_summary(client.events[event_start:]),
                           "invalid_response_retries": sum(retry_counts.values())}
    return ordered, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; never overwrite a run")
    parser.add_argument("--cache-from", type=Path, help="Reuse successful requests from a finished failed run")
    parser.add_argument("--resume", action="store_true", help="Resume a stopped run in its original directory")
    parser.add_argument("--probe-workers", type=int, default=4)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.probe_workers < 1:
        parser.error("--probe-workers must be positive")
    previous = None
    if args.resume:
        if args.cache_from:
            parser.error("--resume and --cache-from cannot be combined")
        previous = json.loads((output / "status.json").read_text())
        if previous["status"] not in ("failed", "interrupted_for_resume"):
            raise ValueError("Resume requires a stopped incomplete run")
    else:
        output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    timing = {}
    protocol = {"status": "initializing", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "method": METHOD, "graph": GRAPH, "probe_workers": args.probe_workers,
                "prompt_sha256": ANCHORED_PROMPT_SHA256,
                "scope": "Full official ArguAna test; our method only; historical SparseCL reference"}
    client = None
    previous_events, previous_wall = [], 0.0

    def stage(name, **extra):
        protocol.update(status=name, wall_seconds=time.perf_counter() - started, **extra)
        write_json(output / "status.json", protocol)
        print(json.dumps({"stage": name, **extra}, ensure_ascii=False), flush=True)

    try:
        if ANCHORED_PROMPT_SHA256 != LOCKED_PROMPT_SHA256:
            raise ValueError("The locked main-method prompt changed")
        model_root = encoders.MODEL_ROOT
        lock = json.loads((model_root / "repro/assets.lock.json").read_text())
        hashes = encoders.verify_assets(model_root, [(row["directory"], row) for row in lock if row["directory"] == "data"])
        docs, queries, qrels, inputs, missing = load_official_arguana(model_root)
        if (len(docs), len(queries), len(missing)) != (8674, 1406, 5):
            raise ValueError("Unexpected official ArguAna release dimensions")
        reference = historical_sparsecl(model_root)
        if args.resume:
            bank = np.load(output / "bank.npy")
            bank_identity = previous["historical_bank"]
            digest = hashlib.sha256(bank.tobytes() + json.dumps([d.memory_id for d in docs]).encode()).hexdigest()[:16]
            if digest != bank_identity["ordered_vector_digest"]:
                raise ValueError("Saved bank/ordered IDs differ from the frozen original")
        else:
            bank, bank_identity = reconstruct_frozen_bank(model_root, docs)
        source_paths = [ROOT / "memory_retrieval" / name for name in
                        ("bipolar_graph.py", "encoders.py", "api.py", "model_lock.json")]
        source_hashes = {str(path): sha256_file(path) for path in source_paths}
        settings = replace(Settings.from_env(), cache_path=output / "probe_cache.sqlite3",
                           llm_model="deepseek-v4.1-flash", llm_temperature=0.0,
                           llm_json_mode=True, llm_reasoning_effort="none")
        settings.require(llm=True)
        protocol.update(dataset_inputs=inputs, queries=len(queries), corpus=len(docs),
                        missing_gold=missing, llm=settings.llm_identity(),
                        source_sha256=source_hashes, historical_bank=bank_identity,
                        official_data_sha256=hashes, historical_reference=reference,
                        evaluation={"library": "BEIR 2.0.0 / pytrec_eval 0.5",
                                    "k": [1, 10], "top10_before_self_id_exclusion": True,
                                    "missing_gold_queries_kept": True, "gold_used_in_scoring": False})
        if args.resume:
            previous_events, previous_wall = resume_artifacts(output, previous, protocol)
        write_json(output / "protocol.json", protocol)
        if args.cache_from:
            previous_events, previous_wall = recover_probe_cache(args.cache_from, output, protocol)
        write_json(output / "ids.json", {"memory_ids": [d.memory_id for d in docs],
                                       "query_ids": [q["query_id"] for q in queries]})
        write_json(output / "qrels.json", qrels)
        (output / "prompt.txt").write_text(ANCHORED_FOUR_DIM_PROMPT, encoding="utf-8")
        shutil.copyfile(Path(__file__), output / Path(__file__).name)
        stage("model_preflight")
        encoder = encoders.FrozenEncoder("bge", model_root)
        if args.resume:
            import torch
            preflight = json.loads((output / "encoding_preflight.json").read_text())
            encoder.identity = json.loads((output / "bge_identity.json").read_text())
            torch.set_num_threads(preflight["chosen_cpu_threads"])
            q_vectors = np.load(output / "queries.npy")
            transition = np.load(output / "transition.npy", mmap_mode="r")
            if q_vectors.shape != (len(queries), 768) or transition.shape != (len(docs), len(docs)):
                raise ValueError("Saved vectors or graph dimensions differ")
            if not all(np.isfinite(a).all() for a in (q_vectors, transition)):
                raise ValueError("Saved vectors or graph contain invalid values")
            timing.update(query_encoding=0.0, graph_build=0.0)
        else:
            preflight = encoding_preflight(encoder, bank, docs, output)
            write_json(output / "bge_identity.json", encoder.identity)
            np.save(output / "bank.npy", bank)
        client = APIClient(settings)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(extract_remaining_probes, client, queries, output, max_workers=args.probe_workers)
            if not args.resume:
                stage("probes_and_query_encoding", cpu_threads=preflight["chosen_cpu_threads"])
                encoded, identity, seconds = encoder.encode_groups({"queries": [q["content"].strip() for q in queries]})
                q_vectors = encoded["queries"]
                timing["query_encoding"] = seconds["queries"]
                np.save(output / "queries.npy", q_vectors)
                graph_started = time.perf_counter()
                stage("building_original_graph")
                transition = build_knn_transition(bank, docs, k=5)
                timing["graph_build"] = time.perf_counter() - graph_started
                np.save(output / "transition.npy", transition)
            while not future.done():
                done = sum(e["status"] == "ok" for e in client.events)
                stage("generating_probes", completed_responses=done, total_queries=len(queries))
                write_json(output / "api_events.json", client.events)
                write_json(output / "api_events_all_attempts.json", previous_events + client.events)
                time.sleep(20)
            records, audit = future.result()
        write_json(output / "anchored_probes.json", records)
        write_json(output / "api_events.json", client.events)
        write_json(output / "api_events_all_attempts.json", previous_events + client.events)
        timing["probe_generation"] = audit["generation"]["wall_seconds"]
        stage("encoding_four_probes")
        encoded, identity, seconds = encoder.encode_groups({
            "violation_probes": [text for record in records for text in record["violation_probes"]]})
        viol_vectors = np.asarray(np.split(encoded["violation_probes"], len(queries)))
        timing["probe_encoding"] = seconds["violation_probes"]
        np.save(output / "violation_probes.npy", viol_vectors)
        audit.update(audit_anchored_probes(queries, records, q_vectors, viol_vectors))
        audit["token_audit"] = identity["token_audit"]
        write_json(output / "anchored_probe_audit.json", audit)
        write_json(output / "bge_identity.json", identity)
        stage("original_main_method_scoring")
        scoring_started = time.perf_counter()
        all_scores = np.lib.format.open_memmap(output / "scores.npy", mode="w+", dtype=np.float32,
                                               shape=(len(queries), len(docs)))
        raw_top10 = {}
        for i, query in enumerate(queries):
            values = compute_bipolar_diffusion_scores(
                q_vectors[i], viol_vectors[i], bank, transition, docs,
                top_seeds=5, alpha=0.2, beta=0.3)[METHOD]
            all_scores[i] = values
            raw_top10[query["query_id"]] = top10_for_official_evaluator(query["query_id"], values, docs)
            if (i + 1) % 100 == 0:
                stage("original_main_method_scoring", scored_queries=i + 1, total_queries=len(queries))
        all_scores.flush()
        del all_scores
        timing["scoring"] = time.perf_counter() - scoring_started
        write_json(output / "raw_top10.json", raw_top10)
        evaluation, per_query = official_evaluation(qrels, raw_top10)
        for row in missing:
            assert per_query[row["query_id"]]["ndcg_cut_10"] == 0
        write_json(output / "per_query_metrics.json", per_query)
        current = evaluation["metrics"]
        previous = reference["metrics"]
        report = {"method": METHOD, "dataset": "official ArguAna", "queries": len(queries), "corpus": len(docs),
                  "missing_gold_queries": len(missing), "evaluation": evaluation,
                  "sparsecl_historical": reference,
                  "delta_vs_sparsecl": {key: current[key] - previous[key] for key in ("NDCG@1", "NDCG@10", "Recall@10")},
                  "timing_seconds": timing, "wall_seconds": time.perf_counter() - started,
                  "total_attempt_wall_seconds": previous_wall + time.perf_counter() - started,
                  "api_usage": usage_summary(previous_events + client.events),
                  "current_attempt_api_usage": usage_summary(client.events),
                  "probe_audit": {k: v for k, v in audit.items() if k != "details"},
                  "bge_bank_reused": True, "sparsecl_rerun": False, "cpu_preflight": preflight,
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("torch", "transformers", "numpy", "beir", "pytrec-eval", "faiss-cpu")}}
        write_json(output / "report.json", report)
        lines = ["# ArguAna 完整方法对比", "",
                 f"官方完整语料库 {len(docs)} 条，Test {len(queries)} 条。5 条缺失正例的查询保留并计零分。",
                 f"主方法保持锁定提示词与 k=5、Top-5、α=0.2、β=0.3；恢复阶段采用 {args.probe_workers} 并发，仅运行 anchored_bipolar_diffusion。",
                 "采用官方流程：先取原始 Top-10，再由 BEIR 排除同 ID 文档，不从第 11 名补位。", "",
                 "| 方法 | NDCG@1 | NDCG@10 | Recall@10 | 结果来源 |",
                 "|---|---:|---:|---:|---|",
                 f"| SparseCL（BGE） | {previous['NDCG@1']:.5f} | {previous['NDCG@10']:.5f} | {previous['Recall@10']:.5f} | 已核验历史完整复现 |",
                 f"| 四维探针＋双极图扩散 | {current['NDCG@1']:.5f} | {current['NDCG@10']:.5f} | {current['Recall@10']:.5f} | 本次服务器完整测试 |", "",
                 f"各次实际运行累计耗时 {report['total_attempt_wall_seconds']:.1f} 秒（本次 {report['wall_seconds']:.1f} 秒）；LLM 用量：{json.dumps(report['api_usage'], ensure_ascii=False)}。",
                 "复用历史 BGE 全库向量，查询与探针新编码。耗时不含历史语料编码，不与本地 SparseCL 历史耗时直接比较。",
                 "探针质量统计为机械标记，未按金标筛选、修复或重新生成。历史 SparseCL 只有汇总指标，无法据此做逐查询配对显著性检验。",
                 "提示词直接用于段落级 ArguAna 查询，未针对该数据集修改。未进行方法参数搜索。", ""]
        (output / "report.zh.md").write_text("\n".join(lines), encoding="utf-8")
        for path, digest in source_hashes.items():
            assert sha256_file(path) == digest, path
        for row in inputs.values():
            assert sha256_file(row["path"]) == row["sha256"]
        for path, digest in reference["files_sha256"].items():
            assert sha256_file(path) == digest
        artifacts = {p.name: sha256_file(p) for p in output.iterdir()
                     if p.is_file() and p.name not in {"status.json", "probe_cache.sqlite3"}}
        write_json(output / "completion.json", {"status": "complete", "artifacts_sha256": artifacts})
        stage("complete", metrics=current, historical_sparsecl_metrics=previous)
    except BaseException as exc:
        if client is not None:
            write_json(output / "api_events.json", client.events)
            write_json(output / "api_events_all_attempts.json", previous_events + client.events)
        stage("failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
