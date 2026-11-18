"""Existing generic retrievers on frozen samples of SparseCL's public dev sets."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_retrieval.api import APIClient, usage_summary
from memory_retrieval.baselines import BM25, rank, reciprocal_rank_fusion
from memory_retrieval.config import Settings
from memory_retrieval.data import Memory, fingerprint

DATA = ROOT / "data/sparsecl_modified_msmarco_hotpotqa"
OUT = ROOT / "runs/experiments/sparsecl-data-general-pilot-20261117"
METHODS = ("bm25", "dense", "hybrid")
KS = (1, 5, 10)
SEED = 20261117
QUERY_COUNT = 50
BATCH_SIZE = 128
WORKERS = 8


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Unexpected global reference: {module}.{name}")


def load_data(dataset):
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    hashes = {row["filename"]: row["sha256"] for row in manifest["files"]}
    corpus, source_hashes = {}, {}
    for split in ("train", "dev", "test"):
        name = f"{dataset}_{split}_retrieval_gpt4_final.pkl"
        path = DATA / dataset / name
        actual = sha(path)
        if actual != hashes[name]:
            raise ValueError(f"Official input changed: {name}")
        source_hashes[name] = actual
        with path.open("rb") as stream:
            c, q, r = (DataUnpickler(stream).load() for _ in range(3))
            if stream.read(1):
                raise ValueError(f"Unexpected trailing data: {name}")
        for doc_id, doc in c.items():
            if doc_id in corpus and corpus[doc_id] != doc:
                raise ValueError(f"Inconsistent corpus ID: {doc_id}")
            corpus[doc_id] = doc
        if split == "dev":
            queries, qrels = q, r
    selected = sorted(random.Random(SEED).sample(sorted(queries), QUERY_COUNT))
    for qid in selected:
        if not qrels[qid] or any(did not in corpus for did in qrels[qid]):
            raise ValueError(f"Missing target for {qid}")
        if any(grade != 1 for grade in qrels[qid].values()):
            raise ValueError("This runner expects binary official relevance grades")
        if qid in qrels[qid]:
            raise ValueError("Self-query is marked relevant")
    return corpus, {qid: queries[qid].strip() for qid in selected}, qrels, source_hashes


def prepare(dataset, settings):
    corpus, queries, qrels, hashes = load_data(dataset)
    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "dataset": dataset, "split": "dev", "query_count": len(queries),
        "sample_seed": SEED, "sampling": "random.sample(sorted(dev_query_ids), 50)",
        "query_ids": list(queries), "distinct_source_groups": len({q.split("-para")[0] for q in queries}),
        "input_sha256": hashes, "corpus_count": len(corpus),
        "corpus_policy": "merge complete train/dev/test corpora; preserve all distinct IDs",
        "text_policy": "text.strip only; no title, instruction, labels or ID sent to encoder",
        "exclude_query_self_id": True, "duplicate_text_policy": "encode exact text once; preserve candidate IDs",
        "embedding": settings.embedding_identity(), "batch_size": BATCH_SIZE, "workers": WORKERS,
        "methods": list(METHODS), "top_k": list(KS), "dense_search": "exact cosine over full corpus",
        "bm25": {"implementation": "memory_retrieval.baselines.BM25", "k1": 1.2, "b": 0.75},
        "hybrid": {"implementation": "memory_retrieval.baselines.reciprocal_rank_fusion", "constant": 60},
        "ties": "descending score then ascending original document ID",
        "baseline_code_sha256": sha(ROOT / "memory_retrieval/baselines.py"),
        "api_code_sha256": sha(ROOT / "memory_retrieval/api.py"),
    }
    plan_path = out / "run_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Resume plan does not match frozen run")
    else:
        write_json(plan_path, plan)
        write_json(out / "selected_queries.json", {
            qid: {"text": text, "qrels": qrels[qid]} for qid, text in queries.items()
        })
    return corpus, queries, qrels, plan, out


def encode(texts, settings, out):
    batch_dir = out / "embedding_batches"
    batch_dir.mkdir(exist_ok=True)
    identity = {"texts_sha256": fingerprint(texts), "count": len(texts),
                "embedding": settings.embedding_identity(), "batch_size": BATCH_SIZE}
    identity_path = out / "embedding_plan.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Embedding cache identity mismatch")
    else:
        write_json(identity_path, identity)
    settings = replace(settings, embedding_batch_size=BATCH_SIZE,
                       cache_path=out / "embedding_cache.sqlite3")
    APIClient(settings)  # Initialize SQLite schema before concurrent use.
    batches = [(i, texts[start:start+BATCH_SIZE])
               for i, start in enumerate(range(0, len(texts), BATCH_SIZE))]

    def job(item):
        i, batch = item
        path = batch_dir / f"{i:05d}.npy"
        if path.exists():
            vector = np.load(path, allow_pickle=False)
            events = []
        else:
            client = APIClient(settings)
            try:
                vector = client.embed(batch, "sparsecl_public_baseline_embedding")
            except Exception:
                write_json(batch_dir / f"{i:05d}.error.json", client.events)
                raise
            events = client.events
            temp = path.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.save(stream, vector, allow_pickle=False)
            temp.replace(path)
        if (vector.ndim != 2 or len(vector) != len(batch) or not np.isfinite(vector).all()
                or not np.allclose(np.linalg.norm(vector, axis=1), 1, atol=1e-5)):
            raise ValueError(f"Invalid embedding batch {i}")
        return i, len(batch), vector.shape[1], events

    completed, dimensions = 0, set()
    started, last_progress = time.monotonic(), 0.0
    with (out / "api_events.jsonl").open("a", encoding="utf-8") as log:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(job, batch) for batch in batches]
            for future in as_completed(futures):
                i, count, dim, events = future.result()
                completed += count
                dimensions.add(dim)
                if len(dimensions) != 1:
                    raise ValueError("Service embedding dimension changed during run")
                for event in events:
                    log.write(json.dumps({"batch": i, **event}, ensure_ascii=False) + "\n")
                log.flush()
                now = time.monotonic()
                if now-last_progress >= 20 or completed == len(texts):
                    progress = {"stage": "embedding", "completed_texts": completed,
                                "total_texts": len(texts), "dimensions": dim,
                                "elapsed_seconds": round(now-started, 1)}
                    write_json(out / "progress.json", progress)
                    print(f"{out.name} {json.dumps(progress)}", flush=True)
                    last_progress = now
    matrix = np.concatenate([np.load(batch_dir / f"{i:05d}.npy", allow_pickle=False) for i, _ in batches])
    if matrix.shape[0] != len(texts):
        raise ValueError("Embedding matrix row count mismatch")
    return matrix


def metrics_for_ranks(gold_ranks):
    count = len(gold_ranks)
    values = {}
    for k in KS:
        hits = sum(r <= k for r in gold_ranks)
        dcg = sum(1 / math.log2(r+1) for r in gold_ranks if r <= k)
        ideal = sum(1 / math.log2(r+1) for r in range(1, min(k, count)+1))
        values[str(k)] = {"hits": hits, "recall": hits/count, "hit": int(hits > 0),
                          "all_found": int(hits == count), "ndcg": dcg/ideal}
    return values


def run(dataset, settings):
    corpus, queries, qrels, plan, out = prepare(dataset, settings)
    if (out / "summary.json").exists():
        print(f"{dataset}: completed results preserved", flush=True)
        return
    ids = list(corpus)
    contents = [corpus[did]["text"].strip() for did in ids]
    texts = list(dict.fromkeys(contents + list(queries.values())))
    text_index = {text: i for i, text in enumerate(texts)}
    doc_text_index = np.array([text_index[text] for text in contents])
    vectors = encode(texts, settings, out)
    memories = [Memory(did, content, {}) for did, content in zip(ids, contents)]
    id_index = {did: i for i, did in enumerate(ids)}
    print(f"{dataset}: building existing BM25 index", flush=True)
    sparse_index = BM25(contents)
    details = []
    order_array = np.empty((len(queries), len(METHODS), len(ids)-1), dtype=np.int32)
    # Official queries in these files share IDs with corpus records.
    if any(qid not in id_index for qid in queries):
        raise ValueError("Unexpected query absent from corpus")
    for qi, (qid, text) in enumerate(queries.items()):
        sparse = sparse_index.score(text)
        dense = (vectors @ vectors[text_index[text]])[doc_text_index].copy()
        # Exclude the query itself before producing all rankings and RRF positions.
        self_index = id_index[qid]
        sparse[self_index] = -np.inf
        dense[self_index] = -np.inf
        fused = reciprocal_rank_fusion(sparse, dense, memories)
        fused[self_index] = -np.inf
        for mi, (method, scores) in enumerate(zip(METHODS, (sparse, dense, fused))):
            order = [i for i in rank(scores, memories) if i != self_index]
            order_array[qi, mi] = order
            inverse = np.empty(len(ids), dtype=np.int32)
            inverse[order] = np.arange(1, len(order)+1)
            gold_ranks = {did: int(inverse[id_index[did]]) for did in qrels[qid]}
            group = qid.split("-para")[0]
            para_ids = [f"{group}-para{i}-gpt4" for i in range(3)]
            para_ranks = {did: int(inverse[id_index[did]]) for did in para_ids
                          if did in id_index and did != qid}
            row = {"query_id": qid, "method": method, "query": text,
                   "gold_ranks": gold_ranks, "same_source_paraphrase_ranks": para_ranks,
                   "metrics": metrics_for_ranks(list(gold_ranks.values())),
                   "top10": [{"rank": p, "id": ids[i], "score": float(scores[i]),
                              "text": contents[i], "is_official_gold": ids[i] in qrels[qid]}
                             for p, i in enumerate(order[:10], 1)]}
            details.append(row)
        write_json(out / "progress.json", {"stage": "ranking", "completed_queries": qi+1,
                                           "total_queries": len(queries)})
        print(f"{dataset} ranking {qi+1}/{len(queries)}", flush=True)
    np.savez_compressed(out / "rankings.npz", orders=order_array,
                        corpus_ids=np.array(ids), query_ids=np.array(list(queries)), methods=np.array(METHODS))
    aggregate = {}
    for method in METHODS:
        rows = [r for r in details if r["method"] == method]
        aggregate[method] = {}
        for k in KS:
            points = [r["metrics"][str(k)] for r in rows]
            aggregate[method][str(k)] = {
                "retrieved_gold_count": sum(x["hits"] for x in points),
                "total_gold_count": sum(len(r["gold_ranks"]) for r in rows),
                "mean_recall": float(np.mean([x["recall"] for x in points])),
                "queries_any_gold": sum(x["hit"] for x in points),
                "queries_all_gold": sum(x["all_found"] for x in points),
                "mean_ndcg": float(np.mean([x["ndcg"] for x in points])),
            }
    events = [json.loads(line) for line in (out / "api_events.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
               "unique_encoded_texts": len(texts), "embedding_dimension": vectors.shape[1],
               "api_usage": usage_summary(events), "metrics": aggregate,
               "script_sha256": sha(Path(__file__)),
               "limitations": ["50 fixed random dev queries; not a full benchmark result",
                               "official qrels unchanged; unjudged candidates are not confirmed negatives",
                               "official synthetic labels are not independently re-annotated",
                               "two queries share a source group in each sample",
                               "gateway model alias and returned dimension recorded; underlying weights not audited"]}
    (out / "per_query.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in details), encoding="utf-8")
    write_json(out / "summary.json", summary)
    write_json(out / "progress.json", {"stage": "complete", "query_count": len(queries)})
    print(json.dumps({"dataset": dataset, "metrics": aggregate}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("msmarco", "hotpotqa", "all"), default="all")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.require(embedding=True)
    datasets = ("msmarco", "hotpotqa") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        if args.prepare_only:
            _, _, _, plan, _ = prepare(dataset, settings)
            print(json.dumps(plan, ensure_ascii=False), flush=True)
        else:
            run(dataset, settings)


if __name__ == "__main__":
    main()
