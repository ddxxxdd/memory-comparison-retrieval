"""Run the pinned SparseCL encoder and HNSW reranker inside its WSL environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path):
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_model_assets(baseline_root):
    lock_path = baseline_root / "repro" / "assets.lock.json"
    lock = read_json(lock_path)
    model_repos = {"SparseCL/BGE-SparseCL-arguana", "BAAI/bge-base-en-v1.5"}
    verified = {}
    for repo in lock:
        if repo["repo"] not in model_repos:
            continue
        for entry in repo["files"]:
            path = baseline_root / repo["directory"] / entry["rfilename"]
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise ValueError(f"SparseCL 模型资源缺失或大小不符：{path}")
            expected = (entry.get("lfs") or {}).get("sha256")
            actual = sha256_file(path) if expected else git_blob_sha1(path)
            if actual != (expected or entry["blobId"]):
                raise ValueError(f"SparseCL 模型资源哈希不符：{path}")
            verified[str(path.relative_to(baseline_root))] = sha256_file(path)
    if len(verified) != 12:
        raise ValueError(f"SparseCL 固定模型资源数不符：expected 12, got {len(verified)}")
    return sha256_file(lock_path), verified


def input_rows(value):
    memories = value.get("memories")
    queries = value.get("queries")
    if not isinstance(memories, list) or not memories or not isinstance(queries, list) or not queries:
        raise ValueError("SparseCL 输入必须含非空 memories 和 queries")
    for label, rows, id_key in (("memory", memories, "memory_id"), ("query", queries, "query_id")):
        ids = []
        for row in rows:
            item_id = row.get(id_key)
            content = row.get("content")
            if not isinstance(item_id, str) or not item_id or not isinstance(content, str) or not content.strip():
                raise ValueError(f"SparseCL {label} 缺少有效 ID 或正文")
            ids.append(item_id)
        if len(ids) != len(set(ids)):
            raise ValueError(f"SparseCL {label} ID 必须唯一")
    return memories, queries


def sparsecl_signature(baseline_root, worker_path, memories, batch_size, cpu_threads,
                       torch, transformers, faiss, numpy):
    from memory_retrieval.sparsecl import (FIXED_ARGUANA_ALPHA, INITIAL_RETRIEVAL_DEPTH,
                                           SPARSECL_SCORE_FORMULA, SPARSECL_UPSTREAM_COMMIT)

    lock_sha, model_files = verify_model_assets(baseline_root)
    packages = {
        "torch": importlib.metadata.version("torch"),
        "transformers": transformers.__version__,
        "faiss-cpu": importlib.metadata.version("faiss-cpu"),
        "numpy": numpy.__version__,
        "beir": importlib.metadata.version("beir"),
        "pytrec-eval": importlib.metadata.version("pytrec-eval"),
    }
    return {
        "protocol": "sparsecl-bge-hoyer-v1",
        "upstream_commit": SPARSECL_UPSTREAM_COMMIT,
        "source_sha256": {
            "sparsecl_models.py": sha256_file(baseline_root / "sparsecl" / "models.py"),
            "official_evaluator.py": sha256_file(baseline_root / "test_contradiction_faiss_final.py"),
            "adapter_worker.py": sha256_file(worker_path),
            "project_scoring_protocol.py": sha256_file(PROJECT_ROOT / "memory_retrieval" / "sparsecl.py"),
            "assets_lock.json": lock_sha,
        },
        "model_revisions": {
            "bge": "BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
            "sparsecl": "SparseCL/BGE-SparseCL-arguana@2797d506cac008fba3be91f7692d1cd3f4ee4c78",
            "files_sha256": model_files,
        },
        "packages": packages,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "precision": "float32",
        "batch_size": batch_size,
        "cpu_threads": cpu_threads,
        "input": "content.strip(); no instructions, titles, source metadata, or labels",
        "pooling": {"bge": "last_hidden_state CLS", "sparsecl": "official our_BertForCL attention-mask average"},
        "max_seq_length": 512,
        "l2_normalize": True,
        "initial_retrieval": {
            "engine": "faiss.IndexHNSWFlat", "dimension": 768, "M": 64,
            "metric": "L2 on normalized BGE embeddings", "efSearch": INITIAL_RETRIEVAL_DEPTH,
            "k": f"min({INITIAL_RETRIEVAL_DEPTH}, memory_count)",
        },
        "scoring": {"formula": SPARSECL_SCORE_FORMULA, "hoyer_zero_l2_lt": 1e-3,
                    "hoyer_zero_l2_score": -1e9, "stable_ties": "retain HNSW result order"},
        "alpha_selection": {"range": [0, 10], "bins": 10, "eps": 0.01,
                             "objective": "dev NDCG@10", "returned_value": "final interval left endpoint"},
        "fixed_arguana_alpha": FIXED_ARGUANA_ALPHA,
        "memory_bank_sha256": canonical_digest([
            {"memory_id": row["memory_id"], "content": row["content"].strip()} for row in memories
        ]),
    }


def encode_texts(texts, tokenizer, model, device, batch_size, torch, pooling):
    import torch.nn.functional as F
    import numpy as np

    output = []
    row_seconds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            started = time.perf_counter()
            inputs = tokenizer(batch_texts, max_length=512, padding=True, truncation=True,
                               return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if pooling == "bge_cls":
                encoded = model(**inputs, output_hidden_states=False, return_dict=True)
                embeddings = encoded.last_hidden_state[:, 0]
                embeddings = F.normalize(embeddings.float(), p=2, dim=1)
            else:
                encoded = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True)
                embeddings = encoded.pooler_output
            output.append(embeddings.float().cpu())
            elapsed = time.perf_counter() - started
            row_seconds.extend([elapsed / len(batch_texts)] * len(batch_texts))
    if not output:
        raise ValueError("SparseCL 不能编码空文本")
    vectors = torch.cat(output).cpu()
    vectors = (vectors / torch.norm(vectors, dim=1, keepdim=True)).numpy().astype(np.float32, copy=False)
    if vectors.ndim != 2 or vectors.shape[1] != 768 or not np.isfinite(vectors).all():
        raise ValueError("SparseCL 编码必须生成有限的 768 维向量")
    return vectors.astype(np.float32, copy=False), row_seconds


def ndcg_for_alpha(alpha, cosine_scores, hoyer_scores, memory_ids, qrels):
    from beir.retrieval.evaluation import EvaluateRetrieval
    from memory_retrieval.sparsecl import fused_order

    gold_by_id = {row["query_id"]: set(row["conflict_old_memory_ids"]) for row in qrels}
    if set(gold_by_id) != set(cosine_scores):
        raise ValueError("SparseCL NDCG 金标必须恰好覆盖全部开发查询")
    rankings = {}
    relevance = {}
    for query_id, targets in gold_by_id.items():
        if not targets:
            continue
        order, scores = fused_order(cosine_scores[query_id], hoyer_scores[query_id], alpha)
        rankings[query_id] = {
            memory_ids[query_id][int(position)]: float(scores[int(position)])
            for position in order[:10]
        }
        relevance[query_id] = {memory_id: 1 for memory_id in targets}
    if not rankings:
        return 0.0
    ndcg, _, _, _ = EvaluateRetrieval.evaluate(relevance, rankings, [1, 10])
    return float(ndcg["NDCG@10"]) * len(rankings) / len(gold_by_id)


def select_alpha(cosine_scores, hoyer_scores, memory_ids, qrels):
    left, right = 0.0, 10.0
    while right - left > 0.01:
        optimal_value = 0.0
        optimal_interval = (0.0, 0.0)
        for bin_index in range(10):
            alpha = left + (bin_index + 0.5) * (right - left) / 10
            value = ndcg_for_alpha(alpha, cosine_scores, hoyer_scores, memory_ids, qrels)
            if value > optimal_value:
                optimal_value = value
                optimal_interval = (left + bin_index * (right - left) / 10,
                                    left + (bin_index + 1) * (right - left) / 10)
        left, right = optimal_interval
    return left


def cache_paths(cache_root, signature):
    digest = canonical_digest(signature)
    return Path(cache_root) / digest, digest


def load_cache(path, signature, memories, faiss, numpy):
    manifest_path = path / "manifest.json"
    if not path.exists():
        return None
    if not manifest_path.is_file():
        raise ValueError(f"SparseCL 缓存目录不完整，拒绝复用：{path}")
    manifest = read_json(manifest_path)
    if manifest.get("signature") != signature:
        raise ValueError(f"SparseCL 缓存配置与当前运行不一致：{path}")
    vectors_path, index_path = path / "library_vectors.npz", path / "bge_hnsw.faiss"
    if sha256_file(vectors_path) != manifest.get("vectors_sha256"):
        raise ValueError("SparseCL 旧库向量缓存哈希不一致")
    if sha256_file(index_path) != manifest.get("index_sha256"):
        raise ValueError("SparseCL HNSW 索引缓存哈希不一致")
    with numpy.load(vectors_path, allow_pickle=False) as saved:
        bge_vectors = saved["bge"].copy()
        sparsecl_vectors = saved["sparsecl"].copy()
    ids = manifest.get("memory_ids")
    expected_ids = [row["memory_id"] for row in memories]
    if ids != expected_ids or bge_vectors.shape != (len(ids), 768) or sparsecl_vectors.shape != (len(ids), 768):
        raise ValueError("SparseCL 缓存的记忆 ID 顺序或向量维度不匹配")
    index = faiss.read_index(str(index_path))
    if index.ntotal != len(ids) or index.d != 768 or index.metric_type != faiss.METRIC_L2:
        raise ValueError("SparseCL 缓存的 FAISS 索引设置或旧库规模不匹配")
    index.hnsw.efSearch = 1000
    return bge_vectors, sparsecl_vectors, index


def save_cache(path, signature, memories, bge_vectors, sparsecl_vectors, index, faiss, numpy):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    if staging.exists():
        raise ValueError(f"SparseCL 临时缓存目录已存在，拒绝覆盖：{staging}")
    staging.mkdir()
    vectors_path, index_path = staging / "library_vectors.npz", staging / "bge_hnsw.faiss"
    numpy.savez(vectors_path, bge=bge_vectors, sparsecl=sparsecl_vectors)
    faiss.write_index(index, str(index_path))
    manifest = {
        "signature": signature,
        "memory_ids": [row["memory_id"] for row in memories],
        "vectors_sha256": sha256_file(vectors_path),
        "index_sha256": sha256_file(index_path),
    }
    write_json(staging / "manifest.json", manifest)
    os.replace(staging, path)


def encode_and_build(value, baseline_root, cache_root, batch_size, cpu_threads,
                     expected_signature, expected_cache_artifacts):
    import faiss
    import numpy as np
    import torch
    import transformers

    from memory_retrieval.sparsecl import INITIAL_RETRIEVAL_DEPTH

    memories, queries = input_rows(value)
    baseline_root = Path(baseline_root).resolve()
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(cpu_threads)
    faiss.omp_set_num_threads(cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signature = sparsecl_signature(baseline_root, Path(__file__).resolve(), memories,
                                   batch_size, cpu_threads, torch, transformers, faiss, np)
    if expected_signature is not None and signature != expected_signature:
        raise ValueError("SparseCL test 设置与开发集冻结签名不一致")
    cache_path, cache_digest = cache_paths(cache_root, signature)
    cached = load_cache(cache_path, signature, memories, faiss, np)
    cache_hit = cached is not None
    if value["mode"] == "test" and not cache_hit:
        raise ValueError("SparseCL test 必须复用 dev 冻结的旧库双向量与 HNSW 索引缓存")
    cache_artifacts = None
    if cache_hit:
        cache_artifacts = {
            "vectors_sha256": sha256_file(cache_path / "library_vectors.npz"),
            "index_sha256": sha256_file(cache_path / "bge_hnsw.faiss"),
        }
        if expected_cache_artifacts is not None and cache_artifacts != expected_cache_artifacts:
            raise ValueError("SparseCL test 缓存文件哈希与 dev 冻结索引不一致")
    bge_vectors = cached[0] if cache_hit else None
    sparsecl_vectors = cached[1] if cache_hit else None
    index = cached[2] if cache_hit else None
    memory_texts = [row["content"].strip() for row in memories]
    query_texts = [row["content"].strip() for row in queries]
    model_load_seconds = bge_library_seconds = sparsecl_library_seconds = 0.0
    index_build_seconds = 0.0

    load_started = time.perf_counter()
    bge_path = baseline_root / "results" / "bge-base-en-v1.5"
    bge_tokenizer = transformers.AutoTokenizer.from_pretrained(str(bge_path), local_files_only=True)
    bge_model = transformers.AutoModel.from_pretrained(str(bge_path), local_files_only=True).to(device)
    bge_model.eval()
    model_load_seconds += time.perf_counter() - load_started
    query_bge, query_bge_seconds = encode_texts(query_texts, bge_tokenizer, bge_model, device,
                                               batch_size, torch, "bge_cls")
    if not cache_hit:
        bge_library, library_row_seconds = encode_texts(
            memory_texts, bge_tokenizer, bge_model, device, batch_size, torch, "bge_cls")
        bge_library_seconds = sum(library_row_seconds)
        bge_vectors = bge_library
        index_started = time.perf_counter()
        index = faiss.IndexHNSWFlat(768, 64)
        index.add(bge_vectors)
        index.hnsw.efSearch = INITIAL_RETRIEVAL_DEPTH
        index_build_seconds = time.perf_counter() - index_started
    del bge_model, bge_tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    load_started = time.perf_counter()
    sparsecl_path = baseline_root / "results" / "BGE-SparseCL-arguana"
    config = transformers.AutoConfig.from_pretrained(str(sparsecl_path), local_files_only=True,
                                                       trust_remote_code=True)
    model_args = SimpleNamespace(pooler_type="avg", temp=0.05, hard_negative_weight=0.0,
                                 do_mlm=False, mlm_weight=0.1, mlp_only_train=False)
    sparsecl_tokenizer = transformers.AutoTokenizer.from_pretrained(str(sparsecl_path),
                                                                     local_files_only=True)
    from sparsecl.models import our_BertForCL
    sparsecl_model = our_BertForCL.from_pretrained(
        str(sparsecl_path), from_tf=False, config=config, use_auth_token=None,
        model_args=model_args, local_files_only=True).to(device)
    sparsecl_model.eval()
    model_load_seconds += time.perf_counter() - load_started
    query_sparsecl, query_sparsecl_seconds = encode_texts(query_texts, sparsecl_tokenizer,
                                                          sparsecl_model, device, batch_size,
                                                          torch, "sparsecl_avg")
    if not cache_hit:
        sparsecl_library, library_row_seconds = encode_texts(
            memory_texts, sparsecl_tokenizer, sparsecl_model, device, batch_size, torch,
            "sparsecl_avg")
        sparsecl_library_seconds = sum(library_row_seconds)
        sparsecl_vectors = sparsecl_library
        save_cache(cache_path, signature, memories, bge_vectors, sparsecl_vectors, index, faiss, np)
        cache_artifacts = {
            "vectors_sha256": sha256_file(cache_path / "library_vectors.npz"),
            "index_sha256": sha256_file(cache_path / "bge_hnsw.faiss"),
        }
    del sparsecl_model, sparsecl_tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    k = min(INITIAL_RETRIEVAL_DEPTH, len(memories))
    query_ids = [row["query_id"] for row in queries]
    memory_ids = [row["memory_id"] for row in memories]
    hnsw_ids, hnsw_seconds = [], []
    for vector in query_bge:
        index.hnsw.efSearch = INITIAL_RETRIEVAL_DEPTH
        started = time.perf_counter()
        _, positions = index.search(vector.reshape(1, -1), k)
        hnsw_seconds.append(time.perf_counter() - started)
        if positions.shape != (1, k) or (positions < 0).any():
            raise ValueError("SparseCL FAISS 初召回没有返回预期数量的合法旧记忆")
        hnsw_ids.append(positions[0].astype(np.int64, copy=False))

    cosine_scores, hoyer_scores, ranked_ids = {}, {}, {}
    scoring_seconds = []
    from memory_retrieval.sparsecl import score_candidates
    for row, query_id, query_cosine, query_sparse, positions in zip(
            queries, query_ids, query_bge, query_sparsecl, hnsw_ids):
        started = time.perf_counter()
        cos, hoyer = score_candidates(query_cosine, bge_vectors[positions],
                                      query_sparse, sparsecl_vectors[positions])
        scoring_seconds.append(time.perf_counter() - started)
        ids = [memory_ids[int(position)] for position in positions]
        cosine_scores[query_id] = cos
        hoyer_scores[query_id] = hoyer
        ranked_ids[query_id] = ids

    alpha_selection_seconds = 0.0
    dev_ndcg_at_10 = None
    if value["mode"] == "dev":
        qrels = value.get("qrels") or []
        if {row["query_id"] for row in qrels} != set(query_ids):
            raise ValueError("SparseCL dev NDCG 标签必须恰好覆盖全部开发查询")
        alpha_started = time.perf_counter()
        alpha = select_alpha(cosine_scores, hoyer_scores, ranked_ids, qrels)
        alpha_selection_seconds = time.perf_counter() - alpha_started
        dev_ndcg_at_10 = ndcg_for_alpha(alpha, cosine_scores, hoyer_scores, ranked_ids, qrels)
        alpha_source = "dev_ndcg_at_10"
    elif value["mode"] == "test":
        alpha = value.get("alpha")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError("SparseCL test 必须接收已冻结的开发集 alpha")
        alpha_source = "dev_frozen"
    else:
        from memory_retrieval.sparsecl import FIXED_ARGUANA_ALPHA
        alpha = FIXED_ARGUANA_ALPHA
        alpha_source = "arguana_fixed_0.18"

    from memory_retrieval.sparsecl import FIXED_ARGUANA_ALPHA, fused_order
    final_k = int(value["final_k"])
    rows = []
    for row, query_id, positions, q_bge_s, q_sparse_s, search_s, score_s in zip(
            queries, query_ids, hnsw_ids, query_bge_seconds, query_sparsecl_seconds,
            hnsw_seconds, scoring_seconds):
        cos = cosine_scores[query_id]
        hoyer = hoyer_scores[query_id]
        ids = ranked_ids[query_id]
        rerank_started = time.perf_counter()
        main_order, main_scores = fused_order(cos, hoyer, alpha)
        fixed_order, fixed_scores = fused_order(cos, hoyer, FIXED_ARGUANA_ALPHA)
        cosine_order = np.arange(len(ids))
        rerank_seconds = score_s + time.perf_counter() - rerank_started

        def candidates(order, scores):
            return [{"memory_id": ids[int(position)], "score": float(scores[int(position)])}
                    for position in order]

        initial = candidates(cosine_order, cos)
        primary = candidates(main_order, main_scores)
        fixed = candidates(fixed_order, fixed_scores)
        cosine = candidates(cosine_order, cos)
        query_total = q_bge_s + q_sparse_s + search_s + rerank_seconds
        result = {
            "query_id": query_id,
            "method": "sparsecl",
            "rerank": True,
            "selection_policy": "top_k",
            "coarse_k": k,
            "final_k": min(final_k, k),
            "library_size": len(memories),
            "coarse_candidates": [{"memory_id": candidate["memory_id"],
                                   "coarse_score": candidate["score"]} for candidate in initial],
            "candidates": primary[:final_k],
            "ranked_candidates": primary,
            "arguana_alpha_ranked_candidates": fixed,
            "cosine_ranked_candidates": cosine,
            "initial_candidates": initial,
            "sparsecl_alpha": float(alpha),
            "sparsecl_alpha_source": alpha_source,
            "judged": False,
            "judgments": [],
            "usage": {},
            "api_events": [],
            "timing": {"extraction": 0.0, "retrieval": float(search_s),
                       "fine_ranking": float(rerank_seconds), "llm_judgment": 0.0,
                       "total_seconds": float(query_total)},
            "timing_basis": "batch_amortized_per_query",
            "sparsecl_timing": {"bge_query_encoding_share_seconds": float(q_bge_s),
                                "sparsecl_query_encoding_share_seconds": float(q_sparse_s),
                                "hnsw_seconds": float(search_s),
                                "hoyer_reranking_seconds": float(rerank_seconds)},
        }
        if abs(float(alpha) - FIXED_ARGUANA_ALPHA) < 1e-12:
            result["arguana_alpha_ranked_candidates"] = None
        rows.append(result)

    costs = {
        "model_load_seconds": model_load_seconds,
        "bge_library_encoding_seconds": bge_library_seconds,
        "sparsecl_library_encoding_seconds": sparsecl_library_seconds,
        "hnsw_index_build_seconds": index_build_seconds,
        "library_build_seconds": bge_library_seconds + sparsecl_library_seconds + index_build_seconds,
        "bge_query_encoding_seconds": sum(query_bge_seconds),
        "sparsecl_query_encoding_seconds": sum(query_sparsecl_seconds),
        "query_encoding_seconds": sum(query_bge_seconds) + sum(query_sparsecl_seconds),
        "hnsw_search_seconds": sum(hnsw_seconds),
        "hoyer_reranking_seconds": sum(row["sparsecl_timing"]["hoyer_reranking_seconds"]
                                        for row in rows),
        "query_retrieval_total_seconds": sum(row["timing"]["total_seconds"] for row in rows),
        "alpha_selection_seconds": alpha_selection_seconds,
        "query_latency_basis": "batch_amortized_per_query",
        "cache_hit": cache_hit,
        "cache_path": str(cache_path),
        "cache_digest": cache_digest,
    }
    return {"signature": signature, "alpha": float(alpha), "alpha_source": alpha_source,
            "dev_ndcg_at_10": dev_ndcg_at_10,
            "initial_retrieval_depth": k, "cache_artifacts": cache_artifacts,
            "costs": costs, "results": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.cpu_threads < 1:
        raise SystemExit("--batch-size 和 --cpu-threads 必须为正整数")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    value = read_json(args.input)
    try:
        result = encode_and_build(value, args.baseline_root, args.cache_root,
                                  args.batch_size, args.cpu_threads,
                                  value.get("expected_signature"),
                                  value.get("expected_cache_artifacts"))
    except Exception as exc:
        print(f"SparseCL 运行失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
    write_jsonl(args.output, result["results"])
    metadata = {key: value for key, value in result.items() if key != "results"}
    write_json(args.metadata, metadata)
    print(f"SparseCL 完成：queries={len(result['results'])}, alpha={result['alpha']:.6f}, "
          f"alpha_source={result['alpha_source']}, cache_hit={result['costs']['cache_hit']}",
          flush=True)


if __name__ == "__main__":
    main()
