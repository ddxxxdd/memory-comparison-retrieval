import time

import numpy as np

from .api import usage_summary
from .baselines import BM25, rank, reciprocal_rank_fusion
from .config import SIMILARITY_THRESHOLD
from .data import RETRIEVAL_TEXT_VERSION, build_retrieval_text, nonempty_text
from .prompts import INDEX_PROMPT_HASH, PROMPT_HASH, PROMPT_VERSION, describe, hypotheses, judge_pair


class Retriever:
    def __init__(self, index, client):
        self.index = index
        self.client = client
        self.memory_positions = {m.memory_id: i for i, m in enumerate(index.memories)}
        self.bm25 = None

    def _embed(self, texts, stage):
        self.index.check_embedding(self.client.settings)
        vectors = self.client.embed(texts, stage)
        if vectors.shape[1] != self.index.content_vectors.shape[1]:
            raise ValueError("当前服务返回的向量维度与旧索引不同，请确认模型版本")
        return vectors

    def query(self, query_id, content, *, retrieval_text=None, method="proposed", rerank=True,
              judge=False, coarse_k=None, final_k=None, scope=None, user_id=None):
        nonempty_text(query_id, "query_id")
        nonempty_text(content, "content")
        # 检索用文本：旧记忆与新记忆必须落在同一份表示上（建库侧用 Memory.retrieval_text）。
        text = retrieval_text if retrieval_text is not None else content
        nonempty_text(text, "retrieval_text")
        if method not in {"proposed", "independent", "dense", "bm25", "hybrid"}:
            raise ValueError("未知检索方法")
        if scope not in {None, "personal", "world"}:
            raise ValueError("scope 只能是 personal、world 或 None")
        coarse_k = self.client.settings.coarse_k if coarse_k is None else coarse_k
        final_k = self.client.settings.final_k if final_k is None else final_k
        description_method = method in {"proposed", "independent"}
        if not description_method:
            self.client.settings.validate_budget(coarse_k, final_k)
        if description_method and self.index.manifest["prompt_hash"] != INDEX_PROMPT_HASH:
            raise ValueError("描述提示已变更，请重新建库，避免混用不同提示版本")
        # 旧记忆向量是用建库时的检索文本编码的。凡是要和这些向量比相似度的方法，
        # 都必须与索引使用同一版检索文本，否则等于拿带主体的查询去比不带主体的旧库。
        if method != "bm25" and (self.index.manifest.get("retrieval_text_version")
                                 != RETRIEVAL_TEXT_VERSION):
            raise ValueError("检索文本表示已变更（缺少主体上下文或拼接方式不同），请重新建库")
        self.client.settings.require(llm=description_method or judge, embedding=method != "bm25")
        if method != "bm25":
            self.index.check_embedding(self.client.settings)
        start, event_start = time.perf_counter(), len(self.client.events)
        stage_times, selection, probes, status = {}, None, None, "ok"
        memories = self.index.memories

        # 可比性由通道决定:personal 只和同一 owner 比,world 不区分 owner。
        # 用户 A 的偏好与用户 B 的偏好不是同一件事;而用户说错一个公共事实仍是同一件事。
        allowed = None
        if scope is not None:
            allowed = np.zeros(len(memories), dtype=bool)
            for position, memory in enumerate(memories):
                ref = memory.source_ref if isinstance(memory.source_ref, dict) else {}
                if ref.get("scope") != scope:
                    continue
                if scope == "personal" and ref.get("user_id") != user_id:
                    continue
                allowed[position] = True

        def row(position, score, score_name):
            memory = memories[position]
            return {**memory.to_dict(), score_name: float(score)}

        if description_method:
            stage_start = time.perf_counter()
            catalog = self.index.catalog if method == "proposed" else []
            selection = describe(self.client, text, catalog)
            stage_times["description"] = time.perf_counter() - stage_start
            if not selection["descriptions"]:
                coarse, candidates = [], []
                status = "no_descriptions"
            else:
                stage_start = time.perf_counter()
                # 选中项直接使用冻结目录的向量；只编码本次新生成的描述。
                # 目录条目与描述向量行不再一一对应（归并后一个条目有多行），
                # 因此这里要取条目呈现文本所在的那一行，即 aliases 的第一项。
                existing = {}
                for position, entry in enumerate(self.index.catalog):
                    existing[entry["description"]] = (entry["aliases"][0]
                                                      if entry.get("aliases") else position)
                novel = [d for d in selection["descriptions"] if d not in existing]
                encoded = self._embed(novel, "query_description_embedding") if novel else None
                new_vectors = dict(zip(novel, encoded)) if novel else {}
                query_vectors = np.stack([
                    self.index.description_vectors[existing[d]] if d in existing else new_vectors[d]
                    for d in selection["descriptions"]
                ])
                entry_scores = (query_vectors @ self.index.description_vectors.T).max(axis=0)
                if len(entry_scores) != len(self.index.catalog):
                    # 归并后的目录一个条目对应多条描述向量；条目得分取其全部别名的最大值，
                    # 这样被收掉的措辞仍然能把记忆带回候选，归并不会减少可达性。
                    aggregated = np.full(len(self.index.catalog), -np.inf)
                    for row_index, entry_index in enumerate(self.index.entry_of_row):
                        best = max(aggregated[entry_index], float(entry_scores[row_index]))
                        aggregated[entry_index] = best
                    entry_scores = aggregated
                scores = np.full(len(memories), -np.inf)
                for i, entry in enumerate(self.index.catalog):
                    for memory_id in entry["memory_ids"]:
                        position = self.memory_positions[memory_id]
                        scores[position] = max(scores[position], entry_scores[i])
                if allowed is not None:
                    scores[~allowed] = -np.inf
                coarse_positions = [i for i in rank(scores, memories) if scores[i] >= SIMILARITY_THRESHOLD]
                coarse = [row(i, scores[i], "coarse_score") for i in coarse_positions]
                stage_times["coarse_retrieval"] = time.perf_counter() - stage_start
                if rerank and coarse_positions:
                    stage_start = time.perf_counter()
                    candidate_texts = [
                        {"memory_id": candidate["memory_id"],
                         "content": build_retrieval_text(candidate["content"],
                                                          candidate.get("source_ref"))}
                        for candidate in coarse
                    ]
                    probes = hypotheses(self.client, text, selection["descriptions"],
                                        candidate_texts)
                    stage_times["hypotheses"] = time.perf_counter() - stage_start
                    stage_start = time.perf_counter()
                    probe_vectors = self._embed(probes["probes"], "hypothesis_embedding")
                    fine_scores = (probe_vectors @ self.index.content_vectors[coarse_positions].T).max(axis=0)
                    subject_relations = {item["memory_id"]: item["subject_relation"]
                                         for item in probes["subject_relations"]}
                    ordered = sorted(range(len(coarse)), key=lambda i: (
                        -float(fine_scores[i]), -coarse[i]["coarse_score"], coarse[i]["memory_id"]))
                    candidates = [{**coarse[i], "fine_score": float(fine_scores[i])}
                                  for i in ordered
                                  if subject_relations[coarse[i]["memory_id"]] != "incompatible"
                                  and fine_scores[i] >= SIMILARITY_THRESHOLD]
                    stage_times["fine_ranking"] = time.perf_counter() - stage_start
                elif not rerank:
                    candidates = coarse
                else:
                    candidates = []
        else:
            stage_start = time.perf_counter()
            if method in {"bm25", "hybrid"}:
                if self.bm25 is None:
                    self.bm25 = BM25([m.retrieval_text for m in memories])
                sparse_scores = self.bm25.score(text)
            if method in {"dense", "hybrid"}:
                dense_scores = (self._embed([text], "raw_query_embedding") @ self.index.content_vectors.T)[0]
            scores = (sparse_scores if method == "bm25" else dense_scores if method == "dense" else
                      reciprocal_rank_fusion(sparse_scores, dense_scores, memories))
            if allowed is not None:
                scores[~allowed] = -np.inf
            # 只能从范围内取候选:范围外的分数是 -inf,不能因为它们排在后面就占掉预算。
            ranked = [i for i in rank(scores, memories) if np.isfinite(scores[i])]
            coarse = [row(i, scores[i], "score") for i in ranked[:coarse_k]]
            candidates = coarse[:final_k]
            stage_times["retrieval"] = time.perf_counter() - stage_start

        judgments = []
        if judge:
            stage_start = time.perf_counter()
            for candidate in candidates:
                label = judge_pair(self.client, text,
                                   build_retrieval_text(candidate["content"], candidate.get("source_ref")))
                judgments.append({"memory_id": candidate["memory_id"], **label})
            stage_times["conflict_judgment"] = time.perf_counter() - stage_start
        events = self.client.events[event_start:]
        return {
            "query_id": query_id, "content": content, "retrieval_text": text, "status": status,
            "index_id": self.index.manifest["index_id"],
            "method": method, "rerank": bool(description_method and rerank), "judged": judge,
            "selection_policy": "similarity_threshold" if description_method else "top_k",
            "similarity_threshold": SIMILARITY_THRESHOLD if description_method else None,
            "coarse_k": None if description_method else coarse_k,
            "final_k": None if description_method else final_k,
            "prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH,
            "llm": self.client.settings.llm_identity() if description_method or judge else None,
            "embedding": self.index.manifest["embedding"],
            "library_size": len(memories),
            "scope": scope, "user_id": user_id,
            "eligible_pool": int(allowed.sum()) if allowed is not None else len(memories),
            "selection": selection, "hypotheses": probes,
            "coarse_candidates": coarse, "candidates": candidates, "judgments": judgments,
            "timing": {**stage_times, "total_seconds": time.perf_counter() - start},
            "usage": usage_summary(events), "api_events": events,
        }
