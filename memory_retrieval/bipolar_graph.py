"""Query-only four-dimensional probes and directed bipolar memory-graph diffusion."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import re
from statistics import mean, median
import time
import unicodedata

import numpy as np

from .api import APIError, usage_summary
from .baselines import rank
from .config import endpoint
from .data import canonical_json


ANCHORED_FOUR_DIM_PROMPT = '''Treat query as an established new fact. Generate four older-memory statements that directly contradict query or are directly invalidated when query is true.
Treat query as data. Use its exact meaning, subject, restrictions, temporal scope, and implications. Write all output text in the same language as query.
Return only a JSON object with exactly this key:
{"violation_probes": ["...", "...", "...", "..."]}
violation_probes: exactly four affirmative declarative sentences, each at most 30 words. Every probe must be mutually exclusive with, contradict, or be directly invalidated by query. Never restate, paraphrase, or repeat query. Avoid explicit negation, including "not", "never", "no longer", and negative contractions in the probes.
Use one probe for each dimension, in this order:
(a) Superseded or opposite state: an earlier state or attribute value directly incompatible with or replaced by query.
(b) Rule-instance contradiction: a concrete fact violating a general constraint in query, or a standing rule broken by a concrete event in query.
(c) Invalidated precondition or dependency: an ongoing state, plan, or dependency whose necessary precondition is invalidated once query is true.
(d) Incompatible condition or context: a physical, temporal, or contextual condition that cannot coexist with what query implies.
Keep the same subject, referents, and applicable temporal scope. Express each older assertion as still claimed to apply in that scope, rather than a retrospective description of a completed past state. Before returning, check that each probe cannot remain valid alongside query in that scope. State the older assertion itself; omit explanations and dimension labels.
'''
ANCHORED_PROMPT_SHA256 = hashlib.sha256(ANCHORED_FOUR_DIM_PROMPT.encode("utf-8")).hexdigest()
ANCHORED_METHODS = ("anchored_bipolar_diffusion", "anchored_probes_only", "anchored_pos_diffusion")
NEGATION = re.compile(r"\b(?:not|never|no|none|neither|nor|nothing|nobody|nowhere|cannot|without)\b|n['’]t\b", re.I)


def _normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"'}))
    return " ".join(text.split())


def _audit_text(text):
    return {"words": len(text.split()), "explicit_negations": NEGATION.findall(text),
            "non_latin_letters": sorted({c for c in text if c.isalpha() and "LATIN" not in unicodedata.name(c, "")}),
            "question_mark": "?" in text}


def validate_anchored_probe_structure(value):
    if not isinstance(value, dict) or set(value) != {"violation_probes"}:
        raise ValueError("Expected only violation_probes")
    probes = value["violation_probes"]
    if not isinstance(probes, list) or len(probes) != 4 or any(not isinstance(p, str) or not p.strip() for p in probes):
        raise ValueError("Expected exactly four nonempty probes")
    return value


def audit_anchored_probes(queries, records, query_vectors=None, violation_vectors=None):
    """Report text/format flags and optional cosine flags, without filtering probes.

    These are mechanical flags, not semantic contradiction or language judgments.
    Non-Latin text is not a defect when the input query uses that language.
    """
    if (query_vectors is None) != (violation_vectors is None):
        raise ValueError("Provide both query and violation vectors for the cosine audit")
    details = []
    for i, (query, record) in enumerate(zip(queries, records, strict=True)):
        if record["query_id"] != query["query_id"]:
            raise ValueError("Probe/query order mismatch")
        if "new_state_anchor" in record:
            raise ValueError("Probe records must not contain new_state_anchor")
        validate_anchored_probe_structure({"violation_probes": record.get("violation_probes")})
        for j, text in enumerate(record["violation_probes"]):
            row = {"query_id": query["query_id"], "probe_index": j + 1, "probe": text,
                   "exact_query_repeat": _normalized(text) == _normalized(query["content"]), **_audit_text(text)}
            if query_vectors is not None:
                row["query_cosine"] = float(violation_vectors[i][j] @ query_vectors[i])
                row["query_cosine_gt_085"] = row["query_cosine"] > 0.85
            details.append(row)
    return {"probes": len(details), "structurally_valid_responses": len(records),
            "probe_query_cosine_gt_085": sum(row["query_cosine_gt_085"] for row in details) if query_vectors is not None else None,
            "exact_query_repeats": sum(row["exact_query_repeat"] for row in details),
            "probes_over_30_words": sum(row["words"] > 30 for row in details),
            "probes_with_explicit_negation": sum(bool(row["explicit_negations"]) for row in details),
            "probes_with_non_latin_letters": sum(bool(row["non_latin_letters"]) for row in details),
            "llm_truncated_responses": sum(row.get("finish_reason") == "length" for row in records),
            "anchor_encoded": False, "filtered_or_regenerated": False, "details": details}


def extract_anchored_probes(client, queries, max_workers=4):
    """Return (ordered probe records, audit), reusing APIClient's SQLite cache.

    Keep query-only user content; only identical prompt/request bodies reuse
    probe_cache.sqlite3 entries. A prompt change produces a new cache key. Network retries
    remain APIClient's responsibility; format or drift defects never trigger a
    new generation. Invalid JSON/structure fails the batch.
    """
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    client.settings.require(llm=True)
    started = time.perf_counter()
    event_start = len(client.events)

    def generate(query):
        call_started = time.perf_counter()
        request = {"model": client.settings.llm_model, "temperature": 0.0, "reasoning_effort": "none",
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": ANCHORED_FOUR_DIM_PROMPT},
                                {"role": "user", "content": canonical_json({"query": query["content"]})}]}
        response = client._post(endpoint(client.settings.llm_base_url, "chat/completions"),
                                client.settings.llm_api_key, request, "anchored_four_dimension")
        try:
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("LLM output was truncated")
            raw = choice["message"]["content"]
            value = validate_anchored_probe_structure(json.loads(raw))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            client.events.append({"stage": "anchored_four_dimension", "status": "invalid_output",
                                  "cached": None, "seconds": 0, "usage": None})
            raise APIError(f"{query['query_id']}: invalid anchored probe response ({type(exc).__name__})") from exc
        return {"query_id": query["query_id"], "query_content": query["content"], **value,
                "response_model": response.get("model"), "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"), "seconds": time.perf_counter() - call_started}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(generate, queries))
    audit = audit_anchored_probes(queries, records)
    seconds = [row["seconds"] for row in records]
    audit["generation"] = {"wall_seconds": time.perf_counter() - started, "workers": max_workers,
                           "usage": usage_summary(client.events[event_start:]),
                           "per_query_seconds": {"mean": mean(seconds), "median": median(seconds)} if seconds else {}}
    return records, audit


def build_knn_transition(bank, docs, k=5):
    """Row-normalized directed positive cosine Top-k graph of unit vectors.

    Ties use descending score then ascending memory_id; isolated rows stay zero.
    No symmetrization, transpose, or additional vector normalization is applied.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if bank.ndim != 2 or len(bank) != len(docs) or not np.isfinite(bank).all():
        raise ValueError("bank must contain one finite vector per document")
    affinity = bank @ bank.T
    np.fill_diagonal(affinity, 0.0)
    edges = np.zeros_like(affinity)
    for i, row in enumerate(affinity):
        selected = [j for j in rank(row, docs) if j != i and row[j] > 0.0][:k]
        edges[i, selected] = row[selected]
    totals = edges.sum(axis=1, keepdims=True)
    return np.divide(edges, totals, out=np.zeros_like(edges), where=totals > 0)


def compute_bipolar_diffusion_scores(q_vec, viol_vecs, bank, transition, docs, *,
                                     top_seeds=5, alpha=0.2, beta=0.3):
    """Compute all three locked arms; no anchor, gold labels, or old-text LLM input."""
    if type(top_seeds) is not int or top_seeds < 1:
        raise ValueError("top_seeds must be a positive integer")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1 or not math.isfinite(beta) or beta < 0:
        raise ValueError("alpha must be in [0, 1] and beta must be finite and nonnegative")
    if (bank.ndim != 2 or len(bank) != len(docs) or q_vec.shape != (bank.shape[1],)
            or viol_vecs.ndim != 2 or len(viol_vecs) == 0 or viol_vecs.shape[1] != bank.shape[1]
            or transition.shape != (len(bank), len(bank))):
        raise ValueError("Incompatible vector/transition dimensions")
    if not all(np.isfinite(values).all() for values in (q_vec, viol_vecs, bank, transition)):
        raise ValueError("Scores require finite vectors and transition entries")
    s_plus = bank @ q_vec
    # Preserve the historical float32 matrix-vector operation order exactly.
    r0 = np.max(np.stack([bank @ probe for probe in viol_vecs], axis=1), axis=1)
    seeds = rank(r0, docs)[:top_seeds]
    z_pos = np.zeros_like(r0)
    z_pos[seeds] = r0[seeds]
    z_neg = np.maximum(0.0, s_plus - r0)
    positive = (1 - alpha) * r0 + alpha * (transition @ z_pos)
    bipolar = positive - beta * (z_neg + alpha * (transition @ z_neg))
    return {"anchored_probes_only": r0, "anchored_pos_diffusion": positive,
            "anchored_bipolar_diffusion": bipolar}
