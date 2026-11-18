"""Run existing raw-text baselines on frozen surface-conflict diagnostic banks."""
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_retrieval.api import APIClient, usage_summary
from memory_retrieval.baselines import BM25, rank, reciprocal_rank_fusion
from memory_retrieval.config import Settings
from memory_retrieval.data import Memory

METHODS = ("bm25", "dense", "hybrid")
KS = (1, 3, 5, 10, 20)
DEFAULT_DATA = ROOT.parent / "research_notes/surface_conflict_diagnostic_2026-11-17"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["artifact_sha256"].items():
        assert sha(dataset / filename) == expected, filename
    rows = {name: read_rows(dataset / (name + ".jsonl")) for name in ("queries", "memories", "banks")}
    qs = {q["query_id"]: q for q in rows["queries"]}
    ms = {m["memory_id"]: m for m in rows["memories"]}
    assert len(qs) == len(rows["queries"]) == 16
    assert len(ms) == len(rows["memories"]) == 1224
    assert len({m['content'] for m in ms.values()}) == len(ms)
    assert not {q['content'] for q in qs.values()} & {m['content'] for m in ms.values()}
    by_case = {b['case_id']: b for b in rows['banks']}
    assert len(by_case) == len(rows['banks']) == 192
    for b in rows["banks"]:
        ids = b['memory_ids']
        assert b['query_id'] in qs and set(ids) <= ms.keys()
        assert len(ids) == len(set(ids)) == b['candidate_count']
        target = b['known_conflict_target_id']
        assert target in ids and ms[target]['scenario_id'] == b['query_id']
        assert ms[target]['variant'] == b['variant']
        assert len(b['known_compatible_ids']) == b['compatible_count']
        assert set(b['known_compatible_ids']) <= set(ids)
        assert all(ms[mid]['scenario_id'] == b['query_id']
                   and ms[mid]['kind'] == 'authored_compatible' for mid in b['known_compatible_ids'])
        assert sum(ms[mid]['kind'] == 'authored_target' for mid in ids) == 1
        other_variant = 'near' if b['variant'] == 'far' else 'far'
        twin = by_case[f"{b['query_id']}-n{b['candidate_count']}-h{b['compatible_count']}-{other_variant}"]
        assert [mid for mid in ids if mid != target] == [mid for mid in twin['memory_ids'] if mid != twin['known_conflict_target_id']]
        assert ids.index(target) == twin['memory_ids'].index(twin['known_conflict_target_id'])
    return manifest, qs, ms, rows['banks']


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['candidate_count'], r['compatible_count'], r['variant'], r['method'])].append(r)
    result = []
    for (n, h, variant, method), values in sorted(groups.items()):
        ranks = [r['target_rank'] for r in values]
        result.append({"candidate_count": n, "compatible_count": h, "variant": variant, "method": method,
                       "query_count": len(values),
                       "hit_counts_at_k": {str(k): sum(x <= k for x in ranks) for k in KS},
                       "hit_rate_at_k": {str(k): sum(x <= k for x in ranks)/len(ranks) for k in KS},
                       "mean_target_rank": statistics.mean(ranks), "median_target_rank": statistics.median(ranks),
                       "mean_reciprocal_rank": statistics.mean(1/x for x in ranks),
                       "known_compatible_ahead_mean": statistics.mean(r['known_compatible_ahead'] for r in values),
                       "unjudged_background_ahead_mean": statistics.mean(r['unjudged_background_ahead'] for r in values),
                       "min_k_for_all_known_targets_posthoc": max(ranks)})
    return result


def render_report(summary, per_case, queries, memories):
    lines = ["# 新构造的表面表达差异冲突诊断：Baseline 结果", "",
             "数据和主条件均在检索前冻结；保留全部成功与失败样例。16 个基础场景，192 个重复实验条件。", "",
             "目标和相容对照为助手构造并审阅，未独立人工复核；公开背景未逐对判定。以下 Hit@K 仅指已确认目标的命中率，不是完整冲突集合的召回率或候选精确率。", "",
             f"Dense 配置：{summary['embedding']['model']}；本次实际向量维度 {summary['embedding_dimension']}。", "",
             "## 主条件：1,000 条候选、12 条已审核相容候选、表达变化的冲突目标", "",
             "| 方法 | Hit@5 | Hit@10 | Hit@20 | 目标平均名次 | 已知相容候选平均排前数 |", "|---|---:|---:|---:|---:|---:|"]
    for r in summary['groups']:
        if (r['candidate_count'], r['compatible_count'], r['variant']) == (1000, 12, 'far'):
            c = r['hit_counts_at_k']
            lines.append(f"| {r['method']} | {c['5']}/16 | {c['10']}/16 | {c['20']}/16 | {r['mean_target_rank']:.2f} | {r['known_compatible_ahead_mean']:.2f} |")
    lines += ["", "## 全部预设条件", "",
              "| 候选数 | 已知相容数 | 目标表达 | 方法 | Hit@5 | Hit@10 | Hit@20 | 平均名次 |",
              "|---:|---:|---|---|---:|---:|---:|---:|"]
    for r in summary['groups']:
        c = r['hit_counts_at_k']
        lines.append(f"| {r['candidate_count']} | {r['compatible_count']} | {r['variant']} | {r['method']} | {c['5']}/16 | {c['10']}/16 | {c['20']}/16 | {r['mean_target_rank']:.2f} |")
    lookup = {(r['query_id'], r['candidate_count'], r['compatible_count'], r['variant'], r['method']): r for r in per_case}
    lines += ["", "## 主条件下逐例的目标名次", "",
              "| 场景 | BM25 near → far | Dense near → far | Hybrid near → far |",
              "|---|---:|---:|---:|"]
    for qid, q in queries.items():
        cells = [f"{lookup[(qid,1000,12,'near',m)]['target_rank']} → {lookup[(qid,1000,12,'far',m)]['target_rank']}" for m in METHODS]
        lines.append(f"| {qid} {q['title_zh']} | " + " | ".join(cells) + " |")
    lines += ["", "## 主条件中落出 Top-10 的样例", ""]
    failures = [r for r in per_case if (r['candidate_count'], r['compatible_count'], r['variant']) == (1000,12,'far') and r['target_rank'] > 10]
    if not failures:
        lines.append("本轮主条件没有已知目标落出 Top-10；不能声称该预算下观察到了漏检。")
    for r in failures:
        lines += [f"### {r['query_id']} / {r['method']}：目标第 {r['target_rank']} 名", "",
                  "查询：" + queries[r['query_id']]['content'], "",
                  "冲突目标：" + memories[r['target_id']]['content'], "",
                  f"排在目标前：{r['known_compatible_ahead']} 条已审核相容记录；{r['unjudged_background_ahead']} 条未逐对审核背景。", ""]
        for c in r['top10'][:3]:
            lines.append(f"- #{c['rank']} [{c['category']}] {memories[c['memory_id']]['content']}")
        lines += [""]
    lines += ["## 解释限制", "",
              "- far 表示经过审核的词汇、句法或事实表示变化；未按低向量分数筛样。",
              "- near/far 目标逻辑内容相同，但还伴随显式程度和长度变化，不能将差异完全归因于词项重合。",
              "- h 是已审核的同题相容候选数；背景中的语义相关性未被排除，h=0 不代表完全没有竞争。",
              "- 候选数增加和相关候选增加是不同对照；所有条件使用同一 baseline 配置和固定 K。",
              "- 每例只有一个已知目标，不能依据 Top-10 其余九项不是该目标，就宣称误报率为 90%。",
              "- 本结果不能估计真实记忆中的失败频率，也不能证明新方法的创新性或效果。", ""]
    return "\n".join(lines)


def run(args):
    dataset, output = args.dataset.resolve(), args.output.resolve()
    manifest, queries, memories, banks = validate(dataset)
    if args.validate_only:
        print(json.dumps({"validated": True, "counts": manifest['counts']}, ensure_ascii=False))
        return
    if (output / 'summary.json').exists():
        raise ValueError("完成的结果不可覆盖。")
    output.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env()
    settings.require(embedding=True)
    plan = {"dataset_manifest_sha256": sha(dataset/'manifest.json'),
            "dataset_hashes": manifest['artifact_sha256'], "embedding": settings.embedding_identity(),
            "methods": METHODS, "top_k": KS}
    plan_path = output / 'run_plan.json'
    # Normalize tuples to JSON lists before comparing a resumed run.
    plan = json.loads(json.dumps(plan))
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding='utf-8')) == plan
    else:
        write_json(plan_path, plan)
    client = APIClient(replace(settings, cache_path=output/'embedding_cache.sqlite3'))
    texts = list(dict.fromkeys([r['content'] for r in memories.values()] + [q['content'] for q in queries.values()]))
    matrices = []
    for start in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[start:start+settings.embedding_batch_size]
        matrix = client.embed(batch, 'surface_conflict_diagnostic_embedding')
        if matrices:
            assert matrix.shape[1] == matrices[0].shape[1]
        matrices.append(matrix)
        print(f"embedding {min(start+len(batch),len(texts))}/{len(texts)} dim={matrix.shape[1]}", flush=True)
    vectors = np.concatenate(matrices)
    np.savez_compressed(output/'embeddings.npz', vectors=vectors)
    write_json(output/'embedding_text_order.json', texts)
    by_text = {text: vectors[i] for i, text in enumerate(texts)}
    per_case = []
    with gzip.open(output/'rankings.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for bank_no, b in enumerate(banks, 1):
            docs = [memories[mid] for mid in b['memory_ids']]
            # Equivalent target versions share a tie-break identity.
            objects = [Memory(d.get('logical_id', d['memory_id']), d['content'], {}) for d in docs]
            contents = [d['content'] for d in docs]
            query = queries[b['query_id']]['content']
            bm25 = BM25(contents).score(query)
            dense = np.stack([by_text[t] for t in contents]) @ by_text[query]
            hybrid = reciprocal_rank_fusion(bm25, dense, objects)
            for method, scores in zip(METHODS, (bm25, dense, hybrid)):
                ordered = []
                for position, index in enumerate(rank(scores, objects), 1):
                    doc = docs[index]
                    mid = doc['memory_id']
                    category = ('known_conflict' if mid == b['known_conflict_target_id'] else
                                'known_compatible' if mid in b['known_compatible_ids'] else 'unjudged_background')
                    ordered.append({"rank": position, "memory_id": mid, "score": float(scores[index]), "category": category})
                assert len(ordered) == b['candidate_count'] and all(math.isfinite(x['score']) for x in ordered)
                target = next(x for x in ordered if x['category'] == 'known_conflict')
                ahead = ordered[:target['rank']-1]
                row = {k: b[k] for k in ('case_id','query_id','candidate_count','compatible_count','variant')}
                row.update(method=method, target_id=target['memory_id'], target_rank=target['rank'],
                           target_score=target['score'],
                           target_tie_count=sum(x['score']==target['score'] for x in ordered)-1,
                           known_compatible_ahead=sum(x['category']=='known_compatible' for x in ahead),
                           unjudged_background_ahead=sum(x['category']=='unjudged_background' for x in ahead),
                           top10=ordered[:10])
                per_case.append(row)
                stream.write(json.dumps({"case_id": b['case_id'], "method": method, "ranked_candidates": ordered}, ensure_ascii=False)+'\n')
            if bank_no % 16 == 0:
                print(f"ranked {bank_no}/{len(banks)} banks", flush=True)
    summary = {"run_at_utc": datetime.now(timezone.utc).isoformat(),
               "dataset_path": str(dataset), **plan, "dataset_counts": manifest['counts'],
               "embedding_dimension": int(vectors.shape[1]), "embedding_usage": usage_summary(client.events),
               "groups": aggregate(per_case), "limitations": manifest['limitations']}
    write_json(output/'summary.json', summary)
    write_json(output/'api_events.json', client.events)
    (output/'per_case.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in per_case), encoding='utf-8')
    (output/'report.zh.md').write_text(render_report(summary, per_case, queries, memories), encoding='utf-8')
    print(json.dumps({"output": str(output), "embedding_dimension": summary['embedding_dimension'],
                      "usage": summary['embedding_usage'], "primary": [r for r in summary['groups'] if (r['candidate_count'],r['compatible_count'],r['variant']) == (1000,12,'far')]},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/experiments/surface-conflict-diagnostic-20261117')
    parser.add_argument('--validate-only', action='store_true')
    run(parser.parse_args())
