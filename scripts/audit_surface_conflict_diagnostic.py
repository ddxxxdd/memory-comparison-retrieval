"""Recompute reported hits from saved complete rankings, without API calls."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'research_notes/surface_conflict_diagnostic_2026-11-17'
RUN = ROOT / 'runs/experiments/surface-conflict-diagnostic-20261117'


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    summary = json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest() == summary['dataset_manifest_sha256']
    for name, digest in manifest['artifact_sha256'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest
    memories = {m['memory_id']: m for m in rows(DATA/'memories.jsonl')}
    queries = {q['query_id']: q for q in rows(DATA/'queries.jsonl')}
    banks = {b['case_id']: b for b in rows(DATA/'banks.jsonl')}
    per_case = {(r['case_id'],r['method']): r for r in rows(RUN/'per_case.jsonl')}
    texts = json.loads((RUN/'embedding_text_order.json').read_text(encoding='utf-8'))
    vectors = np.load(RUN/'embeddings.npz')['vectors']
    assert len(texts) == len(vectors) == 1240 and vectors.shape[1] == 768
    assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
    by_text = {t: vectors[i] for i,t in enumerate(texts)}
    groups = defaultdict(list)
    ties, boundary_ties = [], []
    seen = set()
    scored_candidates = 0
    with gzip.open(RUN/'rankings.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            key = (record['case_id'],record['method'])
            assert key not in seen
            seen.add(key)
            bank = banks[key[0]]
            saved = per_case[key]
            ranked = record['ranked_candidates']
            assert len(ranked) == len(bank['memory_ids'])
            assert {r['memory_id'] for r in ranked} == set(bank['memory_ids'])
            assert [r['rank'] for r in ranked] == list(range(1,len(ranked)+1))
            assert all(a['score'] >= b['score'] for a,b in zip(ranked,ranked[1:]))
            target = next(r for r in ranked if r['memory_id']==bank['known_conflict_target_id'])
            assert target['rank'] == saved['target_rank']
            assert ranked[:10] == saved['top10']
            ahead = ranked[:target['rank']-1]
            known = set(bank['known_compatible_ids'])
            assert sum(r['memory_id'] in known for r in ahead) == saved['known_compatible_ahead']
            assert len(ahead)-saved['known_compatible_ahead'] == saved['unjudged_background_ahead']
            for r in ranked:
                expected = 'known_conflict' if r['memory_id']==target['memory_id'] else 'known_compatible' if r['memory_id'] in known else 'unjudged_background'
                assert r['category'] == expected
            if key[1] == 'dense':
                matrix = np.stack([by_text[memories[r['memory_id']]['content']] for r in ranked])
                scores = matrix @ by_text[queries[bank['query_id']]['content']]
                assert np.allclose(scores,[r['score'] for r in ranked],atol=3e-7)
            equal = [r['rank'] for r in ranked if r['score']==target['score']]
            if len(equal)>1:
                tie = {'case_id':key[0],'method':key[1],'rank_min':min(equal),'rank_max':max(equal)}
                ties.append(tie)
                if any(min(equal)<=k<max(equal) for k in (5,10)):
                    boundary_ties.append(tie)
            group = (bank['candidate_count'],bank['compatible_count'],bank['variant'],key[1])
            groups[group].append(target['rank'])
            scored_candidates += len(ranked)
    assert len(seen) == len(per_case) == 576
    for group in summary['groups']:
        ranks = groups[(group['candidate_count'],group['compatible_count'],group['variant'],group['method'])]
        assert len(ranks) == 16
        for k in (1,3,5,10,20):
            assert sum(r<=k for r in ranks) == group['hit_counts_at_k'][str(k)]
        assert statistics.mean(ranks) == group['mean_target_rank']
    paired = []
    for method in ('bm25','dense','hybrid'):
        entries = []
        for qid in queries:
            near = per_case[(f'{qid}-n1000-h12-near',method)]['target_rank']
            far = per_case[(f'{qid}-n1000-h12-far',method)]['target_rank']
            entries.append((near,far))
        paired.append({'method':method,'worsened':sum(b>a for a,b in entries),
                       'unchanged':sum(b==a for a,b in entries),'improved':sum(b<a for a,b in entries),
                       'near_hit_far_miss_at_5':sum(a<=5<b for a,b in entries)})
    result = {'passed':True,'auditor':'same assistant; separate recomputation script, not an independent human audit',
              'complete_rankings_checked':len(seen),'candidate_scores_checked_for_membership_order_and_labels':scored_candidates,
              'dense_scores_recomputed_from_saved_vectors':192,
              'input_hashes_unchanged':True,'target_ties':ties,'top5_or_top10_boundary_ties':boundary_ties,
              'paired_main_condition':paired}
    (RUN/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='target_ties'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
