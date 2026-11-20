"""Validate the fixed STALE graph formulas on the current 15-query implicit dataset."""

import argparse
from datetime import datetime, timezone
from importlib.util import find_spec
from itertools import combinations
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/experiments/single-pair-15q-bge-20261119'
DATASET = ROOT / 'data/single_pair_implicit_15q_en'
OUTPUT = ROOT / 'runs/experiments/single-pair-15q-graph-bge-20261120'
STALE = ROOT / 'runs/experiments/stale-augmented-10scene-graph-bge-20261120'
DOWNLOADS = ROOT.parents[2] / 'Downloads'
BIPOLAR = ((0.2, 0.3), (0.3, 0.3), (0.3, 0.5))
METHODS = {'dense': 'BGE Dense', 'probes_only': 'probes_only', 'ocsp_qr': 'ocsp_qr γ=0.65',
           'diff_0.15': '单步正扩散 α=0.15',
           **{f'bipolar_{a:g}_{b:g}': f'双极扩散 α={a:g}, β={b:g}' for a, b in BIPOLAR}}

sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '4'
if find_spec('dotenv') is None:
    sys.path.append(str(ROOT / '.venv/Lib/site-packages'))
    import dotenv
    sys.path.pop()

import numpy as np
import pytrec_eval
from memory_retrieval.baselines import rank
from memory_retrieval.data import fingerprint, read_jsonl, sha256_file, write_json
from memory_retrieval.dataset import load_dataset
from memory_retrieval.encoders import MODEL_ROOT, encode_groups
from memory_retrieval.evaluation import ranking_metrics
from memory_retrieval.ocsp import compute_subspace_ocsp_scores
from scripts.run_stale_graph_audit import build_graph


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fixed_scores(bank, query, compatible, probes, transition, documents):
    positive = bank @ query
    initial = np.max(np.stack([bank @ probe for probe in probes], axis=1), axis=1)
    qr = compute_subspace_ocsp_scores(query, compatible, probes, bank, gamma=0.65)['ocsp_qr']
    seeds = rank(initial, documents)[:5]
    z_pos = np.zeros_like(initial)
    z_pos[seeds] = initial[seeds]
    z_neg = np.maximum(0.0, positive - initial)
    prop_pos, prop_neg = transition @ z_pos, transition @ z_neg
    scores = {'dense': positive, 'probes_only': initial, 'ocsp_qr': qr,
              'diff_0.15': (1 - 0.15) * initial + 0.15 * prop_pos}
    for alpha, beta in BIPOLAR:
        scores[f'bipolar_{alpha:g}_{beta:g}'] = (
            (1 - alpha) * initial + alpha * prop_pos - beta * (z_neg + alpha * prop_neg))
    assert all(np.isfinite(values).all() for values in scores.values())
    return scores, seeds, z_pos, z_neg


def audit_groups(dataset, original_queries, original_memories):
    old_queries = {row['query_id']: row for row in read_jsonl(original_queries)}
    old_memories = {row['memory_id']: row for row in read_jsonl(original_memories)}
    assert set(old_memories) == {doc.memory_id for doc in dataset.memories}
    background = [mid for mid, row in old_memories.items() if row['role'] == 'background']
    assert len(background) == 30
    groups = {}
    for query, gold in zip(dataset.queries, dataset.gold):
        qid = query['query_id']
        original = old_queries[qid]
        targets = gold['conflict_old_memory_ids']
        same_topic = original['distractor_ids']
        assert len(targets) == 1 and targets[0] == original['gold_memory_id']
        assert len(same_topic) == len(set(same_topic)) == 12
        assert all(old_memories[mid]['role'] == 'hard_distractor'
                   and old_memories[mid]['linked_query_id'] == qid for mid in same_topic)
        assert not (set(targets) & set(same_topic) or set(targets) & set(background) or set(same_topic) & set(background))
        groups[qid] = {'gold_ids': targets, 'same_topic_distractor_ids': same_topic, 'background_ids': background}
    return {'provenance': 'Original dataset ID/role membership only; all text comes from current English dataset',
            'original_queries': str(original_queries), 'original_memories': str(original_memories), 'queries': groups}


def cosine_audit(affinity, documents, membership):
    positions = {doc.memory_id: i for i, doc in enumerate(documents)}
    totals = {key: [] for key in ('gold_gold', 'gold_same_topic', 'gold_background')}
    per_query = {}
    for qid, group in membership['queries'].items():
        pair_ids = {
            'gold_gold': list(combinations(group['gold_ids'], 2)),
            'gold_same_topic': [(g, d) for g in group['gold_ids'] for d in group['same_topic_distractor_ids']],
            'gold_background': [(g, d) for g in group['gold_ids'] for d in group['background_ids']],
        }
        per_query[qid] = {}
        for key, pairs in pair_ids.items():
            values = [float(affinity[positions[a], positions[b]]) for a, b in pairs]
            totals[key].extend(values)
            per_query[qid][key] = {'pairs': len(values), 'mean': mean(values) if values else None}
    return {'aggregation': 'Pooled query-specific pairs; original 12 distractors/query and 30 designated background entries',
            'gold_gold_note': 'Each query has one gold; other queries gold memories are not co-gold for this query',
            'groups': {key: {'pairs': len(values), 'mean': mean(values) if values else None} for key, values in totals.items()},
            'per_query': per_query}


def evaluate(rankings, gold):
    qrels = {row['query_id']: {mid: 1 for mid in row['conflict_old_memory_ids']} for row in gold}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10'})
    metrics = {}
    for name, ranking in rankings.items():
        metric = ranking_metrics(ranking, gold, (3, 5, 10))
        checked = evaluator.evaluate({qid: {mid: float(len(ids) - i) for i, mid in enumerate(ids)}
                                      for qid, ids in ranking.items()})
        ndcg = mean(row['ndcg_cut_10'] for row in checked.values())
        assert abs(ndcg - metric['ndcg_at_10']) < 1e-12
        metric['pytrec_eval_ndcg_at_10'] = ndcg
        metric['mean_gold_rank'] = mean(position for positions in metric['gold_ranks'].values() for position in positions.values())
        metrics[name] = metric
    return metrics


def render_report(result):
    lines = ['# 15 查询隐式冲突集：固定图扩散跨数据集验证', '',
             '当前 single_pair_implicit_15q_en：15 查询、225 条共享旧记忆、15 个金标，每条查询恰好 1 个金标。', '',
             '复用 single-pair-15q-bge-20261119 的 45 条违例探针和 30 条相容改述。'
             '其 LLM 为 deepseek-v4.1-flash、温度 0、JSON 模式；本次没有 LLM 调用。'
             '原实验未保存向量，本次按原冻结 BAAI/bge-base-en-v1.5 重建并缓存，或使用该重建缓存。'
             'CLS、L2、CPU FP32、batch=4、threads=4、seed=0、max_length=512 均与原配置一致。', '',
             '全部方法共用同一份向量和完整 225 条候选。QR 对照仍使用原句及两条相容改述，γ=0.65。'
             '图采用 STALE 验证脚本同一构图函数：k=5、正相似边、零对角、按行归一化；'
             '正种子为 probes_only 的前 5 项并保留原分数。严格使用 P@z，参数照搬，不叠加 Top-10 重排。', '',
             '| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 金标平均排名↓ |',
             '|---|---:|---:|---:|---:|---:|']
    for name, label in METHODS.items():
        metric = result['metrics'][name]
        hits = [str(metric['at_k'][str(k)]['hits']) for k in (3, 5, 10)]
        lines.append(f"| {label} | {' | '.join(hits)} | {metric['ndcg_at_10']:.4f} | {metric['mean_gold_rank']:.2f} |")
    lines.extend(['', '命中数分母均为 15；NDCG@10 为 15 个查询的平均值，已由 pytrec_eval 复核。', '',
                  '## 旧记忆间相似度', '',
                  '同话题干扰项沿用原始 distractor_ids（每查询 12 条），背景沿用原始 role=background 的 30 条。'
                  '仅使用原始 ID 分组，文本均为当前英文版本；分组不参与构图、种子选择或打分。', '',
                  '| 配对类别 | 平均余弦 | 配对数 |', '|---|---:|---:|'])
    for key, label in (('gold_gold', '同查询金标—金标'), ('gold_same_topic', '金标—同话题干扰'),
                       ('gold_background', '金标—标注背景')):
        group = result['cosine_audit']['groups'][key]
        value = f"{group['mean']:.4f}" if group['mean'] is not None else '不适用'
        lines.append(f"| {label} | {value} | {group['pairs']} |")
    lines.extend(['', '不同查询的金标对应不同事实，不混作同一查询的多金标。'
                  '因此这里无法像 STALE 那样依靠多个同查询金标互相连接来扩散；图也会把同话题相容项连起来。', '',
                  f"三个对照完整排名与原实验一致；Dense 分数最大绝对差 {result['verification']['dense_max_abs_score_difference']:.2g}。"
                  '输入、原实验及主代码哈希保持不变。', '',
                  f"本次 LLM 调用 0 次、SparseCL 推理 0 次，BGE 重建：{result['execution']['bge_reconstructed']}；"
                  f"总耗时 {result['execution']['wall_seconds']:.2f} 秒。", '',
                  '这是固定参数的跨数据集诊断；15 查询集曾经过多轮修订和结果检查，不能当作全新未见测试集。', '',
                  '[完整指标与金标名次](comparison.json) · [完整排名](rankings.json) · [向量缓存](bge_vectors.npz) · '
                  '[相似度分组](audit_groups.json) · [运行配置](run_config.json)', ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--source-run', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--vectors', type=Path, help='Reuse a matching vector cache from this validation')
    parser.add_argument('--original-queries', type=Path, default=DOWNLOADS / 'single_pair_implicit_queries (1).jsonl')
    parser.add_argument('--original-memories', type=Path, default=DOWNLOADS / 'single_pair_memory_bank (1).jsonl')
    args = parser.parse_args(argv)
    started = time.perf_counter()
    source, output = args.source_run.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must be a new directory outside the original experiment')
    original = read_json(source / 'baselines/run_config.json')
    prior = read_json(source / 'ocsp/summary.json')
    stale_config = read_json(STALE / 'run_config.json')
    assert original['status'] == stale_config['status'] == 'completed'
    assert stale_config['k'] == stale_config['positive_seeds'] == 5
    assert 0.15 in stale_config['alphas'] and stale_config['alpha_beta_pairs'] == [list(pair) for pair in BIPOLAR]
    completion = read_json(source / 'baselines/completion.json')
    assert all(sha256_file(source / 'baselines' / name) == digest for name, digest in completion['artifacts_sha256'].items())
    protected = {path: sha256_file(path) for path in source.rglob('*') if path.is_file()}
    protected.update({path: sha256_file(path) for path in (ROOT / 'memory_retrieval').glob('*') if path.is_file()})
    for path in (args.original_queries, args.original_memories, ROOT / 'scripts/run_stale_graph_audit.py', STALE / 'run_config.json'):
        protected[path.resolve()] = sha256_file(path)
    assert sha256_file(ROOT / 'scripts/run_stale_graph_audit.py') == stale_config['script_sha256']
    dataset = load_dataset(args.dataset)
    assert len(dataset.queries) == 15 and len(dataset.memories) == 225 and list(dataset.groups()) == [None]
    assert all(len(row['conflict_old_memory_ids']) == 1 for row in dataset.gold)
    for path in dataset.paths.values():
        assert sha256_file(path) == original['dataset_sha256'][path.name]
        protected[path] = sha256_file(path)
    qids = [query['query_id'] for query in dataset.queries]
    probes = read_json(source / 'ocsp/per_query_details.json')
    assert [row['query_id'] for row in probes] == qids
    assert all(q['content'] == row['query_content'] and len(row['compatible_paraphrases']) == 2
               and len(row['violation_probes']) == 3 for q, row in zip(dataset.queries, probes))
    membership = audit_groups(dataset, args.original_queries.resolve(), args.original_memories.resolve())
    texts = {'memories': [doc.content.strip() for doc in dataset.memories],
             'queries': [q['content'].strip() for q in dataset.queries]}
    for name in ('compatible_paraphrases', 'violation_probes'):
        texts[name] = [text for row in probes for text in row[name]]
    signature = fingerprint({'texts': texts, 'query_ids': qids,
                             'memory_ids': [doc.memory_id for doc in dataset.memories], 'embedding': prior['models']['embedding']})
    output.mkdir(parents=True)
    config = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'dataset': str(args.dataset.resolve()), 'source_run': str(source), 'stale_protocol': str(STALE),
              'methods': METHODS, 'dataset_sha256': original['dataset_sha256'], 'query_ids': qids,
              'probe_details_sha256': sha256_file(source / 'ocsp/per_query_details.json'),
              'embedding': prior['models']['embedding'], 'original_llm': prior['models']['llm'],
              'k': 5, 'positive_seeds': 5, 'alpha': 0.15, 'bipolar_alpha_beta': BIPOLAR, 'qr_gamma': 0.65,
              'transition': stale_config['transition'], 'formula_diff': stale_config['formula_diff'],
              'formula_bipolar': stale_config['formula_bipolar'], 'tie_policy': stale_config['tie_policy'],
              'parameter_tuning': False, 'script_sha256': sha256_file(Path(__file__)),
              'graph_helper_sha256': sha256_file(ROOT / 'scripts/run_stale_graph_audit.py')}
    write_json(output / 'run_config.json', config)
    write_json(output / 'source_manifest.json', {str(path): digest for path, digest in protected.items()})
    write_json(output / 'audit_groups.json', membership)
    write_json(output / 'probes.json', [{key: row[key] for key in ('query_id', 'query_content', 'compatible_paraphrases', 'violation_probes')}
                                        for row in probes])
    if args.vectors:
        with np.load(args.vectors, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            assert metadata['signature'] == signature
            vectors = {name: saved[name].copy() for name in texts}
    else:
        vectors, identity, seconds = encode_groups('bge', texts, MODEL_ROOT)
        metadata = {'signature': signature, 'embedding': identity, 'encoding_seconds': seconds}
    assert metadata['embedding'] == prior['models']['embedding']
    for name, values in vectors.items():
        assert values.shape == (len(texts[name]), 768) and values.dtype == np.float32 and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1.0, rtol=0, atol=1e-6)
    np.savez_compressed(output / 'bge_vectors.npz', **vectors, metadata=np.asarray(json.dumps(metadata)))
    affinity, transition, _ = build_graph(vectors['memories'], dataset.memories, k=5)
    np.savez_compressed(output / 'graph.npz', A=affinity, P=transition,
                        memory_ids=np.asarray([doc.memory_id for doc in dataset.memories]))
    compatible, violations = np.split(vectors['compatible_paraphrases'], 15), np.split(vectors['violation_probes'], 15)
    previous_rankings = read_json(source / 'ocsp/rankings.json')
    previous_dense = {row['query_id']: row for row in read_jsonl(source / 'baselines/arms/dense.jsonl')}
    rankings, score_records, dense_delta = {name: {} for name in METHODS}, {}, 0.0
    for i, qid in enumerate(qids):
        scores, seeds, z_pos, z_neg = fixed_scores(vectors['memories'], vectors['queries'][i], compatible[i], violations[i],
                                                 transition, dataset.memories)
        for name, values in scores.items():
            rankings[name][qid] = [dataset.memories[j].memory_id for j in rank(values, dataset.memories)]
            assert len(set(rankings[name][qid])) == 225
        for name, old_name in (('dense', 'dense'), ('probes_only', 'probes_only_no_ortho'), ('ocsp_qr', 'ocsp_qr')):
            assert rankings[name][qid] == previous_rankings[old_name][qid], f'Control ranking mismatch: {name} {qid}'
        old_scores = {row['memory_id']: row['score'] for row in previous_dense[qid]['ranked_candidates']}
        difference = float(np.max(np.abs(scores['dense'] - [old_scores[doc.memory_id] for doc in dataset.memories])))
        assert difference <= 1e-6
        dense_delta = max(dense_delta, difference)
        score_records[qid] = {'seed_ids': [dataset.memories[j].memory_id for j in seeds], 'z_pos': z_pos.tolist(),
                              'z_neg': z_neg.tolist(), 'scores': {name: values.tolist() for name, values in scores.items()}}
    metrics = evaluate(rankings, dataset.gold)
    audit = cosine_audit(affinity, dataset.memories, membership)
    assert all(sha256_file(path) == digest for path, digest in protected.items())
    result = {'methods': METHODS, 'metrics': metrics, 'cosine_audit': audit,
              'execution': {'llm_calls': 0, 'sparsecl_inference_calls': 0, 'bge_reconstructed': args.vectors is None,
                            'wall_seconds': time.perf_counter() - started},
              'verification': {'control_rankings_match': True, 'dense_max_abs_score_difference': dense_delta,
                               'pytrec_eval_verified': True, 'original_files_unchanged': True}}
    write_json(output / 'comparison.json', result)
    write_json(output / 'cosine_audit.json', audit)
    write_json(output / 'rankings.json', rankings)
    write_json(output / 'scores.json', {'memory_ids': [doc.memory_id for doc in dataset.memories], 'queries': score_records})
    report = render_report(result)
    (output / 'report.zh.md').write_text(report, encoding='utf-8')
    config.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  vector_signature=signature, vector_sha256=sha256_file(output / 'bge_vectors.npz'))
    write_json(output / 'run_config.json', config)
    write_json(output / 'completion.json', {'status': 'completed', 'queries': 15, 'methods': 7,
               'artifacts_sha256': {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}})
    print(report, flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
