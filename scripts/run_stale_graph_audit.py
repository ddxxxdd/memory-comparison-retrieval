"""Offline connectivity audit and one-step diffusion with saved STALE BGE vectors."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/experiments/stale-augmented-10scene-bge-20261119'
PROBE_RUN = ROOT / 'runs/experiments/stale-augmented-10scene-4probe-bge-20261119'
OUTPUT = ROOT / 'runs/experiments/stale-augmented-10scene-graph-bge-20261120'
ALPHAS = (0.15, 0.30, 0.45)
BIPOLAR = ((0.2, 0.3), (0.3, 0.3), (0.3, 0.5))
PAIR_LABELS = {
    'gold_gold': '金标—金标',
    'native_other_gold': '原生 M_old—其他金标',
    'gold_compatible': '金标—compatible',
    'gold_history': '金标—非金标 history',
}

sys.path.insert(0, str(ROOT))
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '4'

import numpy as np
import pytrec_eval
from memory_retrieval.baselines import rank
from memory_retrieval.data import read_jsonl, sha256_file, write_json
from memory_retrieval.dataset import load_dataset
from memory_retrieval.evaluation import ranking_metrics


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def method_labels():
    labels = {}
    for prefix in ('old', 'new'):
        labels[f'{prefix}_probes_only'] = f'{prefix} probes_only'
        for alpha in ALPHAS:
            labels[f'{prefix}_diff_{alpha:g}'] = f'{prefix} 单步 α={alpha:g}'
        for alpha, beta in BIPOLAR:
            labels[f'{prefix}_bipolar_{alpha:g}_{beta:g}'] = f'{prefix} 双极 α={alpha:g}, β={beta:g}'
    return labels


def build_graph(bank, documents, k=5):
    affinity = bank @ bank.T
    np.fill_diagonal(affinity, 0.0)
    edges = np.zeros_like(affinity)
    neighbors = []
    for i, row in enumerate(affinity):
        selected = [j for j in rank(row, documents) if j != i and row[j] > 0.0][:k]
        neighbors.append(selected)
        edges[i, selected] = row[selected]
    totals = edges.sum(axis=1, keepdims=True)
    transition = np.divide(edges, totals, out=np.zeros_like(edges), where=totals > 0)
    assert np.all(np.diag(transition) == 0) and np.all(transition >= 0)
    assert np.all(np.count_nonzero(transition, axis=1) <= k)
    assert np.allclose(transition.sum(axis=1), (totals[:, 0] > 0).astype(float), rtol=0, atol=2e-7)
    return affinity, transition, neighbors


def diffusion_scores(initial, positive, transition, documents):
    seeds = rank(initial, documents)[:5]
    z_pos = np.zeros_like(initial)
    z_pos[seeds] = initial[seeds]
    z_neg = np.maximum(0.0, positive - initial)
    propagated_pos, propagated_neg = transition @ z_pos, transition @ z_neg
    scores = {'probes_only': initial}
    for alpha in ALPHAS:
        scores[f'diff_{alpha:g}'] = (1 - alpha) * initial + alpha * propagated_pos
    for alpha, beta in BIPOLAR:
        scores[f'bipolar_{alpha:g}_{beta:g}'] = (
            (1 - alpha) * initial + alpha * propagated_pos - beta * (z_neg + alpha * propagated_neg))
    assert all(np.isfinite(values).all() for values in scores.values())
    return scores, seeds, z_pos, z_neg


def pair_stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    return {'pairs': int(values.size), 'sum': float(values.sum()),
            'mean': float(values.mean()) if values.size else None}


def audit_scene(affinity, neighbors, documents, query, gold):
    indices = {doc.memory_id: i for i, doc in enumerate(documents)}
    target_ids = set(gold['conflict_old_memory_ids'])
    targets = [i for i, doc in enumerate(documents) if doc.memory_id in target_ids]
    native_ids = [mid for mid, detail in gold['target_details'].items() if detail['source'] == 'stale_native']
    assert len(native_ids) == 1
    native = indices[native_ids[0]]
    others = [i for i in targets if i != native]
    compatible = [i for i, doc in enumerate(documents)
                  if doc.source_ref.get('record_type') == 'synthetic'
                  and doc.source_ref.get('augmentation_group') == 'compatible']
    history = [i for i, doc in enumerate(documents)
               if doc.source_ref.get('record_type') == 'history' and doc.memory_id not in target_ids]
    assert not set(compatible) & set(targets)
    within_gold = affinity[np.ix_(targets, targets)]
    pair_groups = {
        'gold_gold': pair_stats(within_gold[np.triu_indices(len(targets), k=1)]),
        'native_other_gold': pair_stats(affinity[native, others]),
        'gold_compatible': pair_stats(affinity[np.ix_(targets, compatible)]),
        'gold_history': pair_stats(affinity[np.ix_(targets, history)]),
    }
    gold_neighbor_counts = {documents[i].memory_id: sum(j in targets for j in neighbors[i]) for i in targets}
    unjudged = set(gold.get('unjudged_old_memory_ids', []))
    return {
        'query_id': query['query_id'], 'scenario_id': query['scenario_id'], 'type': query['source_ref']['type'],
        'memories': len(documents), 'gold_count': len(targets), 'compatible_count': len(compatible),
        'history_count': len(history), 'history_unjudged_count': sum(documents[i].memory_id in unjudged for i in history),
        'pairs': pair_groups, 'native_memory_id': native_ids[0],
        'native_other_gold_max': float(np.max(affinity[native, others])),
        'native_top5_gold_count': gold_neighbor_counts[native_ids[0]],
        'native_top5_has_other_gold': gold_neighbor_counts[native_ids[0]] > 0,
        'gold_top5_other_gold_counts': gold_neighbor_counts,
        'gold_top5_other_gold_mean': mean(gold_neighbor_counts.values()),
        'native_top5_neighbors': [{'memory_id': documents[i].memory_id, 'cosine': float(affinity[native, i]),
                                   'is_gold': documents[i].memory_id in target_ids} for i in neighbors[native]],
    }


def summarize_audit(scenes):
    groups = {}
    for kind in ('T1', 'T2'):
        selected = [scene for scene in scenes if scene['type'] == kind]
        pairs = {}
        for name in PAIR_LABELS:
            count = sum(scene['pairs'][name]['pairs'] for scene in selected)
            total = sum(scene['pairs'][name]['sum'] for scene in selected)
            pairs[name] = {'pairs': count, 'sum': total, 'mean': total / count if count else None}
        counts = [n for scene in selected for n in scene['gold_top5_other_gold_counts'].values()]
        hits = sum(scene['native_top5_has_other_gold'] for scene in selected)
        groups[kind] = {'scenarios': len(selected), 'gold_count': len(counts), 'pairs': pairs,
                        'native_with_gold_neighbor_count': hits, 'native_with_gold_neighbor_ratio': hits / len(selected),
                        'gold_top5_other_gold_mean': mean(counts),
                        'history_count': sum(scene['history_count'] for scene in selected),
                        'history_unjudged_count': sum(scene['history_unjudged_count'] for scene in selected)}
    return groups


def evaluate(rankings, dataset):
    types = {q['query_id']: q['source_ref']['type'] for q in dataset.queries}
    qrels = {row['query_id']: {mid: 1 for mid in row['conflict_old_memory_ids']} for row in dataset.gold}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10'})
    groups = {kind: {'gold_pairs': count, 'methods': {}} for kind, count in (('T1', 30), ('T2', 32))}
    for method, ranking in rankings.items():
        run = {qid: {mid: float(len(ids) - i) for i, mid in enumerate(ids)} for qid, ids in ranking.items()}
        checked = evaluator.evaluate(run)
        for kind, group in groups.items():
            gold = [row for row in dataset.gold if types[row['query_id']] == kind]
            selected = {row['query_id']: ranking[row['query_id']] for row in gold}
            metrics = ranking_metrics(selected, gold, (3, 5, 10))
            ndcg = mean(checked[qid]['ndcg_cut_10'] for qid in selected)
            assert abs(ndcg - metrics['ndcg_at_10']) < 1e-12
            assert metrics['at_k']['10']['gold_pairs'] == group['gold_pairs']
            metrics['pytrec_eval_ndcg_at_10'] = ndcg
            group['methods'][method] = metrics
    return groups


def render_report(result):
    lines = ['# STALE 10 场景图连通性与单步扩散离线验证', '',
             '复用种子 20261119 的 5 T1 + 5 T2 Dev 场景，共 1,143 条分场景旧记忆。'
             '直接读取已保存的 BGE 向量，旧探针 30 条、新探针 40 条；本次不加载任何模型。', '',
             '图为 A=X@X.T，去掉自环，每行保留最多 5 条正相似边，再按行归一化。'
             '严格使用 P@z，不做对称化、转置或额外归一化。正种子是各自 probes_only 的前 5 个节点，保留原分数。'
             '扩散在整个场景库排序，未叠加上一轮 Top-10 重排。', '',
             '标签只用于图体检和结果评估，不参与构图、选种子或打分。', '',
             '## 图连通性体检', '',
             '均值按同类场景内全部有效配对汇总，金标—金标取无序非自身配对；不计算跨场景配对。'
             'history 指所有未列入金标的 history 条目，包含待定项，并不意味着这些条目都已证实相容。', '',
             '| 配对类别 | T1 平均余弦（配对数） | T2 平均余弦（配对数） |', '|---|---:|---:|']
    for name, label in PAIR_LABELS.items():
        cells = [f"{result['audit']['groups'][kind]['pairs'][name]['mean']:.4f}"
                 f"（{result['audit']['groups'][kind]['pairs'][name]['pairs']}）" for kind in ('T1', 'T2')]
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines.extend(['', '| 类型 | M_old Top-5 至少连到 1 个其他金标 | 每个金标 Top-5 平均其他金标数 | history 中待定项 |',
                  '|---|---:|---:|---:|'])
    for kind, group in result['audit']['groups'].items():
        lines.append(f"| {kind} | {group['native_with_gold_neighbor_count']}/5"
                     f"（{group['native_with_gold_neighbor_ratio']:.0%}） | {group['gold_top5_other_gold_mean']:.4f}"
                     f" | {group['history_unjudged_count']}/{group['history_count']} |")
    lines.extend(['', '场景简称取原 scenario_id 的前 8 位，完整 ID 与邻居列表见 graph_audit.json。', '',
                  '| 类型 | 场景 | M_old—其他金标均值 | 最大值 | M_old Top-5 其他金标数 | 全金标平均其他金标邻居数 |',
                  '|---|---|---:|---:|---:|---:|'])
    for scene in result['audit']['scenes']:
        lines.append(f"| {scene['type']} | {scene['scenario_id'][:8]} | {scene['pairs']['native_other_gold']['mean']:.4f}"
                     f" | {scene['native_other_gold_max']:.4f} | {scene['native_top5_gold_count']}"
                     f" | {scene['gold_top5_other_gold_mean']:.4f} |")
    for kind, group in result['groups'].items():
        lines.extend(['', f"## {kind} 扩散结果（{group['gold_pairs']} 个目标）", '',
                      '| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 原生 M_old Top-10 |',
                      '|---|---:|---:|---:|---:|---:|'])
        for name, label in result['methods'].items():
            metric = group['methods'][name]
            hits = [str(metric['at_k'][str(k)]['hits']) for k in (3, 5, 10)]
            native = metric['at_k']['10']['recall_by_source']['stale_native']['hits']
            lines.append(f"| {label} | {' | '.join(hits)} | {metric['ndcg_at_10']:.4f} | {native}/5 |")
    lines.extend(['', 'NDCG@10 为每类 5 个查询的平均值，所有方法均经 pytrec_eval 复核。'
                  '两版 probes_only 的完整排名和分数与上一轮一致，原实验及主代码哈希保持不变。', '',
                  f"本次 LLM 调用、Embedding 推理、SparseCL 推理均为 0；耗时 {result['execution']['wall_seconds']:.3f} 秒。"
                  '本轮只报告指定参数在同一 Dev 切片上的结果。', '',
                  '[图体检明细](graph_audit.json) · [完整指标](comparison.json) · [完整排名](rankings.json) · '
                  '[分数与种子](scores.json) · [运行配置](run_config.json)', ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=SOURCE)
    parser.add_argument('--probe-run', type=Path, default=PROBE_RUN)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    source, probe_run, output = args.source_run.resolve(), args.probe_run.resolve(), args.output.resolve()
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, probe_run)):
        raise ValueError('Output must be a new directory outside both existing experiments')
    original = read_json(source / 'results/run_config.json')
    probe_config = read_json(probe_run / 'run_config.json')
    assert original['status'] == probe_config['status'] == 'completed'
    assert read_json(source / 'selection.json')['seed'] == probe_config['seed_sampling'] == 20261119
    for folder in (source / 'results', source / 'offline_ablation', probe_run):
        completion = read_json(folder / 'completion.json')
        assert all(sha256_file(folder / name) == digest for name, digest in completion['artifacts_sha256'].items())
    protected = {path: sha256_file(path) for base in (source, probe_run) for path in base.rglob('*') if path.is_file()}
    protected.update({ROOT / 'memory_retrieval' / name: digest for name, digest in original['code_sha256'].items()})
    protected[ROOT / 'memory_retrieval/model_lock.json'] = original['model_lock_sha256']
    assert all(sha256_file(path) == digest for path, digest in protected.items())
    assert original['dataset_sha256'] == probe_config['dataset_sha256']
    assert all(sha256_file(source / 'inputs' / name) == digest for name, digest in original['dataset_sha256'].items())
    dataset = load_dataset(source / 'inputs')
    qids = [q['query_id'] for q in dataset.queries]
    assert qids == original['query_ids'] == probe_config['query_ids']
    assert len(dataset.memories) == 1143 and Counter(q['source_ref']['type'] for q in dataset.queries) == {'T1': 5, 'T2': 5}
    old_texts, new_texts = read_json(source / 'results/per_query_details.json'), read_json(probe_run / 'new_probes.json')
    for texts, count in ((old_texts, 3), (new_texts, 4)):
        assert [row['query_id'] for row in texts] == qids
        assert all(q['content'] == row['query_content'] and len(row['violation_probes']) == count
                   for q, row in zip(dataset.queries, texts))
    vector_path = probe_run / 'bge_vectors.npz'
    with np.load(vector_path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        vectors = {name: saved[name].copy() for name in ('memories', 'queries', 'old_violation_probes', 'new_violation_probes')}
    assert metadata['query_ids'] == qids and metadata['embedding'] == probe_config['new_embedding']
    assert metadata['source_vectors_sha256'] == sha256_file(source / 'offline_ablation/bge_vectors.npz')
    for name, count in (('memories', 1143), ('queries', 10), ('old_violation_probes', 30), ('new_violation_probes', 40)):
        values = vectors[name]
        assert values.shape == (count, 768) and values.dtype == np.float32 and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1.0, rtol=0, atol=1e-6)
    previous_rankings, previous_scores = read_json(probe_run / 'rankings.json'), read_json(probe_run / 'scores.json')
    labels = method_labels()
    output.mkdir(parents=True)
    config = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_run': str(source), 'probe_run': str(probe_run), 'query_ids': qids, 'seed_sampling': 20261119,
              'vector_sha256': sha256_file(vector_path), 'embedding': original['models']['embedding'],
              'dataset_sha256': original['dataset_sha256'], 'k': 5, 'positive_seeds': 5,
              'alphas': ALPHAS, 'alpha_beta_pairs': BIPOLAR, 'methods': labels,
              'formula_diff': '(1-alpha)*r0 + alpha*(P@z_pos)',
              'formula_bipolar': '(1-alpha)*r0 + alpha*(P@z_pos) - beta*(z_neg + alpha*(P@z_neg))',
              'z_neg': 'maximum(0, X@v_q-r0)', 'transition': 'row-normalized positive directed top-5 cosine edges; zero diagonal',
              'zero_degree_policy': 'zero row', 'tie_policy': 'score descending then memory_id ascending',
              'pair_aggregation': 'pooled within-type same-scenario pairs; unordered distinct gold-gold pairs',
              'history_policy': 'record_type history and not a labeled gold; includes unjudged items',
              'script_sha256': sha256_file(Path(__file__))}
    write_json(output / 'run_config.json', config)
    write_json(output / 'source_manifest.json', {str(path): digest for path, digest in protected.items()})
    rankings, score_records, graphs, scene_audits = {name: {} for name in labels}, {}, {}, []
    gold_by_query = {row['query_id']: row for row in dataset.gold}
    probe_vectors = {prefix: np.split(vectors[f'{prefix}_violation_probes'], 10) for prefix in ('old', 'new')}
    control_deltas = {'old': 0.0, 'new': 0.0}
    for scenario_id, (mi, qi) in dataset.groups().items():
        assert len(qi) == 1
        query_index = qi[0]
        query, qid = dataset.queries[query_index], qids[query_index]
        documents = [dataset.memories[i] for i in mi]
        bank = vectors['memories'][mi]
        affinity, transition, neighbors = build_graph(bank, documents)
        graphs[f'{scenario_id}_A'], graphs[f'{scenario_id}_P'] = affinity, transition
        scene_audits.append(audit_scene(affinity, neighbors, documents, query, gold_by_query[qid]))
        positive = bank @ vectors['queries'][query_index]
        ids = [doc.memory_id for doc in documents]
        score_records[qid] = {'scenario_id': scenario_id, 'memory_ids': ids, 's_plus': positive.tolist(), 'probes': {}}
        assert ids == previous_scores[qid]['memory_ids']
        for prefix in ('old', 'new'):
            initial = np.max(np.stack([bank @ probe for probe in probe_vectors[prefix][query_index]], axis=1), axis=1)
            scores, seeds, z_pos, z_neg = diffusion_scores(initial, positive, transition, documents)
            expected = np.asarray(previous_scores[qid][prefix]['probe_score'])
            delta = float(np.max(np.abs(initial - expected)))
            assert delta == 0.0
            control_deltas[prefix] = max(control_deltas[prefix], delta)
            score_records[qid]['probes'][prefix] = {'seed_ids': [ids[i] for i in seeds],
                'z_pos': z_pos.tolist(), 'z_neg': z_neg.tolist(), 'scores': {name: values.tolist() for name, values in scores.items()}}
            for suffix, values in scores.items():
                name = f'{prefix}_{suffix}'
                ordered = [ids[i] for i in rank(values, documents)]
                assert len(set(ordered)) == len(ids)
                rankings[name][qid] = ordered
            assert rankings[f'{prefix}_probes_only'][qid] == previous_rankings[f'{prefix}_probes_only'][qid]
    scene_audits.sort(key=lambda scene: qids.index(scene['query_id']))
    audit = {'aggregation': config['pair_aggregation'], 'history_policy': config['history_policy'],
             'groups': summarize_audit(scene_audits), 'scenes': scene_audits}
    groups = evaluate(rankings, dataset)
    assert all(sha256_file(path) == digest for path, digest in protected.items())
    result = {'methods': labels, 'audit': audit, 'groups': groups,
              'execution': {'llm_calls': 0, 'embedding_inference_calls': 0, 'sparsecl_inference_calls': 0,
                            'wall_seconds': time.perf_counter() - started},
              'verification': {'baseline_max_abs_score_differences': control_deltas, 'baseline_rankings_match': True,
                               'pytrec_eval_verified': True, 'source_files_unchanged': True}}
    np.savez_compressed(output / 'graphs.npz', **graphs)
    write_json(output / 'graph_audit.json', audit)
    write_json(output / 'comparison.json', result)
    write_json(output / 'scores.json', score_records)
    write_json(output / 'rankings.json', rankings)
    report = render_report(result)
    (output / 'report.zh.md').write_text(report, encoding='utf-8')
    config.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'run_config.json', config)
    write_json(output / 'completion.json', {'status': 'completed', 'queries': 10, 'methods': len(labels),
               'artifacts_sha256': {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}})
    print(report, flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
