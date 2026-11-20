"""Compare frozen-probe scoring formulas offline on the saved STALE pilot."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'runs/experiments/stale-augmented-10scene-bge-20261119'
LAMBDAS = (0.25, 0.5, 0.75, 1.0)
TOP10_LAMBDAS = (0.5, 1.0)
SOFT_GAMMAS = (0.25, 0.5)
CUTOFFS = (3, 5, 10)

sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '4'
if find_spec('dotenv') is None:
    sys.path.append(str(ROOT / '.venv/Lib/site-packages'))
    import dotenv
    sys.path.pop()

import numpy as np
from memory_retrieval.baselines import rank
from memory_retrieval.data import fingerprint, read_jsonl, sha256_file, write_json
from memory_retrieval.dataset import load_dataset
from memory_retrieval.encoders import MODEL_ROOT, encode_groups
from memory_retrieval.evaluation import ranking_metrics
from memory_retrieval.ocsp import compute_subspace_ocsp_scores


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def method_labels():
    labels = {'probes_only': 'probes_only', 'ocsp_qr': 'ocsp_qr γ=0.65'}
    labels.update({f'one_sided_{weight:g}': f'单边压制 λ={weight:g}' for weight in LAMBDAS})
    labels.update({f'top10_{weight:g}': f'Top-10 内重排 λ={weight:g}' for weight in TOP10_LAMBDAS})
    labels.update({f'soft_projection_{weight:g}': f'未单位化软投影 γ={weight:g}' for weight in SOFT_GAMMAS})
    return labels


def formula_scores(bank, query_vector, compatible_vectors, violation_vectors):
    positive = bank @ query_vector
    negative = np.stack([bank @ probe for probe in violation_vectors], axis=1)
    scores = {
        'probes_only': np.max(negative, axis=1),
        'ocsp_qr': compute_subspace_ocsp_scores(
            query_vector, compatible_vectors, violation_vectors, bank, gamma=0.65)['ocsp_qr'],
    }
    excess = np.maximum(0.0, positive[:, None] - negative)
    for weight in LAMBDAS:
        scores[f'one_sided_{weight:g}'] = np.max(negative - weight * excess, axis=1)
    residuals = [probe - np.dot(probe, query_vector) * query_vector for probe in violation_vectors]
    residual_scores = np.stack([bank @ residual for residual in residuals], axis=1)
    for weight in SOFT_GAMMAS:
        scores[f'soft_projection_{weight:g}'] = np.max(
            np.maximum(0.0, negative) * (negative + weight * residual_scores), axis=1)
    return scores, positive, negative


def formula_orders(scores, documents):
    orders = {name: rank(values, documents) for name, values in scores.items()}
    original = orders['probes_only']
    for weight in TOP10_LAMBDAS:
        values = scores[f'one_sided_{weight:g}']
        head = sorted(original[:10], key=lambda index: (-float(values[index]), documents[index].memory_id))
        orders[f'top10_{weight:g}'] = head + original[10:]
        assert set(head) == set(original[:10])
    return orders


def load_vectors(texts, original_config, probe_path, args, output):
    signature = fingerprint({
        'inputs': original_config['dataset_sha256'], 'probes': sha256_file(probe_path),
        'embedding': original_config['models']['embedding'],
        'encoder_code': original_config['code_sha256']['encoders.py'],
    })
    if args.vectors:
        with np.load(args.vectors, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            assert metadata['signature'] == signature, 'Vector cache does not match this run'
            vectors = {name: saved[name].copy() for name in texts}
        origin = 'saved_offline_reconstruction'
        vector_path = args.vectors.resolve()
    else:
        vectors, identity, encoding_seconds = encode_groups('bge', texts, args.model_root)
        assert identity == original_config['models']['embedding'], 'BGE configuration differs from the original run'
        metadata = {'signature': signature, 'embedding': identity, 'encoding_seconds': encoding_seconds}
        vector_path = output / 'bge_vectors.npz'
        np.savez_compressed(vector_path, metadata=np.asarray(json.dumps(metadata)), **vectors)
        origin = 'local_bge_reconstruction_original_vectors_not_persisted'
    assert metadata['embedding'] == original_config['models']['embedding']
    for name, values in vectors.items():
        assert values.shape == (len(texts[name]), 768) and values.dtype == np.float32
        assert np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1.0, rtol=0, atol=1e-6)
    return vectors, {'origin': origin, 'path': str(vector_path), 'sha256': sha256_file(vector_path), **metadata}


def check_original(name, query_id, scores, order, documents, original_rows, diagnostics):
    previous = original_rows[name][query_id]['ranked_candidates']
    assert [documents[index].memory_id for index in order] == [row['memory_id'] for row in previous], (
        f'{name} ranking differs from the original run for {query_id}')
    expected = {row['memory_id']: row['score'] for row in previous}
    expected_scores = np.asarray([expected[document.memory_id] for document in documents])
    delta = float(np.max(np.abs(scores - expected_scores)))
    assert delta <= 1e-6, f'{name} score mismatch: {query_id}, max difference {delta}'
    diagnostics[name] = max(diagnostics.get(name, 0.0), delta)


def render_report(summary):
    checks = summary['verification']
    exact = all(value == 0 for value in checks['control_max_abs_score_difference'].values())
    equality = '逐项分数完全相同' if exact else '逐项分数最大误差不超过 1e-6'
    lines = [
        '# STALE 10 场景离线打分消融', '',
        '原有 5 个 T1、5 个 T2 场景，共 1,143 条分场景旧记忆。所有公式使用同一批已保存探针和同一份 BGE 向量。', '',
        f'原实验未将向量落盘，本次按原冻结 BGE、CPU FP32、批大小 4 和原文本顺序离线重建并保存。'
        f'Dense、probes_only、ocsp_qr 的全部排名与原结果一致，{equality}。', '',
        '旧 QR 对照沿用两条相容改述及 γ=0.65。单边压制和软投影只使用原句向量与违例探针。'
        'Top-10 内重排固定 probes_only 的前十候选，尾部保留原顺序，故 Top-10 命中数及原生目标 Top-10 命中数恒定。', '',
        'Top-K 为已标注目标的总命中数，NDCG@10 为每类 5 个查询的平均值。'
        '原生命中数只计 STALE 官方 M_old。', '',
    ]
    for scene_type, group in summary['groups'].items():
        lines.extend([
            f'## {scene_type}', '', f"共 {group['gold_pairs']} 个目标，其中官方原生目标 5 个。", '',
            '| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 原生 M_old Top-10 |',
            '|---|---:|---:|---:|---:|---:|',
        ])
        for name, label in summary['methods'].items():
            metric = group['methods'][name]
            hits = [str(metric['at_k'][str(cutoff)]['hits']) for cutoff in CUTOFFS]
            native = metric['at_k']['10']['recall_by_source']['stale_native']['hits']
            lines.append(f"| {label} | {' | '.join(hits)} | {metric['ndcg_at_10']:.4f} | {native}/5 |")
        lines.append('')
    lines.extend([
        '单边公式中的原句余弦是相似度代理。单边压制的逻辑效果需要由本次结果检验。'
        '本轮在开发切片扫描给定参数，表格保留全部参数结果。', '',
        f"本轮 LLM 请求 {summary['execution']['llm_calls']} 次，SparseCL 推理 {summary['execution']['sparsecl_inference_calls']} 次。"
        f"结果核对与评分总耗时 {summary['execution']['wall_seconds']:.1f} 秒，含需要时的本地 BGE 重建。", '',
        '[逐查询名次与分组指标](comparison.json) · [完整候选排名](rankings.json) · [运行配置](run_config.json)', '',
    ])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-root', type=Path, default=MODEL_ROOT)
    parser.add_argument('--vectors', type=Path, help='Reuse the vector cache produced by this script')
    args = parser.parse_args(argv)
    started = time.perf_counter()
    source_run = args.run_dir.resolve()
    output = (args.output or source_run / 'offline_ablation').resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    result_dir = source_run / 'results'
    original_config = read_json(result_dir / 'run_config.json')
    completion = read_json(result_dir / 'completion.json')
    assert original_config['status'] == completion['status'] == 'completed'
    protected = {result_dir / name: digest for name, digest in completion['artifacts_sha256'].items()}
    protected.update({source_run / 'inputs' / name: digest for name, digest in original_config['dataset_sha256'].items()})
    protected.update({ROOT / 'memory_retrieval' / name: digest for name, digest in original_config['code_sha256'].items()})
    protected[ROOT / 'memory_retrieval/model_lock.json'] = original_config['model_lock_sha256']
    assert all(sha256_file(path) == digest for path, digest in protected.items())
    dataset = load_dataset(source_run / 'inputs')
    query_ids = [query['query_id'] for query in dataset.queries]
    probe_path = result_dir / 'per_query_details.json'
    probes = read_json(probe_path)
    assert query_ids == original_config['query_ids'] == [row['query_id'] for row in probes]
    assert all(query['content'] == row['query_content'] for query, row in zip(dataset.queries, probes))
    assert all(len(row['compatible_paraphrases']) == 2 and len(row['violation_probes']) == 3 for row in probes)
    types = {query['query_id']: query['source_ref']['type'] for query in dataset.queries}
    assert Counter(types.values()) == {'T1': 5, 'T2': 5}
    texts = {'memories': [memory.content.strip() for memory in dataset.memories],
             'queries': [query['content'].strip() for query in dataset.queries]}
    for name in ('compatible_paraphrases', 'violation_probes'):
        texts[name] = [text for row in probes for text in row[name]]
    labels = method_labels()
    output.mkdir(parents=True)
    run_config = {
        'status': 'running', 'source_run': str(source_run), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'methods': labels, 'lambda_values': LAMBDAS, 'top10_lambda_values': TOP10_LAMBDAS,
        'soft_gamma_values': SOFT_GAMMAS, 'old_qr_gamma': 0.65,
        'dataset_sha256': original_config['dataset_sha256'], 'probe_details_sha256': sha256_file(probe_path),
        'script_sha256': sha256_file(Path(__file__)), 'code_sha256': original_config['code_sha256'],
        'tie_policy': 'score descending then memory_id ascending',
        'top10_policy': 'Reorder the probes_only first ten candidates; preserve the remaining tail',
    }
    write_json(output / 'run_config.json', run_config)
    vectors, provenance = load_vectors(texts, original_config, probe_path, args, output)
    run_config['vectors'] = provenance
    write_json(output / 'run_config.json', run_config)
    compatible = np.split(vectors['compatible_paraphrases'], len(query_ids))
    violations = np.split(vectors['violation_probes'], len(query_ids))
    original_rows = {name: {row['query_id']: row for row in read_jsonl(result_dir / 'arms' / f'{name}.jsonl')}
                     for name in ('dense', 'probes_only_no_ortho', 'ocsp_qr')}
    rankings = {name: {} for name in labels}
    control_differences = {}
    for memory_indices, query_indices in dataset.groups().values():
        documents = [dataset.memories[index] for index in memory_indices]
        bank = vectors['memories'][memory_indices]
        for query_index in query_indices:
            qid = query_ids[query_index]
            scores, positive, negative = formula_scores(
                bank, vectors['queries'][query_index], compatible[query_index], violations[query_index])
            orders = formula_orders(scores, documents)
            check_original('dense', qid, positive, rank(positive, documents), documents, original_rows, control_differences)
            for method, original in (('probes_only', 'probes_only_no_ortho'), ('ocsp_qr', 'ocsp_qr')):
                check_original(original, qid, scores[method], orders[method], documents, original_rows, control_differences)
            for weight in LAMBDAS:
                maximum = scores['probes_only']
                equivalent = maximum - weight * np.maximum(0.0, positive - maximum)
                assert np.allclose(scores[f'one_sided_{weight:g}'], equivalent, rtol=0, atol=1e-7)
                assert np.all(scores[f'one_sided_{weight:g}'] <= maximum)
            for name in labels:
                ids = [documents[index].memory_id for index in orders[name]]
                assert len(ids) == len(set(ids)) == len(documents)
                rankings[name][qid] = ids
    groups = {}
    for scene_type in ('T1', 'T2'):
        gold = [row for row in dataset.gold if types[row['query_id']] == scene_type]
        metrics = {name: ranking_metrics({row['query_id']: ranking[row['query_id']] for row in gold}, gold, CUTOFFS)
                   for name, ranking in rankings.items()}
        groups[scene_type] = {'queries': len(gold), 'gold_pairs': sum(len(row['conflict_old_memory_ids']) for row in gold),
                              'methods': metrics}
        for weight in TOP10_LAMBDAS:
            assert metrics[f'top10_{weight:g}']['at_k']['10'] == metrics['probes_only']['at_k']['10']
    assert all(sha256_file(path) == digest for path, digest in protected.items())
    summary = {
        'source_run': str(source_run), 'methods': labels, 'groups': groups,
        'vectors': provenance,
        'execution': {'llm_calls': 0, 'sparsecl_inference_calls': 0, 'bge_reconstructed': args.vectors is None,
                      'wall_seconds': time.perf_counter() - started},
        'verification': {'control_max_abs_score_difference': control_differences, 'all_control_rankings_match': True,
                         'top10_candidate_sets_preserved': True, 'original_artifacts_unchanged': True},
    }
    write_json(output / 'comparison.json', summary)
    write_json(output / 'rankings.json', rankings)
    markdown = render_report(summary)
    (output / 'report.zh.md').write_text(markdown, encoding='utf-8')
    run_config.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat(), execution=summary['execution'])
    write_json(output / 'run_config.json', run_config)
    write_json(output / 'completion.json', {
        'status': 'completed', 'queries': len(query_ids), 'methods': len(labels),
        'artifacts_sha256': {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()},
    })
    print(markdown, flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
