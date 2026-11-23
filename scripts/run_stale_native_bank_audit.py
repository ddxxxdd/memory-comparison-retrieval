"""Offline native-bank comparison and augmented-bank module accounting; no API access."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NETWORK_ATTEMPTS = []


def offline_guard(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        NETWORK_ATTEMPTS.append(event)
        raise RuntimeError("Network access prohibited in this offline experiment")


sys.addaudithook(offline_guard)
import numpy as np
import pytrec_eval
from scipy.stats import wilcoxon
from memory_retrieval.bipolar_graph import (ANCHORED_PROMPT_SHA256, build_knn_transition,
                                           compute_bipolar_diffusion_scores)
from memory_retrieval.data import read_jsonl, sha256_file, write_json, write_jsonl
from memory_retrieval.dataset import load_dataset
from memory_retrieval.evaluation import target_ids
from memory_retrieval.experiment import ranked_row, score_original_text, sparsecl_pools
from memory_retrieval.encoders import MODEL_LOCK
from memory_retrieval.sparsecl import score_candidates, fused_order

BGE_SOURCE = ROOT / "runs/experiments/stale-unseen-60scene-bipolar-seed20261122"
QWEN_SOURCE = ROOT / "runs/experiments/stale-unseen-60scene-qwen-emb-seed20261122"
BASE = ROOT / "data/stale_scenarios/base"
OUTPUT = ROOT / "runs/experiments/stale-unseen-60scene-native-bank-seed20261122"
REPORT = ROOT / "docs/stale_unseen_60scene_native_bank_and_module_audit_seed20261122.md"
PROMPT = "a38b198018c62846780914f1ef90fe2461ee1554c421d42996f54525f93217bf"
PROBES, POSITIVE, FULL = "anchored_probes_only", "anchored_pos_diffusion", "anchored_bipolar_diffusion"
ARMS = {"BGE": ["bm25", "dense", "hybrid", "sparsecl_msmarco", PROBES, POSITIVE, FULL],
        "Qwen": ["dense", "hybrid", PROBES, POSITIVE, FULL]}
NAMES = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid", "sparsecl_msmarco": "SparseCL MS MARCO",
         PROBES: "仅探针", POSITIVE: "正向扩散", FULL: "完整方法"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def check_hashes(root, hashes):
    for name, expected in hashes.items():
        assert sha256_file(root / name) == expected, name


def select_native_rows(augmented, base_rows, selected_sids):
    native, removed = [], []
    for row in augmented:
        assert row["scenario_id"] in selected_sids
        kind = row["source_ref"].get("record_type")
        if kind == "synthetic":
            removed.append(row)
        elif kind in {"history", "M_old"}:
            assert row["source_ref"].get("dataset") == "STALE"
            assert "augmentation_group" not in row["source_ref"]
            native.append(row)
        else:
            raise ValueError("Unknown record provenance: " + str(kind))
    expected = {m["memory_id"]: m for m in base_rows if m["scenario_id"] in selected_sids}
    assert {m["memory_id"]: m for m in native} == expected, "Filtered bank differs from base records"
    counts = Counter(m["scenario_id"] for m in native if m["source_ref"]["record_type"] == "M_old")
    assert counts == {sid: 1 for sid in selected_sids}
    return native, removed


def groups_for(queries):
    return {name: [q["query_id"] for q in queries if name == "all" or q["source_ref"]["type"] == name]
            for name in ("T1", "T2", "all")}


def rank_lists(rows):
    return {qid: [r["memory_id"] for r in row["ranked_candidates"]] for qid, row in rows.items()}


def trec_run(rankings):
    return {qid: {mid: float(len(ids)-i) for i, mid in enumerate(ids)} for qid, ids in rankings.items()}


def native_metrics(rankings, native_ids, queries):
    qrels = {qid: {mid: 1} for qid, mid in native_ids.items()}
    # Recreate evaluators: the installed trec_eval wrapper retains cutoff state.
    result = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recip_rank", "recall_1",
                                                  "recall_3", "recall_5", "recall_10"}).evaluate(trec_run(rankings))
    positions = {qid: ids.index(native_ids[qid])+1 for qid, ids in rankings.items()}
    errors = {"hits": 0, "recall": 0., "mean_rank": 0., "mrr": 0., "ndcg": 0.}
    per_query = {}
    for qid, position in positions.items():
        rr = result[qid]["recip_rank"]
        assert position == int(round(1/rr))
        expected_ndcg = 1/math.log2(position+1) if position <= 10 else 0.
        assert abs(expected_ndcg-result[qid]["ndcg_cut_10"]) < 1e-12
        per_query[qid] = {"rank": position, "reciprocal_rank": 1/position,
                          "ndcg_at_10": result[qid]["ndcg_cut_10"]}
    groups = {}
    for name, qids in groups_for(queries).items():
        counts = {str(k): sum(positions[qid] <= k for qid in qids) for k in (1,3,5,10)}
        group = {"scenes": len(qids), "at_k": {str(k): {"hits": counts[str(k)], "recall": counts[str(k)]/len(qids)}
                                                    for k in (1,3,5,10)},
                 "mean_rank": float(np.mean([positions[qid] for qid in qids])),
                 "mrr": float(np.mean([1/positions[qid] for qid in qids])),
                 "ndcg_at_10": float(np.mean([1/math.log2(positions[qid]+1) if positions[qid]<=10 else 0. for qid in qids]))}
        errors["mean_rank"] = max(errors["mean_rank"], abs(group["mean_rank"]-float(np.mean([round(1/result[qid]["recip_rank"]) for qid in qids]))))
        errors["mrr"] = max(errors["mrr"], abs(group["mrr"]-float(np.mean([result[qid]["recip_rank"] for qid in qids]))))
        errors["ndcg"] = max(errors["ndcg"], abs(group["ndcg_at_10"]-float(np.mean([result[qid]["ndcg_cut_10"] for qid in qids]))))
        for k in (1,3,5,10):
            verified = sum(round(result[qid][f"recall_{k}"]) for qid in qids)
            errors["hits"] = max(errors["hits"], abs(verified-counts[str(k)]))
            errors["recall"] = max(errors["recall"], abs(verified/len(qids)-group["at_k"][str(k)]["recall"]))
        groups[name] = group
    assert max(errors.values()) < 1e-12
    return groups, per_query, errors


def paired_statistics(queries, left, right):
    output = {}
    for group, qids in groups_for(queries).items():
        diff = np.array([left[qid]-right[qid] for qid in qids])
        rounded = np.round(diff, 12)
        test = wilcoxon(rounded, zero_method="pratt", alternative="two-sided", method="approx", correction=False) if np.any(rounded) else None
        rng = np.random.default_rng(20261122)
        if group == "all":
            qtype = {q["query_id"]: q["source_ref"]["type"] for q in queries}
            strata = [np.array([i for i, qid in enumerate(qids) if qtype[qid] == kind]) for kind in ("T1", "T2")]
            draws = np.concatenate([rng.choice(idx, size=(20000,len(idx)),replace=True) for idx in strata],axis=1)
        else:
            draws = rng.integers(0,len(qids),size=(20000,len(qids)))
        ci = np.quantile(diff[draws].mean(axis=1), [.025,.975])
        output[group] = {"scenes": len(qids), "mean_ndcg_difference": float(diff.mean()),
            "wilcoxon_statistic": float(test.statistic) if test else 0., "p_two_sided": float(test.pvalue) if test else 1.,
            "bootstrap_ci95": list(map(float,ci)), "wins": int((rounded>0).sum()),
            "ties": int((rounded==0).sum()), "losses": int((rounded<0).sum()),
            "per_query_difference": dict(zip(qids,map(float,diff)))}
    return output


def transition_counts(before, after, targets):
    details = []
    for qid, ids in targets.items():
        old, new = {mid:i for i,mid in enumerate(before[qid],1)}, {mid:i for i,mid in enumerate(after[qid],1)}
        for mid in sorted(ids):
            a,b = old[mid],new[mid]
            details.append({"query_id": qid,"memory_id":mid,"before_rank":a,"after_rank":b,
                            "rank_change":a-b,"entered_top10":a>10>=b,"exited_top10":a<=10<b,
                            "entered_top3":a>3>=b,"exited_top3":a<=3<b})
    return details


def summarize_transitions(details, before_ndcg, after_ndcg, queries):
    result = {}
    for group,qids in groups_for(queries).items():
        rows = [r for r in details if r["query_id"] in qids]
        delta = np.round([after_ndcg[qid]-before_ndcg[qid] for qid in qids],12)
        summary = {"targets":len(rows),"top10_before":sum(r["before_rank"]<=10 for r in rows),
            "top10_after":sum(r["after_rank"]<=10 for r in rows),"top10_gained":sum(r["entered_top10"] for r in rows),
            "top10_lost":sum(r["exited_top10"] for r in rows),"top3_before":sum(r["before_rank"]<=3 for r in rows),
            "top3_after":sum(r["after_rank"]<=3 for r in rows),"top3_gained":sum(r["entered_top3"] for r in rows),
            "top3_lost":sum(r["exited_top3"] for r in rows),"rank_improved":sum(r["rank_change"]>0 for r in rows),
            "rank_unchanged":sum(r["rank_change"]==0 for r in rows),"rank_worsened":sum(r["rank_change"]<0 for r in rows),
            "ndcg_wins":int((delta>0).sum()),"ndcg_ties":int((delta==0).sum()),"ndcg_losses":int((delta<0).sum()),
            "ndcg_before":float(np.mean([before_ndcg[qid] for qid in qids])),
            "ndcg_after":float(np.mean([after_ndcg[qid] for qid in qids]))}
        assert summary["top10_after"]-summary["top10_before"]==summary["top10_gained"]-summary["top10_lost"]
        assert summary["top3_after"]-summary["top3_before"]==summary["top3_gained"]-summary["top3_lost"]
        result[group] = summary
    return result


def load_source_vectors(source, model):
    if model == "Qwen":
        folder = source/"vectors"
        info = read(folder/"identity.json")
        names = {"memories":"memories","queries":"queries","probes":"violation_probes"}
    else:
        folder = source/"encoders"/("bge" if model=="BGE" else "msmarco")
        info = read(folder/"identity.json")
        names = {"memories":"memories","queries":"queries"}
        if model=="BGE": names["probes"] = "anchored_violation_probes"
    arrays = {}
    for alias,name in names.items():
        path = folder/(name+".npy")
        assert sha256_file(path)==info["vector_sha256"][name]
        arrays[alias]=np.load(path,mmap_mode="r")
    if "probes" in arrays: arrays["probes"]=arrays["probes"].reshape(60,4,arrays["queries"].shape[1])
    return arrays,info


def score_native(dataset, vectors, sparse, backbone, output):
    baseline = [m for m in ARMS[backbone] if m in {"bm25","dense","hybrid"}]
    rows,_=score_original_text(dataset,baseline,vectors,[],.65)
    rows.update({method:{} for method in (PROBES,POSITIVE,FULL)})
    for sid,(mi,qi) in dataset.groups().items():
        docs=[dataset.memories[i] for i in mi]
        bank=vectors["memories"][mi]
        transition=build_knn_transition(bank,docs,k=5)
        path=output/"graphs"/backbone/(sid+".npy")
        path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,transition)
        for i in qi:
            score=compute_bipolar_diffusion_scores(vectors["queries"][i],vectors["probes"][i],bank,transition,
                                                   docs,top_seeds=5,alpha=.2,beta=.3)
            for method in (PROBES,POSITIVE,FULL):
                rows[method][dataset.queries[i]["query_id"]]=ranked_row(dataset.queries[i],docs,method,score[method])
    if backbone=="BGE":
        method="sparsecl_msmarco"
        rows[method]={}
        pools=sparsecl_pools(dataset,vectors)
        alpha=MODEL_LOCK["msmarco"]["alpha"]
        assert alpha==2.19
        for i,q in enumerate(dataset.queries):
            size,idx=pools[i]
            assert size==len(idx) # All scene banks are smaller than the original coarse depth.
            cos,hoyer=score_candidates(vectors["queries"][i],vectors["memories"][idx],sparse["queries"][i],sparse["memories"][idx])
            order,scores=fused_order(cos,hoyer,alpha)
            rows[method][q["query_id"]]={"query_id":q["query_id"],"scenario_id":q["scenario_id"],"method":method,
                "library_size":size,"alpha":alpha,"coarse_k":len(idx),"ranked_candidates":[{
                    "memory_id":dataset.memories[idx[j]].memory_id,"score":float(scores[j]),"cosine":float(cos[j]),"hoyer":float(hoyer[j])} for j in order]}
    for method in ARMS[backbone]:
        write_jsonl(output/"arms"/backbone/(method+".jsonl"),[rows[method][q["query_id"]] for q in dataset.queries])
    return rows


def module_audit(dataset, source_vectors, output):
    queries=dataset.queries
    targets={g["query_id"]:target_ids(g) for g in dataset.gold}
    native={g["query_id"]:[mid for mid,d in g["target_details"].items() if d["source"]=="stale_native"] for g in dataset.gold}
    assert sum(map(len,targets.values()))==340 and sum(map(len,native.values()))==60
    audit={"definition":"Each target counted once per query; NDCG uses the indicated target scope; rank comparisons use full rankings.",
           "backbones":{},"qwen_negative_top10_harm_cases":[]}
    for model,source in (("BGE",BGE_SOURCE),("Qwen",QWEN_SOURCE)):
        saved={method:{r["query_id"]:r for r in read_jsonl(source/"arms"/(method+".jsonl"))} for method in (PROBES,POSITIVE,FULL)}
        rankings={method:rank_lists(r) for method,r in saved.items()}
        audit["backbones"][model]={}
        for scope,labels in (("all_targets",targets),("native_targets",native)):
            qrels={qid:{mid:1 for mid in mids} for qid,mids in labels.items()}
            per={method:{qid:r["ndcg_cut_10"] for qid,r in pytrec_eval.RelevanceEvaluator(qrels,{"ndcg_cut_10"}).evaluate(trec_run(rankings[method])).items()}
                 for method in (PROBES,POSITIVE,FULL)}
            if scope=="all_targets":
                original=read(source/"comparison.json")["per_query_ndcg"]
                for method in per: assert per[method]==original[method]
            scope_result={}
            for name,before,after in (("positive",PROBES,POSITIVE),("negative",POSITIVE,FULL)):
                details=transition_counts(rankings[before],rankings[after],labels)
                scope_result[name]={"before":before,"after":after,
                    "groups":summarize_transitions(details,per[before],per[after],queries),"targets":details}
                if model=="Qwen" and scope=="all_targets" and name=="negative":
                    victims=[r for r in details if r["exited_top10"]]
            audit["backbones"][model][scope]=scope_result
        if model=="Qwen":
            for victim in victims:
                qi=next(i for i,q in enumerate(queries) if q["query_id"]==victim["query_id"])
                q=queries[qi];mi,_=dataset.groups()[q["scenario_id"]]
                docs=[dataset.memories[i] for i in mi]
                loc=next(j for j,d in enumerate(docs) if d.memory_id==victim["memory_id"])
                x=source_vectors[model]["memories"][mi]
                s_plus=x@source_vectors[model]["queries"][qi]
                r0=np.max(np.stack([x@v for v in source_vectors[model]["probes"][qi]],axis=1),axis=1)
                z_neg=np.maximum(0.,s_plus-r0)
                graph=np.load(source/"graphs"/(q["scenario_id"]+".npy"))
                neighbor=.2*(graph@z_neg)
                penalty=.3*(z_neg+neighbor)
                old_scores={r["memory_id"]:r["score"] for r in saved[POSITIVE][q["query_id"]]["ranked_candidates"]}
                new_scores={r["memory_id"]:r["score"] for r in saved[FULL][q["query_id"]]["ranked_candidates"]}
                mid=victim["memory_id"]
                assert abs(old_scores[mid]-float(penalty[loc])-new_scores[mid]) < 1e-7
                neighbors=[{"memory_id":docs[j].memory_id,"weight":float(graph[loc,j]),"excess":float(z_neg[j]),
                            "penalty_contribution":float(.3*.2*graph[loc,j]*z_neg[j])}
                           for j in range(len(docs)) if graph[loc,j]>0 and z_neg[j]>0]
                audit["qwen_negative_top10_harm_cases"].append({**victim,"scenario_id":q["scenario_id"],
                    "type":q["source_ref"]["type"],"query_text":q["content"],"target_text":docs[loc].content,
                    "target_source":dataset.gold[qi]["target_details"][mid]["source"],
                    "s_plus":float(s_plus[loc]),"r0":float(r0[loc]),"similarity_gap":float(s_plus[loc]-r0[loc]),
                    "z_neg":float(z_neg[loc]),"direct_penalty":float(.3*z_neg[loc]),
                    "neighbor_penalty":float(.3*neighbor[loc]),"total_penalty":float(penalty[loc]),
                    "positive_score":old_scores[mid],"bipolar_score":new_scores[mid],"negative_neighbors":neighbors})
    write_json(output/"module_gain_audit.json",audit)
    return audit


def native_table(group,backbone):
    lines=["| 方法 | Top-1 | Top-3 | Top-5 | Top-10 | Mean Rank ↓ | MRR | NDCG@10 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ARMS[backbone]:
        r=group[method]
        hits=[f"{r['at_k'][str(k)]['hits']}/{r['scenes']}（{r['at_k'][str(k)]['recall']:.2%}）" for k in (1,3,5,10)]
        lines.append(f"| {NAMES[method]} | "+" | ".join(hits)+f" | {r['mean_rank']:.2f} | {r['mrr']:.4f} | {r['ndcg_at_10']:.4f} |")
    return lines


def write_report(output,c,a):
    f=c["filter_audit"]
    lines=["# STALE 原生旧库对照与模块收益/误伤分析", "",
        "同一批 seed=20261122 的 30 T1＋30 T2，沿用 240 条原探针与缓存向量。LLM、Qwen Embedding 以及其他 API 调用均为 0；BGE 和 SparseCL 也直接复用缓存，没有重新加载模型编码。", "",
        f"删除 record_type=synthetic 的 {f['removed_synthetic']} 条，只保留 dataset=STALE 且 record_type 为 M_old/history 的条目。过滤后 {f['native_total']} 条，平均每场景 {f['mean_per_scene']:.2f} 条，范围 {f['min_per_scene']}–{f['max_per_scene']}；其中 60 条 M_old、{f['history_total']} 条 history，合成记忆 0 条。每条记录与 data/stale_scenarios/base 逐 ID、逐字段完全一致，60 个官方 M_old 全保留。", "",
        "原生旧库的准确含义：官方 M_old＋项目从 STALE 原始历史对话按既定转换规则摘取的正文；每个 M_new 之前的已完成会话最多保留 3 条用户原句，不是官方直接发布的完整记忆条目列表。转换规则保持原样，本次只去除追加的合成记忆。", "",
        "评测仅将每场景官方 M_old 作为一个目标；保留原生 history 中所有条目，包括扩增轮曾额外标为失效的 7 条历史记录。其余条目不因本次单目标评测而被认定为真实相容。MRR 使用完整排名，Mean Rank 越低越好；NDCG@10 为场景平均。", "",
        "BGE 为原冻结 BAAI/bge-base-en-v1.5；Qwen 请求模型 qwen3-embedding-8b、响应标识 xop3qwen8bembedding，缓存维度均为 768。原向量按 memory_id/query_id 对齐切片，不重新归一化、不重新编码。移除合成条目后各场景重新构图，正余弦有向 Top-5、零对角、行归一化；Top-5 种子、α=0.2、β=0.3。主包、提示词和参数均未改变。SparseCL MS MARCO 使用原冻结检查点向量与 α=2.19；各库小于粗召回深度，候选覆盖全部原生旧库。", ""]
    for backbone in ("BGE","Qwen"):
        for group in ("T1","T2","all"):
            lines += [f"**{backbone} / {'总体' if group=='all' else group}**",""]+native_table(c["native_metrics"][backbone][group],backbone)+[""]
    lines += ["**原生旧库配对显著性（左方法减右方法）**", "",
        "双侧 Wilcoxon、Pratt 零差、正态近似、无连续性修正，检验秩使用差值舍入 12 位；20,000 次配对 Bootstrap，种子 20261122，总体按 T1/T2 分层，百分位 95% CI。总体为主要比较，分组为探索性结果；p 未做多重比较校正。", "",
        "| 底座 | 分组 | 比较 | ΔNDCG | Wilcoxon p | Bootstrap 95% CI | 胜/平/负 |", "|---|---|---|---:|---:|---|---|"]
    for backbone,comparisons in c["significance"].items():
        for label,bygroup in comparisons.items():
            for group,r in bygroup.items():
                lo,hi=r["bootstrap_ci95"]
                lines.append(f"| {backbone} | {group} | {label} | {r['mean_ndcg_difference']:+.4f} | {r['p_two_sided']:.6g} | [{lo:+.4f}, {hi:+.4f}] | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines += ["", "**扩增库模块细账：原排名不变，独立计数**", "",
        "全目标指已有 340 个标注目标；原生目标指其中 60 个官方 M_old。两种范围的 NDCG 分别使用自身金标重新计算；胜/平/负按场景，排名上升/不变/下降按目标。所有跨阈值项逐条列于 module_gain_audit.json，另含 T1/T2 分组。", "",
        "| 底座 | 目标范围 | 模块变化 | Top-10 前→后 | 拉回/推走 | Top-3 前→后 | 拉回/推走 | 排名升/平/降 | NDCG 胜/平/负 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for backbone in ("BGE","Qwen"):
        for scope in ("all_targets","native_targets"):
            for module in ("positive","negative"):
                r=a["backbones"][backbone][scope][module]["groups"]["all"]
                lines.append(f"| {backbone} | {'全部340' if scope=='all_targets' else '原生60'} | {'正扩散−仅探针' if module=='positive' else '负压制−正扩散'} | {r['top10_before']}→{r['top10_after']} | {r['top10_gained']}/{r['top10_lost']} | {r['top3_before']}→{r['top3_after']} | {r['top3_gained']}/{r['top3_lost']} | {r['rank_improved']}/{r['rank_unchanged']}/{r['rank_worsened']} | {r['ndcg_wins']}/{r['ndcg_ties']}/{r['ndcg_losses']} |")
    lines += ["", "**Qwen 负向压制推出 Top-10 的全部目标**", "",
        "差值为 s_plus−r0；正值触发自身扣分，邻居的正差值还会通过 P 产生额外扣分。该差值是相似度启发式，不是逻辑相容判定。以下分析解释分数来源，不重新裁定原标签。", ""]
    for i,r in enumerate(a["qwen_negative_top10_harm_cases"],1):
        lines += [f"{i}. 场景 `{r['scenario_id']}`（{r['type']}），目标 `{r['memory_id']}`（{r['target_source']}），排名 **{r['before_rank']}→{r['after_rank']}**。", "",
            f"   新记忆：{r['query_text']}", "", f"   目标：{r['target_text']}", "",
            f"   s_plus={r['s_plus']:.6f}，r0={r['r0']:.6f}，差值={r['similarity_gap']:+.6f}；直接扣分 {r['direct_penalty']:.6f}，邻域扣分 {r['neighbor_penalty']:.6f}，总扣分 {r['total_penalty']:.6f}。", ""]
    lines += [f"执行耗时 {c['wall_seconds']:.2f} 秒。网络连接尝试 {len(NETWORK_ATTEMPTS)}；LLM/API 新调用 0，新增 Token 0。程序级网络审计钩子禁止连接。", "",
        "复核：`"+json.dumps(c["verification"],ensure_ascii=False)+"`。Mean Rank 通过单金标 pytrec_eval recip_rank 逐场景取倒数恢复名次后独立复核，并非 trec_eval 原生 Mean Rank 指标。", "",
        f"产物：`{output}`。comparison.json 保存原生结果、配对检验和过滤记录；module_gain_audit.json 保存扩增库逐目标细账和误伤案例；inputs/、vectors/、graphs/、arms/ 保存原生实验快照。source_manifest.json 冻结历史输入，历史目录未修改。", ""]
    text="\n".join(lines)
    (output/"report.zh.md").write_text(text,encoding="utf-8")
    REPORT.write_text(text,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    output=args.output.resolve()
    assert not output.exists() and not REPORT.exists(), "Use a new output directory; refusing overwrite"
    started=time.perf_counter()
    old_protocol=read(BGE_SOURCE/"protocol.json")
    assert ANCHORED_PROMPT_SHA256==old_protocol["prompt_sha256"]==PROMPT
    assert old_protocol["graph"]=={"k":5,"top_seeds":5,"alpha":.2,"beta":.3}
    check_hashes(ROOT/"memory_retrieval",old_protocol["code_sha256"])
    for source in (BGE_SOURCE,QWEN_SOURCE):
        assert read(source/"completion.json")["status"]=="complete"
    source_hashes={str(p.relative_to(ROOT)):sha256_file(p) for source in (BGE_SOURCE,QWEN_SOURCE)
                   for p in source.rglob("*") if p.is_file()}
    for name in ("queries.jsonl","memories.jsonl","gold.jsonl","manifest.json"):
        source_hashes[(BASE/name).relative_to(ROOT).as_posix()]=sha256_file(BASE/name)
    augmented=load_dataset(BGE_SOURCE/"inputs")
    queries=read_jsonl(BGE_SOURCE/"inputs/queries.jsonl")
    sids={q["scenario_id"] for q in queries}
    assert len(sids)==60 and Counter(q["source_ref"]["type"] for q in queries)=={"T1":30,"T2":30}
    for name in ("queries.jsonl","memories.jsonl","gold.jsonl"):
        assert sha256_file(BGE_SOURCE/"inputs"/name)==sha256_file(QWEN_SOURCE/"inputs"/name)
    assert sha256_file(BGE_SOURCE/"anchored_probes.json")==sha256_file(QWEN_SOURCE/"anchored_probes.json")
    native,removed=select_native_rows(read_jsonl(BGE_SOURCE/"inputs/memories.jsonl"),read_jsonl(BASE/"memories.jsonl"),sids)
    base_queries={q["query_id"]:q for q in read_jsonl(BASE/"queries.jsonl")}
    assert all(q==base_queries[q["query_id"]] for q in queries)
    base_gold={g["query_id"]:g for g in read_jsonl(BASE/"gold.jsonl")}
    gold=[base_gold[q["query_id"]] for q in queries]
    native_ids={g["query_id"]:target_ids(g)[0] for g in gold}
    assert all(len(target_ids(g))==1 and g["target_details"][target_ids(g)[0]]["source"]=="stale_native" for g in gold)
    for g in augmented.gold:
        assert [mid for mid,d in g["target_details"].items() if d["source"]=="stale_native"]==[native_ids[g["query_id"]]]
    counts=Counter(m["scenario_id"] for m in native)
    types={q["scenario_id"]:q["source_ref"]["type"] for q in queries}
    filtering={"rule":"Keep dataset=STALE AND record_type in {history,M_old}; exclude synthetic; assert exact row equality with selected base bank",
        "native_total":len(native),"history_total":sum(m["source_ref"]["record_type"]=="history" for m in native),
        "native_gold_total":60,"synthetic_remaining":0,"removed_synthetic":len(removed),
        "removed_by_group":dict(Counter(m["source_ref"]["augmentation_group"] for m in removed)),
        "mean_per_scene":len(native)/60,"min_per_scene":min(counts.values()),"max_per_scene":max(counts.values()),
        "by_type":{t:{"scenes":30,"memories":sum(n for sid,n in counts.items() if types[sid]==t),
                      "mean_per_scene":sum(n for sid,n in counts.items() if types[sid]==t)/30} for t in ("T1","T2")},
        "per_scene_counts":dict(counts),"all_native_targets_preserved":True,"exact_base_rows_match":True,
        "base_manifest":read(BASE/"manifest.json")}
    output.mkdir(parents=True)
    for name,rows in (("memories",native),("queries",queries),("gold",gold)):
        write_jsonl(output/"inputs"/(name+".jsonl"),rows)
    for name in ("selection.json","exclusions.json","anchored_probes.json"):
        shutil.copy2(BGE_SOURCE/name,output/name)
    shutil.copy2(Path(__file__),output/"runner.py")
    protocol={"locked_at_utc":datetime.now(timezone.utc).isoformat(),"prompt_sha256":PROMPT,"graph":old_protocol["graph"],
        "source_bge":str(BGE_SOURCE),"source_qwen":str(QWEN_SOURCE),"main_package_sha256":old_protocol["code_sha256"],
        "script_sha256":sha256_file(Path(__file__)),"arms":ARMS,"api_calls_allowed":0,
        "statistics":{"paired_unit":"scene","wilcoxon":"two-sided Pratt approx, no continuity correction, differences rounded 12 decimals",
                      "bootstrap_replicates":20000,"bootstrap_seed":20261122,"bootstrap_ci":"percentile95; overall stratified T1/T2","p_adjustment":"none"}}
    write_json(output/"protocol.json",protocol)
    write_json(output/"source_manifest.json",source_hashes)
    write_json(output/"filter_audit.json",filtering)
    dataset=load_dataset(output/"inputs")
    old_indices={m.memory_id:i for i,m in enumerate(augmented.memories)}
    indices=np.array([old_indices[m.memory_id] for m in dataset.memories])
    write_json(output/"vector_mapping.json",{"memory_ids":[m.memory_id for m in dataset.memories],
        "source_memory_indices":indices.tolist(),"query_ids":[q["query_id"] for q in queries],
        "source_query_indices":list(range(60)),"probes_per_query":4})
    source_vectors={}
    for model,source in (("BGE",BGE_SOURCE),("Qwen",QWEN_SOURCE),("msmarco",BGE_SOURCE)):
        source_vectors[model],info=load_source_vectors(source,model)
        folder=output/"vectors"/model
        folder.mkdir(parents=True)
        for name,x in source_vectors[model].items():
            np.save(folder/(name+".npy"),x[indices] if name=="memories" else x)
        write_json(folder/"identity.json",{"source_identity":info,"sliced_by_memory_id":True,
                    "sha256":{p.name:sha256_file(p) for p in folder.glob('*.npy')}})
    metrics,per_query,errors,significance={},{},{},{}
    for model in ("BGE","Qwen"):
        vectors={k:(v[indices] if k=="memories" else v) for k,v in source_vectors[model].items()}
        sparse={k:(v[indices] if k=="memories" else v) for k,v in source_vectors["msmarco"].items()}
        rows=score_native(dataset,vectors,sparse,model,output)
        metrics[model]={group:{} for group in ("T1","T2","all")}
        per_query[model]={};errors[model]={}
        for method in ARMS[model]:
            measured,pq,err=native_metrics(rank_lists(rows[method]),native_ids,queries)
            for group in measured: metrics[model][group][method]=measured[group]
            per_query[model][method]=pq;errors[model][method]=err
        pairs=[(FULL,"dense"),(POSITIVE,PROBES)]
        if model=="BGE": pairs.insert(1,(FULL,"sparsecl_msmarco"))
        significance[model]={}
        for left,right in pairs:
            label=left+" vs "+right
            significance[model][label]=paired_statistics(queries,{q:r["ndcg_at_10"] for q,r in per_query[model][left].items()},
                                                              {q:r["ndcg_at_10"] for q,r in per_query[model][right].items()})
        print(json.dumps({"backbone_completed":model}),flush=True)
    audit=module_audit(augmented,source_vectors,output)
    check_hashes(ROOT,source_hashes)
    check_hashes(ROOT/"memory_retrieval",old_protocol["code_sha256"])
    assert not NETWORK_ATTEMPTS
    verification={"per_method_errors":errors,"max_metric_error":max(v for bymodel in errors.values() for err in bymodel.values() for v in err.values()),
                  "source_directories_unchanged":True,"main_package_unchanged":True,"llm_calls":0,"embedding_api_calls":0,
                  "network_connection_attempts":0,"local_model_encodings":0}
    comparison={"filter_audit":filtering,"native_metrics":metrics,"per_query":per_query,"significance":significance,
                "verification":verification,"wall_seconds":time.perf_counter()-started,"protocol":protocol}
    write_json(output/"comparison.json",comparison)
    write_json(output/"verification.json",verification)
    write_report(output,comparison,audit)
    write_json(output/"completion.json",{"status":"complete","comparison_sha256":sha256_file(output/"comparison.json"),
        "module_gain_audit_sha256":sha256_file(output/"module_gain_audit.json"),"report_sha256":sha256_file(REPORT),
        "wall_seconds_including_report":time.perf_counter()-started})
    print(json.dumps({"status":"complete","wall_seconds":time.perf_counter()-started,"max_metric_error":verification['max_metric_error']}),flush=True)


if __name__=="__main__":
    main()
