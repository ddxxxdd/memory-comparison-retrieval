"""Paired descriptive analysis of frozen controlled retrieval experiments."""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
from validate_validmem_challenges import ROOT,DATA,RUN,REPORT,KS,METHODS,rows,read_json,dump,file_hash,verify_frozen,digest


def write_csv(path,values):
    with path.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)


def condition(case):
    return (case["module"],case["variant"],case["added_distractors"],case["bank_size"],case["gold_count"],case["wording"])


def numbers(item,k):
    ranks=list(item["result"]["gold_ranks"].values());g=item["case"]["gold_count"]
    hit=sum(r<=k for r in ranks)
    return dict(hit=hit,gold=g,recall=hit/g,full=float(hit==g),zero=float(hit==0),
                capacity=max(g-k,0),extra=min(g,k)-hit,miss=g-hit,non_gold=k-hit,
                first_rank=min(ranks),worst_rank=max(ranks),rank_overhead=max(ranks)-g)


def summarize(group,k):
    vals=[numbers(item,k) for item in group]
    avg=lambda key:float(np.mean([v[key] for v in vals]))
    families=defaultdict(list)
    for item,value in zip(group,vals):families[item["case"]["family"]].append(value["recall"])
    return dict(cases=len(group),anchors=len({i["case"]["anchor_id"] for i in group}),families=len(families),
                hits=sum(v["hit"] for v in vals),gold_pairs=sum(v["gold"] for v in vals),
                recall_micro=sum(v["hit"] for v in vals)/sum(v["gold"] for v in vals),
                recall_macro=avg("recall"),family_balanced_recall=float(np.mean([np.mean(v) for v in families.values()])),
                full_coverage=avg("full"),zero_hit=avg("zero"),capacity_deficit=sum(v["capacity"] for v in vals),
                extra_deficit=sum(v["extra"] for v in vals),extra_deficit_mean=avg("extra"),
                missed_pairs=sum(v["miss"] for v in vals),non_gold=sum(v["non_gold"] for v in vals),
                candidate_mean=k,first_rank_mean=avg("first_rank"),worst_rank_mean=avg("worst_rank"),
                worst_rank_median=float(np.median([v["worst_rank"] for v in vals])),
                worst_rank_p95=float(np.quantile([v["worst_rank"] for v in vals],.95)),
                rank_overhead_mean=avg("rank_overhead"))


def comparison_specs(experiment):
    if experiment=="attribute_specific":
        return [("same_entity_attribute_vs_other_entity",{"module":"attribute_specific","variant":"other_entity_other_attribute"},{"module":"attribute_specific","variant":"same_entity_other_attribute"}),
                ("attribute_injection_vs_none",{"module":"attribute_specific","variant":"no_injection"},{"module":"attribute_specific","variant":"same_entity_other_attribute"}),
                ("restatement_vs_none",{"module":"attribute_specific","variant":"no_injection"},{"module":"attribute_specific","variant":"same_value_restatement"})]
    specs=[]
    for d in (1,5,15):
        specs.append((f"identity_value_match_d{d}",{"module":"identity","variant":"other_entity_old_value","added_distractors":d},
                      {"module":"identity","variant":"other_entity_query_value","added_distractors":d}))
        specs.append((f"identity_injection_d{d}",{"module":"identity","variant":"no_injection"},
                      {"module":"identity","variant":"other_entity_query_value","added_distractors":d}))
    specs.append(("same_entity_mention_vs_other_entity",{"module":"attribute","variant":"other_entity_mention"},
                  {"module":"attribute","variant":"same_entity_mention"}))
    for wording in ("original","paraphrase","chinese"):
        specs.append((f"wording_{wording}",{"module":"wording","wording":"canonical"},{"module":"wording","wording":wording}))
    for n in (200,1000):
        specs.append((f"scale_50_to_{n}",{"module":"scale","bank_size":50},{"module":"scale","bank_size":n}))
    for g in (2,4,6):
        specs.append((f"cardinality_1_to_{g}",{"module":"cardinality","gold_count":1},{"module":"cardinality","gold_count":g}))
    return specs


def paired(items,experiment):
    output=[];anchor_output=[]
    for name,before_selector,after_selector in comparison_specs(experiment):
        matches=lambda c,s:all(c[key]==value for key,value in s.items())
        for method in METHODS:
            before={(i["case"]["anchor_id"],i["case"]["seed"]):i for i in items if i["result"]["method"]==method and matches(i["case"],before_selector)}
            after={(i["case"]["anchor_id"],i["case"]["seed"]):i for i in items if i["result"]["method"]==method and matches(i["case"],after_selector)}
            assert set(before)==set(after) and before,(name,method)
            for k in KS:
                changes=defaultdict(list);extra_changes=defaultdict(list);family={}
                for key in sorted(before):
                    b,a=numbers(before[key],k),numbers(after[key],k)
                    changes[key[0]].append(a["recall"]-b["recall"])
                    extra_changes[key[0]].append(a["extra"]-b["extra"])
                    family[key[0]]=after[key]["case"]["family"]
                anchor_means={aid:float(np.mean(v)) for aid,v in changes.items()}
                family_changes=defaultdict(list)
                for aid,delta in anchor_means.items():
                    family_changes[family[aid]].append(delta)
                    anchor_output.append(dict(experiment=experiment,comparison=name,method=method,k=k,anchor_id=aid,
                                              family=family[aid],mean_recall_change=delta,seed_changes=json.dumps(changes[aid]),
                                              extra_deficit_change=float(np.mean(extra_changes[aid]))))
                family_means=np.array([np.mean(v) for v in family_changes.values()])
                if len(family_means)>=4:
                    generator=np.random.default_rng(int(digest([experiment,name,method,k])[:8],16))
                    draws=generator.choice(family_means,(2000,len(family_means)),replace=True).mean(axis=1)
                    lo,hi=[float(v) for v in np.quantile(draws,[.025,.975])]
                else:lo=hi=None
                nongrocery=[v for aid,v in anchor_means.items() if family[aid]!="grocery"]
                output.append(dict(experiment=experiment,comparison=name,method=method,k=k,anchors=len(changes),families=len(family_means),
                                   anchor_mean_recall_change=float(np.mean(list(anchor_means.values()))),
                                   family_balanced_recall_change=float(family_means.mean()),family_cluster_ci_low=lo,family_cluster_ci_high=hi,
                                   non_grocery_recall_change=float(np.mean(nongrocery)) if nongrocery else None,
                                   harmed_anchors=sum(v< -1e-12 for v in anchor_means.values()),
                                   helped_anchors=sum(v>1e-12 for v in anchor_means.values()),
                                   harmed_in_all_five_seeds=sum(all(v< -1e-12 for v in values) for values in changes.values()),
                                   extra_deficit_change=float(np.mean([np.mean(v) for v in extra_changes.values()])),
                                   before_recall=float(np.mean([numbers(i,k)["recall"] for i in before.values()])),
                                   after_recall=float(np.mean([numbers(i,k)["recall"] for i in after.values()]))))
    return output,anchor_output


def analyze_results():
    verify_frozen()
    experiments=[("primary",DATA,RUN),("attribute_specific",ROOT/"data/validmem_attribute_controls_v1",ROOT/"runs/experiments/validmem-attribute-controls-20261112")]
    all_metrics=[];family_metrics=[];seed_metrics=[];comparisons=[];anchor_changes=[];metadata={}
    for experiment,data,run in experiments:
        completion=read_json(run/"completion.json")
        assert completion["status"]=="complete"
        assert all(file_hash(run/p)==d for p,d in completion["result_hashes"].items())
        manifest=read_json(data/"manifest.json")
        assert all(file_hash(data/p)==d for p,d in manifest["files_sha256"].items())
        cases={r["case_id"]:r for r in rows(data/"cases.jsonl")}
        results=rows(run/"per_case.jsonl")
        assert len(results)==4*len(cases) and len({(r["case_id"],r["method"]) for r in results})==len(results)
        items=[]
        for r in results:
            c=cases[r["case_id"]]
            assert set(r["gold_ranks"])==set(c["gold_ids"])
            assert all(isinstance(v,int) and 1<=v<=c["bank_size"] for v in r["gold_ranks"].values())
            items.append({"case":c,"result":r})
        grouped=defaultdict(list);by_family=defaultdict(list);by_seed=defaultdict(list)
        for item in items:
            c,r=item["case"],item["result"]
            key=(*condition(c),r["method"])
            grouped[key].append(item);by_family[(*key,c["family"])].append(item);by_seed[(*key,c["seed"])].append(item)
        for mapping,output,extra in ((grouped,all_metrics,None),(by_family,family_metrics,"family"),(by_seed,seed_metrics,"seed")):
            for key,group in sorted(mapping.items()):
                base=dict(zip(("module","variant","added_distractors","bank_size","gold_count","wording","method"),key[:7]))
                if extra:base[extra]=key[-1]
                for k in KS:
                    value=summarize(group,k)
                    assert value["missed_pairs"]==value["capacity_deficit"]+value["extra_deficit"]
                    output.append(dict(experiment=experiment,**base,k=k,**value))
        pair,anchors=paired(items,experiment);comparisons+=pair;anchor_changes+=anchors
        metadata[experiment]={"dataset":manifest,"completion":completion,"qwen":read_json(run/"qwen_encoding.json"),
                              "local":read_json(run/"local_encoding.json"),"repeat":read_json(run/"repeat_rankings.json"),
                              "natural_replay":read_json(run/"natural_replay.json")}
    REPORT.mkdir(parents=True,exist_ok=True)
    write_csv(REPORT/"metrics.csv",all_metrics);write_csv(REPORT/"by_family.csv",family_metrics)
    write_csv(REPORT/"by_seed.csv",seed_metrics);write_csv(REPORT/"paired_effects.csv",comparisons)
    write_csv(REPORT/"paired_anchor_effects.csv",anchor_changes)
    dump(REPORT/"analysis.json",{"metadata":metadata,"metrics":all_metrics,"paired_effects":comparisons})
    lines=["# 可复算结果表", "", "每行先在相同条件下对查询/种子汇总；种子不是独立事实。R 为 macro Recall；主表全部为固定同一 G，micro 与 macro 一致。", "",
           "| 实验 | 条件 | 方法 | k | anchor数 | R | 分族等权 R | 全覆盖 | 漏检 | 容量缺口 | 额外缺口 | 最差金标名次中位数 |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in all_metrics:
        if r["k"] not in (1,5,10):continue
        cond=f"{r['module']}/{r['variant']}; d={r['added_distractors']}, N={r['bank_size']}, G={r['gold_count']}"
        lines.append(f"| {r['experiment']} | {cond} | {r['method']} | {r['k']} | {r['anchors']} | {r['recall_macro']:.2%} | {r['family_balanced_recall']:.2%} | {r['full_coverage']:.2%} | {r['missed_pairs']} | {r['capacity_deficit']} | {r['extra_deficit']} | {r['worst_rank_median']:.1f} |")
    (REPORT/"statistics.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"metric_rows":len(all_metrics),"paired_comparisons":len(comparisons),"anchor_effect_rows":len(anchor_changes),"status":"complete"}),flush=True)


if __name__=="__main__":analyze_results()
