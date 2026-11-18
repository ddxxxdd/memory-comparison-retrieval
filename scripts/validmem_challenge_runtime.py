"""Execution and reporting for the frozen challenge probes; no baseline edits."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from validate_validmem_challenges import (
    ROOT, DATA, RUN, REPORT, SOURCE, SEEDS, KS, METHODS, FAMILIES,
    read_json, rows, dump, jsonl, digest, file_hash, verify_frozen,
)


def load_inputs():
    manifest=verify_frozen()
    texts=rows(DATA/"texts.jsonl")
    memories=rows(DATA/"memories.jsonl")
    banks={r["bank_id"]:r for r in rows(DATA/"banks.jsonl")}
    cases=rows(DATA/"cases.jsonl")
    return manifest,texts,memories,banks,cases


def repeat_panel(cases):
    eligible=[c for c in cases if c["module"]=="wording" and c["wording"]=="canonical" and c["seed"]==17]
    if not eligible:
        eligible=[c for c in cases if c["variant"]=="same_entity_other_attribute" and c["seed"]==17]
    selected=[]
    for family in sorted(FAMILIES):
        selected.append(min((c for c in eligible if c["family"]==family),key=lambda c:digest(c["anchor_id"])))
    rest=[c for c in eligible if c not in selected]
    selected+=sorted(rest,key=lambda c:digest(c["anchor_id"]))[:3]
    return selected


def preflight():
    manifest,texts,memories,banks,cases=load_inputs()
    RUN.mkdir(parents=True,exist_ok=True)
    panel=repeat_panel(cases)
    bymid={r["memory_id"]:r for r in memories}
    repeat_ids=sorted({tid for c in panel for tid in [c["query_text_id"],bymid[c["gold_ids"][0]]["text_id"]]})
    value={"dataset_manifest_sha256":file_hash(DATA/"manifest.json"),
           "protocol_sha256":file_hash(REPORT/"protocol.zh.md"),
           "repeat_case_ids":[c["case_id"] for c in panel],"repeat_text_ids":repeat_ids,
           "budgets":list(KS),"seeds":list(SEEDS),"alpha":0.18,"max_bank_size":1000,
           "label_fields_sent_to_encoders":False}
    path=RUN/"preflight.json"
    if path.exists():assert read_json(path)==value,"Frozen protocol/panel changed"
    else:dump(path,value)
    return manifest,texts,memories,banks,cases,value


def embed_qwen():
    import numpy as np
    from memory_retrieval.api import APIClient,usage_summary
    from memory_retrieval.config import Settings
    manifest,texts,memories,banks,cases,pref=preflight()
    settings=Settings.from_env()
    previous=read_json(ROOT/"runs/experiments/validmem-v7-main-20261030/run_config.json")
    assert settings.embedding_identity()==previous["settings"]["embedding"],"Model identity differs from archived Dense"
    settings=replace(settings,cache_path=RUN/"qwen_cache.sqlite3",embedding_batch_size=32,timeout=60,max_retries=2)
    target=RUN/"qwen_vectors.npz"
    if target.exists():
        info=read_json(RUN/"qwen_encoding.json")
        assert info["vectors_sha256"]==file_hash(target)
        print("Qwen vectors already complete and verified",flush=True)
        return
    chunks=[texts[i:i+32] for i in range(0,len(texts),32)]
    events=[]
    parts={}
    started=time.perf_counter()

    def encode(index,chunk):
        client=APIClient(settings)
        matrix=client.embed([r["content"] for r in chunk],"challenge_unique_texts")
        return index,matrix,client.events

    try:
        index,matrix,ev=encode(0,chunks[0])
        parts[index]=matrix;events+=ev
        print(f"Qwen connection verified; {len(texts)} texts in {len(chunks)} batches",flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(encode,i,chunk) for i,chunk in enumerate(chunks) if i]
            for future in as_completed(futures):
                index,matrix,ev=future.result();parts[index]=matrix;events+=ev
                if len(parts)%10==0:print(f"Qwen batches {len(parts)}/{len(chunks)}",flush=True)
        vectors=np.concatenate([parts[i] for i in range(len(chunks))])
        bytid={r["text_id"]:r for r in texts}
        # A new cache forces an independent network encoding of the preselected panel.
        repeat_client=APIClient(replace(settings,cache_path=RUN/"qwen_repeat_cache.sqlite3"))
        repeated=repeat_client.embed([bytid[tid]["content"] for tid in pref["repeat_text_ids"]],"repeat_panel")
        positions={r["text_id"]:i for i,r in enumerate(texts)}
        first=vectors[[positions[tid] for tid in pref["repeat_text_ids"]]]
        np.savez_compressed(target,vectors=vectors,repeat=repeated)
        # This is a comparability diagnostic, not a license to mix old and new features.
        old_index=read_json(ROOT/"runs/experiments/validmem-v7-main-20261030/index/index.json")
        old_vec=np.load(ROOT/"runs/experiments/validmem-v7-main-20261030/index/vectors.npz")["contents"]
        bytext={r["content"]:i for i,r in enumerate(texts)}
        fresh=vectors[[bytext[r["content"]] for r in old_index["memories"]]]
        info={"model":settings.embedding_model,"identity_matches_archived_run":True,
              "text_count":len(texts),"dimension":int(vectors.shape[1]),"network_workers":4,
              "seconds":time.perf_counter()-started,"usage":usage_summary(events),
              "repeat_usage":usage_summary(repeat_client.events),"repeat_count":len(first),
              "repeat_max_abs_difference":float(np.max(np.abs(first-repeated))),
              "repeat_min_cosine":float(np.min(np.sum(first*repeated,axis=1))),
              "original_bank_min_cosine":float(np.min(np.sum(old_vec*fresh,axis=1))),
              "original_bank_mean_cosine":float(np.mean(np.sum(old_vec*fresh,axis=1))),
              "all_experiment_vectors_fresh_or_this_run_cache":True,"vectors_sha256":file_hash(target)}
        dump(RUN/"qwen_encoding.json",info)
        print(json.dumps(info,ensure_ascii=False),flush=True)
    except Exception as exc:
        dump(RUN/"qwen_failure.json",{"type":type(exc).__name__,"message":str(exc),"completed_batches":len(parts)})
        raise


def encode_local(texts,pref):
    import numpy as np
    import torch
    import transformers
    from sparsecl_worker import encode_texts,verify_model_assets
    target=RUN/"local_vectors.npz"
    if target.exists():
        info=read_json(RUN/"local_encoding.json")
        assert info["vectors_sha256"]==file_hash(target)
        return
    baseline=ROOT.parent/"baselines/SparseCL"
    sys.path.insert(0,str(baseline))
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    torch.set_num_threads(4)
    lock_hash,asset_hashes=verify_model_assets(baseline)
    device=torch.device("cpu")
    bytid={r["text_id"]:r for r in texts}
    all_text=[r["content"].strip() for r in texts]
    repeated=[bytid[tid]["content"].strip() for tid in pref["repeat_text_ids"]]
    positions={r["text_id"]:i for i,r in enumerate(texts)}
    panel_positions=[positions[tid] for tid in pref["repeat_text_ids"]]
    arrays={};info={"assets_lock_sha256":lock_hash,"model_assets_sha256":asset_hashes,
                   "text_count":len(texts),"batch_size":4,"cpu_threads":4,"device":"cpu","models":{}}
    for kind,folder,pooling in (("bge","bge-base-en-v1.5","bge_cls"),("sparsecl","BGE-SparseCL-arguana","sparsecl_avg")):
        path=baseline/"results"/folder
        tokenizer=transformers.AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        if kind=="bge":
            model=transformers.AutoModel.from_pretrained(str(path),local_files_only=True).to(device)
        else:
            from sparsecl.models import our_BertForCL
            config=transformers.AutoConfig.from_pretrained(str(path),local_files_only=True,trust_remote_code=True)
            args=SimpleNamespace(pooler_type="avg",temp=0.05,hard_negative_weight=0.0,do_mlm=False,mlm_weight=0.1,mlp_only_train=False)
            model=our_BertForCL.from_pretrained(str(path),from_tf=False,config=config,use_auth_token=None,model_args=args,local_files_only=True).to(device)
        model.eval()
        print(f"Encoding {kind}: {len(all_text)} unique texts",flush=True)
        started=time.perf_counter()
        vectors,_=encode_texts(all_text,tokenizer,model,device,4,torch,pooling)
        repeat,_=encode_texts(repeated,tokenizer,model,device,4,torch,pooling)
        arrays[kind]=vectors;arrays[kind+"_repeat"]=repeat
        info["models"][kind]={"seconds":time.perf_counter()-started,
                              "repeat_max_abs_difference":float(np.max(np.abs(vectors[panel_positions]-repeat))),
                              "repeat_min_cosine":float(np.min(np.sum(vectors[panel_positions]*repeat,axis=1)))}
        del model,tokenizer
        print(f"Finished {kind}",flush=True)
    np.savez_compressed(target,**arrays)
    info["vectors_sha256"]=file_hash(target)
    dump(RUN/"local_encoding.json",info)


def run_local():
    import numpy as np
    import faiss
    from memory_retrieval.baselines import BM25,rank,reciprocal_rank_fusion
    from memory_retrieval.sparsecl import score_candidates,fused_order
    from memory_retrieval.data import Memory
    manifest,texts,memories,banks,cases,pref=preflight()
    encode_local(texts,pref)
    if not (RUN/"qwen_vectors.npz").exists():
        print("Local encodings ready; rerun run-local after Qwen encoding completes",flush=True)
        return
    if (RUN/"completion.json").exists():
        completion=read_json(RUN/"completion.json")
        assert all(file_hash(RUN/name)==v for name,v in completion["result_hashes"].items())
        print("Completed rankings verified; no rerun needed",flush=True)
        return
    faiss.omp_set_num_threads(4)
    local=np.load(RUN/"local_vectors.npz")
    qwen=np.load(RUN/"qwen_vectors.npz")
    vectors={"qwen":qwen["vectors"],"bge":local["bge"],"sparsecl":local["sparsecl"]}
    bytid={r["text_id"]:i for i,r in enumerate(texts)}
    bymid={r["memory_id"]:r for r in memories}
    text_by_id={r["text_id"]:r["content"] for r in texts}
    case_by_id={c["case_id"]:c for c in cases}
    repeat_positions={tid:i for i,tid in enumerate(pref["repeat_text_ids"])}
    repeat_arrays={"qwen":qwen["repeat"],"bge":local["bge_repeat"],"sparsecl":local["sparsecl_repeat"]}

    @lru_cache(maxsize=8)
    def bank_index(bank_id):
        ids=banks[bank_id]["memory_ids"]
        positions=np.array([bytid[bymid[mid]["text_id"]] for mid in ids])
        records=[Memory(mid,texts[int(p)]["content"],{}) for mid,p in zip(ids,positions)]
        bm25=BM25([r.content for r in records])
        index=faiss.IndexHNSWFlat(768,64)
        index.add(vectors["bge"][positions])
        index.hnsw.efSearch=1000
        return ids,positions,records,bm25,index

    # Score candidates with the original pairwise routine; the initial pool and
    # stable tie order are still obtained from each bank's actual HNSW index.
    @lru_cache(maxsize=16)
    def pair_scores(query_tid):
        q=bytid[query_tid]
        return score_candidates(vectors["bge"][q],vectors["bge"],vectors["sparsecl"][q],vectors["sparsecl"])

    def retrieve(case,repeated=False):
        ids,positions,records,bm25,index=bank_index(case["bank_id"])
        tid=case["query_text_id"];qi=bytid[tid]
        sparse=bm25.score(text_by_id[tid])
        qvec=vectors["qwen"][qi]
        mem_qwen=vectors["qwen"][positions]
        q_bge=vectors["bge"][qi]
        mem_bge=vectors["bge"][positions]
        q_sp=vectors["sparsecl"][qi]
        mem_sp=vectors["sparsecl"][positions]
        if repeated:
            qvec=repeat_arrays["qwen"][repeat_positions[tid]]
            q_bge=repeat_arrays["bge"][repeat_positions[tid]]
            q_sp=repeat_arrays["sparsecl"][repeat_positions[tid]]
            mem_qwen=mem_qwen.copy();mem_bge=mem_bge.copy();mem_sp=mem_sp.copy()
            for i,mid in enumerate(ids):
                mtid=bymid[mid]["text_id"]
                if mtid in repeat_positions:
                    ri=repeat_positions[mtid]
                    mem_qwen[i]=repeat_arrays["qwen"][ri]
                    mem_bge[i]=repeat_arrays["bge"][ri]
                    mem_sp[i]=repeat_arrays["sparsecl"][ri]
            index=faiss.IndexHNSWFlat(768,64);index.add(mem_bge);index.hnsw.efSearch=1000
        dense=(qvec.reshape(1,-1)@mem_qwen.T)[0]
        hybrid=reciprocal_rank_fusion(sparse,dense,records)
        _,initial=index.search(q_bge.reshape(1,-1),min(1000,len(ids)))
        initial=initial[0]
        assert len(set(initial.tolist()))==min(1000,len(ids)) and np.min(initial)>=0
        if repeated:
            cosine,hoyer=score_candidates(q_bge,mem_bge[initial],q_sp,mem_sp[initial])
        else:
            all_cos,all_hoyer=pair_scores(tid)
            cosine,hoyer=all_cos[positions[initial]],all_hoyer[positions[initial]]
        order,sp_scores=fused_order(cosine,hoyer,0.18)
        result={}
        for method,values in (("bm25",sparse),("dense",dense),("hybrid",hybrid)):
            indices=rank(values,records)
            result[method]=([ids[i] for i in indices],[float(values[i]) for i in indices])
        result["sparsecl"]=([ids[int(initial[i])] for i in order],[float(sp_scores[i]) for i in order])
        return result

    started=time.perf_counter()
    details=[];panel_first={}
    with gzip.open(RUN/"rankings.jsonl.gz","wt",encoding="utf-8",compresslevel=3) as handle:
        for i,case in enumerate(cases):
            result=retrieve(case)
            if case["case_id"] in pref["repeat_case_ids"]:panel_first[case["case_id"]]=result
            for method,(ids,scores) in result.items():
                assert len(ids)==len(set(ids))==case["bank_size"]
                ranks={mid:j for j,mid in enumerate(ids,1)}
                gold_ranks={mid:ranks[mid] for mid in case["gold_ids"]}
                details.append({"case_id":case["case_id"],"method":method,"gold_ranks":gold_ranks,
                                "top1":ids[0],"top5":ids[:5],"top10":ids[:10]})
                handle.write(json.dumps({"case_id":case["case_id"],"method":method,"ranked_ids":ids,"scores":scores},separators=(",",":"))+"\n")
            if (i+1)%250==0:print(f"Ranked {i+1}/{len(cases)} cases ({time.perf_counter()-started:.1f}s)",flush=True)
    jsonl(RUN/"per_case.jsonl",details)
    repeat_results=[]
    for case_id in pref["repeat_case_ids"]:
        second=retrieve(case_by_id[case_id],repeated=True)
        first=panel_first[case_id]
        for method in METHODS:
            before=first[method][0];after=second[method][0];gold=set(case_by_id[case_id]["gold_ids"])
            repeat_results.append({"case_id":case_id,"method":method,
                                   "same_full_ranking":before==after,
                                   "same_top10":before[:10]==after[:10],
                                   "gold_ranks_before":{mid:before.index(mid)+1 for mid in gold},
                                   "gold_ranks_after":{mid:after.index(mid)+1 for mid in gold},
                                   "recall_changed_at": [k for k in KS if len(set(before[:k])&gold)!=len(set(after[:k])&gold)]})
    dump(RUN/"repeat_rankings.json",repeat_results)
    # Fresh replay of every eligible, unfiltered source query on the complete original
    # bank. This retains natural hard negatives rather than showing controlled banks only.
    source_mem=rows(SOURCE/"memories.jsonl")
    source_q={r["query_id"]:r for r in rows(SOURCE/"queries.jsonl")}
    source_g={r["query_id"]:r for r in rows(SOURCE/"gold.jsonl")}
    bytext={r["content"]:r["text_id"] for r in texts}
    natural_bank="natural-1225"
    banks[natural_bank]={"bank_id":natural_bank,"memory_ids":[r["memory_id"] for r in source_mem]}
    archived={}
    for method in METHODS:
        run="validmem-v7-sparsecl-exploratory-20261031-retry1" if method=="sparsecl" else "validmem-v7-main-20261030"
        archived[method]={r["query_id"]:r for r in rows(ROOT/"runs/experiments"/run/"arms"/f"{method}.jsonl")}
    replay=[]
    replay_anchors=[] if manifest.get("skip_natural_replay") else rows(DATA/"anchors.jsonl")
    for anchor in replay_anchors:
        qid=anchor["anchor_id"]
        case={"bank_id":natural_bank,"query_text_id":bytext[source_q[qid]["content"]]}
        result=retrieve(case)
        gold=set(source_g[qid]["conflict_old_memory_ids"])
        for method,(ids,scores) in result.items():
            old=archived[method][qid]
            old_ids=[r["memory_id"] for r in old["ranked_candidates" if method=="sparsecl" else "candidates"]]
            replay.append({"query_id":qid,"family":anchor["query_fact"]["attribute"],"method":method,
                           "gold_ranks":{mid:(ids.index(mid)+1 if mid in ids else None) for mid in gold},
                           "archived_gold_ranks":{mid:(old_ids.index(mid)+1 if mid in old_ids else None) for mid in gold},
                           "same_top10":ids[:10]==old_ids[:10],
                           "same_gold_at_5":set(ids[:5])&gold==set(old_ids[:5])&gold,
                           "same_gold_at_10":set(ids[:10])&gold==set(old_ids[:10])&gold})
    dump(RUN/"natural_replay.json",replay)
    verify_frozen()
    output_names=["rankings.jsonl.gz","per_case.jsonl","repeat_rankings.json","natural_replay.json"]
    completion={"status":"complete","cases":len(cases),"method_case_count":len(details),
                "ranking_seconds":time.perf_counter()-started,"dataset_manifest_sha256":file_hash(DATA/"manifest.json"),
                "runtime_sha256":file_hash(__file__),"result_hashes":{name:file_hash(RUN/name) for name in output_names}}
    dump(RUN/"completion.json",completion)
    print(json.dumps(completion,ensure_ascii=False),flush=True)


def analyze():
    from validmem_challenge_analysis import analyze_results
    analyze_results()
