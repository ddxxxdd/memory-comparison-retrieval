"""Frozen, source-derived controlled probes; existing baseline algorithms are imported.

All new data and results live outside the original ValidMem datasets and runs.
Phases: prepare, embed-qwen, run-local, analyze (see the frozen protocol).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import random
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/validmem_challenge_validation_v1"
RUN = ROOT / "runs/experiments/validmem-challenge-validation-20261112"
REPORT = ROOT / "reports/validmem_challenge_validation_2026-11-12"
SOURCE = ROOT / "data/validmem_v7"
SEEDS = (17, 29, 43, 71, 101)
KS = (1, 2, 3, 5, 10, 20, 50)
METHODS = ("bm25", "dense", "hybrid", "sparsecl")
FAMILIES = {
    "grocery": ("preferred grocery store", "first choice of grocery store", "首选的杂货店"),
    "coffee_brand": ("preferred coffee brand", "first choice of coffee brand", "首选的咖啡品牌"),
    "coffee_shop": ("favorite coffee shop", "first choice of coffee shop", "最喜欢的咖啡店"),
    "beverage": ("preferred morning beverage", "first choice of morning beverage", "首选的早晨饮品"),
    "vacation": ("preferred vacation destination", "first choice of vacation destination", "首选的度假目的地"),
    "workout": ("preferred workout location", "first choice of workout location", "首选的健身地点"),
    "delivery": ("preferred delivery option", "first choice of delivery option", "首选的配送方式"),
    "study": ("preferred study tool", "first choice of study tool", "首选的学习工具"),
    "shoe": ("preferred shoe size", "first choice of shoe size", "首选的鞋码"),
}
ALIASES = {values[0]: family for family, values in FAMILIES.items()}
ALIASES.update({"favorite coffee brand": "coffee_brand", "primary grocery store": "grocery", "default grocery store": "grocery"})
ATTRIBUTE = "(?:" + "|".join(re.escape(s) for s in sorted(ALIASES, key=len, reverse=True)) + ")"
NAME = r"[A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*"
PATTERNS = [
    re.compile(rf"^(?P<name>{NAME})'s (?P<attribute>{ATTRIBUTE}) (?:is(?: now)?|set to|(?:has )?(?:changed|updated|switched) to) (?P<value>.+)$"),
    re.compile(rf"^(?P<name>{NAME}) (?:changed|updated) (?:their )?(?P<attribute>{ATTRIBUTE}) to (?P<value>.+)$"),
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng(*parts):
    return random.Random(int(digest(parts)[:16], 16))


def parse_fact(text):
    for pattern in PATTERNS:
        match = pattern.fullmatch(text.strip().rstrip("."))
        if match:
            value = match["value"].strip()
            return {"entity": match["name"], "attribute": ALIASES[match["attribute"]], "value": value}
    return None


def norm(value):
    return value.strip(" .\"'").casefold()


def render(fact, style="canonical"):
    entity, attr, value = fact["entity"], fact["attribute"], fact["value"]
    canonical, paraphrase, chinese = FAMILIES[attr]
    if style == "canonical":
        return f"{entity}'s {canonical} is {value}."
    if style == "paraphrase":
        return f"For {entity}, the {paraphrase} is {value}."
    if style == "chinese":
        return f"{entity}{chinese}是{value}。"
    if style == "mentioned":
        return f"{entity} mentioned {value} in a conversation."
    raise ValueError(style)


def source_snapshot():
    paths = list(SOURCE.glob("*"))
    paths += [ROOT / "memory_retrieval" / name for name in ("baselines.py", "pipeline.py", "sparsecl.py", "api.py")]
    paths += [ROOT / "scripts" / name for name in ("run_experiment.py", "sparsecl_worker.py")]
    for run in ("validmem-v7-main-20261030", "validmem-v7-sparsecl-exploratory-20261031-retry1"):
        base = ROOT / "runs/experiments" / run
        paths += list((base / "arms").glob("*.jsonl")) + [base / "run_config.json"]
    paths += [ROOT / "reports/validmem_failure_analysis_2026-11-12/report.zh.md"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in paths if p.is_file()}


def prepare():
    if (DATA / "manifest.json").exists():
        raise RuntimeError("Frozen dataset already exists; refusing to regenerate after observing scores")
    memories = {r["memory_id"]: r for r in rows(SOURCE / "memories.jsonl")}
    queries = {r["query_id"]: r for r in rows(SOURCE / "queries.jsonl")}
    gold = {r["query_id"]: r for r in rows(SOURCE / "gold.jsonl")}
    anchors, exclusions = [], []
    for qid, query in sorted(queries.items()):
        fact = parse_fact(query["content"])
        old = [(mid, parse_fact(memories[mid]["content"])) for mid in gold[qid]["conflict_old_memory_ids"]]
        if not fact or not old or any(not item or item["entity"] != fact["entity"] or
                                      item["attribute"] != fact["attribute"] or norm(item["value"]) == norm(fact["value"])
                                      for mid,item in old):
            exclusions.append({"query_id":qid,"reason":"outside explicit preference grammar or gold pair not verified by that grammar"})
            continue
        # Select a target without consulting any saved rank or failure category.
        target_id,target = rng("target",qid).choice(sorted(old))
        anchors.append({"anchor_id":qid,"source_query":query["content"],"query_fact":fact,
                        "source_target_id":target_id,"source_target":memories[target_id]["content"],
                        "target_fact":target,"source_gold_ids":[mid for mid,_ in old]})
    assert anchors and len({a["anchor_id"] for a in anchors})==len(anchors)
    entities = sorted({a["query_fact"]["entity"] for a in anchors})
    assert len(entities) > 16
    value_sources = defaultdict(dict)
    for row in list(memories.values()) + list(queries.values()):
        fact = parse_fact(row["content"])
        if fact:
            value_sources[fact["attribute"]].setdefault(norm(fact["value"]), {
                "value":fact["value"],"source_id":row.get("memory_id",row.get("query_id")),"source_text":row["content"]})
    texts, controlled_memories, banks, cases = {}, {}, {}, []

    def text_id(content):
        key = "T-" + digest(content)[:20]
        prior = texts.setdefault(key, {"text_id":key,"content":content})
        assert prior["content"]==content
        return key

    # Reused original background memory records. Only their text enters the baseline.
    for mid,row in memories.items():
        controlled_memories[mid] = {"memory_id":mid,"text_id":text_id(row["content"]),
                                    "source_memory_id":mid,"role":"source_background"}

    def add_memory(fact, style="canonical", source_ids=()):
        tid = text_id(render(fact,style))
        mid = "C-" + digest([fact,style])[:20]
        controlled_memories.setdefault(mid,{"memory_id":mid,"text_id":tid,"fact":fact,"style":style,
                                            "source_ids":list(source_ids),"role":"controlled"})
        return mid

    def make_case(anchor, module, variant, seed, foreground, query_text, n=200, d=0, g=1, style="canonical"):
        entity = anchor["query_fact"]["entity"]
        # Original background lacks the explicit named target subject. Local v7 has
        # no alias linking; exclude exact foreground texts as well to avoid duplicates.
        foreground_texts = {controlled_memories[mid]["text_id"] for mid in foreground}
        eligible = [mid for mid,r in memories.items() if entity.casefold() not in r["content"].casefold()
                    and (not parse_fact(r["content"]) or parse_fact(r["content"])["attribute"]!=anchor["query_fact"]["attribute"])
                    and controlled_memories[mid]["text_id"] not in foreground_texts]
        order = sorted(eligible,key=lambda mid:digest(["background",anchor["anchor_id"],seed,mid]))
        assert len(order)>=n-len(foreground),(anchor["anchor_id"],len(order),n)
        selected = list(foreground) + order[:n-len(foreground)]
        assert len(set(selected))==n
        # Order is a fixed hash of IDs, independent of gold status and score.
        selected.sort(key=lambda mid:digest(["bank-order",seed,mid]))
        bank_id = "B-" + digest(selected)[:20]
        banks.setdefault(bank_id,{"bank_id":bank_id,"memory_ids":selected})
        qfact = anchor["query_fact"]
        targets=[]
        relation_counts=Counter()
        for mid in selected:
            memory=controlled_memories[mid]
            if memory["role"]=="source_background":
                assert entity.casefold() not in memories[mid]["content"].casefold()
                relation_counts["different_explicit_subject_or_no_target_subject"]+=1
                continue
            old=memory["fact"]
            same_entity=old["entity"]==qfact["entity"]
            same_attribute=memory["style"]!="mentioned" and old["attribute"]==qfact["attribute"]
            if same_entity and same_attribute and norm(old["value"])!=norm(qfact["value"]):
                targets.append(mid); relation_counts["conflict"]+=1
            elif not same_entity: relation_counts["different_subject"]+=1
            elif not same_attribute: relation_counts["mention_does_not_assert_preference"]+=1
            else: relation_counts["same_value"]+=1
        assert len(targets)==g,(anchor["anchor_id"],module,variant,targets,g)
        key=[anchor["anchor_id"],module,variant,seed,n,d,g,style]
        cases.append({"case_id":"P-"+digest(key)[:20],"anchor_id":anchor["anchor_id"],
                      "family":qfact["attribute"],"module":module,"variant":variant,"seed":seed,
                      "bank_id":bank_id,"query_text_id":text_id(query_text),"bank_size":n,
                      "added_distractors":d,"gold_count":g,"wording":style,
                      "gold_ids":sorted(targets),"relation_counts":dict(relation_counts)})

    for anchor in anchors:
        qid=anchor["anchor_id"]
        qfact,ofact=anchor["query_fact"],anchor["target_fact"]
        target=add_memory(ofact,source_ids=[anchor["source_target_id"]])
        query=render(qfact)
        for seed in SEEDS:
            others=sorted([x for x in entities if x!=qfact["entity"]],key=lambda name:digest(["other-name",qid,seed,name]))[:15]
            hard=[add_memory({**qfact,"entity":name},source_ids=[qid]) for name in others]
            old_value=[add_memory({**ofact,"entity":name},source_ids=[anchor["source_target_id"]]) for name in others]
            make_case(anchor,"identity","no_injection",seed,[target],query,d=0)
            for d in (1,5,15):
                make_case(anchor,"identity","other_entity_query_value",seed,[target]+hard[:d],query,d=d)
                make_case(anchor,"identity","other_entity_old_value",seed,[target]+old_value[:d],query,d=d)
            # Same speaker vs different speaker, with exactly the same mention statement.
            mentioned_same=add_memory(qfact,"mentioned",[qid])
            mentioned_other=add_memory({**qfact,"entity":others[0]},"mentioned",[qid])
            same_value=add_memory(qfact,source_ids=[qid])
            for variant,mid in (("same_entity_mention",mentioned_same),("other_entity_mention",mentioned_other),
                                ("same_value_restatement",same_value)):
                make_case(anchor,"attribute",variant,seed,[target,mid],query,d=1)
            # Query wording alone changes; bank, gold, identity and value remain fixed.
            for style in ("original","canonical","paraphrase","chinese"):
                content=anchor["source_query"] if style=="original" else render(qfact,style)
                make_case(anchor,"wording",style,seed,[target]+hard[:5],content,d=5,style=style)
            # Bank scale alone changes, with nested deterministic background prefixes.
            for n in (50,200,1000):
                make_case(anchor,"scale","fixed_five_added_distractors",seed,[target]+hard[:5],query,n=n,d=5)
            # Distinct observed values; no duplicate gold sentence inflation.
            values=[v for key,v in sorted(value_sources[qfact["attribute"]].items()) if key!=norm(qfact["value"])]
            values=sorted(values,key=lambda v:digest(["gold-value",qid,seed,v["value"]]))
            if len(values)>=6:
                targets=[add_memory({**qfact,"value":v["value"]},source_ids=[v["source_id"]]) for v in values[:6]]
                for g in (1,2,4,6):
                    make_case(anchor,"cardinality","distinct_observed_values",seed,targets[:g]+hard[:5],query,d=5,g=g)
    assert len({c["case_id"] for c in cases})==len(cases)
    # Paired cases use the same sample rank order; replacements differ only at the
    # foreground positions (except exact duplicate exclusions, audited explicitly).
    case_groups=defaultdict(list)
    for c in cases: case_groups[c["anchor_id"],c["module"],c["seed"],c["added_distractors"]].append(c)
    pair_bank_checks=[]
    for key,group in case_groups.items():
        if key[1] not in {"identity","attribute","wording"}:continue
        background_sets=[{mid for mid in banks[c["bank_id"]]["memory_ids"] if mid in memories} for c in group]
        equal=all(s==background_sets[0] for s in background_sets)
        pair_bank_checks.append({"group":list(key),"same_background":equal})
    assert all(c["same_background"] for c in pair_bank_checks),"Exact-duplicate exclusions changed a paired background"
    jsonl(DATA/"anchors.jsonl",anchors)
    jsonl(DATA/"excluded_queries.jsonl",exclusions)
    jsonl(DATA/"texts.jsonl",sorted(texts.values(),key=lambda x:x["text_id"]))
    jsonl(DATA/"memories.jsonl",sorted(controlled_memories.values(),key=lambda x:x["memory_id"]))
    jsonl(DATA/"banks.jsonl",sorted(banks.values(),key=lambda x:x["bank_id"]))
    jsonl(DATA/"cases.jsonl",cases)
    dump(DATA/"value_sources.json",dict(value_sources))
    dump(DATA/"source_hashes.json",source_snapshot())
    manifest={"protocol":"validmem-challenge-validation-v1","date":"2026-11-12",
              "selection":"all queries accepted by explicit grammar; no scores read in prepare",
              "seeds":SEEDS,"budgets":KS,"anchors":len(anchors),"families":dict(Counter(a["query_fact"]["attribute"] for a in anchors)),
              "cases":len(cases),"cases_by_module":dict(Counter(c["module"] for c in cases)),
              "unique_banks":len(banks),"unique_texts":len(texts),"controlled_memories":len(controlled_memories),
              "background_pair_checks":len(pair_bank_checks),"all_paired_backgrounds_identical":True,
              "label_basis":"Deterministic entity/attribute/value rule in explicit single-current-preference frames; not LLM pair classification",
              "human_reviewed":False,"source_query_count":181,
              "files_sha256":{p.name:file_hash(p) for p in sorted(DATA.glob('*')) if p.is_file()},
              "builder_sha256":file_hash(__file__)}
    dump(DATA/"manifest.json",manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


def verify_frozen():
    manifest=read_json(DATA/"manifest.json")
    for name,expected in manifest["files_sha256"].items():
        assert file_hash(DATA/name)==expected,(name,"dataset changed")
    for path,expected in read_json(DATA/"source_hashes.json").items():
        assert file_hash(ROOT/path)==expected,(path,"original source changed")
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["prepare","embed-qwen","run-local","analyze"])
    args=parser.parse_args()
    if args.phase=="prepare":prepare()
    else:
        from validmem_challenge_runtime import embed_qwen, run_local, analyze
        if args.phase=="embed-qwen":embed_qwen()
        elif args.phase=="run-local":run_local()
        else:analyze()


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    main()
