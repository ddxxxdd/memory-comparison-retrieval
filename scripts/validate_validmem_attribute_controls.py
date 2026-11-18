"""Additional explicit attribute pairs, fixed before inspecting main retrieval metrics."""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path
import validate_validmem_challenges as core

core.DATA=core.ROOT/"data/validmem_attribute_controls_v1"
core.RUN=core.ROOT/"runs/experiments/validmem-attribute-controls-20261112"
core.REPORT=core.ROOT/"reports/validmem_challenge_validation_2026-11-12/attribute_controls"
core.FAMILIES={name:(name,name,name) for name in ("purchase_preference","event_registration","essay_guidelines","adult_child_fee")}


def prepare():
    from validate_validmem_challenges import rows,dump,jsonl,digest,file_hash,DATA,SOURCE,SEEDS
    if (DATA/"manifest.json").exists():raise RuntimeError("Refuse to rebuild frozen additional controls")
    source_mem={r["memory_id"]:r for r in rows(SOURCE/"memories.jsonl")}
    primary=rows(core.ROOT/"data/validmem_challenge_validation_v1/anchors.jsonl")
    owners=sorted({a["query_fact"]["entity"] for a in primary if len(a["query_fact"]["entity"].split())==2},key=lambda n:digest(["attribute-owner",n]))[:7]
    texts={};memories={};banks={};cases=[];anchors=[]
    def tid(text):
        key="T-"+digest(text)[:20];texts[key]={"text_id":key,"content":text};return key
    for mid,m in source_mem.items():memories[mid]={"memory_id":mid,"text_id":tid(m["content"]),"role":"source_background"}
    def mid(text,entity,attribute,value):
        key="A-"+digest([text,entity,attribute,value])[:20]
        memories[key]={"memory_id":key,"text_id":tid(text),"role":"controlled","fact":{"entity":entity,"attribute":attribute,"value":value}}
        return key
    def sentence(family,name,value,other=False):
        if family=="purchase_preference":
            return f"{name} recently purchased apples from {value}." if other else f"{name}'s preferred grocery store is {value}."
        if family=="event_registration":
            return f"Registration for the community event organized by {name} closes on day {value}." if other else f"The community event organized by {name} is scheduled for day {value}."
        if family=="essay_guidelines":
            return f"The guidelines for the history essay assigned by {name} are available until day {value}." if other else f"The history essay assigned by {name} is due on day {value}."
        return f"{'Child' if other else 'Adult'} entry to the park managed by {name} costs ${value} per person."
    recipes={
        "purchase_preference":{"sources":["M-A-0211","M-A-0213"],"slots":["preferred_store","purchase_location"],"pairs":[("Walmart","Costco"),("Target","Whole Foods"),("GreenGrocers","Trader Joe's")],"proof":"Buying apples at a store does not assert that it is the preferred grocery store."},
        "event_registration":{"sources":["M-B-0106","M-B-0114","M-B-0207"],"slots":["event_date","registration_closing_date"],"pairs":[("45","35"),("50","20"),("55","15")],"proof":"Registration closes at or before the newly scheduled event; its deadline is a different attribute from the event date."},
        "essay_guidelines":{"sources":["M-B-0157","M-B-0199","M-B-0201"],"slots":["submission_deadline","guideline_availability_end"],"pairs":[("80","75"),("90","50"),("100","35")],"proof":"Guideline availability and essay submission deadline can have distinct values; no implication equates them."},
        "adult_child_fee":{"sources":["M-C-0185","M-C-0186","M-C-0187"],"slots":["adult_entry_fee","child_entry_fee"],"pairs":[("15","5"),("10","5"),("15","10")],"proof":"Adult and child prices have separate applicability conditions. Child price is no greater than the old adult price and equals the new adult price."},
    }
    # Every number/place is already present in the source corpus. The assignment of
    # named organizers/roles is explicitly counterfactual and never treated as natural.
    source_text="\n".join(m["content"] for m in source_mem.values())
    for recipe in recipes.values():
        assert all(value in source_text for pair in recipe["pairs"] for value in pair)
    for family,recipe in recipes.items():
        for i,owner in enumerate(owners[:6]):
            other=owners[(i+1)%len(owners)]
            old,new=recipe["pairs"][i%3]
            aid=f"AF-{family}-{i+1:02d}"
            qtext=sentence(family,owner,new)
            target=mid(sentence(family,owner,old),owner,recipe["slots"][0],old)
            same_other_attr=mid(sentence(family,owner,new,True),owner,recipe["slots"][1],new)
            other_other_attr=mid(sentence(family,other,new,True),other,recipe["slots"][1],new)
            same_value=mid(qtext,owner,recipe["slots"][0],new)
            anchors.append({"anchor_id":aid,"query_fact":{"entity":owner,"attribute":family,"value":new},
                            "target_value":old,"source_ids":recipe["sources"],"recipe":recipe,
                            "query_text":qtext,"target_text":sentence(family,owner,old),
                            "non_conflict_text":sentence(family,owner,new,True),"review":"fixed explicit slot proof; not human-reviewed natural gold"})
            for seed in SEEDS:
                # Fully explicit other-person backgrounds, no unqualified world facts.
                eligible=[]
                for source_id,m in source_mem.items():
                    named=re.match(r"^([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*)'s ",m["content"])
                    if source_id.startswith("M-A-") and named and owner.casefold() not in m["content"].casefold() and other.casefold() not in m["content"].casefold():eligible.append(source_id)
                eligible.sort(key=lambda x:digest([aid,seed,x]))
                assert len(eligible)>=199
                for variant,extra in (("no_injection",[]),("same_entity_other_attribute",[same_other_attr]),
                                      ("other_entity_other_attribute",[other_other_attr]),("same_value_restatement",[same_value])):
                    ids=[target]+extra+eligible[:199-len(extra)]
                    ids.sort(key=lambda x:digest(["bank-order",seed,x]))
                    bid="B-"+digest(ids)[:20];banks[bid]={"bank_id":bid,"memory_ids":ids}
                    # Recompute relation from explicit slots; no LLM label inference.
                    positive=[m for m in ids if memories[m]["role"]=="controlled" and memories[m]["fact"]["entity"]==owner
                              and memories[m]["fact"]["attribute"]==recipe["slots"][0] and memories[m]["fact"]["value"]!=new]
                    assert positive==[target]
                    cases.append({"case_id":"P-"+digest([aid,seed,variant])[:20],"anchor_id":aid,"family":family,
                                  "module":"attribute_specific","variant":variant,"seed":seed,"bank_id":bid,
                                  "query_text_id":tid(qtext),"bank_size":200,"added_distractors":len(extra),
                                  "gold_count":1,"wording":"canonical","gold_ids":[target]})
    jsonl(DATA/"texts.jsonl",sorted(texts.values(),key=lambda r:r["text_id"]))
    jsonl(DATA/"memories.jsonl",sorted(memories.values(),key=lambda r:r["memory_id"]))
    jsonl(DATA/"banks.jsonl",sorted(banks.values(),key=lambda r:r["bank_id"]))
    jsonl(DATA/"cases.jsonl",cases);jsonl(DATA/"anchors.jsonl",anchors)
    dump(DATA/"source_hashes.json",core.source_snapshot())
    dump(DATA/"manifest.json",{"protocol":"validmem-attribute-controls-v1","date":"2026-11-12",
         "anchors":len(anchors),"families":dict(Counter(a["query_fact"]["attribute"] for a in anchors)),
         "cases":len(cases),"unique_texts":len(texts),"skip_natural_replay":True,"human_reviewed":False,
         "selection":"All four predeclared slot pairs crossed with six hash-selected source names; no retrieval scores inspected",
         "files_sha256":{p.name:file_hash(p) for p in sorted(DATA.glob('*')) if p.is_file()},"builder_sha256":file_hash(__file__)})
    print(json.dumps(read_manifest(),ensure_ascii=False,indent=2))


def read_manifest():return core.read_json(core.DATA/"manifest.json")


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    phase=sys.argv[1]
    if phase=="prepare":prepare()
    else:
        import validmem_challenge_runtime as runtime
        if phase=="embed-qwen":runtime.embed_qwen()
        elif phase=="run-local":runtime.run_local()
        else:raise ValueError(phase)
