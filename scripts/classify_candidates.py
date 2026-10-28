"""Classify every returned candidate so低纯度不会被误读成"召回错了"。

对每条写入,候选只落进以下几类之一:

  conflict        该写入真正的冲突旧记忆(金标)
  same_subject    同一主体、同一事项,但未被标冲突(例如只提到某店有优惠,未断言取值)
  other_subject   其他主体的同事项记录 —— 文本几乎一样,但按定义不是冲突
  same_user_slot  同一主体、不同事项
  unrelated       其他一切
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    run = args.run_dir.resolve()
    dataset = args.dataset.resolve()
    bank = {m["memory_id"]: m for m in read_jsonl(dataset / "memories.jsonl")}
    queries = {q["query_id"]: q for q in read_jsonl(dataset / "queries.jsonl")}
    gold = {g["query_id"]: g for g in read_jsonl(dataset / "gold.jsonl")}

    report = {"run": run.name, "arms": {}}
    for path in sorted((run / "arms").glob("*.jsonl")):
        buckets = collections.Counter()
        for row in read_jsonl(path):
            qid = row["query_id"]
            targets = set(gold[qid]["conflict_old_memory_ids"])
            qref = queries[qid]["source_ref"]
            qslot = (qref["conflict_entity"], qref["conflict_attribute"])
            for cand in row["candidates"]:
                mid = cand["memory_id"]
                mem = bank.get(mid)
                if mid in targets:
                    buckets["conflict"] += 1
                    continue
                if mem is None:
                    buckets["missing"] += 1
                    continue
                mref = mem["source_ref"]
                mslot = (mref["conflict_entity"], mref["conflict_attribute"])
                if mref["user_id"] != qref["user_id"]:
                    buckets["other_subject"] += 1
                elif mslot == qslot:
                    buckets["same_subject"] += 1
                else:
                    buckets["same_user_slot"] += 1
        total = sum(buckets.values())
        report["arms"][path.stem] = {"buckets": dict(buckets)}
        print(f"=== {path.stem} ===")
        print(f"  候选总数              {total}")
        for key in ("conflict", "same_subject", "other_subject", "same_user_slot", "missing"):
            print(f"  {key:<22}{buckets[key]:>6}")
        if total:
            print(f"  真冲突占候选           {buckets['conflict']/total:.3f}")
            print(f"  其他主体同名占候选       {buckets['other_subject']/total:.3f}")
        print()

    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("json written:", args.json_out)


if __name__ == "__main__":
    main()