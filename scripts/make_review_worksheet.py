"""Build a human-review worksheet from a scored run.

One row per returned candidate of the sampled writes, with the bucket it falls in and
the text on both sides. The point of the worksheet is to settle the cross-scenario
same-slot records: they are the only class where the source provides no label, and they
dominate the difference between "purity 0.08" and "the retriever returned the wrong
memory".

Rows are ordered so that unresolved pairs come first.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BUCKET_ORDER = ["other_subject", "gold_conflict", "same_subject",
                "same_entity_other", "unrelated", "missing"]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--arm", default="proposed")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    run = args.run_dir.resolve()
    dataset = args.dataset.resolve()
    memories = {m["memory_id"]: m for m in read_jsonl(dataset / "memories.jsonl")}
    gold = {g["query_id"]: g for g in read_jsonl(dataset / "gold.jsonl")}
    queries = {q["query_id"]: q for q in read_jsonl(dataset / "queries.jsonl")}
    results = read_jsonl(run / "arms" / f"{args.arm}.jsonl")

    rows = []
    for row in results:
        qid = row["query_id"]
        g, q = gold[qid], queries[qid]
        ref = q["source_ref"]
        qslot = (ref["conflict_entity"], ref["conflict_attribute"])
        conflicts = set(g.get("conflict_old_memory_ids") or [])
        for cand in row["candidates"]:
            mid = cand["memory_id"]
            mem = memories[mid]
            mref = mem["source_ref"]
            mslot = (mref["conflict_entity"], mref["conflict_attribute"])
            if mid in conflicts:
                bucket = "gold_conflict"
            elif mref["user_id"] != ref["user_id"]:
                bucket = "other_subject"
            elif mslot == qslot:
                bucket = "same_subject"
            elif mslot[0] == qslot[0]:
                bucket = "same_entity_other"
            else:
                bucket = "unrelated"
            rows.append({
                "bucket": bucket,
                "query_id": qid,
                "new_content": q["content"],
                "new_scenario": ref["scenario_id"],
                "new_created_day": ref.get("created_day"),
                "old_memory_id": mid,
                "old_content": mem["content"],
                "old_scenario": mref.get("scenario_id"),
                "old_created_day": mref.get("created_day"),
                "old_role": mref.get("source_role"),
                "slot": f"{qslot[0]} | {qslot[1]}",
                "conflict_old_memory_ids": sorted(conflicts),
                "human_label": "",
                "human_note": "",
            })

    rows.sort(key=lambda r: (BUCKET_ORDER.index(r["bucket"]), r["query_id"], r["old_memory_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = collections.Counter(r["bucket"] for r in rows)
    print(f"worksheet rows {len(rows)} -> {args.output}")
    for bucket in BUCKET_ORDER:
        print(f"  {bucket:<20} {counts[bucket]:>4}")
    print()
    print("需要人工判定的行(bucket=same_slot_other):", counts["same_slot_other"])
    print("填写 human_label(conflict / no_conflict),再据此更新金标。")


if __name__ == "__main__":
    main()