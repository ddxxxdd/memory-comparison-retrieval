"""Exact confusion matrix for the frozen judge against the dataset's gold labels.

Separates two questions that are easy to conflate:
  A. retrieval  — did the candidate set contain the old memories the write supersedes?
  B. judgment   — did the frozen judge call those pairs "conflict"?

The frozen judge is defined on logical contradiction and explicitly calls a clear
state change "no_conflict". ValidMem Part A calls the same relation a supersession
conflict. So B is expected to be lower than A, and the size of that gap is the result
worth reporting, not a bug to hide.
"""

import collections
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
run = Path(sys.argv[1])
dataset = Path(sys.argv[2])
arm = sys.argv[3] if len(sys.argv) > 3 else "proposed"


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


gold = {g["query_id"]: g for g in read_jsonl(dataset / "gold.jsonl")}
results = read_jsonl(run / "arms" / f"{arm}.jsonl")

retrieved = total_conf = missed_retrieval = 0
tp = fn_judged_nc = fn_judged_unc = fn_not_retrieved = 0
positive_retrieved = 0
extra_conflict = 0
extra_total = 0
per_query = []

for row in results:
    qid = row["query_id"]
    g = gold[qid]
    conflicts = set(g.get("gold_conflict_old_memory_ids") or [])
    candidates = {c["memory_id"] for c in row["candidates"]}
    verdicts = {j["memory_id"]: j["conflict_label"] for j in row.get("judgments") or []}

    hit = conflicts & candidates
    total_conf += len(conflicts)
    retrieved += len(hit)
    positive_retrieved += len(hit)
    fn_not_retrieved += len(conflicts - candidates)

    for mid in hit:
        label = verdicts.get(mid)
        if label == "conflict":
            tp += 1
        elif label == "uncertain":
            fn_judged_unc += 1
        else:
            fn_judged_nc += 1

    for mid, label in verdicts.items():
        if mid in conflicts:
            continue
        extra_total += 1
        if label == "conflict":
            extra_conflict += 1

    per_query.append({
        "query_id": qid,
        "conflicts": len(conflicts),
        "retrieved": len(hit),
        "judged_conflict": sum(1 for m in hit if verdicts.get(m) == "conflict"),
        "judged_no_conflict": sum(1 for m in hit if verdicts.get(m) == "no_conflict"),
        "judged_uncertain": sum(1 for m in hit if verdicts.get(m) == "uncertain"),
        "not_retrieved": len(conflicts - candidates),
    })

print(f"run {run.name}  arm {arm}")
print()
print("A. retrieval stage (candidate set vs superseded set)")
print(f"   superseded pairs total        {total_conf}")
print(f"   present in candidate set      {retrieved}   recall {retrieved/total_conf:.3f}")
print(f"   missing from candidate set    {fn_not_retrieved}")
print()
print("B. judgment stage (frozen judge on the retrieved superseded pairs)")
print(f"   judged conflict               {tp}")
print(f"   judged no_conflict            {fn_judged_nc}")
print(f"   judged uncertain              {fn_judged_unc}")
print(f"   end-to-end conflict recall    {tp/total_conf:.3f}  ({tp}/{total_conf})")
print()
print(f"   judge conflict calls on non-gold candidates: {extra_conflict}/{extra_total}"
      f"  ({extra_conflict/max(extra_total,1):.3f})")
print()
print("per write:")
print(f"   {'query':<12}{'conflicts':>10}{'retrieved':>11}{'conf':>7}{'no_conf':>9}{'uncert':>8}{'missed':>8}")
for q in per_query:
    print(f"   {q['query_id']:<12}{q['conflicts']:>10}{q['retrieved']:>11}"
          f"{q['judged_conflict']:>7}{q['judged_no_conflict']:>9}"
          f"{q['judged_uncertain']:>8}{q['not_retrieved']:>8}")

summary = {
    "run": run.name, "arm": arm,
    "superseded_pairs": total_conf,
    "retrieval_recall": retrieved / total_conf,
    "judge_conflict": tp,
    "judge_no_conflict": fn_judged_nc,
    "judge_uncertain": fn_judged_unc,
    "not_retrieved": fn_not_retrieved,
    "end_to_end_conflict_recall": tp / total_conf,
    "judge_positive_rate_on_non_gold": extra_conflict / max(extra_total, 1),
    "per_query": per_query,
}
out = run / f"judge_gap_{arm}.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("\njson written:", out)