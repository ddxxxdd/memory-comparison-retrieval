"""Score a run against conflict-only gold.

The stage-1 question is narrow: did the method put every conflicting old memory into the
candidate set, and how much of the candidate set is actually conflicting?

Reported per arm
----------------
conflict recall          |C n G| / |G|          G = conflict_old_memory_ids
full coverage            share of writes with G subset of C
conflict purity          |C n G| / |C|          how many candidates really conflict
avg candidates           mean |C|
library reduction        1 - mean|C| / bank size
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def score(gold_rows, results, library):
    by_id = {r["query_id"]: r for r in results}
    total = {"conflicts": 0, "found": 0, "full": 0, "writes": 0, "candidates": 0}
    per_query = []

    for gold in gold_rows:
        qid = gold["query_id"]
        if qid not in by_id:
            continue
        targets = set(gold["conflict_old_memory_ids"])
        candidates = {row["memory_id"] for row in by_id[qid]["candidates"]}
        hit = candidates & targets
        total["conflicts"] += len(targets)
        total["found"] += len(hit)
        total["full"] += int(targets <= candidates)
        total["writes"] += 1
        total["candidates"] += len(candidates)
        per_query.append({"query_id": qid, "conflicts": sorted(targets),
                          "candidates": len(candidates),
                          "recalled": sorted(hit), "missed": sorted(targets - candidates),
                          "false_candidates": len(candidates - targets)})

    ratio = lambda a, b: a / b if b else None
    return {
        "writes": total["writes"],
        "conflict_pairs": total["conflicts"],
        "conflict_recall": ratio(total["found"], total["conflicts"]),
        "full_coverage": ratio(total["full"], total["writes"]),
        "conflict_purity": ratio(total["found"], total["candidates"]),
        "avg_candidates": ratio(total["candidates"], total["writes"]),
        "library_reduction": 1 - (total["candidates"] / total["writes"]) / library if library else None,
        "per_query": per_query,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    run = args.run_dir.resolve()
    gold = read_jsonl(run / "input" / "gold.jsonl")
    dataset_gold = {g["query_id"]: g for g in read_jsonl(args.dataset / "gold.jsonl")}
    for row in gold:
        if "conflict_old_memory_ids" not in row:
            row["conflict_old_memory_ids"] = dataset_gold[row["query_id"]]["conflict_old_memory_ids"]
    library = json.loads((run / "run_config.json").read_text(encoding="utf-8"))["dataset"]["memory_count"]

    report = {"run": run.name, "library_size": library, "arms": {}}
    print(f"run       {run.name}")
    print(f"library   {library} old memories")
    print(f"writes    {len(gold)}")
    print()
    header = f"{'arm':<14}{'conf recall':>13}{'full cover':>12}{'conf purity':>13}{'avg cand':>10}{'reduction':>11}"
    print(header)
    print("-" * len(header))
    for path in sorted((run / "arms").glob("*.jsonl")):
        metrics = score(gold, read_jsonl(path), library)
        report["arms"][path.stem] = metrics
        print(f"{path.stem:<14}{metrics['conflict_recall']:>13.3f}"
              f"{metrics['full_coverage']:>12.3f}{metrics['conflict_purity']:>13.3f}"
              f"{metrics['avg_candidates']:>10.1f}{metrics['library_reduction']:>11.4f}")

    print()
    for arm, metrics in report["arms"].items():
        missed = [q for q in metrics["per_query"] if q["missed"]]
        print(f"{arm}: {len(missed)}/{metrics['writes']} 条写入漏掉冲突")
        for item in missed[:6]:
            print(f"    {item['query_id']}: missed {item['missed']} "
                  f"(候选 {item['candidates']}, 冲突 {len(item['conflicts'])})")

    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\njson written:", args.json_out)


if __name__ == "__main__":
    main()