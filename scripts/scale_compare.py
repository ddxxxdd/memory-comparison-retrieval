"""Controlled scale comparison: same writes, two library sizes.

Reports per arm and per library the conflict recall, full-coverage rate, candidate
count, purity and library reduction, using the run_config of each run for the library
size. The point is to separate "does the method still find the superseded memories"
from "how many unrelated memories come along".
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"C:\Users\dzp\project\paper\memory-comparison-retrieval")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def score(run_dir, dataset):
    gold = {g["query_id"]: g for g in read_jsonl(dataset / "gold.jsonl")}
    ids = [json.loads(l)["query_id"] for l in (run_dir / "input" / "queries.jsonl").open(encoding="utf-8") if l.strip()]
    library = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["dataset"]["memory_count"]
    out = {}
    for path in sorted((run_dir / "arms").glob("*.jsonl")):
        rows = {r["query_id"]: r for r in read_jsonl(path)}
        conflicts = found = full = 0
        cand = required = hit = 0
        for qid in ids:
            g = gold[qid]
            target = set(g.get("gold_conflict_old_memory_ids") or [])
            compare = set(g.get("gold_compare_ids") or [])
            got = {c["memory_id"] for c in rows[qid]["candidates"]}
            conflicts += len(target)
            found += len(target & got)
            full += int(target <= got)
            cand += len(got)
            required += len(compare)
            hit += len(got & compare)
        out[path.stem] = {
            "library": library, "queries": len(ids), "conflict_pairs": conflicts,
            "conflict_recall": found / conflicts,
            "full_coverage": full / len(ids),
            "avg_candidates": cand / len(ids),
            "purity": hit / cand if cand else None,
            "reduction": 1 - (cand / len(ids)) / library,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", type=Path, required=True)
    ap.add_argument("--large", type=Path, required=True)
    ap.add_argument("--small-dataset", type=Path, required=True)
    ap.add_argument("--large-dataset", type=Path, required=True)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    a = score(args.small, args.small_dataset)
    b = score(args.large, args.large_dataset)
    libs = sorted({v["library"] for v in a.values()} | {v["library"] for v in b.values()})

    print("同一批新记忆写入,两个库容下的主方法对比")
    print()
    header = (f"{'arm':<10}{'library':>9}{'conf recall':>12}{'full cover':>12}"
              f"{'avg cand':>10}{'purity':>9}{'reduction':>11}")
    print(header)
    print("-" * len(header))
    rows = {}
    for arm in ("proposed", "dense", "bm25"):
        for label, table in (("small", a), ("large", b)):
            if arm not in table:
                continue
            m = table[arm]
            rows[f"{arm}@{m['library']}"] = m
            print(f"{arm:<10}{m['library']:>9}{m['conflict_recall']:>12.3f}"
                  f"{m['full_coverage']:>12.3f}{m['avg_candidates']:>10.1f}"
                  f"{m['purity']:>9.3f}{m['reduction']:>11.4f}")
    print()
    print("库容从 281 增到 1223(4.35x):")
    for arm in ("proposed", "dense", "bm25"):
        if arm in a and arm in b:
            print(f"  {arm:<10} conflict recall {a[arm]['conflict_recall']:.3f} -> "
                  f"{b[arm]['conflict_recall']:.3f}   "
                  f"avg candidates {a[arm]['avg_candidates']:.1f} -> {b[arm]['avg_candidates']:.1f}   "
                  f"reduction {a[arm]['reduction']:.4f} -> {b[arm]['reduction']:.4f}")

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\njson written:", args.json_out)


if __name__ == "__main__":
    main()