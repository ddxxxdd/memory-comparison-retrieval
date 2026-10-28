"""Print recall and write latency across saved library-size evaluations."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reports", nargs="+", type=Path, help="evaluate 输出的 metrics JSON 文件")
args = parser.parse_args()

rows = []
for path in args.reports:
    report = json.loads(path.read_text(encoding="utf-8"))
    retrieval = report["candidate_retrieval"]
    latency = report["write_latency_ms"]["total"]
    rows.append((report["library_size"], retrieval["candidate_recall"], latency["p50"], latency["p95"], path.name))

print("library_size,candidate_recall,write_p50_ms,write_p95_ms,report")
for size, recall, p50, p95, name in sorted(rows, key=lambda row: (row[0] or 0, row[4])):
    print(f"{size if size is not None else ''},{recall if recall is not None else ''},"
          f"{p50 if p50 is not None else ''},{p95 if p95 is not None else ''},{name}")
