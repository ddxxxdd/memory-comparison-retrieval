"""Export each STALE scenario as a standalone experiment dataset."""

import argparse
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_retrieval.data import Memory, load_queries, read_jsonl, validate_memories, write_json, write_jsonl
from memory_retrieval.evaluation import validate_dataset
from scripts.convert_stale_scenarios import DATA_ROOT, HISTORY_POLICY
from scripts.augment_stale_scenarios import validate_background_gold


def export_scenarios(dataset):
    dataset = Path(dataset).resolve()
    memory_rows = read_jsonl(dataset / "memories.jsonl")
    queries = load_queries(dataset / "queries.jsonl")
    gold = read_jsonl(dataset / "gold.jsonl")
    memories = [Memory.from_dict(row) for row in memory_rows]
    validate_memories(memories)
    banks = validate_dataset(memories, queries, gold)
    if None in banks or len(queries) != len(banks) or {q["scenario_id"] for q in queries} != set(banks):
        raise ValueError("STALE 导出需要每个场景恰好对应一个旧库和一条新记忆")

    raw_by_id = {row["memory_id"]: row for row in memory_rows}
    gold_by_query = {row["query_id"]: row for row in gold}
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}
    if manifest.get("history_policy") == HISTORY_POLICY:
        for query in queries:
            validate_background_gold([raw_by_id[m.memory_id] for m in banks[query["scenario_id"]]],
                                     gold_by_query[query["query_id"]])
    output = dataset / "scenarios"
    if output.resolve() != output or (output.exists() and not (output / "index.json").is_file()):
        raise ValueError(f"导出目标不是可重建的场景目录 {output}")

    index = []
    with TemporaryDirectory(dir=dataset, prefix="export-scenarios-") as temp:
        staging = Path(temp).resolve()
        if staging.parent != dataset:
            raise ValueError(f"临时导出目录必须位于数据集内 {dataset}")
        staged_output = staging / "scenarios"
        for query in queries:
            scenario_id = query["scenario_id"]
            scenario_type = query["source_ref"]["type"]
            if scenario_type not in ("T1", "T2") or not re.fullmatch(r"[A-Za-z0-9_-]+", scenario_id):
                raise ValueError(f"场景类型或 ID 无法用作目录名 {scenario_type}/{scenario_id}")
            relative_path = Path(scenario_type) / scenario_id
            folder = staged_output / relative_path
            folder.mkdir(parents=True)

            query_id = query["query_id"]
            bank = banks[scenario_id]
            write_jsonl(folder / "memories.jsonl", (raw_by_id[m.memory_id] for m in bank))
            write_jsonl(folder / "queries.jsonl", [query])
            write_jsonl(folder / "gold.jsonl", [gold_by_query[query_id]])
            index.append({"scenario_id": scenario_id, "scenario_type": scenario_type, "query_id": query_id,
                          "M_new": query["content"], "memory_count": len(bank), "path": relative_path.as_posix()})
        write_json(staged_output / "index.json", index)

        previous = staging / "previous"
        try:
            if output.exists():
                output.rename(previous)
            staged_output.rename(output)
        finally:
            if previous.exists() and not output.exists():
                previous.rename(output)
    print(f"{output} | {len(index)} scenarios, {len(memories)} old memories", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", default=[DATA_ROOT / "base", DATA_ROOT / "augmented"],
                        help="要导出的数据集目录，默认导出 STALE 原版和扩增版")
    args = parser.parse_args(argv)
    for dataset in args.dataset:
        export_scenarios(dataset)


if __name__ == "__main__":
    main()
