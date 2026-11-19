"""Load the three experiment inputs and restrict the candidate bank per query."""

from dataclasses import dataclass
from pathlib import Path
import random

from .data import Memory, nonempty_text, read_jsonl, validate_memories
from .evaluation import validate_dataset


def select_queries(queries, limit, seed, query_ids=None, scenario_ids=None):
    """Explicit IDs take precedence over random sampling and preserve requested order."""
    if query_ids:
        wanted = list(dict.fromkeys(query_ids))
        by_id = {row["query_id"]: row for row in queries}
        missing = [qid for qid in wanted if qid not in by_id]
        if missing:
            raise ValueError(f"--query-ids 中有查询不在数据集中:{missing}")
        return [by_id[qid] for qid in wanted]
    if scenario_ids:
        by_scenario = {sid: [] for sid in scenario_ids}
        for row in queries:
            scenario_id = row.get("scenario_id")
            if scenario_id in by_scenario:
                by_scenario[scenario_id].append(row)
        missing = [sid for sid, rows in by_scenario.items() if not rows]
        if missing:
            raise ValueError(f"--scenario-ids 指定的场景 {missing} 不在数据集中")
        return [row for rows in by_scenario.values() for row in rows]
    if limit is None or limit >= len(queries):
        return list(queries)
    if limit < 1:
        raise ValueError("--limit 必须大于 0")
    return random.Random(seed).sample(queries, limit)


@dataclass
class Dataset:
    paths: dict
    memories: list[Memory]
    queries: list[dict]
    gold: list[dict]

    def groups(self):
        """Indices into the selected memory/query arrays, in input order."""
        by_scene = {}
        for i, memory in enumerate(self.memories):
            by_scene.setdefault(memory.scenario_id, ([], []))[0].append(i)
        for i, query in enumerate(self.queries):
            by_scene[query.get("scenario_id")][1].append(i)
        return by_scene


def load_dataset(directory, *, scenario_ids=None, query_ids=None, limit=0, seed=20261014,
                 queries_file=None, memories_file=None, gold_file=None):
    directory = Path(directory).resolve()
    paths = {name: Path(override or directory / f"{name}.jsonl").resolve()
             for name, override in (("queries", queries_file), ("memories", memories_file), ("gold", gold_file))}
    memories = [Memory.from_dict({**row, "source_ref": row.get("source_ref") or "text",
                                  "content": row.get("content") or row.get("text")})
                for row in read_jsonl(paths["memories"])]
    validate_memories(memories)
    queries = read_jsonl(paths["queries"])
    for row in queries:
        nonempty_text(row.get("query_id"), "query_id")
        row["content"] = nonempty_text(row.get("content") or row.get("query_text"), "query content")
        if row.get("scenario_id") is not None:
            nonempty_text(row["scenario_id"], "scenario_id")
    if not queries or len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("查询不能为空，且 query_id 必须唯一")
    if paths["gold"].is_file() or gold_file or not all("gold_memory_id" in q for q in queries):
        gold = read_jsonl(paths["gold"])
        for row in gold:
            if "conflict_labels" in row and "gold_compare_ids" not in row and "conflict_old_memory_ids" not in row:
                row["gold_compare_ids"] = list(row["conflict_labels"])
    else:
        gold = [{"query_id": q["query_id"], "conflict_old_memory_ids": [q["gold_memory_id"]]} for q in queries]
        del paths["gold"]
    validate_dataset(memories, queries, gold)
    selected = select_queries(queries, limit or None, seed, query_ids, scenario_ids)
    scenes = {q.get("scenario_id") for q in selected}
    gold_by_query = {row["query_id"]: row for row in gold}
    return Dataset(paths, [m for m in memories if m.scenario_id in scenes], selected,
                   [gold_by_query[q["query_id"]] for q in selected])
