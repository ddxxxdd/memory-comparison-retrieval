"""Build the unaugmented STALE write-time dataset, one old bank per scenario."""

import argparse
import json
from collections import Counter
from pathlib import Path
import re
from statistics import median
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_retrieval.data import fingerprint, nonempty_text, sha256_file, write_json, write_jsonl


STALE_URL = "https://huggingface.co/datasets/STALEproj/STALE/resolve/main/T1_T2_400_FULL.json"
DATA_ROOT = ROOT / "data" / "stale_scenarios"
V2_ROOT = ROOT / "data" / "stale_scenarios_v2"
HISTORY_POLICY = "rule_only_background"
SENTENCE_BOUNDARY = re.compile(r'(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<=[.!?])\s+|\n+')
PERSONAL = re.compile(r"\b(?:I|my|we|our)\b", re.I)
NON_FACT_START = re.compile(
    r"^(?:thanks\b|thank you\b|if\b|what if\b|imagine\b|suppose\b|let's\b|"
    r"(?:yeah|yes|right|okay|ok|cool|great|sure|wow)[!.,\s]|"
    r"i (?:think|wonder|see|agree|guess|suspect|might|may|would|need|want|should|could)\b|"
    r"i'm (?:thinking|considering|trying|looking|wondering|curious|not sure)\b|"
    r"(?:please|could|can|would) you\b|(?:could|can|should|do|did|will) i\b|"
    r"(?:it|this|that|these|those|they)\b)",
    re.I,
)
META_TALK = re.compile(
    r"\b(?:follow-up (?:question|questions)|thanks for|your (?:advice|reply|response|suggestion|suggestions)|"
    r"could you|can you|would you|do you think)\b", re.I,
)


def iter_json_array(path, chunk_size=1024 * 1024):
    """Read a top-level JSON array without loading the full source into memory."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as stream:
        buffer = ""
        position = 0
        eof = False

        def fill():
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = stream.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        fill()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"{path} 顶层必须是 JSON 数组")
        position += 1

        while True:
            while True:
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position < len(buffer) or eof:
                    break
                fill()
            if position >= len(buffer):
                raise ValueError(f"{path} JSON 数组意外结束")
            if buffer[position] == "]":
                return
            try:
                row, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"{path} 含无效或不完整的 JSON 数组项") from None
                fill()
                continue
            if not isinstance(row, dict):
                raise ValueError(f"{path} 数组项必须是 JSON 对象")
            position = end
            yield row


def text_key(text):
    return " ".join(re.findall(r"\w+", text.casefold().replace("’", "'")))


def memory_row(scenario_id, content, record_type, **source):
    digest = fingerprint([scenario_id, text_key(content)])[:24]
    return {"memory_id": f"stale_m_{digest}", "scenario_id": scenario_id, "content": content,
            "source_ref": {"dataset": "STALE", "record_type": record_type, **source}}


def history_claims(session, *, all_sentences=False):
    """Extract eligible user sentences in dialogue order."""
    selected = []
    for turn_index, turn in enumerate(session):
        if turn["role"] != "user":
            continue
        for excerpt in SENTENCE_BOUNDARY.split(turn["content"]):
            excerpt = excerpt.strip()
            normalized = excerpt.replace("’", "'")
            if (7 <= len(excerpt.split()) <= 55 and excerpt.endswith((".", "!"))
                    and "?" not in excerpt and PERSONAL.search(excerpt)
                    and not NON_FACT_START.search(normalized) and not META_TALK.search(normalized)):
                selected.append((turn_index, excerpt))
                if not all_sentences:
                    break
    return selected


def convert_scenario(row, *, version="v1"):
    if version not in ("v1", "v2"):
        raise ValueError(f"Unknown STALE extraction version: {version}")
    scenario_id = nonempty_text(row.get("uid"), "STALE uid")
    old = nonempty_text(row.get("M_old"), "STALE M_old")
    new = nonempty_text(row.get("M_new"), "STALE M_new")
    old_session, new_session = row["relevant_session_index"]
    sessions = row["haystack_session"]
    if not 0 <= old_session < new_session < len(sessions):
        raise ValueError(f"{scenario_id}: STALE 会话顺序无效")
    native = memory_row(scenario_id, old, "M_old", session_index=old_session)
    query = {"query_id": f"stale_q_{scenario_id}", "scenario_id": scenario_id, "content": new,
             "source_ref": {"dataset": "STALE", "record_type": "M_new",
                            "type": row["type"], "session_index": new_session}}
    gold = {"query_id": query["query_id"], "conflict_old_memory_ids": [native["memory_id"]],
            "target_details": {native["memory_id"]: {"source": "stale_native", "relation": "official_stale"}},
            "fully_reviewed": False, "human_reviewed": False}
    memories = [native]
    seen = {text_key(old), text_key(new)}
    anchor = text_key(old)
    claims = [history_claims(turns, all_sentences=version == "v2") for turns in sessions[:new_session]]
    # Filter the dialogue once. Reserve V1's occurrences before admitting other matches.
    for v1_limits in ((True, False) if version == "v2" else (True,)):
        for session_index, excerpts in enumerate(claims):
            count = 0
            previous_turn = None
            for turn_index, content in excerpts:
                if v1_limits and turn_index == previous_turn:
                    continue
                previous_turn = turn_index
                key = text_key(content)
                if key in seen or anchor in key or key in anchor:
                    continue
                seen.add(key)
                memories.append(memory_row(scenario_id, content, "history",
                                           session_index=session_index, turn_index=turn_index))
                count += 1
                if v1_limits and count == 3:
                    break
    return sorted(memories, key=lambda item: item["memory_id"]), query, gold


def library_stats(memories):
    sizes = list(Counter(row["scenario_id"] for row in memories).values())
    return {"min": min(sizes), "median": median(sizes), "max": max(sizes),
            "mean": sum(sizes) / len(sizes)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="STALE T1_T2_400_FULL.json")
    parser.add_argument("--version", choices=("v1", "v2"), default="v1",
                        help="v2 includes every matching sentence in every completed pre-write session")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.output = args.output or (V2_ROOT if args.version == "v2" else DATA_ROOT) / "base"
    if args.output.exists():
        raise ValueError(f"输出目录已存在：{args.output.resolve()}")
    memories, queries, gold = [], [], []
    for source_row in iter_json_array(args.source):
        old_bank, query, targets = convert_scenario(source_row, version=args.version)
        memories.extend(old_bank)
        queries.append(query)
        gold.append(targets)
    if not queries or len({row["scenario_id"] for row in queries}) != len(queries):
        raise ValueError("STALE 场景不能为空或重复")
    stats = {"source": STALE_URL, "source_sha256": sha256_file(args.source),
             "scenarios": len(queries), "scenario_types": dict(Counter(q["source_ref"]["type"] for q in queries)),
             "memories": len(memories), "memories_per_scenario": library_stats(memories),
             "history_extraction": "Up to 3 original user excerpts per completed session before the M_new session.",
             "gold_coverage": "Official M_old only. Background excerpts have no semantic labels in this version."}
    if args.version == "v2":
        stats.update(extraction_version="v2", history_policy=HISTORY_POLICY, history_extraction=(
            "All eligible user sentences in completed pre-write sessions, deduplicated within each scenario."))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix="convert-") as stage:
        dataset = Path(stage) / "dataset"
        dataset.mkdir()
        for name, rows in (("memories", memories), ("queries", queries), ("gold", gold)):
            write_jsonl(dataset / f"{name}.jsonl", rows)
        write_json(dataset / "manifest.json", stats)
        dataset.rename(args.output.resolve())
    print(f"{args.output.resolve()} | {len(queries)} scenarios, {len(memories)} old memories | "
          f"bank sizes: {stats['memories_per_scenario']}", flush=True)


if __name__ == "__main__":
    main()
