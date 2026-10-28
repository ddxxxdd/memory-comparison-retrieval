"""Convert MemConflict, HaluMem, and STALE into one shared retrieval dataset."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MEMCONFLICT = WORKSPACE_ROOT / "tmp" / "MemConflict_Step4_4.jsonl"
DEFAULT_HALUMEM = WORKSPACE_ROOT / "tmp" / "HaluMem-Medium.jsonl"
DEFAULT_STALE = WORKSPACE_ROOT / "tmp" / "T1_T2_400_FULL.json"
DEFAULT_OUTPUT = ROOT / "data" / "combined_sources_v1"
MEMCONFLICT_URL = "https://github.com/TaoZhen1110/MemConflict/blob/main/Data/Step4_4.jsonl"
HALUMEM_URL = "https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Medium.jsonl"
STALE_URL = "https://huggingface.co/datasets/STALEproj/STALE/resolve/main/T1_T2_400_FULL.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no} 必须是 JSON 对象")
                yield row


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


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows),
        encoding="utf-8",
    )


def display_value(value):
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def display_path(path):
    return " > ".join(part.replace("_", " ") for part in path.split("."))


def render_claim(name, fact, *, related=False):
    subject = f"A related entity associated with {name}" if related else name
    return f"{subject} | {display_path(fact['path'])}: {display_value(fact['value'])}"


def extract_static_facts(row):
    facts = []
    for session in row.get("Full_Session_Chain", []):
        for item in session.get("Static_Conflict_Information", []):
            role = item.get("Role")
            if role not in {"Point_A", "Point_B", "Distractor"}:
                continue
            facts.append({
                "role": role,
                "conflict_id": item.get("Conflict_ID"),
                "path": item.get("Target_Field_Path"),
                "value": item.get("Value"),
                "session_index": int(session["Session_ID"]),
                "date": session.get("Date"),
            })
    return facts


def convert_memconflict(path):
    source_hash = sha256_file(path)
    rows = list(iter_jsonl(path))
    if len(rows) != 30:
        raise ValueError(f"MemConflict 固定来源应包含 30 个 profile，实际为 {len(rows)}")

    memories, queries, gold = [], [], []
    for profile in rows:
        profile_id = profile["ID"]
        short_id = profile_id.replace("-", "")[:8]
        name = profile["Fixed_Profile"]["Name"]
        facts = extract_static_facts(profile)
        point_a = [fact for fact in facts if fact["role"] == "Point_A"]
        point_b = [fact for fact in facts if fact["role"] == "Point_B"]
        if len(point_a) != 12 or len(point_b) != 12:
            raise ValueError(f"{profile_id}: Point_A 和 Point_B 都应各有 12 条")
        first_b_session = min(fact["session_index"] for fact in point_b)
        if max(fact["session_index"] for fact in point_a) >= first_b_session:
            raise ValueError(f"{profile_id}: Point_A 未全部早于 Point_B")

        a_by_conflict = {fact["conflict_id"]: fact for fact in point_a}
        b_by_conflict = {fact["conflict_id"]: fact for fact in point_b}
        if len(a_by_conflict) != len(point_a) or set(a_by_conflict) != set(b_by_conflict):
            raise ValueError(f"{profile_id}: Point_A/Point_B Conflict_ID 未一一对应")
        for conflict_id, b_fact in b_by_conflict.items():
            a_fact = a_by_conflict[conflict_id]
            if a_fact["path"] != b_fact["path"] or display_value(a_fact["value"]) == display_value(b_fact["value"]):
                raise ValueError(f"{profile_id}/{conflict_id}: 来源配对不是同字段异值")

        distractors = [fact for fact in facts
                       if fact["role"] == "Distractor" and fact["session_index"] < first_b_session]
        memory_facts = [(fact, "user") for fact in point_a]
        memory_facts.extend((fact, "related_entity") for fact in distractors)
        profile_memories, memory_meta, point_a_ids = [], {}, {}
        distractor_index = 0
        for fact, scope in sorted(memory_facts,
                                  key=lambda pair: (pair[0]["session_index"], pair[0]["role"], pair[0]["path"])):
            if scope == "user":
                memory_id = f"m_mc_{short_id}_a_{fact['conflict_id'].lower()}"
                point_a_ids[fact["conflict_id"]] = memory_id
            else:
                distractor_index += 1
                memory_id = f"m_mc_{short_id}_d_{distractor_index:03d}"
            source_ref = {
                "dataset": "MemConflict", "source_file": MEMCONFLICT_URL,
                "source_file_sha256": source_hash, "profile_id": profile_id,
                "profile_name": name, "session_index": fact["session_index"],
                "date": fact["date"], "role": fact["role"], "scope": scope,
                "conflict_id": fact["conflict_id"], "target_field_path": fact["path"],
            }
            item = {"memory_id": memory_id,
                    "content": render_claim(name, fact, related=scope == "related_entity"),
                    "source_ref": source_ref}
            profile_memories.append(item)
            memory_meta[memory_id] = {**fact, "scope": scope}

        memories.extend(profile_memories)
        for fact in sorted(point_b, key=lambda item: (item["session_index"], item["conflict_id"])):
            query_id = f"q_mc_{short_id}_{fact['conflict_id'].lower()}"
            query = {
                "query_id": query_id, "content": render_claim(name, fact),
                "source_ref": {
                    "dataset": "MemConflict", "source_file": MEMCONFLICT_URL,
                    "source_file_sha256": source_hash, "profile_id": profile_id,
                    "profile_name": name, "session_index": fact["session_index"],
                    "date": fact["date"], "role": fact["role"],
                    "conflict_id": fact["conflict_id"], "target_field_path": fact["path"],
                },
            }
            same_field_ids = [memory["memory_id"] for memory in profile_memories
                              if memory_meta[memory["memory_id"]]["scope"] == "user"
                              and memory_meta[memory["memory_id"]]["path"] == fact["path"]]
            target_id = point_a_ids[fact["conflict_id"]]
            if target_id not in same_field_ids:
                raise ValueError(f"{profile_id}/{fact['conflict_id']}: 缺少 Point_A 配对")
            old_value = a_by_conflict[fact["conflict_id"]]["value"]
            labels = {}
            for memory_id in same_field_ids:
                memory = memory_meta[memory_id]
                if memory["conflict_id"] == fact["conflict_id"]:
                    labels[memory_id] = "conflict"
                elif display_value(memory["value"]) == display_value(old_value):
                    labels[memory_id] = "conflict"
                elif display_value(memory["value"]) == display_value(fact["value"]):
                    labels[memory_id] = "no_conflict"
                else:
                    labels[memory_id] = "uncertain"
            queries.append(query)
            gold.append({"query_id": query_id, "fully_reviewed": True, "human_reviewed": False,
                         "gold_compare_ids": same_field_ids, "conflict_labels": labels,
                         "label_source": "MemConflict source Conflict_ID and target field"})
    return memories, queries, gold


def is_update(value):
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def stable_id(prefix, user_id, text):
    suffix = hashlib.sha256(f"{user_id}\0{text}".encode("utf-8")).hexdigest()[:16]
    user = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{user}_{suffix}"


def convert_halumem(path):
    source_hash = sha256_file(path)
    memories, queries, links = [], [], []
    memory_by_user_content = {}
    for user in iter_jsonl(path):
        user_id = user.get("uuid")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("HaluMem user 缺少 uuid")
        for session_index, session in enumerate(user.get("sessions", [])):
            for point in session.get("memory_points", []):
                content = point.get("memory_content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"HaluMem {user_id}/{session_index}: memory_content 为空")
                if is_update(point.get("is_update")):
                    originals = point.get("original_memories") or []
                    if not originals:
                        raise ValueError(f"HaluMem update {user_id}/{session_index}/{point.get('index')} 没有 original_memories")
                    if not all(isinstance(item, str) and item.strip() for item in originals):
                        raise ValueError("HaluMem original_memories 必须是非空文本数组")
                    query_id = stable_id("q_halu", user_id,
                                         f"{session_index}:{point.get('index')}:{content}")
                    query = {"query_id": query_id, "content": content,
                             "source_ref": {"dataset": "HaluMem", "source_file": HALUMEM_URL,
                                            "source_file_sha256": source_hash, "user_id": user_id,
                                            "session_index": session_index, "memory_index": point.get("index"),
                                            "memory_type": point.get("memory_type"),
                                            "memory_source": point.get("memory_source"),
                                            "timestamp": point.get("timestamp")}}
                    links.append((query, list(dict.fromkeys(item.strip() for item in originals))))
                    continue
                key = (user_id, content.strip())
                if key in memory_by_user_content:
                    continue
                memory_id = "m_halu_" + stable_id("u", user_id, content.strip())
                memory = {"memory_id": memory_id, "content": content.strip(),
                          "source_ref": {"dataset": "HaluMem", "source_file": HALUMEM_URL,
                                         "source_file_sha256": source_hash, "user_id": user_id,
                                         "record_type": "non_update_memory_point",
                                         "session_index": session_index, "memory_index": point.get("index"),
                                         "memory_type": point.get("memory_type"),
                                         "memory_source": point.get("memory_source"),
                                         "timestamp": point.get("timestamp")}}
                memory_by_user_content[key] = memory
                memories.append(memory)

    queries, gold, jobs = [], [], []
    for query, originals in links:
        user_id = query["source_ref"]["user_id"]
        compare_ids = []
        for original in originals:
            key = (user_id, original)
            memory = memory_by_user_content.get(key)
            if memory is None:
                memory = {"memory_id": "m_halu_" + stable_id("u", user_id, original),
                          "content": original,
                          "source_ref": {"dataset": "HaluMem", "source_file": HALUMEM_URL,
                                         "source_file_sha256": source_hash, "user_id": user_id,
                                         "record_type": "original_memory_link"}}
                memory_by_user_content[key] = memory
                memories.append(memory)
            compare_ids.append(memory["memory_id"])
            jobs.append((query["query_id"], query["content"], memory["memory_id"], memory["content"]))
        queries.append(query)
        gold.append({"query_id": query["query_id"], "fully_reviewed": True, "human_reviewed": False,
                     "gold_compare_ids": compare_ids, "conflict_labels": {}, "label_source": "llm_judge"})
    return memories, queries, gold, jobs


def convert_stale(path):
    source_hash = sha256_file(path)
    memories, queries, gold, jobs = [], [], [], []
    for row in iter_json_array(path):
        uid, old, new = row.get("uid"), row.get("M_old"), row.get("M_new")
        if not all(isinstance(value, str) and value.strip() for value in (uid, old, new)):
            raise ValueError("STALE 每项必须含非空 uid、M_old、M_new")
        memory_id = f"m_stale_{uid}"
        query_id = f"q_stale_{uid}"
        memories.append({"memory_id": memory_id, "content": old,
                         "source_ref": {"dataset": "STALE", "source_file": STALE_URL,
                                        "source_file_sha256": source_hash, "uid": uid,
                                        "record_type": "M_old", "type": row.get("type")}})
        queries.append({"query_id": query_id, "content": new,
                        "source_ref": {"dataset": "STALE", "source_file": STALE_URL,
                                       "source_file_sha256": source_hash, "uid": uid,
                                       "record_type": "M_new", "type": row.get("type")}})
        gold.append({"query_id": query_id, "fully_reviewed": True, "human_reviewed": False,
                     "gold_compare_ids": [memory_id], "conflict_labels": {}, "label_source": "llm_judge"})
        jobs.append((query_id, new, memory_id, old))
    return memories, queries, gold, jobs


def judge_jobs(jobs, judge_fn, workers=4):
    labels = {}
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge_fn, query, memory): (query_id, memory_id)
                   for query_id, query, memory_id, memory in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            query_id, memory_id = futures[future]
            try:
                result = future.result()
                label = result["conflict_label"] if isinstance(result, dict) else result
                if label not in {"conflict", "no_conflict", "uncertain"}:
                    raise ValueError(f"非法冲突标签：{label}")
                labels[(query_id, memory_id)] = label
            except Exception as exc:
                errors.append({"query_id": query_id, "memory_id": memory_id,
                               "error": f"{type(exc).__name__}: {exc}"})
            if completed % 100 == 0 or completed == len(futures):
                print(f"[pair labels] 完成 {completed}/{len(futures)}", flush=True)
    if errors:
        first = errors[0]
        raise RuntimeError(f"{len(errors)}/{len(jobs)} 个配对判定失败；首个失败为 "
                           f"{first['query_id']}/{first['memory_id']}: {first['error']}")
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memconflict", type=Path, default=DEFAULT_MEMCONFLICT)
    parser.add_argument("--halumem", type=Path, default=DEFAULT_HALUMEM)
    parser.add_argument("--stale", type=Path, default=DEFAULT_STALE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env", type=Path, help="API 环境文件；默认项目根目录 .env")
    parser.add_argument("--label-workers", type=int, default=4)
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if args.label_workers < 1:
        raise SystemExit("--label-workers 必须大于 0")
    paths = [args.memconflict.resolve(), args.halumem.resolve(), args.stale.resolve()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("来源文件缺失：" + ", ".join(missing))
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"输出目录非空，不覆盖已有文件：{output_dir}")

    print("转换 MemConflict...", flush=True)
    mc_memories, mc_queries, mc_gold = convert_memconflict(paths[0])
    print("转换 HaluMem Medium...", flush=True)
    halu_memories, halu_queries, halu_gold, halu_jobs = convert_halumem(paths[1])
    print("转换 STALE...", flush=True)
    stale_memories, stale_queries, stale_gold, stale_jobs = convert_stale(paths[2])
    jobs = halu_jobs + stale_jobs
    print(f"共需模型判定 {len(jobs)} 对（HaluMem {len(halu_jobs)}，STALE {len(stale_jobs)}）。", flush=True)

    from memory_retrieval.api import APIClient
    from memory_retrieval.config import Settings
    from memory_retrieval.prompts import judge_pair

    settings = Settings.from_env(args.env)
    settings.require(llm=True)
    client = APIClient(settings)
    labels = judge_jobs(jobs, lambda query, memory: judge_pair(client, query, memory), args.label_workers)
    for rows in (halu_gold, stale_gold):
        for row in rows:
            row["conflict_labels"] = {
                memory_id: labels[(row["query_id"], memory_id)]
                for memory_id in row["gold_compare_ids"]
            }

    memories = mc_memories + halu_memories + stale_memories
    queries = mc_queries + halu_queries + stale_queries
    gold = mc_gold + halu_gold + stale_gold
    memory_ids = [row["memory_id"] for row in memories]
    query_ids = [row["query_id"] for row in queries]
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError("合并后的 memory_id 不唯一")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("合并后的 query_id 不唯一")
    if {row["query_id"] for row in gold} != set(query_ids):
        raise ValueError("合并后的查询与金标不一致")
    known_ids = set(memory_ids)
    if any(set(row["gold_compare_ids"]) - known_ids for row in gold):
        raise ValueError("合并金标引用了旧库中不存在的记忆")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "memories.jsonl", memories)
    write_jsonl(output_dir / "queries.jsonl", queries)
    write_jsonl(output_dir / "gold.jsonl", gold)
    label_counts = {label: sum(value == label for row in gold for value in row["conflict_labels"].values())
                    for label in ("conflict", "no_conflict", "uncertain")}
    print(json.dumps({"output_dir": str(output_dir), "memory_count": len(memories),
                      "query_count": len(queries), "gold_count": len(gold),
                      "llm_judged_pairs": len(jobs), "conflict_labels": label_counts,
                      "files": ["memories.jsonl", "queries.jsonl", "gold.jsonl"]},
                     ensure_ascii=False, indent=2), flush=True)
    return output_dir


if __name__ == "__main__":
    main()
