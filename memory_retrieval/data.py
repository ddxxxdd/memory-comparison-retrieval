from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是有效 JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} 必须是 JSON 对象")
            records.append(row)
    return records


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def nonempty_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    return value


@dataclass(frozen=True)
class Memory:
    memory_id: str
    content: str
    source_ref: str | dict
    scenario_id: str | None = None

    @classmethod
    def from_dict(cls, row):
        memory_id = nonempty_text(row.get("memory_id"), "memory_id")
        content = nonempty_text(row.get("content"), "content")
        ref = row.get("source_ref")
        if not isinstance(ref, (str, dict)) or not ref:
            raise ValueError(f"{memory_id}: source_ref 必须是非空字符串或对象")
        scenario_id = row.get("scenario_id")
        if scenario_id is not None:
            nonempty_text(scenario_id, "scenario_id")
        return cls(memory_id, content, ref, scenario_id)

    def to_dict(self):
        row = asdict(self)
        if self.scenario_id is None:
            row.pop("scenario_id")
        return row


def validate_memories(memories):
    ids = [m.memory_id for m in memories]
    if not ids:
        raise ValueError("旧记忆库不能为空")
    if len(ids) != len(set(ids)):
        raise ValueError("memory_id 必须唯一；相同内容的不同旧记录仍应使用不同 ID")


def memory_banks(memories):
    banks = {}
    for memory in memories:
        banks.setdefault(memory.scenario_id, []).append(memory)
    if None in banks and len(banks) > 1:
        raise ValueError("旧记忆不能混用分场景和全库两种组织方式")
    return banks


def load_queries(path):
    rows = read_jsonl(path)
    ids = []
    for row in rows:
        ids.append(nonempty_text(row.get("query_id"), "query_id"))
        nonempty_text(row.get("content"), "content")
        if row.get("scenario_id") is not None:
            nonempty_text(row["scenario_id"], "scenario_id")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("查询不能为空，且 query_id 必须唯一")
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
