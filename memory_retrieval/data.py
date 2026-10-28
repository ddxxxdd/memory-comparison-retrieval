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

    @classmethod
    def from_dict(cls, row):
        memory_id = nonempty_text(row.get("memory_id"), "memory_id")
        content = nonempty_text(row.get("content"), "content")
        ref = row.get("source_ref")
        if not isinstance(ref, (str, dict)) or not ref:
            raise ValueError(f"{memory_id}: source_ref 必须是非空字符串或对象")
        return cls(memory_id, content, ref)

    @property
    def retrieval_text(self):
        return build_retrieval_text(self.content, self.source_ref)

    def to_dict(self):
        return asdict(self)


SUBJECT_TEMPLATE = "主体上下文：{subject}。\n记忆内容：{content}"

# 检索文本的表示版本：拼接方式或所用字段一旦变化就要 +1。
# 旧记忆向量是用建库时的这份文本编码的，查询侧必须与它同版本，否则两端不在同一表示空间。
RETRIEVAL_TEXT_VERSION = 1


def subject_context(source_ref):
    """检索可用的主体上下文：个人记忆用匿名用户编号，公共记忆用"公共事实"。

    只读 scope 与 user_id。source_ref 里的 conflict_entity、conflict_attribute、
    topic、source_role 属于评测标注，不得进入检索输入。
    """
    ref = source_ref if isinstance(source_ref, dict) else {}
    scope = ref.get("scope")
    if scope == "personal":
        user = ref.get("user_id")
        return f"用户 {user}" if user else "未标注用户"
    if scope == "world":
        return "公共事实"
    return None


def build_retrieval_text(content, source_ref):
    """把来源上下文与原文拼成检索输入，原始 content 不改。

    旧记忆建索引与新记忆发起查询必须用同一份文本，否则两边不在同一表示空间。
    公共事实不套个人主体；正文自身讲第三人时保留原文主体。
    """
    subject = subject_context(source_ref)
    if subject is None:
        return content
    return SUBJECT_TEMPLATE.format(subject=subject, content=content)


def validate_memories(memories):
    ids = [m.memory_id for m in memories]
    if not ids:
        raise ValueError("旧记忆库不能为空")
    if len(ids) != len(set(ids)):
        raise ValueError("memory_id 必须唯一；相同内容的不同旧记录仍应使用不同 ID")


def load_queries(path):
    rows = read_jsonl(path)
    ids = []
    for row in rows:
        ids.append(nonempty_text(row.get("query_id"), "query_id"))
        nonempty_text(row.get("content"), "content")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("查询不能为空，且 query_id 必须唯一")
    return rows
