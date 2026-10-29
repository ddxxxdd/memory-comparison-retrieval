"""把 v5 数据集改造成 v6：明确本人主体、保留第三方主体并改写过敏陈述。

两条规则：
1. 个人记忆（scope=personal）中明确指向本人的泛称/第一人称，使用所属用户的固定化名；
   学生、患者等第三方泛称不与记录用户合并。公共记忆不动。
2. 药物过敏：允许"多种过敏并存"的记录会削弱冲突标签，正文改成"仅对 X 过敏"，
   这样同一主体的不同过敏原构成本身互斥的冲突。

原始 v5 不覆盖，输出到 data/validmem_v6；用户→化名映射写进 manifest，便于复核。
"""

import json
import hashlib
import re
from pathlib import Path

BASE = Path(r"C:\Users\dzp\project\paper\memory-comparison-retrieval")
SRC = BASE / "data" / "validmem_v5"
DST = BASE / "data" / "validmem_v6"

FIRST = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy",
         "Jack", "Karen", "Liam", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rachel",
         "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zack", "Amber",
         "Brian", "Chloe", "Derek"]
LAST = ["Chen", "Davis", "Evans", "Fisher", "Garcia", "Hughes", "Irwin", "Jensen",
        "Klein", "Lopez", "Miller", "Nelson", "Owen", "Patel", "Quinn", "Reed",
        "Silva", "Turner", "Upton", "Vargas"]

# 开头是这些泛称时，直接替换成化名
GENERIC_SUBJECT = [
    (re.compile(r"^The user's\b", re.I), "possessive"),
    (re.compile(r"^User's\b", re.I), "possessive"),
    (re.compile(r"^The user\b", re.I), "agent"),
    (re.compile(r"^User\b", re.I), "agent"),
    (re.compile(r"^The patient's\b", re.I), "possessive"),
    (re.compile(r"^Patient's\b", re.I), "possessive"),
    (re.compile(r"^The patient\b", re.I), "agent"),
    (re.compile(r"^Patient\b", re.I), "agent"),
    (re.compile(r"^My\b", re.I), "possessive"),
    (re.compile(r"^I\b", re.I), "first-person"),
    (re.compile(r"^用户"), "zh"),
    (re.compile(r"^我(?![们])"), "zh-agent"),
]

# 无人称开头：动词开头直接接名字，名词短语用所有格
VERB_START = re.compile(
    r"^(Completed|Prefers?|Switched|Changed|Started|Reported|Moved|Visited|Bought|"
    r"Finished|Enrolled|Booked|Upgraded|Rescheduled|Cancelled|Attended|Joined|"
    r"Submitted|Decided|Chose|Updated|Recommended|Asked|Requested|Planned)\b")

NAME_START = re.compile(r"^(?:[A-Z][a-z]+(?: [A-Z][a-z]+)?)(?:'s| is| has| was| lives| moved| works|"
                        r" prefers| reported| switched| completed| started| said)\b")
# 正文里已经出现具体人名时不再套化名，避免出现"Uma Miller John Doe"这种双主体。
# 只认人名，不把 Trader Joe's、Whole Foods 这类品牌当成主体。
EXPLICIT_NAME = re.compile(r"\b(?:John|Jane|David|Daniel|Tom|Lily|Emma|Alice|Bob|Carol|Sam|Jack|"
                           r"Mia|Noah|James|Mary|Robert|Sarah)\b")
BRAND_START = {"Walmart", "Costco", "Trader", "Whole", "Target", "Nike", "Amazon", "MegaStore",
               "Instacart", "GreenMart", "FreshGrocer", "MarketSquare", "Sam's", "Brilliant"}

ALLERGY = re.compile(r"(allerg\w*|过敏)", re.I)
ALLERGY_ALREADY_EXCLUSIVE = re.compile(r"(only|仅|no known|no allergies|无已知|没有.*过敏|无过敏)", re.I)

# 这些来源句式无法只靠通用主语替换保持原事实关系，按原意改成自然的患者/写入者表述。
CONTENT_OVERRIDES = {
    "M-C-0012": "The patient is prescribed 10mg of medication X once daily.",
    "M-C-0013": "The patient is prescribed 20mg of medication X once daily.",
    "M-C-0014": "The patient's prescription was adjusted to 15mg of medication X once daily.",
    "M-C-0155": "The patient is prescribed Medication A for condition X.",
    "M-C-0255": "The patient is prescribed 20mg of Medication X for hypertension.",
    "M-C-0263": "The patient is prescribed 10mg of Medication A daily.",
    "M-C-0267": "The patient is prescribed 500mg of Medication X daily.",
    "M-C-0331": "Grace Silva reported side effects from Naproxen and was prescribed Paracetamol.",
    "TC-A-0086": "Olivia Chen updated the patient's prescription to cefuroxime 250mg twice daily.",
}


def name_for(user_id, mapping):
    if user_id not in mapping:
        index = len(mapping)
        mapping[user_id] = f"{FIRST[index % len(FIRST)]} {LAST[(index // len(FIRST)) % len(LAST)]}"
    return mapping[user_id]


GENERIC_TOKENS = {"User", "The", "Patient", "My", "I", "A", "An", "This", "That", "Updated",
                  "Preferred", "Recommended", "Blood", "Home", "Email", "Phone", "Favorite",
                  "Yoga", "Team", "Project", "Student", "Assignment", "Exam", "Library",
                  "Summer", "Winter", "Community"}


def add_subject(text, name):
    """把正文改成带明确主体的表述。已有人名的主体原样返回。"""
    stripped = text.strip()
    # 学生/患者是正文中的事实主体，不能仅凭记录所属用户推断二者是同一人。
    if re.match(r"^(?:The\s+)?(?:student|patient)\b", stripped, re.I):
        return text

    # 修复来源中已给出人名、但仍残留第一人称所有格的句子。
    if re.match(rf"^{re.escape(name)}\b", stripped, re.I):
        return re.sub(r"\bmy\b", "their", text, flags=re.I)

    for pattern, kind in GENERIC_SUBJECT:
        match = pattern.match(stripped)
        if not match:
            continue
        rest = stripped[match.end():]
        # 泛称后面已经跟着具体人名（如 "Patient John Doe"）时保留原主体。
        if (EXPLICIT_NAME.search(rest)
                or re.search(rf"\b{re.escape(name)}\b", rest, re.I)):
            return text
        if kind == "possessive":
            return f"{name}'s{rest}"
        if kind == "first-person":
            rest = re.sub(r"^\s+am\b", " is", rest, count=1, flags=re.I)
            rest = re.sub(r"\bmy\b", "their", rest, flags=re.I)
            return f"{name}{rest}"
        return f"{name}{rest}"
    if NAME_START.match(stripped) and stripped.split()[0] not in GENERIC_TOKENS:
        return text
    if EXPLICIT_NAME.search(stripped):
        return text
    if re.match(r"^[\u4e00-\u9fff]", stripped):
        # 中文句首是动词时直接接名字（"张三完成了…"），名词短语才加"的"（"张三的住址是…"）。
        verb = re.match(r"^(喜欢|偏好|完成|决定|计划|预约|取消|参加|提交|更新|更换|搬|购买|买|看|"
                        r"学习|学|煮|读|写|考|复习|练习|认为|开始|通过|加入|注册|报名|换|选择|"
                        r"放弃|继续|停止|减少|增加|推荐|查看|逛|整理|收|寄|发送|下载|安装|维修|"
                        r"打扫|准备|安排|预订|排队|参加|听说|发现|记得)", stripped)
        if re.match(r"^(昨天|今天|明天|上周|本周|下周|周末|晚上|早上|上午|下午|最近|目前|现在|"
                    r"上个月|这个月|今年|去年)", stripped):
            verb = True
        return f"{name}{stripped}" if verb else f"{name}的{stripped}"
    # "The gym is closed ..." 这类以 The 开头的名词句，去掉 The 再加所有格。
    if re.match(r"^The (?!user\b)", stripped, re.I):
        rest = re.sub(r"^The\s+", "", stripped, count=1, flags=re.I)
        return f"{name}'s {rest[0].lower() + rest[1:]}"
    # "Updated allergy: ..." 这类以标签开头的句子用所有格，并把标签首字母小写。
    label = re.match(r"^([A-Z][A-Za-z ]*):", stripped)
    if label:
        lowered = stripped[0].lower() + stripped[1:]
        return f"{name}'s {lowered}"
    first = stripped.split()[0]
    keep_case = first in BRAND_START
    lowered = stripped if keep_case else stripped[0].lower() + stripped[1:]
    if VERB_START.match(stripped):
        lowered = re.sub(r"\bmy\b", "their", lowered, flags=re.I)
        return f"{name} {lowered}"
    return f"{name}'s {lowered}"


def exclusive_allergy(text):
    """把"对 X 过敏"改成"仅对 X 过敏"，不动药名本身。"""
    if not ALLERGY.search(text):
        return text
    text = re.sub(r"\bis only allergic to\b", "is allergic only to", text, count=1, flags=re.I)
    instead = re.match(
        r"^(?P<prefix>.*?\breported\s+)allergy to (?P<allergen>[^.!?;]+?)\s+instead"
        r"(?P<punct>[.!?]?)$", text, re.I)
    if instead:
        prefix = instead["prefix"]
        if prefix.lower() == "patient reported ":
            prefix = "The patient reported "
        return (f"{prefix}an allergy only to {instead['allergen'].strip()}"
                f"{instead['punct']}")
    reported = re.match(
        r"^(?P<prefix>.*?\breported\s+)(?:an?\s+)?allergy to "
        r"(?P<allergen>[^.!?;]+?)(?:\s+only)?(?P<punct>[.!?]?)$", text, re.I)
    if reported:
        prefix = reported["prefix"]
        if prefix.lower() == "patient reported ":
            prefix = "The patient reported "
        return (f"{prefix}an allergy only to {reported['allergen'].strip()}"
                f"{reported['punct']}")
    patient = re.match(
        r"^(?P<subject>(?:The )?patient(?:\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)?) has "
        r"(?:an? )?(?:known )?allergy to (?P<allergen>[^.!?;]+?)"
        r"(?:\s+only)?(?P<punct>[.!?]?)$",
        text, re.I)
    if patient:
        return (f"{patient['subject']} is allergic only to {patient['allergen'].strip()}"
                f"{patient['punct']}")
    if ALLERGY_ALREADY_EXCLUSIVE.search(text):
        return text
    noted = re.match(
        r"^(?P<subject>.+?)'s allergy to (?P<allergen>[^.!?;]+?)"
        r"(?P<note>\s+(?:noted|reported|recorded|documented)\b.*)?(?P<punct>[.!?]?)$",
        text, re.I)
    if noted:
        subject = noted["subject"].strip()
        if subject.lower() == "patient":
            subject = "The patient"
        note = f", as {noted['note'].strip()}" if noted["note"] else ""
        return (f"{subject}'s only known allergy is "
                f"{noted['allergen'].strip()}{note}{noted['punct']}")
    label = re.match(
        r"^(?P<subject>.+?)'s (?:(?:known|updated) )?allergy(?: updated)?"
        r"(?:\s+(?:to|is)|:)\s*(?P<allergen>[^.!?;]+?)(?P<punct>[.!?]?)$",
        text, re.I)
    if label:
        subject = label["subject"].strip()
        if subject.lower() == "patient":
            subject = "The patient"
        return (f"{subject}'s only known allergy is {label['allergen'].strip()}"
                f"{label['punct'] or '.'}")
    history = re.match(
        r"^(?P<subject>(?:The )?patient) has a history of (?:a )?"
        r"(?P<allergen>[^.!?;]+?) allergy(?P<punct>[.!?]?)$", text, re.I)
    if history:
        subject = history["subject"]
        if subject.lower() == "patient":
            subject = "The patient"
        return (f"{subject}'s history includes an allergy only to "
                f"{history['allergen'].strip()}{history['punct'] or '.'}")
    file_note = re.match(
        r"^(?P<subject>(?:The )?patient) had an allergy to (?P<allergen>[^.!?;]+?)"
        r" reported on file(?P<punct>[.!?]?)$", text, re.I)
    if file_note:
        subject = file_note["subject"]
        if subject.lower() == "patient":
            subject = "The patient"
        return (f"{subject} had an allergy only to {file_note['allergen'].strip()}, "
                f"as reported on file{file_note['punct'] or '.'}")
    if re.search(r"[\u4e00-\u9fff]", text):
        new = re.sub(r"对([^，。；]*)过敏", r"仅对\1过敏", text, count=1)
        return new if new != text else text
    for pattern in (r"\bis allergic to\b", r"\bare allergic to\b", r"\bwas allergic to\b"):
        if re.search(pattern, text, re.I):
            rewritten = re.sub(
                pattern, lambda m: m.group(0)[:-len("allergic to")] + "allergic only to",
                text, count=1, flags=re.I)
            if re.match(r"^Patient is allergic\b", rewritten):
                rewritten = "The " + rewritten[0].lower() + rewritten[1:]
            return rewritten
    if re.search(r"\ballergy is\b", text, re.I):
        return re.sub(r"\ballergy is\b", "only allergy is", text, count=1, flags=re.I)
    if re.search(r"'s allergy to\b", text, re.I):
        return re.sub(r"'s allergy to\b", "'s only allergy to", text, count=1, flags=re.I)
    # 其余句式把 only 收在句末，保住药名同时表达"仅此一种"。
    return re.sub(r"([.!?]?)\s*$", r" only\1", text, count=1)


def transform(records, mapping, stats, kind):
    out = []
    for row in records:
        row_id = row.get("memory_id") or row.get("query_id")
        ref = row.get("source_ref") or {}
        content = row.get("content") or ""
        new = content
        if kind == "memory" and ref.get("scope") == "personal" and ref.get("user_id"):
            named = add_subject(content, name_for(ref["user_id"], mapping))
            if named != content:
                stats["subject_added"] += 1
            new = named
        elif kind == "query" and ref.get("scope") == "personal" and ref.get("user_id"):
            named = add_subject(content, name_for(ref["user_id"], mapping))
            if named != content:
                stats["subject_added"] += 1
            new = named
        elif kind == "query":
            # 公共写入也可能缺主体，但主体是公共对象，不加人名
            new = content
        fixed = exclusive_allergy(new)
        if fixed != new:
            stats["allergy_rewritten"] += 1
        corrected = CONTENT_OVERRIDES.get(row_id, fixed)
        if corrected != fixed:
            stats["sentence_corrected"] += 1
        fixed = corrected
        out.append({**row, "content": fixed})
    return out


def main():
    mapping, stats = {}, {"subject_added": 0, "allergy_rewritten": 0, "sentence_corrected": 0}
    memories = [json.loads(line) for line in (SRC / "memories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = [json.loads(line) for line in (SRC / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    new_memories = transform(memories, mapping, stats, "memory")
    new_queries = transform(queries, mapping, stats, "query")

    DST.mkdir(parents=True, exist_ok=True)
    for name, rows in (("memories.jsonl", new_memories), ("queries.jsonl", new_queries)):
        (DST / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                encoding="utf-8")
    gold_path = DST / "gold.jsonl"
    if not gold_path.exists():
        gold_path.write_bytes((SRC / "gold.jsonl").read_bytes())

    manifest = json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))
    manifest["dataset_id"] = "validmem_v6"
    manifest.pop("dataset", None)
    manifest["conversion"]["content"] = (
        "只将明确指向记录用户本人的泛称/第一人称替换为固定化名；已具名主体及学生、患者等第三方主体保留；"
        "公共正文不改；过敏陈述改为明确的唯一已知过敏原。")
    manifest["modifications"] = {
        "from": "validmem_v5",
        "rule_subject": "只将明确指向记录用户本人的泛称/第一人称替换为固定化名；已有具名主体及学生、患者等第三方主体保留，公共正文不改",
        "rule_allergy": "含过敏的正文明确表达只有一个已知过敏原；药名保留，修正文末 instead only 等错误表达",
        "stats": stats,
        "user_names": mapping,
    }
    manifest["files"] = {
        name: hashlib.sha256((DST / name).read_bytes()).hexdigest()
        for name in ("memories.jsonl", "queries.jsonl", "gold.jsonl")
    }
    (DST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print("输出:", DST)
    print("统计:", stats, "用户数:", len(mapping))


if __name__ == "__main__":
    main()
