"""Create a content-based, single-bank ValidMem repair from validmem_v6.

The previous version used scope and user IDs to define who could conflict.
This repair keeps those values only in a provenance sidecar and builds the
active memory/query records and gold from their text, with conservative review
queues for identity, matter, condition, or value ambiguities.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "validmem_v6"
OUTPUT = ROOT / "data" / "validmem_v7"
REVIEW_OUTPUT = OUTPUT / "manual_review.jsonl"
RENAMES = {"John Doe": "Mara Voss"}
USER_CONFIRMED_NON_CONFLICTS = {
    ("TW-0001", "M-C-0171"): "用户裁定新会员 20% 优惠与会员费 80% 通用优惠暂不计为冲突。",
    ("TW-0001", "M-C-0172"): "用户确认沿用 20% 优惠的判断：新会员 25% 优惠与会员费 80% 通用优惠暂不计为冲突。",
    ("TC-A-0165", "M-A-0434"): "用户裁定：计划学习 Biology 与最终确定 Data Science 不算冲突。",
    ("TC-A-0086", "M-A-0208"): "用户确认两条处方记忆涉及不同患者，不冲突。",
    ("TC-A-0089", "M-A-0217"): "用户确认两条处方记忆涉及不同患者，不冲突。",
}
INHERITED_NON_CONFLICT_PAIRS = {
    ("TW-0001", "M-C-0171"),
    ("TW-0001", "M-C-0172"),
    ("TC-A-0086", "M-A-0208"),
    ("TC-A-0089", "M-A-0217"),
}
USER_CONFIRMED_CONFLICTS = {
    ("TW-0013", "M-C-0188"): "用户裁定为同一教材事项的不同版本：Physics 101 第 5 版与 Physics Concepts 第 2 版。",
    ("TW-0022", "M-B-0014"): "用户裁定同一春季清仓促销中，部分商品与全部商品适用范围不同，标为冲突。",
}
CONTENT_REWRITES = {
    "M-B-0014": {
        "before": "Spring clearance sale: 50% off select items",
        "after": "Spring clearance sale: only select items are 50% off.",
        "reason": "用户要求明确表达只有部分商品打 5 折，以凸显其与全部商品打 5 折的范围差异",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def canonical(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", str(value or "").casefold()).replace("_", " ")).strip()


def rewrite_name(text: str) -> str:
    for old, new in RENAMES.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.I)
    return text


def subject_key(row: dict) -> str | None:
    """Resolve only the identity cases licensed by the user's rules.

    Exact names are global identifiers; unqualified singular role nouns form
    their own shared identity. A named person is not silently merged with an
    unqualified role noun.
    """
    text = row["content"].strip()
    patient_name = re.match(r"^Patient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
    if patient_name:
        return f"name:{norm(patient_name.group(1))}"
    user_label = re.match(r"^(?:The\s+)?User\s+([A-Z])\b", text)
    if user_label:
        return f"name:user {norm(user_label.group(1))}"

    gym_name = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Gym)\b", text)
    if gym_name:
        return f"name:{norm(gym_name.group(1))}"

    generic_subjects = (
        "patient", "student", "user", "gym", "group", "team", "teacher",
        "doctor", "professor", "customer", "client", "employee", "manager",
    )
    for term in generic_subjects:
        match = re.match(rf"^(?:the\s+)?{term}(?:'s)?\b(?:\s+)?", text, re.I)
        if not match:
            continue
        remainder = text[match.end():].lstrip(" '")
        if term == "student" and re.match(r"(?:feedback|club|course|project|study|affairs)\b", remainder, re.I):
            continue
        if term == "user" and re.match(r"(?:guide|manual|profile|account page)\b", remainder, re.I):
            continue
        if term == "group" and re.match(r"(?:project|study|meeting|activity)\b", remainder, re.I):
            continue
        if term == "team" and re.match(r"(?:lunch|outing|project|meeting)\b", remainder, re.I):
            continue
        if term in {"teacher", "doctor", "professor", "customer", "client", "employee", "manager"} and re.match(r"[A-Z][a-z]+\b", remainder):
            return f"name:{norm(term + ' ' + remainder.split()[0])}"
        return f"generic:{term}"

    possessive_name = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:'s\b|的)", text)
    if possessive_name:
        return f"name:{norm(possessive_name.group(1))}"

    full_name_subject = re.match(
        r"^([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:is|has|prefers|likes|change[sd]?|updated|started|moved|switched|uses|needs|plans|finalizes|relocated|scheduled|recommended|increased|prescribed|lives|feedback|review|announced|mentions|suggests|teaches|works|manages)\b",
        text,
    )
    if full_name_subject:
        return f"name:{norm(full_name_subject.group(1))}"
    topic = norm((row.get("source_ref") or {}).get("topic"))
    if topic:
        # World facts often omit a grammatical subject (e.g. a textbook or
        # public promotion). Their stable topic is the shared public entity.
        return f"topic:{topic}"
    return None


def replace_generic_subject(text: str, term: str, alias: str) -> str:
    """Give an unqualified lead subject a stable, content-visible identity."""
    prefix = rf"^(?:the\s+)?{re.escape(term)}"
    match = re.match(prefix + r"('s)?\b", text, re.I)
    if not match:
        return text
    possessive = bool(match.group(1))
    suffix = "'s" if possessive else ""
    return alias + suffix + text[match.end():]


def normalized_item(row: dict) -> str:
    ref = row.get("source_ref") or {}
    text = row["content"].casefold()
    subject = subject_key(row)
    entity = norm(ref.get("conflict_entity"))
    attribute = norm(ref.get("conflict_attribute"))

    if re.search(r"\bonly likes?\b|\blikes? only\b|只喜欢", text):
        return "exclusive likes"

    # In several source rows conflict_entity is the person/role rather than
    # the item. Remove that identity component; keep actual item entities such
    # as Medication X, team_lunch, or chosen_major.
    if subject and subject.startswith("name:") and entity == subject[5:]:
        entity = ""
    if subject and subject.startswith("generic:"):
        generic = subject[8:]
        if entity == generic or entity.startswith(generic + " "):
            entity = ""

    if "blood pressure" in text:
        return "blood pressure"
    if "allerg" in text:
        return "allergy"
    if "gym membership" in text and re.search(r"\b(number|id)\b", text):
        return "gym membership identifier"
    if "email" in text and "work email" not in text:
        return "email"
    if "favorite coffee shop" in text:
        return "favorite coffee shop"
    if ("chosen major" in text or "major as" in text or "major is" in text
            or ("chosen major" in attribute and "major" in text)):
        return "chosen major"
    if ref.get("topic"):
        topic = norm(ref["topic"])
        if "price" in topic and "%" in text:
            return "topic:" + topic + "|discount"
        if "price" in topic and re.search(r"(?:\$|\b\d+(?:\.\d+)?\s*(?:yuan|dollars?)\b)", text):
            return "topic:" + topic + "|price"
        if "price" in topic:
            return "topic:" + topic + "|offer"
        return "topic:" + topic

    if attribute in {"primary email", "email address"}:
        attribute = "email"
    if attribute in {"gym membership id", "gym membership number"}:
        attribute = "gym membership identifier"
    if attribute in {"favorite coffee shop", "favorite coffee shop name"}:
        attribute = "favorite coffee shop"

    return "|".join(part for part in (entity, attribute) if part)


def value_key(row: dict, item: str) -> str | None:
    text = row["content"].casefold()
    if item == "blood pressure":
        match = re.search(r"\b(\d{2,3}\s*/\s*\d{2,3})\b", text)
        return re.sub(r"\s+", "", match.group(1)) if match else None
    if item == "allergy":
        if re.search(r"\bno known allergies\b", text):
            return "none"
        match = re.search(
            r"(?:allergic(?: only)? to|allergy(?: only)? to|allergy is)\s+([^.,;]+)", text)
        return norm(match.group(1)) if match else None
    if item == "email":
        match = re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text)
        return match.group(0) if match else None
    if item == "gym membership identifier":
        match = re.search(r"(?:number|id)\s+is\s+([a-z0-9_-]+)", text)
        return match.group(1) if match else None
    if "dosage" in item or "dosage" in text or "prescribed medication" in text:
        match = re.search(r"\b(\d+(?:\.\d+)?\s*mg)\b", text)
        return re.sub(r"\s+", "", match.group(1)) if match else None
    if item == "chosen major":
        # A future plan is not treated as a conflicting asserted value. A
        # stated change or finalized major has an extractable value.
        if re.search(r"\bplans? to study\b", text):
            return None
        match = re.search(
            r"(?:finalizes? (?:their )?major as|(?:changes?|changed) (?:their )?major to|major (?:is|as|to))\s+([^.,;]+)",
            text,
        )
        return norm(match.group(1)) if match else None
    if item == "exclusive likes":
        match = re.search(r"(?:only likes?|likes? only)\s+([^.,;]+)", text)
        if match:
            return norm(match.group(1))
        match = re.search(r"只喜欢\s*([^，。；,;]+)", row["content"])
        return norm(match.group(1)) if match else None
    if item.startswith("topic:") and item.endswith("|discount"):
        match = re.search(r"(\d+(?:\.\d+)?\s*%)", text)
        return re.sub(r"\s+", "", match.group(1)) if match else None
    return None


def audit_value_key(row: dict, item: str) -> str | None:
    """Extract a conservative value for auditing inherited labels.

    The stricter value_key remains the gate for new automatic labels. This
    broader extractor only checks already-labeled pairs and leaves the raw
    text in gold_audit.jsonl for traceability.
    """
    value = value_key(row, item)
    if value is not None:
        if item.startswith("topic:") and item.endswith("|discount"):
            condition = value_condition(row)
            return f"{value}|{condition}" if condition else value
        return value
    text = row["content"].strip()
    patterns = (
        r"\bevery\s+([A-Za-z]+(?:\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)",
        r"\b(?:valid until|due (?:on|by)|expires? (?:on|at)?|deadline (?:is|set for))\s+(.+)$",
        r"\bset for\s+(.+)$",
        r"\bscheduled\s+((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b.*)$",
        r"\bassigned\s+(.+?)\s+as\b",
        r"\b(?:rescheduled|reassigned|scheduled|changed|updated|switched|moved|relocated|increased|decreased|extended|upgraded|reduced|assigned|set|adjusted)\b.*?\bto\s+(.+)$",
        r"\b(?:is now|are now|was|were|is|are|uses|use|lives in|becomes|became|scheduled for|scheduled on|scheduled at|recommended|prescribed)\s+(.+)$",
        r"将在\s*(.+?)(?:举行|举办|进行|开始|召开)",
        r"(?:更新为|改为|更换为|调整为|变为|设置为|转为|地址是|地址为|是)\s*([^，。；]+)",
        r":\s*(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        candidate = match.group(1).strip().strip(" .,:;。；")
        candidate = re.sub(r"^(?:now|the|a|an|as)\s+", "", candidate, flags=re.I)
        normalized = norm(candidate)
        if normalized:
            return normalized
    return None


def audit_pair(query: dict, memory: dict, origin: str, status: str, reason: str) -> dict:
    query_item = normalized_item(query)
    memory_item = normalized_item(memory)
    return {
        "query_id": query["query_id"],
        "memory_id": memory["memory_id"],
        "origin": origin,
        "status": status,
        "query_subject": subject_key(query),
        "memory_subject": subject_key(memory),
        "subject_match": subject_key(query) == subject_key(memory),
        "query_item": query_item,
        "memory_item": memory_item,
        "item_match": query_item == memory_item,
        "query_value": audit_value_key(query, query_item),
        "memory_value": audit_value_key(memory, memory_item),
        "query_content": query["content"],
        "memory_content": memory["content"],
        "reason": reason,
    }


def value_condition(row: dict) -> str | None:
    text = row["content"].casefold()
    conditions = (
        (r"\bnew (?:members|sign[- ]?ups?)\b", "new_members"),
        (r"\bselect items\b", "select_items"),
        (r"\ball items\b", "all_items"),
        (r"\bstorewide\b", "storewide"),
        (r"\bmembership fees\b", "membership_fees"),
        (r"\bgold members\b", "gold_members"),
    )
    matches = [label for pattern, label in conditions if re.search(pattern, text)]
    return "+".join(matches) if matches else None


def subject_relation(left: dict, right: dict) -> str:
    lsub, rsub = subject_key(left), subject_key(right)
    if lsub and rsub and lsub == rsub:
        return "same"
    if lsub and rsub and lsub.startswith("name:") and rsub.startswith("name:"):
        return "different"
    if lsub and rsub and lsub.startswith("generic:") and rsub.startswith("generic:"):
        return "different"
    if lsub and rsub and lsub.split(":", 1)[0] != rsub.split(":", 1)[0]:
        return "uncertain"  # named versus unqualified subject
    return "unknown"


def main() -> None:
    if OUTPUT.exists():
        existing_manifest_path = OUTPUT / "manifest.json"
        if not existing_manifest_path.exists():
            raise SystemExit(f"输出目录已存在且没有本脚本生成的 manifest，不覆盖: {OUTPUT}")
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("dataset_id") != "validmem_v7" or existing_manifest.get("source", {}).get("derived_from") != "validmem_v6":
            raise SystemExit(f"输出目录不是此前由本脚本生成的 v7，不覆盖: {OUTPUT}")

    memories = read_jsonl(SOURCE / "memories.jsonl")
    queries = read_jsonl(SOURCE / "queries.jsonl")
    old_gold = read_jsonl(SOURCE / "gold.jsonl")
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    by_memory = {row["memory_id"]: row for row in memories}

    provenance = []
    alias_audit = []
    all_rows = [("memory", memories, "memory_id"), ("query", queries, "query_id")]
    for kind, rows, id_field in all_rows:
        for row in rows:
            provenance.append({
                "record_type": kind,
                "record_id": row[id_field],
                "source_ref": row.get("source_ref", {}),
                "original_content": row["content"],
            })
            row["content"] = rewrite_name(row["content"])
            if kind == "memory" and row[id_field] in CONTENT_REWRITES:
                rewrite = CONTENT_REWRITES[row[id_field]]
                if row["content"] != rewrite["before"]:
                    raise ValueError(f"unexpected source text for approved rewrite: {row[id_field]}")
                row["content"] = rewrite["after"]
            ref = row.get("source_ref") or {}
            if ref.get("conflict_entity") == "John Doe":
                ref["conflict_entity"] = RENAMES["John Doe"]

    # If a subject class has both named and anonymous mentions for the same
    # matter, use the source scenario's stable name for each anonymous mention.
    # If the class is anonymous-only, one shared alias represents that entity.
    item_names = {normalized_item(row) for _, rows, _ in all_rows for row in rows
                  if (subject_key(row) or "").startswith("name:")}
    mixed_generic_terms = {
        (subject_key(row) or "").split(":", 1)[1]
        for _, rows, _ in all_rows for row in rows
        if (subject_key(row) or "").startswith("generic:")
        and normalized_item(row) in item_names
    }
    shared_aliases = {
        "patient": "Avery Flint",
        "student": "Briar Quill",
        "user": "Cedar Rowan",
        "gym": "Northstar Gym",
        "group": "Sable Group",
        "team": "Orchid Team",
        "teacher": "Marlow Reed",
        "doctor": "Kestrel Medical",
        "professor": "Dorian Vale",
        "customer": "Juniper Client",
        "client": "Juniper Client",
        "employee": "Talon Avery",
        "manager": "Indigo Park",
    }
    first_names = (
        "Arden", "Bellamy", "Callista", "Dorian", "Elowen", "Fenwick", "Galen", "Halcyon",
        "Isolde", "Juniper", "Kestrel", "Lumen", "Marcelline", "Nerys", "Oberon", "Peregrine",
        "Quill", "Reverie", "Sable", "Tamsin", "Ulric", "Vesper", "Wystan", "Xanthe", "Yarrow",
        "Zephyr", "Alaric", "Briony", "Cormac", "Delphine", "Evander", "Fiora", "Gwyndolyn",
        "Hesper", "Ione", "Jareth", "Keelin", "Lysander", "Maelle", "Niall", "Odelia", "Phaedra",
        "Riven", "Sorrel", "Thalia", "Ulysses", "Virelai", "Winona", "Xiomara", "Ysabel", "Zorion",
    )
    last_names = (
        "Ashbourne", "Brightwell", "Cloudmere", "Duskwood", "Everhart", "Fable", "Goldcrest", "Hawkridge",
        "Ironwood", "Junefield", "Kingswell", "Larkspur", "Moonvale", "Northcott", "Oakenfold", "Pinecrest",
        "Quicksilver", "Ravenshade", "Starling", "Thornfield", "Umberley", "Valehart", "Wintermere", "Xenwood",
        "Yarrowby", "Zephyrine", "Ainsworth", "Brookhaven", "Clearwater", "Deepwell", "Emberton", "Foxglove",
        "Greenbriar", "Holloway", "Ivystone", "Jarvis", "Kestrelton", "Lightfoot", "Morningside", "Nightingale",
        "Oakheart", "Penrose", "Quillane", "Rosewood", "Silverton", "Tumbleton", "Underhill", "Vintner",
        "Westbrook", "Yardley", "Zinnwood",
    )
    mixed_scenarios = sorted({
        (term, str((row.get("source_ref") or {}).get("user_id") or f"{kind}:{row[id_field]}"))
        for kind, rows, id_field in all_rows for row in rows
        if (subject_key(row) or "").startswith("generic:")
        and (subject_key(row) or "").split(":", 1)[1] in mixed_generic_terms
        for term in [(subject_key(row) or "").split(":", 1)[1]]
    })
    scenario_aliases = {}
    for index, key in enumerate(mixed_scenarios):
        block, offset = divmod(index, len(first_names))
        first = first_names[offset]
        last = last_names[(offset * 20 + block) % len(last_names)]
        alias = f"{first} {last}"
        if key[0] == "gym":
            alias = f"{first} {last} Gym"
        scenario_aliases[key] = alias
    for kind, rows, id_field in all_rows:
        for row in rows:
            generic = subject_key(row)
            if not generic or not generic.startswith("generic:"):
                continue
            term = generic.split(":", 1)[1]
            ref = row.get("source_ref") or {}
            scenario = str(ref.get("user_id") or f"{kind}:{row[id_field]}")
            if term in mixed_generic_terms:
                alias = scenario_aliases[(term, scenario)]
            else:
                alias = shared_aliases.get(term, f"Shared {term.title()}")
            row["content"] = replace_generic_subject(row["content"], term, alias)
            alias_audit.append({
                "record_type": kind,
                "record_id": row[id_field],
                "generic_subject": term,
                "scenario_key": scenario if term in mixed_generic_terms else "shared_anonymous_entity",
                "assigned_name": alias,
                "rule": "per source user_id because this class also has named mentions"
                        if term in mixed_generic_terms else "one shared name because this class is anonymous-only",
            })
    for _, rows, _ in all_rows:
        for row in rows:
            ref = row.get("source_ref") or {}
            # Never expose a channel or source-owner boundary in active input.
            ref.pop("scope", None)
            ref.pop("user_id", None)
            ref.pop("source_role", None)

    queries_by_id = {row["query_id"]: row for row in queries}
    memories_by_id = {row["memory_id"]: row for row in memories}
    gold_by_id = {row["query_id"]: set(row["conflict_old_memory_ids"])
                  for row in old_gold}
    gold_reasons = {qid: {} for qid in gold_by_id}
    review_pairs: dict[tuple[str, str], dict] = {}
    gold_audit = []

    for pair in INHERITED_NON_CONFLICT_PAIRS:
        qid, mid = pair
        if qid not in gold_by_id or mid not in gold_by_id[qid]:
            raise ValueError(f"expected user-confirmed non-conflict missing from inherited gold: {qid}/{mid}")
    for (qid, mid) in USER_CONFIRMED_CONFLICTS:
        if qid not in gold_by_id or mid not in gold_by_id[qid]:
            raise ValueError(f"expected user-confirmed conflict missing from inherited gold: {qid}/{mid}")

    for old in old_gold:
        qid = old["query_id"]
        query = queries_by_id[qid]
        for mid in old["conflict_old_memory_ids"]:
            memory = memories_by_id[mid]
            pair = (qid, mid)
            q_subject, m_subject = subject_key(query), subject_key(memory)
            q_item, m_item = normalized_item(query), normalized_item(memory)
            q_value = audit_value_key(query, q_item)
            m_value = audit_value_key(memory, m_item)

            if pair in USER_CONFIRMED_NON_CONFLICTS:
                gold_by_id[qid].discard(mid)
                reason = USER_CONFIRMED_NON_CONFLICTS[pair]
                gold_audit.append(audit_pair(query, memory, "inherited_v6", "not_conflict_user_confirmed", reason))
                continue
            if pair in USER_CONFIRMED_CONFLICTS:
                reason = USER_CONFIRMED_CONFLICTS[pair]
                gold_reasons[qid][mid] = reason
                gold_audit.append(audit_pair(query, memory, "inherited_v6", "conflict_user_confirmed", reason))
                continue
            if q_subject and m_subject and q_subject != m_subject:
                gold_by_id[qid].discard(mid)
                reason = f"主体姓名/身份不同：{q_subject} 与 {m_subject}；按规则不构成同一主体冲突。"
                gold_audit.append(audit_pair(query, memory, "inherited_v6", "not_conflict_subject_mismatch", reason))
                continue
            if not q_subject or not m_subject:
                gold_by_id[qid].discard(mid)
                reason = "正文和稳定 topic 均不足以解析主体身份，待人工确认。"
                review_pairs[pair] = {
                    "query_id": qid, "memory_id": mid, "reason": reason,
                    "query_content": query["content"], "memory_content": memory["content"],
                    "query_value": q_value, "memory_value": m_value,
                }
                gold_audit.append(audit_pair(query, memory, "inherited_v6", "manual_review_subject", reason))
                continue
            if q_item != m_item:
                gold_by_id[qid].discard(mid)
                reason = f"同一主体但事项解析不同：{q_item} 与 {m_item}，暂不标冲突。"
                review_pairs[pair] = {
                    "query_id": qid, "memory_id": mid, "reason": reason,
                    "query_content": query["content"], "memory_content": memory["content"],
                    "query_value": q_value, "memory_value": m_value,
                }
                gold_audit.append(audit_pair(query, memory, "inherited_v6", "manual_review_item", reason))
                continue
            if q_value is not None and m_value is not None and q_value == m_value:
                gold_by_id[qid].discard(mid)
                reason = "主体和事项相同，但抽取到的取值相同，不构成冲突。"
                gold_audit.append(audit_pair(query, memory, "inherited_v6", "not_conflict_same_value", reason))
                continue

            if q_value is not None and m_value is not None:
                reason = "复核正文主体与事项一致，且抽取到不同取值；保留冲突标签。"
                status = "conflict_revalidated_values_differ"
            else:
                reason = "复核正文主体与事项一致；值解析器未覆盖此属性，保留 v6 已有冲突标签并保留原文供审计。"
                status = "conflict_revalidated_subject_item_value_unparsed"
            gold_reasons[qid][mid] = reason
            gold_audit.append(audit_pair(query, memory, "inherited_v6", status, reason))

    inherited_pairs = {
        (row["query_id"], mid)
        for row in old_gold for mid in row["conflict_old_memory_ids"]
    }
    for (qid, mid), reason in USER_CONFIRMED_NON_CONFLICTS.items():
        if (qid, mid) not in inherited_pairs:
            gold_audit.append(audit_pair(
                queries_by_id[qid], memories_by_id[mid], "review_candidate",
                "not_conflict_user_confirmed", reason,
            ))

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for memory in memories:
        subject = subject_key(memory)
        item = normalized_item(memory)
        if subject and item:
            groups[(subject, item)].append(memory)

    automatic_additions = 0
    skipped_same_values = 0

    for query in queries:
        qid = query["query_id"]
        q_subject = subject_key(query)
        q_item = normalized_item(query)
        if not q_subject or not q_item:
            continue
        for memory in groups.get((q_subject, q_item), []):
            mid = memory["memory_id"]
            if (qid, mid) in USER_CONFIRMED_NON_CONFLICTS:
                continue
            if mid in gold_by_id[qid] or (qid, mid) in review_pairs:
                continue
            query_value = value_key(query, q_item)
            memory_value = value_key(memory, q_item)
            if query_value is not None and memory_value is not None:
                if query_value == memory_value:
                    skipped_same_values += 1
                    continue
                # A gym promotion's generic discount and a conditional
                # new-member discount can coexist; keep condition-bearing
                # cases for explicit review.
                if q_item.startswith("topic:") and q_item.endswith("|discount"):
                    q_condition = value_condition(query)
                    m_condition = value_condition(memory)
                    if q_condition != m_condition:
                        key = (qid, mid)
                        review_pairs[key] = {
                            "query_id": qid, "memory_id": mid,
                            "reason": "discount values differ, but the applicability conditions may differ",
                            "query_content": query["content"], "memory_content": memory["content"],
                            "query_value": query_value, "memory_value": memory_value,
                        }
                        continue
                gold_by_id[qid].add(mid)
                reason = "正文主体与事项一致，抽取到不同取值；标为冲突。"
                gold_reasons[qid][mid] = reason
                gold_audit.append(audit_pair(
                    query, memory, "automatic_addition", "conflict_auto_values_differ", reason
                ))
                automatic_additions += 1
                continue

            # Topic-level overlap without extractable values is not enough to
            # promote a non-gold candidate or request review; inherited golds
            # are audited separately above.
            continue

    # Surface remaining named/unqualified identity boundaries for review.
    for query in queries:
        qid = query["query_id"]
        q_subject, q_item = subject_key(query), normalized_item(query)
        if not q_subject or not q_item:
            continue
        for memory in memories:
            mid = memory["memory_id"]
            if (qid, mid) in USER_CONFIRMED_NON_CONFLICTS or (qid, mid) in USER_CONFIRMED_CONFLICTS:
                continue
            if mid in gold_by_id[qid] or normalized_item(memory) != q_item:
                continue
            relation = subject_relation(query, memory)
            if relation != "uncertain":
                continue
            qv, mv = value_key(query, q_item), value_key(memory, q_item)
            if qv is None or mv is None or qv == mv:
                continue
            review_pairs.setdefault((qid, mid), {
                "query_id": qid, "memory_id": mid,
                "reason": "same item and different extracted values, but one subject is named and the other is unqualified",
                "query_content": query["content"], "memory_content": memory["content"],
                "query_value": qv, "memory_value": mv,
            })

    audited_pairs = {(row["query_id"], row["memory_id"]) for row in gold_audit}
    for (qid, mid), review in review_pairs.items():
        if (qid, mid) not in audited_pairs:
            gold_audit.append(audit_pair(
                queries_by_id[qid], memories_by_id[mid], "candidate_review",
                "manual_review", review["reason"],
            ))

    dropped_writes = []
    empty_gold_qids = {qid for qid, ids in gold_by_id.items() if not ids}
    source_gold_by_id = {row["query_id"]: row["conflict_old_memory_ids"] for row in old_gold}
    for qid in sorted(empty_gold_qids):
        query = queries_by_id[qid]
        dropped_writes.append({
            "query_id": qid,
            "content": query["content"],
            "removed_conflict_old_memory_ids": sorted(source_gold_by_id.get(qid, [])),
            "reason": "逐对复核后所有原冲突标签均不满足同一主体；活动冲突数据不保留空金标写入。原文仍在 provenance.jsonl。",
        })
    queries = [query for query in queries if query["query_id"] not in empty_gold_qids]

    gold_rows = []
    for old in old_gold:
        qid = old["query_id"]
        if qid in empty_gold_qids:
            continue
        ids = sorted(gold_by_id[qid])
        gold_rows.append({
            "query_id": qid,
            "conflict_old_memory_ids": ids,
            "fully_reviewed": False,
            "human_reviewed": False,
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("memories.jsonl", memories), ("queries.jsonl", queries),
                       ("gold.jsonl", gold_rows)):
        (OUTPUT / name).write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
    (OUTPUT / "provenance.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in provenance), encoding="utf-8")
    (OUTPUT / "entity_aliases.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in alias_audit), encoding="utf-8")
    (REVIEW_OUTPUT).write_text(
        "".join(canonical(row) + "\n" for row in sorted(
            review_pairs.values(), key=lambda row: (row["query_id"], row["memory_id"]))),
        encoding="utf-8")
    pair_rationales = [
        {"query_id": qid, "memory_id": mid, "rationale": reason}
        for qid, reasons in gold_reasons.items()
        for mid, reason in reasons.items()
    ]
    (OUTPUT / "pair_label_rationales.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in sorted(
            pair_rationales, key=lambda row: (row["query_id"], row["memory_id"]))),
        encoding="utf-8")
    (OUTPUT / "gold_audit.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in sorted(
            gold_audit, key=lambda row: (row["query_id"], row["memory_id"]))),
        encoding="utf-8")

    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    prior_rewrites = manifest.get("rewrites", {"count": 0, "items": []})
    prior_rewrites["items"] = [item for item in prior_rewrites.get("items", [])
                                if item.get("memory_id") not in CONTENT_REWRITES]
    prior_rewrites["items"].extend({
        "memory_id": memory_id,
        "before": rewrite["before"],
        "after": rewrite["after"],
        "reason": rewrite["reason"],
    } for memory_id, rewrite in CONTENT_REWRITES.items())
    prior_rewrites["count"] = len(prior_rewrites["items"])
    audit_status_counts = Counter(row["status"] for row in gold_audit)
    manifest.update({
        "dataset_id": "validmem_v7",
        "task": "一个统一的冻结旧记忆库 + 独立新记忆写入；冲突 = 同一主体、同一事项的不同值。",
        "source": {**manifest["source"], "derived_from": "validmem_v6"},
        "identity_and_conflict_policy": {
            "active_identity_source": "memory/query content only",
            "same_exact_name": "相同姓名在所有场景中视为同一主体",
            "different_names": "姓名不同则视为不同主体；不做近似姓名合并",
            "unqualified_subjects": "匿名-only 类别共用一个姓名；若同类也有具名主体，每个匿名场景生成新姓名，不与记录者自动合并",
            "alias_mapping": "混合类别按原始 user_id 分场景生成的新姓名及匿名-only 共用姓名记录在 entity_aliases.jsonl",
            "conflict": "同一主体、同一事项、取值不同即冲突；不按 personal/world 或 source user 分割",
            "same_value": "取值相同的重述不计冲突",
            "time": "不使用 created_day 将旧值排除；created_day 仅为溯源字段",
        },
        "label_definition": {
            "conflict": "同一主体 + 同一事项 + 不同取值",
            "same_name": "精确姓名相同即可跨场景匹配，不按原 user_id 或 scope 区分",
            "different_name": "不同姓名是不同主体；不做 John/John Doe 这类近似姓名推断",
            "unqualified_subject": "匿名-only 主语全库统一；同时存在具名实体的类别按场景分配唯一匿名主体姓名",
            "same_value": "同事项同值的改写或重复不算冲突",
            "conditions": "若适用条件不同可能并存，需人工复核，不自动标冲突",
            "time": "新旧顺序不改变不同值标签；时间字段只用于溯源",
        },
        "renamed_entities": [{"from": old, "to": new, "reason": "与 John 区分为不同主体，并降低姓名相似性"}
                             for old, new in RENAMES.items()],
        "review_status": {
            "fully_reviewed": False,
            "human_reviewed": False,
            "pending_pair_count": len(review_pairs),
            "meaning": "自动添加仅覆盖可抽取到不同取值的明确同主体同事项配对；其余边界配对在 manual_review.jsonl，裁定后再更新金标和状态。",
        },
        "human_decisions": ([
            {"query_id": qid, "memory_id": mid, "label": "conflict", "reason": reason}
            for (qid, mid), reason in USER_CONFIRMED_CONFLICTS.items()
        ] + [
            {"query_id": qid, "memory_id": mid, "label": "not_conflict_for_current_labeling", "reason": reason}
            for (qid, mid), reason in USER_CONFIRMED_NON_CONFLICTS.items()
        ]),
        "revalidation_dropped_writes": dropped_writes,
        "gold_audit": {
            "file": "gold_audit.jsonl",
            "inherited_pair_count": sum(len(row["conflict_old_memory_ids"]) for row in old_gold),
            "status_counts": dict(audit_status_counts),
            "meaning": "每个继承金标都重新检查正文主体与事项；不同主体已排除，无法解析的边界进入人工复核，完整证据保存在 gold_audit.jsonl。",
        },
        "rewrites": prior_rewrites,
        "totals": {
            **manifest.get("totals", {}),
            "memory_count": len(memories),
            "query_count": len(queries),
            "conflict_pairs": sum(len(row["conflict_old_memory_ids"]) for row in gold_rows),
            "writes_without_conflict": 0,
            "writes_dropped_after_revalidation": len(dropped_writes),
            "automatic_cross_scenario_additions": automatic_additions,
            "same_value_candidates_excluded": skipped_same_values,
            "pending_manual_review_pairs": len(review_pairs),
            "inherited_gold_pairs_revalidated": sum(len(row["conflict_old_memory_ids"]) for row in old_gold),
            "inherited_subject_mismatch_pairs_excluded": audit_status_counts.get("not_conflict_subject_mismatch", 0),
        },
        "checks": {
            "scope_owner_absent_from_active_records": "memories/queries source_ref no longer contains scope or user_id",
            "provenance_separate": "original source_ref and content saved in provenance.jsonl",
            "same_name_identity": "exact content names are resolved globally",
            "different_names_not_merged": "similar or partial names are not merged automatically",
            "inherited_gold_revalidated": "all inherited pairs rechecked against content-resolved subjects and normalized items; decisions recorded in gold_audit.jsonl",
            "every_write_has_conflict": "empty-gold writes removed after subject revalidation; active writes all retain at least one conflict",
        },
    })
    manifest.pop("channels", None)
    manifest.pop("scope_labels", None)
    manifest.pop("world_topics", None)
    manifest.pop("modifications", None)
    for key in (
        "personal_writes", "world_writes", "world_memories", "personal_memories",
        "background_personal", "background_world", "distinct_users",
        "slots_shared_by_multiple_users", "bank_role_counts",
    ):
        manifest["totals"].pop(key, None)
    manifest["conversion"] = {
        "active_identity": "只从正文解析；source_ref 不含 scope 或 user_id",
        "provenance": "原始 source_ref 与原文独立保存在 provenance.jsonl",
        "name_canonicalization": "相同姓名全局一致；匿名主体按主体类别是否另有具名实体决定共用或按场景取名；John Doe 统一重命名为 Mara Voss",
        "new_memory": "保留 validmem_v6 查询文本",
    }
    manifest["files"] = {
        name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
        for name in ("memories.jsonl", "queries.jsonl", "gold.jsonl", "provenance.jsonl", "entity_aliases.jsonl",
                     "manual_review.jsonl", "pair_label_rationales.jsonl", "gold_audit.jsonl")
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    summary = Counter({"automatic_additions": automatic_additions,
                       "same_value_candidates_excluded": skipped_same_values,
                       "manual_review_pairs": len(review_pairs),
                       "writes_dropped_after_revalidation": len(dropped_writes),
                       "conflict_pairs": sum(len(row["conflict_old_memory_ids"]) for row in gold_rows)})
    print(json.dumps(dict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
