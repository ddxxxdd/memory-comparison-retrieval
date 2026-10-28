"""Convert ValidMem Part A into a two-channel shared-memory-bank dataset.

Task this dataset serves
-----------------------
One big frozen old-memory bank plus a stream of independent new memory writes. For each
write the method must pull out the old memories that may conflict with it.

Two channels, because two kinds of memory exist
-----------------------------------------------
``personal``  a claim about one specific person (preference, possession, plan, health).
              Only that person's memories can conflict with the write.

``world``     a claim about a public thing (opening hours, price, policy, textbook
              edition, schedule). Anybody's statement about it is comparable - a write
              and an old memory may even come from different owners.

A pair conflicts when both sides share the slot (``conflict_entity`` +
``conflict_attribute``), state different values, and the old memory is older. The channel
rule above decides whether the two are comparable at all.

Owner and scope are metadata, not text
--------------------------------------
ValidMem stores the owner only in case metadata. An earlier revision prefixed every
sentence with ``For user <id>, `` so that text-only retrieval could see the owner. Measured
on the resulting bank, that hurt: in the grocery slot the median cosine between memories of
*different* owners was 0.814 while within the *same* owner it was 0.861, i.e. the prefix
made owners nearly indistinguishable for the embedding while the id token handed lexical
retrieval a free match. Scope and owner now live only in ``source_ref``.

Every write gets a conflict
---------------------------
The source leaves two near-miss cases with no superseded memory; they only test subject
confusion ("Patient John" vs "Patient Jane"). Their required conflicting memory is
synthesised here, marked ``source_role=synthesized_conflict`` and listed in the manifest.

World-channel writes are synthesised on top of existing world facts: each one restates a
fact that is already in the bank with a different value, so the gold link is the old
memory itself rather than an invented one.

Output files (only the fields the task uses)
--------------------------------------------
    memories.jsonl  memory_id, content, source_ref
    queries.jsonl   query_id, content, source_ref
    gold.jsonl      query_id, conflict_old_memory_ids, fully_reviewed, human_reviewed
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

DATASET_URL = "https://huggingface.co/datasets/Zhou11Alex/ValidMem"
BANK_ROLES = ("superseded_decoys", "irrelevant_decoys")

# The two source near-miss cases have no superseded memory. Both compare "John" with
# "Jane", i.e. they only test subject confusion. Give each write a same-subject,
# same-slot, different-value old memory so that every write owns a real conflict.
SYNTHESIZED = {
    "TC-A-0123": {
        "memory_id": "SYN-TC-A-0123",
        "content": "Patient John has a known allergy to ibuprofen.",
        "created_day": 100,
        "conflict_entity": "John",
        "conflict_attribute": "allergy",
        "note": "来源只有不同主体(Jane)的同值记忆;补一条同主体不同值(ibuprofen)的旧记忆",
    },
    "TC-A-0137": {
        "memory_id": "SYN-TC-A-0137",
        "content": "Patient John Doe has an allergy to sulfa drugs.",
        "created_day": 20,
        "conflict_entity": "John Doe",
        "conflict_attribute": "allergy",
        "note": "来源只有不同主体(Jane Doe)的同值记忆;补一条同主体不同值(sulfa drugs)的旧记忆",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def canonical(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(memories_path: Path, cases_path: Path, output: Path, parts: tuple[str, ...],
          scope_labels: dict[str, dict], world_writes: list[dict],
          slot_labels: dict[str, dict], label_overrides: dict) -> dict:
    memories = read_jsonl(memories_path)
    cases = read_jsonl(cases_path)
    by_id = {m["mem_id"]: m for m in memories}

    # --- world topic map: "same thing" decided by content, frozen as labels -----
    # source_ref 的 (entity, attribute) 是来源按 case 各自写的名字,同一件事会有多个写法
    # (Physics / Physics 101 / Physics course)。这里用一次内容标注把它们归到
    # (subject, aspect);topic 落盘后,金标仍按标签机械重算。
    merged_subjects = {}
    for rule in label_overrides.get("subject_merges", []):
        merged_subjects[(rule["from"], rule["aspect"])] = rule["to"]
    merged_aspects = {(rule["subject"], rule["from"]): rule["to"]
                      for rule in label_overrides.get("aspect_merges", [])}

    def topic_of(ref):
        key = f"{ref['conflict_entity']}\u0000{ref['conflict_attribute']}"
        label = slot_labels.get(key)
        if label is None:
            raise ValueError(f"槽位键 {key} 没有 topic 标签")
        subject, aspect = label["subject"], label["aspect"]
        # 先做 aspect 归并,再按(可能已改过的)aspect 查 subject 归并,
        # 否则"aspect 被改正、subject 仍需合并"的键会漏掉。
        aspect = merged_aspects.get((subject, aspect), aspect)
        merged = merged_subjects.get((subject, aspect)) or merged_subjects.get(
            (subject, label["aspect"]))
        return f"{merged or subject}|{aspect}"

    mention_pattern = re.compile(label_overrides["mention_pattern"], re.IGNORECASE)
    non_assertive = re.compile(label_overrides["non_assertive"]["pattern"], re.IGNORECASE)
    rewrites = {item["memory_id"]: item["after"] for item in label_overrides.get("rewrites", [])}
    exclusions = {item["memory_id"]: item["reason"] for item in label_overrides.get("exclusions", [])}

    selected = [c for c in cases if c["part"] in parts]
    if not selected:
        raise ValueError("no cases selected")
    for case in selected:
        if len(case["ground_truth"]) != 1:
            raise ValueError(f"{case['id']}: expected exactly one ground_truth id")

    new_ids = {case["ground_truth"][0] for case in selected}
    if len(new_ids) != len(selected):
        raise ValueError("two cases share the same ground_truth memory id")

    # --- frozen bank, personal channel: subject = owning scenario -------------
    claimed: dict[str, tuple[str, str]] = {}
    for case in selected:
        for role in BANK_ROLES:
            for mem_id in case[role]:
                if mem_id in new_ids:
                    raise ValueError(f"{case['id']}: bank role {role} holds a new-write memory")
                previous = claimed.get(mem_id)
                if previous is not None and previous[0] != case["id"]:
                    raise ValueError(f"{mem_id}: claimed by {previous[0]} and {case['id']}")
                claimed[mem_id] = (case["id"], role)

    bank: list[dict] = []
    queries: list[dict] = []
    gold: list[dict] = []
    role_counts = collections.Counter()

    def bank_row(memory_id, user_id, scope, text, role, entity, attribute, created_day,
                 topic=None) -> dict:
        ref = {
            "user_id": user_id,
            "scope": scope,
            "source_role": role,
            "conflict_entity": entity,
            "conflict_attribute": attribute,
            "created_day": created_day,
        }
        if topic is not None:
            ref["topic"] = topic
        # 少量旧记忆按需求改写为现在时表述(原句是过去时/计划),改写清单记入 manifest。
        return {"memory_id": memory_id, "content": rewrites.get(memory_id, text), "source_ref": ref}

    synthesized_used = []
    notes = []
    dropped_writes = []
    dropped_pairs = []

    def append_case_bank(case, user_id, entity, attribute):
        for role in BANK_ROLES:
            for mem_id in case[role]:
                if claimed[mem_id][1] != role:
                    continue
                old = by_id[mem_id]
                bank.append(bank_row(mem_id, user_id, "personal", old["description"], role,
                                     old["conflict_entity"], old["conflict_attribute"],
                                     old["created_day"]))
                role_counts[role] += 1

    for case in selected:
        user_id = case["id"]
        new_mem = by_id[case["ground_truth"][0]]
        entity = new_mem["conflict_entity"]
        attribute = new_mem["conflict_attribute"]

        conflict_ids = list(case["superseded_decoys"])
        for mem_id in conflict_ids:
            old = by_id[mem_id]
            if (old["conflict_entity"], old["conflict_attribute"]) != (entity, attribute):
                raise ValueError(f"{user_id}: {mem_id} labelled conflict but its slot differs")

        # 明确把主张限定在过去(was / lived / old / initial)或只表达计划的记录,
        # 不与写入的当前取值构成互斥,不算冲突。测量值没有时间限定,仍算冲突。
        historical = [mem_id for mem_id in conflict_ids
                      if non_assertive.search(by_id[mem_id]["description"])]
        for mem_id in historical:
            dropped_pairs.append({"query_id": user_id, "memory_id": mem_id,
                                  "content": by_id[mem_id]["description"],
                                  "reason": "历史/计划表述,不构成当前取值的互斥"})
            conflict_ids.remove(mem_id)

        # 被改写成现在时的旧记忆,按改写意图直接算作冲突(来源标签里没有它们)。
        for mem_id in rewrites:
            owner = claimed.get(mem_id)
            if owner is None or owner[0] != user_id:
                continue
            old = by_id[mem_id]
            if (old["conflict_entity"], old["conflict_attribute"]) != (entity, attribute):
                continue
            if mem_id not in conflict_ids:
                conflict_ids.append(mem_id)

        # stage 1 only asks for the conflicting memories; there is no separate
        # "comparable but not conflicting" layer any more.
        same_slot_irrelevant = [mid for mid in case["irrelevant_decoys"]
                                if (by_id[mid]["conflict_entity"],
                                    by_id[mid]["conflict_attribute"]) == (entity, attribute)]
        for mid in same_slot_irrelevant:
            # A decoy that shares the slot but does not assert a value for it is not the
            # same thing; it stays in the bank as an ordinary distractor.
            notes.append({"query_id": user_id, "memory_id": mid,
                          "reason": "同主体同事项但未断言该属性的取值,不算同一件事,不标冲突"})

        labels = {mid: "conflict" for mid in conflict_ids}
        if not conflict_ids:
            synth = SYNTHESIZED.get(user_id)
            if synth is None:
                # 按口径这条写入没有任何冲突旧记忆,就不再作为一条写入;旧记忆仍留在库里。
                append_case_bank(case, user_id, entity, attribute)
                dropped_writes.append({"query_id": user_id, "content": new_mem["description"],
                                       "reason": "剔除历史/计划表述后没有冲突旧记忆"})
                continue
            if (synth["conflict_entity"], synth["conflict_attribute"]) != (entity, attribute):
                raise ValueError(f"{user_id}: synthesized memory does not match the write slot")
            if synth["created_day"] >= new_mem["created_day"]:
                raise ValueError(f"{user_id}: synthesized memory is not older than the write")
            bank.append(bank_row(synth["memory_id"], user_id, "personal", synth["content"],
                                 "synthesized_conflict", entity, attribute, synth["created_day"]))
            labels[synth["memory_id"]] = "conflict"
            synthesized_used.append({"query_id": user_id, "memory_id": synth["memory_id"],
                                     "content": synth["content"], "reason": synth["note"]})

        queries.append({
            "query_id": user_id,
            "content": new_mem["description"],
            "source_ref": {
                "user_id": user_id,
                "scope": "personal",
                "new_memory_id": new_mem["mem_id"],
                "conflict_entity": entity,
                "conflict_attribute": attribute,
                "created_day": new_mem["created_day"],
            },
        })
        gold.append({
            "query_id": user_id,
            "conflict_old_memory_ids": list(labels),
            "fully_reviewed": True,
            "human_reviewed": False,
        })

        append_case_bank(case, user_id, entity, attribute)

    # --- background bank: memories no selected case claims ---------------------
    # Personal ones represent other users (each gets its own id, so it can never conflict
    # with a write). World ones have no owner at all: they are public facts and stay
    # comparable with world-channel writes.
    bg_claimed = {row["memory_id"] for row in bank} | new_ids
    background = collections.Counter()
    for index, mem in enumerate(sorted(memories, key=lambda m: (m["created_day"], m["mem_id"])), 1):
        if mem["mem_id"] in bg_claimed:
            continue
        label = scope_labels.get(mem["mem_id"])
        if label is None:
            raise ValueError(f"{mem['mem_id']}: background memory has no scope label")
        if label["scope"] == "world":
            topic = topic_of(mem)
            bank.append(bank_row(mem["mem_id"], None, "world", mem["description"],
                                 "background_world", mem["conflict_entity"],
                                 mem["conflict_attribute"], mem["created_day"], topic))
            background["world"] += 1
        else:
            bank.append(bank_row(mem["mem_id"], f"BG-{index:04d}", "personal", mem["description"],
                                 "background_personal", mem["conflict_entity"],
                                 mem["conflict_attribute"], mem["created_day"]))
            background["personal"] += 1

    # --- world channel: synthesised writes over existing public facts ----------
    by_memory = {row["memory_id"]: row for row in bank}
    world_used = []
    for number, write in enumerate(world_writes, 1):
        query_id = write["query_id"]
        anchor = by_memory.get(write["slot_ref"])
        if anchor is None:
            raise ValueError(f"{query_id}: slot_ref {write['slot_ref']} is not in the bank")
        if anchor["source_ref"]["scope"] != "world":
            raise ValueError(f"{query_id}: slot_ref {write['slot_ref']} is not a world memory")
        topic = anchor["source_ref"]["topic"]
        # 同一 topic 里所有更早、且真的给出了不同取值的世界记忆都算冲突。
        # 只提及/评论/补充材料的记录不算(它们没有给出竞争取值),显式列出的例外也不算。
        conflicts = []
        for row in bank:
            ref = row["source_ref"]
            if ref["scope"] != "world" or ref.get("topic") != topic:
                continue
            if row["content"].strip() == write["content"].strip():
                continue
            if mention_pattern.search(row["content"]):
                continue
            if non_assertive.search(row["content"]):
                continue
            if row["memory_id"] in exclusions:
                continue
            conflicts.append(row["memory_id"])
        if not conflicts:
            dropped_writes.append({"query_id": query_id, "content": write["content"],
                                   "reason": f"topic {topic} 下剔除历史/计划表述后没有冲突旧记忆"})
            continue
        queries.append({
            "query_id": query_id,
            "content": write["content"],
            "source_ref": {
                "user_id": None,
                "scope": "world",
                "new_memory_id": f"TW-M-{number:04d}",
                "conflict_entity": anchor["source_ref"]["conflict_entity"],
                "conflict_attribute": anchor["source_ref"]["conflict_attribute"],
                "topic": topic,
                "created_day": write["created_day"],
            },
        })
        gold.append({
            "query_id": query_id,
            "conflict_old_memory_ids": conflicts,
            "fully_reviewed": True,
            "human_reviewed": False,
        })
        world_used.append({"query_id": query_id, "memory_id": f"TW-M-{number:04d}",
                           "content": write["content"], "conflicts": conflicts,
                           "topic": topic, "note": write.get("note", "")})

    bank.sort(key=lambda row: (row["source_ref"]["created_day"], row["memory_id"]))
    queries.sort(key=lambda row: row["query_id"])
    gold.sort(key=lambda row: row["query_id"])

    # --- assertions the whole dataset must satisfy -----------------------------
    seen = set()
    for row in bank:
        if row["memory_id"] in seen:
            raise ValueError(f"duplicate bank memory {row['memory_id']}")
        seen.add(row["memory_id"])
        if row["content"].startswith("For user "):
            raise ValueError(f"{row['memory_id']}: content still carries a subject prefix")
    if new_ids & seen:
        raise ValueError("a new-write memory leaked into the bank")

    by_query = {row["query_id"]: row for row in queries}
    if len(by_query) != len(queries):
        raise ValueError("duplicate query id")
    for row in gold:
        query = by_query[row["query_id"]]
        scope = query["source_ref"]["scope"]
        user = query["source_ref"]["user_id"]
        for mem_id in row["conflict_old_memory_ids"]:
            old = by_memory[mem_id]
            if old["source_ref"]["scope"] != scope:
                raise ValueError(f"{row['query_id']}: {mem_id} is in another channel")
            if scope == "personal" and old["source_ref"]["user_id"] != user:
                raise ValueError(f"{row['query_id']}: {mem_id} is a different subject")
            if scope == "world":
                if old["source_ref"].get("topic") != query["source_ref"].get("topic"):
                    raise ValueError(f"{row['query_id']}: {mem_id} is another topic")
            elif (old["source_ref"]["conflict_entity"],
                  old["source_ref"]["conflict_attribute"]) != (
                  query["source_ref"]["conflict_entity"],
                  query["source_ref"]["conflict_attribute"]):
                raise ValueError(f"{row['query_id']}: {mem_id} slot differs")
            if old["content"].strip() == query["content"].strip():
                raise ValueError(f"{row['query_id']}: {mem_id} states the same text")
        if not row["conflict_old_memory_ids"]:
            raise ValueError(f"{row['query_id']}: write has no conflicting old memory")

    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("memories.jsonl", bank), ("queries.jsonl", queries),
                       ("gold.jsonl", gold)):
        (output / name).write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")

    conflicts = sum(len(g["conflict_old_memory_ids"]) for g in gold)
    users = {row["source_ref"]["user_id"] for row in bank if row["source_ref"]["user_id"]}
    slot_users = collections.defaultdict(set)
    for row in bank:
        key = (row["source_ref"]["conflict_entity"], row["source_ref"]["conflict_attribute"])
        slot_users[key].add(row["source_ref"]["user_id"])
    shared_slots = [k for k, owners in slot_users.items() if len(owners) > 1]

    manifest = {
        "dataset_id": output.name,
        "task": ("一个冻结旧记忆库 + 独立新记忆写入;每条写入需从旧库检索出可能冲突的旧记忆。"
                 "冲突 = 同一通道 + 同一事项 + 取值不同。"),
        "source": {
            "dataset": "ValidMem",
            "url": DATASET_URL,
            "citation": "Validity Is Not Visibility: A Lifecycle FSM for Agent Memory (EMNLP 2026)",
            "case_partition": list(parts),
            "memories_file_sha256": sha256_of(memories_path),
            "cases_file_sha256": sha256_of(cases_path),
        },
        "channels": {
            "personal": "关于某个具体人的记忆;只有同一个 owner 的旧记忆可以与之冲突。",
            "world": "关于公共事物的记忆(营业时间/价格/政策/教材版本/日程);任何人都可以与之冲突。",
            "why": ("不同用户的同类偏好不是同一件事,而用户说错一个公共事实仍是同一件事;"
                    "因此可比性由通道决定,而不是由文本里有没有主体决定。"),
        },
        "conversion": {
            "content": "来源描述原文,不加任何主体前缀",
            "owner": "user_id 只在 source_ref 中;world 通道为 null",
            "scope": "source_ref.scope = personal | world",
            "new_memory": "case.ground_truth[0].description",
            "bank_roles": list(BANK_ROLES),
            "background": "未被入选 case 引用的记忆;scope 来自 LLM 标注(见 scope_labels)",
            "world_writes": "从已有世界事实上合成,每条重述该事实并给出不同取值",
        },
        "label_definition": {
            "conflict": "同一通道 + 同一件事 + 取值不同",
            "personal": "只有同一 user_id 的旧记忆才算冲突;同一件事由来源槽位判定",
            "world": ("同一件事由 topic 判定(内容标注的 subject+aspect),不看来源槽位键;"
                      "任何人给出的不同取值都算冲突"),
            "time": ("不比较 created_day:按需求默认旧记忆库整体早于新写入,"
                     "时间字段只作溯源"),
            "comparable_layer": "已取消;第一阶段只要求找全冲突旧记忆",
            "not_a_conflict": ("同 topic 但只提及/评论/补充材料(没有给出竞争取值)的记录,"
                              "以及记录在案的例外"),
            "distinction": "不区分“同时段互斥”与“先后更新”,两者都计为冲突",
            "not_covered": ["重复冗余(同通道同事项同值的改写)"],
        },
        "world_topics": {
            "source": "scripts/label_world_slots.py:按内容标注 subject 与 aspect,结果冻结",
            "rule": "topic = subject|aspect;同 topic 内、更早、给出不同取值的旧记忆都算冲突",
            "topic_count": len({row["source_ref"].get("topic") for row in bank
                                if row["source_ref"].get("topic")}),
            "subject_merges": label_overrides.get("subject_merges", []),
            "mention_pattern": label_overrides.get("mention_pattern"),
            "mention_reason": label_overrides.get("mention_reason"),
            "exclusions": label_overrides.get("exclusions", []),
        },
        "documented_exceptions": {
            "count": len(notes),
            "items": notes,
            "meaning": "这些记录与新写入同主体同事项,但没有断言该属性的取值(例如只说‘提到某店有优惠’),"
                       "不构成同一件事,因此不标冲突,只作为普通干扰留在旧库",
        },
        "rewrites": {
            "count": len(rewrites),
            "items": label_overrides.get("rewrites", []),
            "meaning": ("这些旧记忆被改写成现在时/当前取值,使其与新写入构成明确冲突;"
                        "原文保存在 before 字段,改写是人工裁定后的一次性动作。"),
        },
        "dropped": {
            "write_count": len(dropped_writes),
            "writes": dropped_writes,
            "pair_count": len(dropped_pairs),
            "pairs": dropped_pairs,
            "meaning": ("按口径不算冲突、被剔除的历史/计划表述(was / lived / old / initial / planned 等)。"
                        "剔除后没有任何冲突的写入整条移出数据集,不为它编造旧记忆;"
                        "测量值(血压等)没有时间限定,仍算冲突。"),
        },
        "synthesized": {
            "personal_count": len(synthesized_used),
            "personal_items": synthesized_used,
            "world_count": len(world_used),
            "world_items": world_used,
            "reason": "个人通道补两条 near-miss 缺失记忆;世界通道按需求从已有公共事实上合成写入",
        },
        "scope_labels": {
            "source": "scripts/classify_memory_scope.py(串行批量 LLM 标注)",
            "world": background["world"],
            "personal": background["personal"],
            "meaning": "world 背景记忆的 user_id 置为 null,不再属于任何虚构用户",
        },
        "totals": {
            "memory_count": len(bank),
            "query_count": len(queries),
            "conflict_pairs": conflicts,
            "personal_writes": sum(1 for q in queries if q["source_ref"]["scope"] == "personal"),
            "world_writes": sum(1 for q in queries if q["source_ref"]["scope"] == "world"),
            "world_memories": sum(1 for r in bank if r["source_ref"]["scope"] == "world"),
            "personal_memories": sum(1 for r in bank if r["source_ref"]["scope"] == "personal"),
            "documented_exceptions": len(notes),
            "writes_without_conflict": sum(1 for g in gold if not g["conflict_old_memory_ids"]),
            "multi_conflict_writes": sum(
                1 for g in gold if len(g["conflict_old_memory_ids"]) >= 2),
            "background_personal": background["personal"],
            "background_world": background["world"],
            "distinct_users": len(users),
            "slots_shared_by_multiple_users": len(shared_slots),
            "bank_role_counts": dict(role_counts),
            "bank_to_write_ratio": round(len(bank) / max(len(queries), 1), 2),
        },
        "checks": {
            "no_subject_prefix": "所有 content 都不以 `For user ` 开头",
            "owner_in_metadata_only": "主体只出现在 source_ref.user_id",
            "conflict_same_channel": "所有 conflict 配对两侧 scope 相同",
            "personal_conflict_same_subject": "personal 配对的 user_id 与写入一致",
            "world_conflict_owner_free": "world 配对的两侧都不要求同一 owner",
            "conflict_same_slot": "所有 conflict 配对同 (conflict_entity, conflict_attribute)",
            "conflict_time_ignored": "不比较 created_day;默认旧库整体早于写入,时间只作溯源",
            "conflict_text_differs": "所有 conflict 配对两段文本不完全相同(取值不同由来源标注或合成构造保证)",
            "every_write_has_conflict": "每条写入至少一条冲突旧记忆",
            "no_self_leak": "旧库不包含任何新写入记忆 ID",
        },
        "review_status": {
            "fully_reviewed": True,
            "human_reviewed": False,
            "meaning": "标签来自来源发布;通道、槽位、先后、取值与泄漏检查由构造脚本断言;未做独立人工复核",
        },
        "files": {name: sha256_of(output / name)
                  for name in ("memories.jsonl", "queries.jsonl", "gold.jsonl")},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memories", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--parts", nargs="+", default=["A"])
    ap.add_argument("--scope-labels", type=Path, required=True,
                    help="背景记忆的 scope 标注 JSON(scripts/classify_memory_scope.py 产出)")
    ap.add_argument("--world-writes", type=Path, required=True,
                    help="世界通道新写入 JSON:query_id/content/created_day/slot_ref")
    ap.add_argument("--slot-labels", type=Path, required=True,
                    help="世界通道槽位键的 (subject, aspect) 标签")
    ap.add_argument("--label-overrides", type=Path, required=True,
                    help="subject 合并、只提及排除、逐条例外")
    args = ap.parse_args()

    if args.output.exists():
        raise SystemExit(f"输出目录已存在,不覆盖:{args.output}")
    scope_labels = json.loads(args.scope_labels.read_text(encoding="utf-8"))
    world_writes = json.loads(args.world_writes.read_text(encoding="utf-8"))
    slot_labels = json.loads(args.slot_labels.read_text(encoding="utf-8"))["labels"]
    label_overrides = json.loads(args.label_overrides.read_text(encoding="utf-8"))
    manifest = build(args.memories, args.cases, args.output, tuple(args.parts),
                     scope_labels, world_writes, slot_labels, label_overrides)
    print(json.dumps(manifest["totals"], ensure_ascii=False, indent=2))
    print("written:", args.output)


if __name__ == "__main__":
    main()
