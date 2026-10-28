"""建库阶段的可判定约束：抽取结果校验、目录归并。

这里只放"能从数据本身判定"的规则：
不读数据集标注、不带领域词表、不猜测哪一段文字是取值。
每条规则都对应提示词里已经写明的硬性要求，因此对任何记忆库都成立。
"""

from collections import defaultdict
import re

import numpy as np

CJK = re.compile(r"[\u3400-\u9fff]")
NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def is_degenerate(text):
    """描述去标点后没有实际内容。"""
    return not NON_WORD.sub("", text).strip()


def audit(contents, rows, *, content_vectors=None, template_threshold=None):
    """校验一批抽取结果，返回违规清单。

    contents 与 rows 按 index 对齐；rows 每项为 {"index", "descriptions"}。
    content_vectors 给出时，额外检查"内容近重复的记忆必须共用描述"。
    """
    positions = {row["index"] for row in rows}
    violations = []
    for row in rows:
        content = contents[row["index"]]
        if not row["descriptions"]:
            violations.append({"index": row["index"], "rule": "missing",
                               "detail": "这条记忆没有任何描述，永远无法被召回"})
            continue
        for text in row["descriptions"]:
            if is_degenerate(text):
                violations.append({"index": row["index"], "description": text, "rule": "degenerate",
                                   "detail": "描述除了标点/空白没有实际内容"})
            elif len(text) > max(2 * len(content), 200):
                violations.append({"index": row["index"], "description": text, "rule": "too_long",
                                   "detail": f"描述 {len(text)} 字，比原文（{len(content)} 字）长一倍以上；"
                                             "描述应是事项，不是原文副本"})
    if content_vectors is not None:
        violations.extend(_split_pairs(rows, content_vectors, template_threshold))
    return [v for v in violations if v["index"] in positions]


def _split_pairs(rows, vectors, threshold):
    """同一批里内容近重复、却没有共用描述的记忆，记为诊断项。

    内容相似不足以证明"同一事项"：不同主体的同类事项（两个用户各自的过敏史、
    各自的住址）本来就会高相似，强行要求共用一份描述会把主体区别抹掉。
    因此这里只留档供事后查看，不进入重写，也不作为"必须统一描述"的依据。
    """
    if threshold is None or len(rows) < 2:
        return []
    sims = vectors @ vectors.T
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if sims[i, j] < threshold:
                continue
            left = set(rows[i]["descriptions"])
            right = set(rows[j]["descriptions"])
            if left & right:
                continue
            out.append({
                "index": rows[i]["index"], "other_index": rows[j]["index"],
                "rule": "same_event_split", "similarity": round(float(sims[i, j]), 4),
                "detail": (f"本条与第 {rows[j]['index']} 条内容相似度 {sims[i, j]:.2f}，"
                           f"描述不同（仅记录，不强制统一）：{sorted(left)} 对 {sorted(right)}"),
            })
    return out


def summarize(violations, total):
    counts = defaultdict(int)
    for violation in violations:
        counts[violation["rule"]] += 1
    return {"memories": total, "violations": len(violations), "by_rule": dict(sorted(counts.items()))}


# 只记录、不触发重写的规则。内容相似度不能证明"同一事项"：不同主体的同类事项
# （两个用户各自的过敏史或住址）相似度同样很高，强制统一描述会把主体区别抹掉。
# 这类项留在审计里供事后查看，但不要求模型改写。
DIAGNOSTIC_RULES = frozenset({"same_event_split"})


def enforceable(violations):
    """交给重写轮的只有可判定规则，诊断项不参与。"""
    return [violation for violation in violations if violation["rule"] not in DIAGNOSTIC_RULES]


def _max_distinct(left_block, right_block, left_ids, right_ids):
    """两个条目的记忆之间最大的内容相似度，只比"不同的两条记忆"。

    同一条记忆同时挂在两个条目上，说明它同时提供两件事（日期和地点），
    这不是"同一事项被拆开"，不能据此归并。
    """
    products = left_block @ right_block.T
    different = np.asarray([[a != b for b in right_ids] for a in left_ids], dtype=bool)
    return float(products[different].max()) if different.any() else 0.0


class _Groups:
    """并查集：归并关系可传递，同时记录每组是靠哪些判据连起来的。"""

    def __init__(self, count):
        self.parent = list(range(count))
        self.rules = [set() for _ in range(count)]

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right, rule):
        a, b = self.find(left), self.find(right)
        if a == b:
            self.rules[a].add(rule)
            return
        if b < a:
            a, b = b, a
        self.parent[b] = a
        self.rules[a] |= self.rules[b] | {rule}


def merge_entries(catalog, description_vectors, content_vectors, memory_positions, *,
                  description_threshold, template_threshold):
    """把同一事项的多个描述收成一个条目，返回 (新目录, 归并报告)。

    两条判据都只看库内数据：
      一、两条描述的向量相似度达到 description_threshold：同一事项的两种措辞；
      二、两个条目各自挂着的旧记忆里，存在一对内容相似度达到 template_threshold：
          同一事项的两个取值。取值写进描述时条目会被拆开，这一条负责收回。

    归并只改变"目录里怎么呈现"，不丢任何一条检索路径：被收掉的措辞仍然作为别名
    留在 aliases 里参与打分，因此归并前后任何记忆的可达性只增不减。
    条目对外呈现的文本取"挂着记忆最多"的那一份（并列取短、再取编号小），
    保留下来的措辞是库里实际复用最多的写法。
    """
    count = len(catalog)
    report = {"description_threshold": description_threshold,
              "template_threshold": template_threshold,
              "entries_before": count, "entries_after": count, "merged_count": 0,
              "links": [], "merged_groups": []}
    if count < 2:
        return [dict(entry, aliases=[position]) for position, entry in enumerate(catalog)], report

    groups = _Groups(count)
    if description_threshold is not None and description_vectors.shape[0] == count:
        sims = description_vectors @ description_vectors.T
        rows, columns = np.triu_indices(count, 1)
        for left, right in zip(rows.tolist(), columns.tolist()):
            if sims[left, right] < description_threshold:
                continue
            groups.union(left, right, "description_duplicate")
            report["links"].append({"rule": "description_duplicate",
                                    "left": catalog[left]["id"], "right": catalog[right]["id"],
                                    "similarity": round(float(sims[left, right]), 4)})
    if template_threshold is not None and content_vectors.shape[0] == len(memory_positions):
        blocks = [content_vectors[[memory_positions[m] for m in entry["memory_ids"]]]
                  for entry in catalog]
        ids = [entry["memory_ids"] for entry in catalog]
        for left in range(count):
            for right in range(left + 1, count):
                if groups.find(left) == groups.find(right):
                    continue
                similarity = _max_distinct(blocks[left], blocks[right], ids[left], ids[right])
                if similarity < template_threshold:
                    continue
                groups.union(left, right, "memory_template")
                report["links"].append({"rule": "memory_template",
                                        "left": catalog[left]["id"], "right": catalog[right]["id"],
                                        "similarity": round(similarity, 4)})

    clusters = defaultdict(list)
    for position in range(count):
        clusters[groups.find(position)].append(position)
    order = sorted(clusters, key=lambda root: min(clusters[root]))
    # 条目文本取复用最多的一份；并列时取更短、再取编号更小的，保证可复现。
    chosen = {root: min(members, key=lambda i: (-len(catalog[i]["memory_ids"]),
                                                len(catalog[i]["description"]), catalog[i]["id"]))
              for root, members in clusters.items()}

    new_catalog = []
    for root in order:
        members = sorted(clusters[root])
        memory_ids = list(dict.fromkeys(m for i in members for m in catalog[i]["memory_ids"]))
        # aliases 的第一项必须是条目对外呈现的那一行文本，其余是保留的别名。
        aliases = [chosen[root]] + [i for i in members if i != chosen[root]]
        entry = {"id": f"D{len(new_catalog) + 1:06d}",
                 "description": catalog[chosen[root]]["description"],
                 "memory_ids": memory_ids,
                 "aliases": aliases}
        new_catalog.append(entry)
        if len(members) > 1:
            report["merged_groups"].append({
                "into": entry["id"],
                "from": [catalog[i]["id"] for i in members],
                "descriptions": [catalog[i]["description"] for i in members],
                "memory_count": len(memory_ids),
                "rules": sorted(groups.rules[root]),
            })
    report["entries_after"] = len(new_catalog)
    report["merged_count"] = sum(len(g["from"]) - 1 for g in report["merged_groups"])
    return new_catalog, report
