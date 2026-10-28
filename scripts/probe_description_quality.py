"""诊断"旧记忆→描述事项"这一层本身是否成立，不跑检索。

三条互不依赖的检查：

  static  目录形态：条目数、单记忆条目占比、近重复待合并对数、准入出口计数。
  converge 同事项收敛：对每条写入走生产同一条 describe()，看它选中的条目
           与 gold 冲突旧记忆的条目是否有交集；同时算全库非 gold 旧记忆的
           共享率作为对照。只有"冲突对明显高于随机对"才说明事项层在区分
           "同一件事"与"两件不同的事"。
  leak     值泄漏：抽条目问判定器，描述本身是否已经包含该记忆给出的取值。

用法：
  python scripts/probe_description_quality.py --index runs/indexes/x --mode static
  python scripts/probe_description_quality.py --index runs/indexes/x \
      --dataset data/validmem_v5 --mode converge --report reports/x.md
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_retrieval.api import APIClient, APIError
from memory_retrieval.config import DESCRIPTION_DUPLICATE_THRESHOLD, Settings
from memory_retrieval.data import build_retrieval_text, read_jsonl, write_json
from memory_retrieval.prompts import describe

DIFFUSION_THRESHOLD = 0.90


def _pct(numerator, denominator):
    return f"{numerator}/{denominator} ({numerator / denominator:.3f})" if denominator else "0/0"


def load_index(path):
    directory = Path(path)
    manifest = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    vectors = np.load(directory / "vectors.npz")
    return manifest, vectors


def static_report(manifest, vectors):
    catalog = manifest["catalog"]
    sizes = np.array([len(entry["memory_ids"]) for entry in catalog])
    canonical = np.array([entry["aliases"][0] for entry in catalog])
    matrix = vectors["descriptions"][canonical]
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    sims = matrix @ matrix.T
    upper = sims[np.triu_indices(len(catalog), k=1)]
    return {
        "memories": len(manifest["memories"]),
        "entries": len(catalog),
        "entries_per_memory": round(len(catalog) / len(manifest["memories"]), 4),
        "singleton_entries": int((sizes == 1).sum()),
        "singleton_ratio": round(float((sizes == 1).mean()), 4),
        "mean_memories_per_entry": round(float(sizes.mean()), 4),
        "max_memories_per_entry": int(sizes.max()),
        "duplicate_pairs": int((upper >= DESCRIPTION_DUPLICATE_THRESHOLD).sum()),
        "near_duplicate_pairs": int((upper >= DIFFUSION_THRESHOLD).sum()),
        "values": {key: value for key, value in manifest.get("values", {}).items()
                   if key != "items"},
        "quality": {key: value for key, value in manifest.get("quality", {}).items()
                    if key != "batches"},
        "prompt_version": manifest.get("prompt_version"),
        "prompt_hash": manifest.get("prompt_hash", "")[:12],
    }


def entry_map(catalog):
    """旧记忆 → 它挂着的条目集合。"""
    owners = {}
    for entry in catalog:
        for memory_id in entry["memory_ids"]:
            owners.setdefault(memory_id, set()).add(entry["id"])
    return owners


def eligible(memories, partners):
    """全库中除 gold 冲突对象外的旧记忆，用作事项共享率对照池。"""
    return [memory["memory_id"] for memory in memories if memory["memory_id"] not in partners]


def converge_report(manifest, dataset, settings, *, limit=None, seed=20261027):
    catalog = manifest["catalog"]
    owners = entry_map(catalog)
    # 运行时用的是解析后的描述文本（选中项与"确认无差异"的复用项都会被替换成目录原文），
    # 只数 selected_ids 会把这类复用漏成空选择，因此按文本反查条目。
    by_text = {}
    for entry in catalog:
        by_text.setdefault(entry["description"], set()).add(entry["id"])
    memories = read_jsonl(Path(dataset) / "memories.jsonl")
    queries = read_jsonl(Path(dataset) / "queries.jsonl")
    gold = {row["query_id"]: row for row in read_jsonl(Path(dataset) / "gold.jsonl")}
    random.Random(seed).shuffle(queries)
    if limit:
        queries = queries[:limit]

    client = APIClient(settings)
    rows, totals = [], {"pairs": 0, "shared": 0, "control_pairs": 0, "control_shared": 0,
                        "empty_selection": 0, "partner_without_entry": 0}
    by_channel = {}
    for position, query in enumerate(queries, 1):
        label = gold.get(query["query_id"])
        partners = set(label["conflict_old_memory_ids"]) if label else set()
        result = describe(client, build_retrieval_text(query["content"], query.get("source_ref")),
                          catalog)
        selected = set(result["selected_ids"])
        texts = result.get("descriptions") or []
        reached = set(selected)
        for text in texts:
            reached |= by_text.get(text, set())
        scope = query["source_ref"].get("scope")
        channel = by_channel.setdefault(scope, {"pairs": 0, "shared": 0,
                                                "control_pairs": 0, "control_shared": 0})
        shared_partners = [old for old in partners if owners.get(old, set()) & reached]
        missing_entry = [old for old in partners if not owners.get(old)]
        pool = eligible(memories, partners)
        control = [old for old in pool if owners.get(old, set()) & reached]

        totals["pairs"] += len(partners)
        totals["shared"] += len(shared_partners)
        totals["control_pairs"] += len(pool)
        totals["control_shared"] += len(control)
        totals["empty_selection"] += int(not texts)
        totals["partner_without_entry"] += len(missing_entry)
        channel["pairs"] += len(partners)
        channel["shared"] += len(shared_partners)
        channel["control_pairs"] += len(pool)
        channel["control_shared"] += len(control)
        rows.append({
            "query_id": query["query_id"], "scope": scope, "selected": len(selected),
            "descriptions": len(texts), "reached": len(reached),
            "partners": len(partners), "shared": len(shared_partners),
            "missed_partners": [old for old in partners if not owners.get(old, set()) & reached],
            "partner_entries": [len(owners.get(old, ())) for old in partners],
            "missing_entry": len(missing_entry),
            "control_pairs": len(pool), "control_shared": len(control),
            "hit": bool(shared_partners),
        })
        print(f"[converge] {position}/{len(queries)} {query['query_id']} "
              f"选中 {len(selected)} 条，命中 {len(shared_partners)}/{len(partners)}", flush=True)

    summary = {
        "conflict_pairs": totals["pairs"],
        "conflict_shared": totals["shared"],
        "conflict_rate": round(totals["shared"] / totals["pairs"], 4) if totals["pairs"] else None,
        "control_pairs": totals["control_pairs"],
        "control_shared": totals["control_shared"],
        "control_rate": (round(totals["control_shared"] / totals["control_pairs"], 4)
                         if totals["control_pairs"] else None),
        "queries_with_empty_selection": totals["empty_selection"],
        "partners_without_entry": totals["partner_without_entry"],
        "by_channel": by_channel,
        "api": {"calls": len(client.events),
                "cache_hits": sum(event["cached"] is True for event in client.events),
                "errors": sum(event["status"] != "ok" for event in client.events)},
    }
    if summary["conflict_rate"] is not None and summary["control_rate"] is not None:
        if summary["control_rate"] == 0:
            summary["lift"] = "inf" if summary["conflict_rate"] else None
        else:
            summary["lift"] = round(summary["conflict_rate"] / summary["control_rate"], 3)
    return {"summary": summary, "rows": rows}


LEAK_PROMPT = """你在核查一份记忆索引的描述质量。memories 是挂在该条目上的原始记忆，
description 是索引为它们建立的"事项描述"。

问题只有一个：description 本身是否已经包含了这些记忆给出的具体取值（答案）？

判定口径：
- 只描述"问的是什么"（对象、待比较事项、识别实例所需的限定）→ 未泄漏。
- 写出了答案本身（具体地址、数字、日期、人名、结论、状态取值等）→ 泄漏。
- 日期、人名是否算答案要看含义："项目A第一期"这类限定不算答案；
  "截止日期是 3 月 5 日"里的 3 月 5 日算答案。

answer_in_description 必须是从 description 中**逐字复制**的连续片段，程序会校验它确实出现在
description 里；校验不过就说明这条判定不可核查。若描述里找不到构成答案的片段，就是未泄漏：
leak 填 false、该字段填空字符串。不要凭印象写理由，只引用你实际看到的文字。

只输出 JSON：{"leak": true/false, "answer_in_description": "逐字片段或空字符串", "reason": "一句话"}"""


def _verbatim(fragment, description):
    """引用必须是描述里的连续片段；返回 None 表示这条判定不可核查。"""
    fragment = " ".join(str(fragment).split())
    if not fragment:
        return ""
    return fragment if fragment in " ".join(description.split()) else None


def leak_report(manifest, settings, *, sample=120, seed=20261027):
    catalog = manifest["catalog"]
    by_id = {memory["memory_id"]: memory["content"] for memory in manifest["memories"]}
    pickable = [entry for entry in catalog if entry.get("aliases") is not None]
    random.Random(seed).shuffle(pickable)
    picked = pickable[:sample]
    client = APIClient(settings)
    hits, unchecked, rows = 0, 0, []
    for position, entry in enumerate(picked, 1):
        contents = [by_id[memory_id] for memory_id in entry["memory_ids"] if memory_id in by_id]
        payload = {"description": entry["description"],
                   "memories": [{"content": text} for text in contents]}

        def validate(value, description=entry["description"]):
            if not isinstance(value, dict) or not isinstance(value.get("leak"), bool):
                raise ValueError("必须返回 leak 布尔值和 reason")
            if value["leak"] and _verbatim(value.get("answer_in_description", ""),
                                          description) in (None, ""):
                raise ValueError("answer_in_description 必须是从 description 逐字复制的连续片段；"
                                 "若描述里没有构成答案的片段，应判 leak=false")
            return value

        try:
            result = client.json_call("leak", LEAK_PROMPT, payload, validate)
        except APIError as exc:
            # 判定器拿不出可核查引用是合法出口，逐条记下继续，不中断整轮。
            unchecked += 1
            rows.append({"id": entry["id"], "description": entry["description"], "leak": None,
                         "answer": "", "verbatim": False, "reason": f"未给出可核查引用：{exc}",
                         "memories": contents[:1]})
            print(f"[leak] {position}/{len(picked)} {entry['id']} unverifiable", flush=True)
            continue
        answer = _verbatim(result.get("answer_in_description", ""), entry["description"])
        leak = bool(result.get("leak"))
        # 判为泄漏却拿不出可核查引用：不计入泄漏，单列出来事后核对。
        if leak and answer in (None, ""):
            leak, unchecked = False, unchecked + 1
        hits += int(leak)
        rows.append({"id": entry["id"], "description": entry["description"], "leak": leak,
                     "answer": answer,
                     "verbatim": answer is not None,
                     "reason": result.get("reason", ""),
                     "memories": contents[:1]})
        print(f"[leak] {position}/{len(picked)} {entry['id']} leak={leak}", flush=True)
    judged = len(picked) - unchecked
    return {"summary": {"sampled": len(picked), "judged": judged, "leaked": hits,
                        "unverifiable": unchecked,
                        "leak_rate": round(hits / judged, 4) if judged else None,
                        "api": {"calls": len(client.events),
                                "errors": sum(e["status"] != "ok" for e in client.events)}},
            "rows": rows}


def value_report(manifest):
    """事项—取值分离是否落到字面上：新增项的 value 有没有出现在 description 里。

    只是线索不是判决：必要的实例限定也可能由数字构成（"第5章"）。
    但它能给出一个免模型、可重复的上界口径。
    """
    items = [item for item in manifest.get("values", {}).get("items", [])
             if item.get("value")]
    exact, digits, examples = 0, 0, []
    for item in items:
        value = " ".join(str(item["value"]).split())
        description = " ".join(str(item["description"]).split())
        if value and value in description:
            exact += 1
            if len(examples) < 8:
                examples.append({"value": value, "description": description, "mode": "整段值"})
            continue
        shared = {token for token in DIGITS.findall(value)} & {token for token in DIGITS.findall(description)}
        if shared:
            digits += 1
            if len(examples) < 8:
                examples.append({"value": value, "description": description,
                                 "mode": "共有数字 " + "/".join(sorted(shared))})
    return {"items_with_value": len(items),
            "value_in_description": exact,
            "shared_number": digits,
            "either": exact + digits,
            "either_rate": round((exact + digits) / len(items), 4) if items else None,
            "examples": examples}


def render(path, sections):
    lines = ["# 描述事项生成诊断", ""]
    for title, body in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(body, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--mode", default="static",
                        choices=["static", "converge", "leak", "values"])
    parser.add_argument("--sample", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--output", default=None, help="把完整结果写成 JSON")
    parser.add_argument("--env", default=None)
    args = parser.parse_args()

    manifest, vectors = load_index(args.index)
    sections = []
    detail = {}
    try:
        if args.mode == "static":
            detail = static_report(manifest, vectors)
            sections.append(("目录形态", detail))
        elif args.mode == "values":
            detail = value_report(manifest)
            sections.append(("事项—取值分离", detail))
        elif args.mode == "converge":
            settings = Settings.from_env(args.env)
            settings.require(llm=True, embedding=False)
            detail = converge_report(manifest, args.dataset, settings, limit=args.limit)
            sections.append(("同事项收敛", detail["summary"]))
        elif args.mode == "leak":
            settings = Settings.from_env(args.env)
            settings.require(llm=True, embedding=False)
            detail = leak_report(manifest, settings, sample=args.sample)
            sections.append(("值泄漏", detail["summary"]))
    except (APIError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps({key: value for key, value in detail.items() if key != "rows"},
                     ensure_ascii=False, indent=2))
    if args.report:
        render(args.report, sections)
        print(f"报告：{args.report}")
    if args.output:
        write_json(Path(args.output), detail)
        print(f"明细：{args.output}")


if __name__ == "__main__":
    main()
