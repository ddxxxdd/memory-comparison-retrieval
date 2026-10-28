from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import quality
from .api import usage_summary
from .config import MEMORY_TEMPLATE_THRESHOLD
from .data import (RETRIEVAL_TEXT_VERSION, Memory, fingerprint, validate_memories,
                   write_json)
from .prompts import INDEX_PROMPT_HASH, PROMPT_VERSION, describe, describe_batch


DESCRIPTION_MODES = ("independent", "reuse", "batch_reuse")
FORMAT_VERSION = 3


@dataclass
class MemoryIndex:
    memories: list[Memory]
    catalog: list[dict]
    description_vectors: np.ndarray
    content_vectors: np.ndarray
    manifest: dict

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        # 版本 1 是没有"校验+归并"的旧目录；仍可加载，只用于复现旧实验。
        if meta.get("format_version") not in {1, 2, FORMAT_VERSION}:
            raise ValueError("不支持的索引版本")
        saved_id = meta.pop("index_id")
        if fingerprint(meta) != saved_id:
            raise ValueError("索引元数据已改变，请重新建库")
        meta["index_id"] = saved_id
        vector_path = directory / "vectors.npz"
        if hashlib.sha256(vector_path.read_bytes()).hexdigest() != meta["vectors_sha256"]:
            raise ValueError("索引向量文件已改变，请重新建库")
        memories = [Memory.from_dict(row) for row in meta["memories"]]
        validate_memories(memories)
        catalog = meta["catalog"]
        ids = {m.memory_id for m in memories}
        if not catalog or len({c["id"] for c in catalog}) != len(catalog):
            raise ValueError("描述目录为空或编号重复")
        linked = set()
        for entry in catalog:
            if not entry["memory_ids"] or set(entry["memory_ids"]) - ids:
                raise ValueError("描述目录的旧记忆引用无效")
            linked.update(entry["memory_ids"])
        if linked != ids:
            raise ValueError("存在没有描述入口的旧记忆")
        with np.load(vector_path, allow_pickle=False) as vectors:
            dv, cv = vectors["descriptions"], vectors["contents"]
        # 归并后的条目用一个 aliases 列表指向它保留的全部描述向量行；
        # 这些行必须恰好被划分完，否则打分时会有向量没人用。
        aliased = [entry["aliases"] for entry in catalog if "aliases" in entry]
        if aliased:
            if len(aliased) != len(catalog):
                raise ValueError("描述目录里 aliases 必须全部存在")
            flat = [row for rows_ in aliased for row in rows_]
            if sorted(flat) != list(range(len(dv))):
                raise ValueError("aliases 未覆盖全部描述向量行，或存在重复引用")
        if dv.ndim != 2 or cv.ndim != 2 or cv.shape[1] != dv.shape[1] or cv.shape[0] != len(memories):
            raise ValueError("向量数量/维度与索引元数据不匹配")
        for array in (dv, cv):
            if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5):
                raise ValueError("索引向量必须归一化且有限")
            array.setflags(write=False)
        return cls(memories, catalog, dv, cv, meta)

    def check_embedding(self, settings):
        if self.manifest["embedding"] != settings.embedding_identity():
            raise ValueError("当前向量服务/模型与离线索引不同，请使用原配置或重新建库")

    @cached_property
    def entry_of_row(self):
        """描述向量行 -> 目录条目下标。没有 aliases 的旧索引是一一对应。"""
        rows = np.full(len(self.description_vectors), -1, dtype=np.int64)
        for position, entry in enumerate(self.catalog):
            for row in entry.get("aliases", [position]):
                rows[row] = position
        return rows


def _attach_entries(catalog, by_text, texts, memory_id):
    """把一条记忆挂到它引用的事项上；新文本按出现顺序编号入目录。"""
    for text in texts:
        if text not in by_text:
            entry = {"id": f"D{len(catalog) + 1:06d}", "description": text, "memory_ids": []}
            catalog.append(entry)
            by_text[text] = entry
        by_text[text]["memory_ids"].append(memory_id)


def value_audit(extraction):
    """把每条记忆抽取到的 value 与 description 一起留档。

    索引里保留两者，事后可以核对"描述是否仍然含着取值"，不必重跑模型；
    只有 description 进入目录与描述向量，value 不参与检索，也不参与目录去重。
    """
    seen, items = 0, []
    for record in extraction:
        seen += len(record.get("selected_ids") or [])
        for item in record.get("new_items") or []:
            items.append({"memory_id": record["memory_id"],
                          "value": item.get("value"),
                          "description": item.get("description")})
    return {"selected": seen, "new": len(items), "items": items}


def _fallback_descriptions(client, memory, catalog, *, repair=True):
    """批量模式偶尔对某条记忆返回空。退回单条抽取，保证这条记忆仍有检索入口。

    只用当前真实目录重试一次，不再退回空目录抽取：目录非空时把上下文清空，
    模型就无法复用已有事项，等于凭空造一份新措辞。
    仍为空则停止建库。质量重写由调用方的 repair 开关决定，兜底不能自己把它打开，
    否则 --no-repair 的消融仍会多出一次重写调用并改变目录。兜底自身的违规随结果返回。
    """
    result = describe(client, memory.retrieval_text, catalog, check=repair)
    if not result["descriptions"]:
        raise ValueError(f"旧记录 {memory.memory_id} 未提取到事项，已停止建库；请核查原内容")
    return {"selected_ids": result["selected_ids"],
            "new_items": result["new_items"],
            "descriptions": result["descriptions"],
            "violations": result["violations"]}


def _extract_batched(memories, client, batch_size, repair_rounds=1):
    """批式复用：每批一次请求，批间引用已有条目，批内同一事项收敛到同一份文本。

    批次之间必须按顺序推进（后一批要用前一批结束时的目录）；
    批内结果按记忆原始顺序合并，保证条目编号与串行结果一样可复现。
    每批先编码本条批记忆的内容，用它校验"内容近重复的记忆是否共用了描述"；
    校验阈值是固定的先验（MEMORY_TEMPLATE_THRESHOLD），与是否启用目录归并无关。
    """
    catalog, by_text, extraction = [], {}, []
    fallbacks, audit_log = [], []
    initial_rules, remaining_rules, fallback_rules = {}, {}, {}
    content_vectors = []
    for start in range(0, len(memories), batch_size):
        batch = memories[start:start + batch_size]
        contents = [memory.retrieval_text for memory in batch]
        batch_vectors = client.embed(contents, "batch_content_embedding")
        content_vectors.append(batch_vectors)
        result = describe_batch(client, contents, catalog,
                                content_vectors=batch_vectors,
                                template_threshold=MEMORY_TEMPLATE_THRESHOLD,
                                repair_rounds=repair_rounds)
        for violation in result["initial_violations"]:
            initial_rules[violation["rule"]] = initial_rules.get(violation["rule"], 0) + 1
        by_index = {row["index"]: row for row in result["results"]}
        working, batch_fallback = [], False
        for local, memory in enumerate(batch):
            row = by_index[local]
            if row["descriptions"]:
                local_violations = [v for v in result["violations"] if v["index"] == local]
            else:
                fallback = _fallback_descriptions(client, memory, catalog,
                                                  repair=bool(repair_rounds))
                row = {"index": local, **fallback}
                fallbacks.append(memory.memory_id)
                batch_fallback = True
                # 兜底把这一条的结果整个换掉了，它的违规也要跟着换，否则索引里
                # 记的是被替换前那一批的违规；source 标明来自单条兜底。
                local_violations = [{**v, "index": local, "source": "fallback"}
                                    for v in fallback["violations"]]
                for violation in local_violations:
                    fallback_rules[violation["rule"]] = fallback_rules.get(violation["rule"], 0) + 1
            working.append({"index": local, "memory_id": memory.memory_id, "row": row,
                            "violations": local_violations})
            # 立即挂入目录：同一批后面的兜底要能看到这一条刚生成的描述，否则同一事项
            # 会被两条记忆各写一份措辞；跨行重审只依赖 descriptions，不受挂接影响。
            _attach_entries(catalog, by_text, row["descriptions"], memory.memory_id)
        if batch_fallback:
            # 兜底替换了整行结果，跨行约束必须按最终行集重算：既不能留下已失效的成对
            # 违规，也不能丢掉仍然成立的成对违规。
            cross_row = [violation for violation in quality.audit(
                             contents,
                             [{"index": item["index"], "descriptions": item["row"]["descriptions"]}
                              for item in working],
                             content_vectors=batch_vectors,
                             template_threshold=MEMORY_TEMPLATE_THRESHOLD)
                         if violation["rule"] == "same_event_split"]
            for item in working:
                item["violations"] = [v for v in item["violations"] if v["rule"] != "same_event_split"]
            by_local = {item["index"]: item for item in working}
            for violation in cross_row:
                target = by_local.get(violation["index"])
                if target is not None:
                    target["violations"].append(violation)
        remaining = 0
        for item in working:
            row = item["row"]
            for violation in item["violations"]:
                remaining_rules[violation["rule"]] = remaining_rules.get(violation["rule"], 0) + 1
            remaining += len(item["violations"])
            extraction.append({"memory_id": item["memory_id"], "selected_ids": row["selected_ids"],
                               "new_items": row["new_items"],
                               "descriptions": row["descriptions"],
                               "violations": item["violations"]})
        if result["initial_violations"]:
            audit_log.append({"batch": start // batch_size, "memories": len(batch),
                              "initial": len(result["initial_violations"]),
                              "remaining": remaining,
                              "repaired": bool(result["repairs"]),
                              "rules": sorted({v["rule"] for v in result["initial_violations"]})})
        print(f"[batch] {min(start + batch_size, len(memories))}/{len(memories)} 条记忆，"
              f"目录 {len(catalog)} 个条目，重写 {len(audit_log)} 批，兜底 {len(fallbacks)} 条", flush=True)
    if fallbacks:
        print(f"[batch] 退回单条抽取的记忆: {len(fallbacks)} 条（均保证有检索入口）", flush=True)
    audit = {"audit_threshold": MEMORY_TEMPLATE_THRESHOLD, "repair_rounds": repair_rounds,
             "repaired_batches": sum(1 for row in audit_log if row["repaired"]),
             "flagged_batches": len(audit_log),
             "initial_by_rule": dict(sorted(initial_rules.items())),
             "remaining_by_rule": dict(sorted(remaining_rules.items())),
             "fallback_by_rule": dict(sorted(fallback_rules.items())),
             "batches": audit_log}
    return catalog, by_text, extraction, fallbacks, audit, np.concatenate(content_vectors, axis=0)


def build_index(memories, client, directory, mode="independent", batch_size=None, *,
                description_threshold=None, template_threshold=None, repair_rounds=1):
    """建库。

    M1（抽取校验与重写）由 repair_rounds 控制，默认开启：只修可判定的违规，
    不改变模型原本正确的输出。
    M2（目录归并）默认关闭：实测在同一份抽取结果上只加归并，37 条写入的候选召回从
    0.980 掉到 0.961（两次基线运行都稳定在 0.980），因为归并后模型只看到一份条目文本，
    失去了选中具体措辞的入口。传 description_threshold / template_threshold 可开启。
    """
    validate_memories(memories)
    if mode not in DESCRIPTION_MODES:
        raise ValueError("description-mode 必须是 " + "、".join(DESCRIPTION_MODES))
    if mode == "batch_reuse" and (not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("batch_reuse 必须指定正数 batch_size")
    directory = Path(directory)
    if directory.exists():
        raise ValueError("索引目录已存在；为保持旧库冻结，请使用新的输出目录")
    client.settings.require(llm=True, embedding=True)
    start = time.perf_counter()
    event_start = len(client.events)
    batch_fallbacks, quality_report, batch_content_matrix = [], {}, None
    if mode == "batch_reuse":
        catalog, by_text, extraction, batch_fallbacks, quality_report, batch_content_matrix = _extract_batched(
            memories, client, batch_size, repair_rounds)
    else:
        catalog, by_text, extraction = [], {}, []
        for memory in memories:
            result = describe(client, memory.retrieval_text, catalog if mode == "reuse" else [],
                              check=bool(repair_rounds))
            if not result["descriptions"]:
                raise ValueError(f"旧记录 {memory.memory_id} 未提取到事项，已停止建库；请核查原内容")
            extraction.append({"memory_id": memory.memory_id, **result})
            _attach_entries(catalog, by_text, result["descriptions"], memory.memory_id)
    if batch_content_matrix is None:
        vectors = client.embed([c["description"] for c in catalog] + [m.retrieval_text for m in memories],
                               "index_embedding")
        description_matrix, content_matrix = vectors[:len(catalog)], vectors[len(catalog):]
    else:
        description_matrix = client.embed([c["description"] for c in catalog], "index_embedding")
        content_matrix = batch_content_matrix
    positions = {memory.memory_id: i for i, memory in enumerate(memories)}
    catalog, consolidation = quality.merge_entries(
        catalog, description_matrix, content_matrix, positions,
        description_threshold=description_threshold, template_threshold=template_threshold)
    print(f"[consolidate] 描述条目 {consolidation['entries_before']} -> {consolidation['entries_after']}"
          f"（收掉 {consolidation['merged_count']} 个：措辞重复 "
          f"{sum(1 for link in consolidation['links'] if link['rule'] == 'description_duplicate')} 条，"
          f"取值拆分 {sum(1 for link in consolidation['links'] if link['rule'] == 'memory_template')} 条；"
          "被收掉的措辞保留为别名，仍参与打分）", flush=True)
    # 所有预处理完成后才创建目录，失败时不会产生看似可用的索引。
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "vectors.npz", descriptions=description_matrix, contents=content_matrix)
    events = client.events[event_start:]
    meta = {
        "format_version": FORMAT_VERSION, "description_mode": mode,
        "batch_size": batch_size if mode == "batch_reuse" else None,
        "batch_fallbacks": batch_fallbacks,
        "prompt_version": PROMPT_VERSION, "prompt_hash": INDEX_PROMPT_HASH,
        "retrieval_text_version": RETRIEVAL_TEXT_VERSION,
        "llm": client.settings.llm_identity(), "embedding": client.settings.embedding_identity(),
        "memories": [m.to_dict() for m in memories], "catalog": catalog,
        "extraction": extraction,
        "values": value_audit(extraction),
        "quality": quality_report,
        "consolidation": consolidation,
        "vectors_sha256": hashlib.sha256((directory / "vectors.npz").read_bytes()).hexdigest(),
        "build": {"seconds": time.perf_counter() - start, "usage": usage_summary(events), "events": events},
    }
    meta["index_id"] = fingerprint(meta)
    write_json(directory / "index.json", meta)
    return MemoryIndex.load(directory)
