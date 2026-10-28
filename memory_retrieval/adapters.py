"""适配器只读取内容和来源，不生成描述，不判断冲突。"""

import json
from pathlib import Path
from string import Formatter

from .data import Memory, nonempty_text, read_jsonl, validate_memories


def load_text(path):
    memories = [Memory.from_dict(row) for row in read_jsonl(path)]
    validate_memories(memories)
    return memories


def load_graph(path):
    """读取明确 schema 的属性图；模板必须使用所有限定属性，拒绝静默丢失。"""
    path = Path(path)
    graph = json.loads(path.read_text(encoding="utf-8-sig"))
    nodes, edges = {}, {}
    for node in graph["nodes"]:
        node_id = nonempty_text(node.get("id"), "node.id")
        if node_id in nodes:
            raise ValueError(f"重复图节点 ID: {node_id}")
        nodes[node_id] = nonempty_text(node.get("text"), "node.text")
    for edge in graph["edges"]:
        edge_id = nonempty_text(edge.get("id"), "edge.id")
        if edge_id in edges:
            raise ValueError(f"重复图边 ID: {edge_id}")
        edges[edge_id] = edge

    templates = graph["relation_templates"]
    memories = []
    for record in graph["memories"]:
        edge_ids = record["edge_ids"]
        if not isinstance(edge_ids, list) or not edge_ids or len(edge_ids) != len(set(edge_ids)):
            raise ValueError("图记忆的 edge_ids 必须为非空且无重复的列表")
        contents, node_ids = [], []
        for edge_id in edge_ids:
            edge = edges[edge_id]
            if set(edge) - {"id", "subject", "relation", "object", "qualifiers"}:
                raise ValueError(f"{edge_id}: 额外语义属性应放在 qualifiers，并在模板中表达")
            qualifiers = edge.get("qualifiers", {})
            if not isinstance(qualifiers, dict) or {"subject", "object"} & set(qualifiers):
                raise ValueError(f"{edge_id}: qualifiers 无效")
            for key, val in qualifiers.items():
                if not isinstance(val, str):
                    raise ValueError(f"{edge_id}: 限定属性 {key} 必须提供可读字符串")
            values = {"subject": nodes[edge["subject"]], "object": nodes[edge["object"]], **qualifiers}
            template = nonempty_text(templates[edge["relation"]], "关系模板")
            fields = set()
            for _, field, spec, conversion in Formatter().parse(template):
                if field is not None:
                    if not field.isidentifier() or spec or conversion:
                        raise ValueError("图模板只允许简单的 {字段名} 占位符")
                    fields.add(field)
            if fields != set(values):
                raise ValueError(f"{edge_id}: 模板必须完整表达 subject、object 和全部 qualifiers")
            contents.append(template.format(**values))
            node_ids.extend([edge["subject"], edge["object"]])
        memories.append(Memory.from_dict({
            "memory_id": record["memory_id"],
            "content": "".join(contents),
            "source_ref": {
                "path": str(path.resolve()), "edge_ids": edge_ids,
                "node_ids": list(dict.fromkeys(node_ids)),
            },
        }))
    validate_memories(memories)
    return memories
