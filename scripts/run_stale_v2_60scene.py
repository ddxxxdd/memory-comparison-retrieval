"""Repeat the STALE 60-scene comparison on v2 without resampling or tuning."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import shutil

from run_stale_unseen_60scene import (BASELINES, DEFAULT_OUTPUT as V1_RUN, NAMES, OUR, ROOT,
    atomic_json, evaluate, main as run_comparison, read, verify_protocol, write_report)
from memory_retrieval.bipolar_graph import ANCHORED_FOUR_DIM_PROMPT
from memory_retrieval.config import Settings
from memory_retrieval.data import read_jsonl, sha256_file, write_jsonl
from memory_retrieval.dataset import load_dataset

V2_DATA = ROOT / "data/stale_scenarios_v2/augmented"
V2_RUN = ROOT / "runs/experiments/stale-v2-60scene-bipolar-seed20261122"
V2_REPORT = ROOT / "reports/stale_v2_60scene_bipolar_seed20261122_experiment.md"
INPUT_NAMES = ("queries.jsonl", "memories.jsonl", "gold.jsonl")


def prepare(output):
    protocol = verify_protocol(V1_RUN)
    assert protocol["methods"] == BASELINES + OUR
    settings = Settings.from_env()
    settings.require(llm=True)
    assert settings.llm_identity() == protocol["llm"], "LLM settings differ from v1"
    manifest = read(V2_DATA / "manifest.json")
    assert manifest["extraction_version"] == "v2" and manifest["history_policy"] == "rule_only_background"
    assert all(manifest["inherited_files_sha256"][name] == protocol["source_inputs_sha256"][name] for name in INPUT_NAMES)
    selection = read(V1_RUN / "selection.json")
    assert selection["seed"] == 20261122
    scenes = selection["scenarios"]
    scene_ids = [record["scenario_id"] for record in scenes]
    assert len(set(scene_ids)) == len(scene_ids) == 60
    v1 = load_dataset(V1_RUN / "inputs")
    v2 = load_dataset(V2_DATA, scenario_ids=scene_ids)
    assert v2.queries == v1.queries
    assert [query["query_id"] for query in v2.queries] == [record["query_id"] for record in scenes]
    assert Counter(query["source_ref"]["type"] for query in v2.queries) == {"T1": 30, "T2": 30}
    old_memories = {memory.memory_id: memory for memory in v1.memories}
    new_memories = {memory.memory_id: memory for memory in v2.memories}
    assert all(new_memories.get(mid) == memory for mid, memory in old_memories.items())
    added = [memory for mid, memory in new_memories.items() if mid not in old_memories]
    assert all(memory.source_ref["dataset"] == "STALE" and memory.source_ref["record_type"] == "history"
               for memory in added)

    removed = {}
    counts = {}
    for record, old, new in zip(scenes, v1.gold, v2.gold, strict=True):
        assert old["query_id"] == new["query_id"] == record["query_id"]
        retained = {mid: detail for mid, detail in old["target_details"].items() if detail["source"] != "history_review"}
        native = [mid for mid, detail in retained.items() if detail["source"] == "stale_native"]
        assert native == [record["native_memory_id"]]
        assert new["target_details"] == retained and set(new["conflict_old_memory_ids"]) == set(retained)
        assert new["unjudged_old_memory_ids"] == old["unjudged_old_memory_ids"]
        removed.update({mid: record["scenario_id"] for mid in old["target_details"] if mid not in retained})
    for group in ("T1", "T2", "all"):
        selected = [record for record in scenes if group == "all" or record["type"] == group]
        ids = {record["scenario_id"] for record in selected}
        query_ids = {record["query_id"] for record in selected}
        counts[group] = {"scenes": len(ids), "v1_memories": sum(m.scenario_id in ids for m in v1.memories),
                         "v2_memories": sum(m.scenario_id in ids for m in v2.memories),
                         "added_history": sum(m.scenario_id in ids for m in added),
                         "removed_history_targets": sum(sid in ids for sid in removed.values()),
                         "v2_targets_by_source": dict(Counter(detail["source"] for row in v2.gold
                            if row["query_id"] in query_ids for detail in row["target_details"].values()))}

    completion = read(V1_RUN / "completion.json")
    assert completion["status"] == "complete"
    previous_hashes = completion["artifacts_sha256"]
    previous_files = ["selection.json", "exclusions.json", "protocol.json", "comparison.json", "anchored_probes.json",
                      *(f"inputs/{name}" for name in INPUT_NAMES),
                      *(f"arms/{method}.jsonl" for method in BASELINES + OUR)]
    for name in previous_files:
        assert sha256_file(V1_RUN / name) == previous_hashes[name], name
    protected = {f"memory_retrieval/{name}": digest for name, digest in protocol["code_sha256"].items()}
    protected["memory_retrieval/model_lock.json"] = protocol["model_lock_sha256"]
    protected.update({f"{protocol['source_dataset']}/{name}": digest for name, digest in protocol["source_inputs_sha256"].items()})
    protected.update({(V1_RUN / name).relative_to(ROOT).as_posix(): previous_hashes[name] for name in previous_files})
    for path in [*(V2_DATA / name for name in (*INPUT_NAMES, "manifest.json")),
                 Path(__file__).resolve(), ROOT / "scripts/run_stale_unseen_60scene.py"]:
        protected[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for name, digest in protected.items():
        assert sha256_file(ROOT / name) == digest, name

    output.mkdir(parents=True)
    for name in ("selection.json", "exclusions.json"):
        shutil.copyfile(V1_RUN / name, output / name)
    shutil.copyfile(Path(__file__), output / "v2_driver.py")
    (output / "prompt.txt").write_bytes(ANCHORED_FOUR_DIM_PROMPT.encode("utf-8"))
    for name, rows in zip(INPUT_NAMES, (v2.queries, [m.to_dict() for m in v2.memories], v2.gold), strict=True):
        write_jsonl(output / "inputs" / name, rows)
    mapping = {"source_run": str(V1_RUN), "source_dataset": str(V2_DATA), "scene_ids_in_order": scene_ids,
               "sample_seed": selection["seed"], "new_sampling": False, "queries_and_native_unchanged": True,
               "all_v1_memories_inherited": True, "counts": counts, "removed_history_target_scenes": removed}
    atomic_json(output / "v2_alignment.json", mapping)
    atomic_json(output / "source_manifest.json", protected)
    atomic_json(output / "protocol.json", {**protocol, "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": V2_DATA.relative_to(ROOT).as_posix(), "source_run": str(V1_RUN), "dataset_version": "v2",
        "source_inputs_sha256": {name: sha256_file(V2_DATA / name) for name in INPUT_NAMES},
        "input_sha256": {name: sha256_file(output / "inputs" / name) for name in INPUT_NAMES},
        "prior_verification": "Same selected scenario IDs and queries; v1 official and synthetic gold unchanged",
        "v2_alignment_sha256": sha256_file(output / "v2_alignment.json")})
    verify_protocol(output)


def report_v2(output, comparison):
    alignment = read(output / "v2_alignment.json")
    assert sha256_file(output / "v2_alignment.json") == comparison["protocol"]["v2_alignment_sha256"]
    v1 = load_dataset(V1_RUN / "inputs")
    v2 = load_dataset(output / "inputs")
    previous = {arm: {row["query_id"]: row for row in read_jsonl(V1_RUN / "arms" / f"{arm}.jsonl")}
                for arm in BASELINES + OUR}
    common, _, verification = evaluate(replace(v1, gold=v2.gold), previous)
    groups = comparison["groups"]
    v1_probes = read(V1_RUN / "anchored_probes.json")
    v2_probes = read(output / "anchored_probes.json")
    assert [r["query_id"] for r in v1_probes] == [r["query_id"] for r in v2_probes]
    identical = sum(a["violation_probes"] == b["violation_probes"] for a, b in zip(v1_probes, v2_probes))
    atomic_json(output / "v1_rescored_on_v2_gold.json", {"groups": common, "verification": verification,
                "exact_probe_lists_repeated": identical, "new_model_calls_for_v1_rescoring": 0})

    group_counts = alignment["counts"]
    intro = ["# STALE v2 同场景 60 场景对照实验", "",
             "完全复用 v1 的 60 个场景 ID、顺序与查询（seed=20261122，30 T1 + 30 T2）。无重抽样；这是同场景跨版本复测，不是新独立 Test。各方法只检索自己的场景旧库。", "",
             "| 分组 | v1 旧记忆 | v2 旧记忆 | 新增历史 | 去掉的历史标签 | v2 官方/合成目标 |",
             "|---|---:|---:|---:|---:|---:|"]
    for group in ("T1", "T2", "all"):
        row = group_counts[group]
        sources = row["v2_targets_by_source"]
        intro.append(f"| {group} | {row['v1_memories']} | {row['v2_memories']} | {row['added_history']} | "
                     f"{row['removed_history_targets']} | {sources['stale_native']}/{sources['synthetic_review']} |")
    intro += ["", "v2 原样继承 v1 旧记忆和合成项，仅追加规则抽取的背景历史；被去掉历史标签的旧记忆仍可被检索。合成目标只有模型复核，不是逐条人工标注。", ""]

    total = groups["all"]["methods"]
    full = total[OUR[-1]]
    def hits(row):
        return row["at_k"]["10"]["hits"]
    strongest_r = max(BASELINES, key=lambda arm: hits(total[arm]))
    strongest_n = max(BASELINES, key=lambda arm: total[arm]["ndcg_at_10"])
    appendix = ["## v1 / v2 同金标口径复测", "",
                "下表用 v2 保留的 333 个目标重计 v1 已保存排名，不重跑 v1 方法。v2 按要求用全新缓存重新生成探针；"
                f"60 个查询中有 {identical} 个四探针列表与 v1 完全相同。探针方法的跨版本差值不能完全归因于增加背景历史。", "",
                "| 方法 | Top-10 命中 v1 → v2（/333） | NDCG@10 v1 → v2 | 官方 M_old@10 v1 → v2（/60） |",
                "|---|---:|---:|---:|"]
    for arm in BASELINES + OUR:
        before, after = common["all"]["methods"][arm], total[arm]
        native = [r["at_k"]["10"]["recall_by_source"]["stale_native"]["hits"] for r in (before, after)]
        appendix.append(f"| {NAMES[arm]} | {hits(before)} → {hits(after)} | "
                        f"{before['ndcg_at_10']:.4f} → {after['ndcg_at_10']:.4f} | {native[0]} → {native[1]} |")
    appendix += ["", "## 结果分析", "",
                 f"- v2 总体 Top-10：完整方法命中 {hits(full)} 条；六个基线中最高的是 {NAMES[strongest_r]}（{hits(total[strongest_r])} 条）。"
                 f"完整方法 NDCG@10 为 {full['ndcg_at_10']:.4f}，最高基线 {NAMES[strongest_n]} 为 {total[strongest_n]['ndcg_at_10']:.4f}。",
                 f"- v2 模块比较：仅探针 {hits(total[OUR[0]])} → 仅正扩散 {hits(total[OUR[1]])} → 双极完整方法 {hits(full)} 条 Top-10 命中。"
                 "三者共用探针与 BGE 向量，不应把探针带来的提升都归于图扩散。",
                 "- 所有指标都按已标注目标计分。未标注历史未被确认相容；Top-10 已标注命中占比不能当作真实候选纯度。", ""]
    return write_report(output, comparison, introduction=intro, appendix=appendix,
                        report_path=V2_REPORT, show_ablations=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=V2_RUN)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.resume:
        prepare(output)
    if not args.prepare_only:
        run_comparison(["--output", str(output), *(["--resume"] if args.resume else [])], report_writer=report_v2)


if __name__ == "__main__":
    main()
