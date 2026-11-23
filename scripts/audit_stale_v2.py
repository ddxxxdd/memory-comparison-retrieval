"""Audit STALE V2 against the raw source, frozen V1 files and scene exports."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_retrieval.data import read_jsonl, sha256_file, write_json
from scripts.augment_stale_scenarios import (
    background_metadata, check_published, dataset_hashes, inheritance_counts, load_scene_data, prepare_inheritance,
)
from scripts.convert_stale_scenarios import (
    DATA_ROOT, HISTORY_POLICY, V2_ROOT, convert_scenario, history_claims, iter_json_array, memory_row, text_key,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_v1_hashes(v1, snapshot):
    before = json.loads(Path(snapshot).read_text(encoding="utf-8-sig"))
    require(Path(before["v1_root"]).resolve() == v1.resolve(), "V1 snapshot belongs to another directory")
    actual = {p.relative_to(v1).as_posix(): sha256_file(p)
              for p in sorted(v1.rglob("*")) if p.is_file()}
    expected = before["v1_files_sha256"]
    changed = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
    require(not changed, f"V1 files changed: {changed[:10]} ({len(changed)} total)")
    return len(actual)


def verify_exports(dataset, queries, banks, gold):
    directory = dataset / "scenarios"
    index = json.loads((directory / "index.json").read_text(encoding="utf-8-sig"))
    expected_index, expected_files = [], {"index.json"}
    for query in queries:
        sid, qid = query["scenario_id"], query["query_id"]
        kind = query["source_ref"]["type"]
        relative = f"{kind}/{sid}"
        expected_index.append({"scenario_id": sid, "scenario_type": kind, "query_id": qid,
                               "M_new": query["content"], "memory_count": len(banks[sid]), "path": relative})
        for name, rows in (("memories", banks[sid]), ("queries", [query]), ("gold", [gold[qid]])):
            filename = f"{relative}/{name}.jsonl"
            require(read_jsonl(directory / filename) == rows,
                    f"{dataset.name}/{filename}: scene file differs from aggregate")
            expected_files.add(filename)
    require(index == expected_index, f"{dataset}: scene index differs from aggregate")
    actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    require(actual_files == expected_files, f"{dataset}: missing or extra scene files")
    return len(index)


def verify_preserved_work(snapshot):
    before = json.loads(Path(snapshot).read_text(encoding="utf-8-sig"))
    root = Path(before["work_root"]).resolve()
    for name, digest in before["files_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), "Preserved-work snapshot contains a path outside its root")
        require(path.is_file() and sha256_file(path) == digest, f"Preserved cache or checkpoint changed: {name}")
    return len(before["files_sha256"])


def audit(v1, v2, source, *, snapshot=None, final=False, preserved_work_snapshot=None):
    v1, v2, source = Path(v1).resolve(), Path(v2).resolve(), Path(source).resolve()
    hashes_verified = verify_v1_hashes(v1, snapshot) if snapshot else None
    queries, old_base, old_base_gold = load_scene_data(v1 / "base")
    prior_queries, old_augmented, old_augmented_gold = load_scene_data(v1 / "augmented")
    base_queries, base, base_gold = load_scene_data(v2 / "base")
    require(queries == prior_queries == base_queries, "V1/V2 queries differ")
    require(old_base_gold == base_gold, "V2 base gold differs from V1")
    inherited = prepare_inheritance(v1 / "augmented", queries, base, base_gold)
    manifest = json.loads((v1 / "augmented" / "manifest.json").read_text(encoding="utf-8-sig"))
    raw_sha = sha256_file(source)
    for directory in (v1 / "base", v2 / "base"):
        require(json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))["source_sha256"] == raw_sha,
                f"{directory}: raw source hash differs")

    by_sid = {q["scenario_id"]: q for q in queries}
    source_ids, checked_history = set(), 0
    for raw in iter_json_array(source):
        sid = raw["uid"]
        require(sid in by_sid and sid not in source_ids, f"Unexpected or repeated raw scenario: {sid}")
        source_ids.add(sid)
        expected_v1, raw_query, raw_gold = convert_scenario(raw)
        require(expected_v1 == old_base[sid] and raw_query == by_sid[sid]
                and raw_gold == old_base_gold[raw_query["query_id"]], f"{sid}: V1 extraction differs from raw source")
        keyed = {text_key(r["content"]): r for r in base[sid]}
        require(len(keyed) == len(base[sid]), f"{sid}: duplicate normalized V2 text")
        require(all(keyed.get(text_key(r["content"])) == r for r in old_base[sid]),
                f"{sid}: V1 IDs, content or provenance were changed during normalized-text alignment")
        cutoff = raw["relevant_session_index"][1]
        occurrences = {(si, ti, text) for si, session in enumerate(raw["haystack_session"][:cutoff])
                       for ti, text in history_claims(session, all_sentences=True)}
        anchor, new_key = text_key(raw["M_old"]), text_key(raw["M_new"])
        eligible = {text_key(text) for _, _, text in occurrences if text_key(text) != new_key
                    and anchor not in text_key(text) and text_key(text) not in anchor}
        history = [r for r in base[sid] if r["source_ref"]["record_type"] == "history"]
        require({text_key(r["content"]) for r in history} == eligible,
                f"{sid}: missing or extra eligible raw history sentences")
        for row in base[sid]:
            ref = row["source_ref"]
            require(row["scenario_id"] == sid and row["memory_id"] == memory_row(sid, row["content"], "history")["memory_id"],
                    f"{sid}: invalid scenario or normalized memory ID")
            require(type(ref["session_index"]) is int and 0 <= ref["session_index"] < cutoff,
                    f"{sid}: history crosses the write-time boundary")
            if ref["record_type"] == "history":
                require((ref["session_index"], ref["turn_index"], row["content"]) in occurrences,
                        f"{sid}: history provenance is not an eligible original user sentence")
                checked_history += 1
            else:
                require(ref["record_type"] == "M_old" and ref["session_index"] == raw["relevant_session_index"][0]
                        and row["content"] == raw["M_old"].strip(), f"{sid}: invalid official old memory")
    require(source_ids == set(by_sid), "Raw source does not cover all scenarios")
    counts = inheritance_counts(inherited.values())
    report = {"stage": "final" if final else "preflight", "scenarios": len(queries),
              "scenario_types": dict(Counter(q["source_ref"]["type"] for q in queries)),
              "v1_files_hash_verified": hashes_verified, "raw_source_sha256": raw_sha,
              "v2_base_memories": sum(map(len, base.values())), **counts,
              "expected_augmented_memories": counts["inherited_memories"] + counts["new_history"],
              "checked_original_history_provenance": checked_history,
              "v1_base_records_and_gold_inherited": True, "queries_inherited": True,
              "scene_isolation_verified": True, "write_time_boundary_verified": True,
              "all_eligible_history_included": True}
    if final:
        expected = background_metadata(inherited.values(), dataset_hashes(v2 / "base"),
                                       dataset_hashes(v1 / "augmented", manifest=True))
        check_published(v2 / "augmented", queries, inherited, expected)
        augmented_queries, augmented, gold = load_scene_data(v2 / "augmented")
        augmented_manifest = json.loads((v2 / "augmented" / "manifest.json").read_text(encoding="utf-8-sig"))
        for key, old_key in (("inherited_generation_model", "generation_model"),
                             ("inherited_synthetic_review_model", "review_model")):
            require(augmented_manifest.get(key) == manifest.get(old_key), "Inherited synthetic model provenance changed")
        per_scene, removed_ids = [], []
        for query in queries:
            qid = query["query_id"]
            added_ids = {r["memory_id"] for r in inherited[qid].added_history}
            require(not added_ids & (set(gold[qid]["target_details"]) | set(gold[qid].get("unjudged_old_memory_ids", []))),
                    f"{qid}: added history received a target or pending label")
            removed = [mid for mid, label in old_augmented_gold[qid]["target_details"].items()
                       if label["source"] == "history_review"]
            require(all(any(r["memory_id"] == mid for r in augmented[query["scenario_id"]]) for mid in removed),
                    f"{qid}: removing history gold also removed its memory")
            removed_ids.extend(removed)
            per_scene.append({"scenario_id": query["scenario_id"], "scenario_type": query["source_ref"]["type"],
                              "new_history": len(added_ids), "removed_history_target_ids": removed,
                              "gold_targets": len(gold[qid]["conflict_old_memory_ids"]),
                              "inherited_unjudged": len(gold[qid].get("unjudged_old_memory_ids", [])),
                              "added_history_targets": 0, "added_history_unjudged": 0})
        for key, value in {"memories": sum(map(len, augmented.values())), "gold_targets": counts["inherited_targets"],
                           "history_unjudged": counts["inherited_unjudged"]}.items():
            require(augmented_manifest[key] == value, f"Incorrect augmented manifest count: {key}")
        sources = Counter(label["source"] for row in gold.values() for label in row["target_details"].values())
        require(set(sources) <= {"stale_native", "synthetic_review"}, "History or unknown target source remains")
        require(dict(sources) == augmented_manifest["gold_target_sources"], "Incorrect gold source counts")
        by_type = {kind: {key: sum(row[key] for row in per_scene if row["scenario_type"] == kind)
                          for key in ("new_history", "gold_targets", "inherited_unjudged")}
                   for kind in report["scenario_types"]}
        report.update(v2_augmented_memories=sum(map(len, augmented.values())),
                      v1_augmented_memories_inherited=True, synthetic_records_and_labels_inherited=True,
                      gold_matches_v1_after_history_filter=True, legacy_pending_and_flags_preserved=True, new_synthetic=0,
                      history_policy=HISTORY_POLICY, model_calls=0, reviewed_new_history=0,
                      history_review_targets=0, added_history_targets=0, added_history_unjudged=0,
                      added_history_by_scenario_type=by_type, final_targets=sum(sources.values()),
                      final_unjudged=counts["inherited_unjudged"], gold_target_sources=dict(sources),
                      removed_history_target_ids=sorted(removed_ids), new_history_unlisted=counts["new_history"],
                      base_scenes_exported=verify_exports(v2 / "base", queries, base, base_gold),
                      augmented_scenes_exported=verify_exports(v2 / "augmented", augmented_queries, augmented, gold),
                      aggregates_match_scene_files=True, scenario_stats=per_scene)
    if preserved_work_snapshot:
        report["preserved_cache_and_checkpoint_files"] = verify_preserved_work(preserved_work_snapshot)
    if snapshot:
        verify_v1_hashes(v1, snapshot)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v1", type=Path, default=DATA_ROOT)
    parser.add_argument("--v2", type=Path, default=V2_ROOT)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--preserved-work-snapshot", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.final and not args.snapshot:
        parser.error("Final acceptance requires the pre-build V1 file-hash snapshot")
    if args.report and args.report.resolve().is_relative_to(args.v1.resolve()):
        parser.error("The audit report must be outside V1")
    report = audit(args.v1, args.v2, args.source, snapshot=args.snapshot,
                   final=args.final, preserved_work_snapshot=args.preserved_work_snapshot)
    if args.report:
        write_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("scenario_stats", "removed_history_target_ids")},
                     ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()
