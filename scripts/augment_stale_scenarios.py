"""Review synthetic memories for V1 or inherit V2 offline with rule-only history."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import re
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_retrieval.api import APIClient, APIError
from memory_retrieval.config import Settings
from memory_retrieval.data import Memory, fingerprint, load_queries, nonempty_text, read_jsonl, validate_memories, write_json, sha256_file, write_jsonl
from memory_retrieval.evaluation import validate_dataset, validate_gold
from scripts.convert_stale_scenarios import DATA_ROOT, HISTORY_POLICY, V2_ROOT, library_stats, memory_row, text_key
from scripts.stale_augmentation_prompts import GENERATION, REVIEW


NEGATION = re.compile(r"\b(?:no|not|never|neither|nor|none|nothing|nobody|cannot|without|"
                      r"nope|used to|instead of)\b|n['’]t\b", re.I)
ADMITS_COMPATIBILITY = re.compile(
    r"\b(?:could still|might still|may still|can still|is compatible with|is consistent with|"
    r"does not (?:contradict|conflict|establish|invalidate)|not (?:a conflict|contradicted)|"
    r"may (?:continue|remain|be valid)|if the (?:membership|subscription) (?:lapses|expired))\b", re.I)
STOP_WORDS = set("""a an and are as at be been being but by can could did do does for from get got
    had has have he her here him his how i if in into is it its just ll m me my of on or our out re
    s she so some t than that the their them then there these they this those through to up us ve
    was we were what when where which who will with would you your now since lately really still
    recently usually going only""".split())
GENERATION_GROUPS = ("compatible", "implicit")
TARGET_RELATIONS = {"conflict", "stale"}
REVIEW_RELATIONS = TARGET_RELATIONS | {"compatible", "uncertain", "unsuitable"}


def content_words(text):
    return set(text_key(text).split()) - STOP_WORDS


def lexical_overlap(old, new):
    left, right = content_words(old), content_words(new)
    shared = len(left & right)
    jaccard = shared / max(1, len(left | right))
    containment = shared / max(1, min(len(left), len(right)))
    return shared, jaccard, containment


def generation_contract(result, requested):
    groups = dict.fromkeys(GENERATION_GROUPS, ())
    for group in requested:
        values = result.get(group, [])
        if not isinstance(values, list):
            raise ValueError(f"{group} must be an array of strings")
        for item in values:
            nonempty_text(item, group)
        groups[group] = values[:12]
    return groups


def review_contract(result, count):
    labels = result["candidates"]
    ids = [row["id"] for row in labels]
    if len(ids) != count:
        raise ValueError("Review must cover all synthetic candidates")
    if any(type(item) is not int or not 0 <= item < count for item in ids):
        raise ValueError("Review IDs must match supplied integer IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Review IDs must be unique across labels")
    for row in labels:
        if row["relation"] not in REVIEW_RELATIONS:
            raise ValueError("Unrecognized review relation")
        nonempty_text(row["reason"], "reason")
    return result


def review_batch(client, payload, texts):
    try:
        return client.json_call(
            "stale_blind_review", REVIEW, {**payload, "candidates": [{"id": i, "text": text} for i, text in enumerate(texts)]},
            lambda result: review_contract(result, len(texts)))
    except APIError as exc:
        retryable = re.search(r"HTTP (?:408|5\d\d)\b|超时|格式修复|截断", str(exc))
        if not retryable or not texts:
            raise
        if len(texts) == 1:
            return {"candidates": [{"id": 0, "relation": "uncertain", "reason": "Review unavailable after retries."}]}

    # Use smaller inputs for a failed review. The request IDs start at zero for
    # each batch and are mapped back to the original scene's positions.
    size = (len(texts) + 1) // 2
    merged = {"candidates": []}
    for offset in range(0, len(texts), size):
        result = review_batch(client, payload, texts[offset:offset + size])
        merged["candidates"].extend({**item, "id": item["id"] + offset} for item in result["candidates"])
    return merged


def generate_candidates(client, payload, existing, feedback, round_index, requested):
    batch = client.json_call(
        "stale_author", GENERATION,
        {**payload, "round": round_index, "requested_groups": requested, "feedback": feedback},
        lambda result: generation_contract(result, requested))
    seen = {text_key(row) for row in existing} | {text_key(payload["new_memory"])}
    candidates, rejected = [], []
    for group in GENERATION_GROUPS:
        for content in batch[group]:
            content = " ".join(content.split())
            key = text_key(content)
            if key in seen:
                continue
            if not 6 <= len(key.split()) <= 65:
                continue
            shared, jaccard, containment = lexical_overlap(content, payload["new_memory"])
            if group == "implicit" and (NEGATION.search(content)
                    or shared > 1 and (jaccard > 0.15 or containment > 0.25)):
                rejected.append({"content": content, "reason": "Overt negation or too much shared wording."})
                continue
            seen.add(key)
            candidates.append({"content": content, "intended": group, "lexical_jaccard": jaccard})
    return candidates, rejected


def validate_scene(query, bank, gold, result):
    """A completed scene keeps its base bank and contains 3–6 reviewed rows per added group."""
    old_by_id = {row["memory_id"]: row for row in bank}
    new_by_id = {row["memory_id"]: row for row in result["memories"]}
    if (len(new_by_id) != len(result["memories"])
            or any(new_by_id.get(mid) != row for mid, row in old_by_id.items())):
        raise ValueError(f"{query['query_id']}: 扩增版必须保留原版旧库")
    details = result["gold"]["target_details"]
    if any(details.get(mid) != label for mid, label in gold["target_details"].items()):
        raise ValueError(f"{query['query_id']}: 扩增版必须保留官方金标")
    validate_background_gold(result["memories"], result["gold"])
    counts = Counter()
    for mid in new_by_id.keys() - old_by_id.keys():
        row = new_by_id[mid]
        group = row["source_ref"].get("augmentation_group")
        if group not in GENERATION_GROUPS or row["source_ref"].get("record_type") != "synthetic":
            raise ValueError(f"{mid}: 扩增类型无效")
        counts[group] += 1
        if (mid in details) != (group == "implicit"):
            raise ValueError(f"{mid}: 扩增类型与金标不一致")
        if group == "implicit":
            shared, jaccard, containment = lexical_overlap(row["content"], query["content"])
            if (NEGATION.search(row["content"]) or shared > 1 and (jaccard > 0.15 or containment > 0.25)
                    or details[mid]["relation"] not in TARGET_RELATIONS):
                raise ValueError(f"{mid}: 合成隐式目标未通过形式校验")
    if any(not 3 <= counts[group] <= 6 for group in GENERATION_GROUPS):
        raise ValueError(f"{query['query_id']}: 每类新增记忆必须有 3–6 条")
    return counts


def validate_background_gold(bank, gold):
    """Gold contains only the official old memory and reviewed synthetic targets."""
    records = {row["memory_id"]: row for row in bank}
    validate_gold([gold], set(records))
    for mid, label in gold["target_details"].items():
        ref = records[mid]["source_ref"]
        native = (ref["record_type"] == "M_old" and label["source"] == "stale_native"
                  and label["relation"] == "official_stale")
        synthetic = (ref["record_type"] == "synthetic" and ref.get("augmentation_group") == "implicit"
                     and label["source"] == "synthetic_review" and label["relation"] in TARGET_RELATIONS)
        if not (native or synthetic):
            raise ValueError(f"{mid}: gold must contain only official or reviewed synthetic targets")


def background_only_gold(gold, bank):
    """Remove legacy history targets, preserving memories, pending IDs and all other gold fields."""
    records = {row["memory_id"]: row for row in bank}
    validate_gold([gold], set(records))
    removed = {mid for mid, label in gold["target_details"].items() if label["source"] == "history_review"}
    if any(records[mid]["source_ref"]["record_type"] != "history" for mid in removed):
        raise ValueError("history_review label does not refer to original history")
    result = {**gold, "conflict_old_memory_ids": [mid for mid in gold["conflict_old_memory_ids"] if mid not in removed],
              "target_details": {mid: label for mid, label in gold["target_details"].items() if mid not in removed}}
    validate_background_gold(bank, result)
    return result


def review_scene(settings, review_settings, query, bank, gold, rounds):
    scenario_id = query["scenario_id"]
    native, = [row for row in bank if row["source_ref"]["record_type"] == "M_old"]
    payload = {"new_memory": query["content"], "official_old_memory": native["content"]}
    generator, reviewer = APIClient(settings), APIClient(review_settings)
    accepted = {group: [] for group in GENERATION_GROUPS}
    feedback = {}
    rejected_counts = Counter()
    for round_index in range(rounds):
        added = [item["content"] for rows in accepted.values() for item in rows]
        requested = [group for group in GENERATION_GROUPS if len(accepted[group]) < 3]
        try:
            candidates, rejected = generate_candidates(
                generator, payload, [item["content"] for item in bank] + added, feedback, round_index, requested)
            random.Random(f"{scenario_id}:{round_index}").shuffle(candidates)
            labels = []
            if candidates:
                review = review_batch(reviewer, {**payload, "existing_added": added},
                                      [item["content"] for item in candidates])
                labels = review["candidates"]
        except APIError as exc:
            if "格式修复" not in str(exc) and "截断" not in str(exc):
                raise
            rejected_counts["invalid_output"] += 1
            continue
        for label in labels:
            candidate = candidates[label["id"]]
            group, relation = candidate["intended"], label["relation"]
            expected = (relation in TARGET_RELATIONS and not ADMITS_COMPATIBILITY.search(label["reason"])
                        if group == "implicit" else relation == "compatible")
            if expected and len(accepted[group]) < 6:
                accepted[group].append({**candidate, "relation": relation, "reason": label["reason"]})
            elif not expected:
                rejected_counts[relation] += 1
                rejected.append({"content": candidate["content"], "reason": label["reason"]})
        if round_index >= 3 and len(accepted["implicit"]) < 3:
            feedback_note = ("Use a fact the write expressly commits to. Drop the official old claim as a "
                             "generation anchor when the write leaves that particular value open. "
                             "Try a different consequence, class-wide restriction, or concrete event constraint.")
        else:
            feedback_note = ""
        if min(map(len, accepted.values())) >= 3:
            break
        feedback = {"kept": [item["content"] for rows in accepted.values() for item in rows],
                    "rejected": rejected, "needed": {key: max(0, 3 - len(rows))
                                                       for key, rows in accepted.items()}}
        if feedback_note:
            feedback["new_approach"] = feedback_note
    counts = {key: len(rows) for key, rows in accepted.items()}
    if min(counts.values()) < 3:
        raise ValueError(f"{scenario_id}: {rounds} 轮后只通过 {counts}，需补足至少各 3 条")

    gold = background_only_gold(gold, bank)
    target_details = dict(gold["target_details"])
    memories = list(bank)
    for group, items in accepted.items():
        for item in items:
            memory = memory_row(scenario_id, item["content"], "synthetic",
                                augmentation_group=group)
            memories.append(memory)
            if group == "implicit":
                target_details[memory["memory_id"]] = {
                    "source": "synthetic_review", "relation": item["relation"],
                    "reason": item["reason"], "lexical_jaccard": item["lexical_jaccard"]}
    targets = {**gold, "conflict_old_memory_ids": list(target_details),
               "target_details": target_details, "review_status": "model_reviewed",
               "unjudged_old_memory_ids": list(gold.get("unjudged_old_memory_ids", []))}
    return {"memories": sorted(memories, key=lambda row: row["memory_id"]), "gold": targets,
            "audit": {"generated": counts, "rounds": round_index + 1,
                      "rejected_synthetic_labels": dict(rejected_counts)}}


def load_scene_data(directory):
    queries = load_queries(directory / "queries.jsonl")
    rows = read_jsonl(directory / "memories.jsonl")
    gold = read_jsonl(directory / "gold.jsonl")
    banks = defaultdict(list)
    for row in rows:
        banks[row["scenario_id"]].append(row)
    if len(queries) != len(banks) or {q["scenario_id"] for q in queries} != set(banks):
        raise ValueError(f"{directory}: expected exactly one query per scene bank")
    records = [Memory.from_dict(row) for row in rows]
    validate_memories(records)
    validate_dataset(records, queries, gold)
    return queries, banks, {row["query_id"]: row for row in gold}


@dataclass
class InheritedScene:
    memories: list
    gold: dict
    added_history: list


def prepare_inheritance(directory, queries, banks, base_gold):
    prior_queries, old_banks, old_gold = load_scene_data(directory)
    if {q["query_id"]: q for q in prior_queries} != {q["query_id"]: q for q in queries}:
        raise ValueError("V1 and V2 must contain the same queries and scenario IDs")
    scenes = {}
    for query in queries:
        sid, qid = query["scenario_id"], query["query_id"]
        old_rows = old_banks[sid]
        original = [r for r in old_rows if r["source_ref"]["record_type"] != "synthetic"]
        validate_scene(query, original, base_gold[qid],
                       {"memories": old_rows, "gold": background_only_gold(old_gold[qid], old_rows)})
        base = {r["memory_id"]: r for r in banks[sid]}
        if any(base.get(r["memory_id"]) != r for r in original):
            raise ValueError(f"{qid}: missing or changed V1 base record in V2")
        if any(r["memory_id"] in base for r in old_rows if r["source_ref"]["record_type"] == "synthetic"):
            raise ValueError(f"{qid}: V2 history collides with an inherited synthetic ID")
        native, = [r for r in banks[sid] if r["source_ref"]["record_type"] == "M_old"]
        if base_gold[qid]["conflict_old_memory_ids"] != [native["memory_id"]]:
            raise ValueError(f"{qid}: inconsistent official old memory")
        old_ids = {r["memory_id"] for r in old_rows}
        added = sorted((r for mid, r in base.items() if mid not in old_ids), key=lambda r: r["memory_id"])
        if any(r["source_ref"]["record_type"] != "history" or
               not 0 <= r["source_ref"]["session_index"] < query["source_ref"]["session_index"] for r in added):
            raise ValueError(f"{qid}: only pre-write history may be added")
        scenes[qid] = InheritedScene(old_rows, old_gold[qid], added)
    return scenes


def inherit_background_scene(scene):
    return {"memories": sorted(scene.memories + scene.added_history, key=lambda r: r["memory_id"]),
            "gold": background_only_gold(scene.gold, scene.memories)}


def inheritance_counts(scenes):
    scenes = list(scenes)
    removed = sum(label["source"] == "history_review" for scene in scenes
                  for label in scene.gold["target_details"].values())
    original_targets = sum(len(scene.gold["conflict_old_memory_ids"]) for scene in scenes)
    return {"inherited_memories": sum(len(s.memories) for s in scenes),
            "inherited_synthetic": sum(r["source_ref"]["record_type"] == "synthetic" for s in scenes for r in s.memories),
            "v1_targets": original_targets, "removed_history_targets": removed,
            "inherited_targets": original_targets - removed,
            "inherited_unjudged": sum(len(s.gold.get("unjudged_old_memory_ids", [])) for s in scenes),
            "new_history": sum(len(s.added_history) for s in scenes)}


def background_metadata(scenes, base_hashes, prior_hashes):
    return {**inheritance_counts(scenes), "new_synthetic": 0, "extraction_version": "v2",
            "history_policy": HISTORY_POLICY, "reviewed_new_history": 0, "added_history_targets": 0,
            "added_history_unjudged": 0, "model_calls": 0, "history_review_targets": 0,
            "base_files_sha256": base_hashes, "inherited_files_sha256": prior_hashes}


def check_published(directory, queries, inherited, expected_metadata):
    actual_queries, banks, gold = load_scene_data(directory)
    if actual_queries != queries:
        raise ValueError("Published V2 queries differ from the input")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    if any(manifest.get(k) != v for k, v in expected_metadata.items()):
        raise ValueError("Published V2 manifest does not match these inputs")
    for query in queries:
        qid = query["query_id"]
        scene = inherited[qid]
        expected = inherit_background_scene(scene)
        if expected["gold"] != gold[qid] or expected["memories"] != banks[query["scenario_id"]]:
            raise ValueError(f"{qid}: published records or labels differ from V1 plus background-only history")
    print(f"Verified {len(queries)} published V2 scenes in {directory.resolve()}", flush=True)


def dataset_hashes(directory, *, manifest=False):
    names = ["memories.jsonl", "queries.jsonl", "gold.jsonl"]
    if manifest:
        names.append("manifest.json")
    return {name: sha256_file(directory / name) for name in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--inherit-v1", type=Path, metavar="AUGMENTED",
                        help="Construct V2 offline; inherit memories and synthetic labels, remove history targets")
    parser.add_argument("--check-only", action="store_true",
                        help="Check V2 inputs and, if present, the published output; offline and read-only")
    parser.add_argument("--resume-from", type=Path, action="append", default=[],
                        help="Reuse completed per-scene checkpoints from a verified run with the same prompts and models")
    parser.add_argument("--env", type=Path, help="Model environment; defaults to project .env")
    parser.add_argument("--review-env", type=Path,
                        help="V1 synthetic review configuration; unused by offline V2 inheritance")
    parser.add_argument("--review-model", help="V1 synthetic review model; unused by offline V2 inheritance")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--limit", type=int, help="Optional reproducible pilot sample size")
    args = parser.parse_args(argv)
    if min(args.workers, args.rounds) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers, rounds and limit must be positive")
    if args.check_only and not args.inherit_v1:
        parser.error("--check-only requires --inherit-v1")
    root = V2_ROOT if args.inherit_v1 else DATA_ROOT
    args.base = args.base or root / "base"
    args.work_dir = args.work_dir or root / "work"
    args.output = args.output or root / (f"pilot-{args.limit}" if args.inherit_v1 and args.limit else "augmented")
    if not args.check_only and args.output.exists():
        raise ValueError(f"输出目录已存在：{args.output.resolve()}")
    manifest_path = args.base / "manifest.json"
    base_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}
    if (base_manifest.get("extraction_version") == "v2") != bool(args.inherit_v1):
        parser.error("Use a --version v2 base together with --inherit-v1")
    base_hashes = dataset_hashes(args.base)
    queries, banks, base_gold = load_scene_data(args.base)
    inherited = {}
    if args.inherit_v1:
        prior_hashes = dataset_hashes(args.inherit_v1, manifest=True)
        previous = json.loads((args.inherit_v1 / "manifest.json").read_text(encoding="utf-8-sig"))
        if previous.get("extraction_version") == "v2":
            raise ValueError("Expected the V1 augmented dataset")
        inherited = prepare_inheritance(args.inherit_v1, queries, banks, base_gold)
    if args.limit and args.limit < len(queries):
        queries = random.Random(20261119).sample(queries, args.limit)
    signatures = {"history_policy": HISTORY_POLICY, "script_sha256": sha256_file(Path(__file__))}
    if args.inherit_v1:
        scenes = [inherited[q["query_id"]] for q in queries]
        expected_metadata = background_metadata(scenes, base_hashes, prior_hashes)
        inherited_counts = inheritance_counts(scenes)
        print(json.dumps({"scenarios": len(queries), **inherited_counts,
                          "expected_memories": inherited_counts["inherited_memories"] + inherited_counts["new_history"]}),
              flush=True)
        if args.check_only:
            if args.output.exists():
                check_published(args.output, queries, inherited, expected_metadata)
            return
        signatures.update(extraction_version="v2", augmentation_mode="inherit_v1",
                          inherited_generation_model=previous.get("generation_model"),
                          inherited_synthetic_review_model=previous.get("review_model"))
        ordered = [inherit_background_scene(scene) for scene in scenes]
        for query, result in zip(queries, ordered):
            validate_scene(query, banks[query["scenario_id"]], base_gold[query["query_id"]], result)
    else:
        settings = Settings.from_env(args.env)
        settings.require(llm=True)
        review_settings = Settings.from_env(args.review_env) if args.review_env else settings
        if args.review_model:
            review_settings = replace(review_settings, llm_model=args.review_model)
        signatures.update(generation_model=settings.llm_identity(), generation_prompt_sha256=fingerprint(GENERATION),
                          review_prompt_sha256=fingerprint(REVIEW))
        settings = replace(settings, cache_path=args.work_dir / "api_cache.sqlite3")
        review_settings.require(llm=True)
        signatures["review_model"] = review_settings.llm_identity()
        work = args.work_dir / fingerprint(signatures)[:16]
        work.mkdir(parents=True, exist_ok=True)
        if any(not path.is_dir() for path in args.resume_from):
            raise ValueError("--resume-from must point to an existing checkpoint directory")
        review_settings = replace(review_settings, cache_path=args.work_dir / "api_cache.sqlite3")
        review_settings = replace(review_settings, max_retries=min(review_settings.max_retries, 1),
                                  timeout=min(review_settings.timeout, 65))

        def process(query):
            bank = banks[query["scenario_id"]]
            gold = base_gold[query["query_id"]]
            filename = fingerprint([query, bank, gold, signatures]) + ".json"
            checkpoint = next((path / filename for path in (work, *args.resume_from)
                               if (path / filename).is_file()), None)
            if checkpoint is not None:
                result = json.loads(checkpoint.read_text(encoding="utf-8"))
            else:
                result = review_scene(settings, review_settings, query, bank, gold, args.rounds)
            validate_scene(query, bank, gold, result)
            if checkpoint is None:
                with TemporaryDirectory(dir=work, prefix="scene-") as stage:
                    temporary = Path(stage) / "scene.json"
                    write_json(temporary, result)
                    temporary.replace(work / filename)
            return result

        complete, failures = {}, {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(process, q): q["query_id"] for q in queries}
            for job in as_completed(jobs):
                query_id = jobs[job]
                try:
                    complete[query_id] = job.result()
                except (APIError, ValueError, OSError) as exc:
                    failures[query_id] = str(exc)
                    print(f"[STALE] {query_id}: {exc}", flush=True)
                    if isinstance(exc, APIError) and re.search(r"HTTP (?:401|402|403|404)\b", str(exc)):
                        for pending in jobs:
                            pending.cancel()
                        break
                done = len(complete) + len(failures)
                if done % 10 == 0 or done == len(queries):
                    print(f"[STALE] {done}/{len(queries)} scenarios, {len(failures)} failed", flush=True)
        if failures:
            write_json(work / "failures.json", failures)
            raise SystemExit(f"{len(failures)} failures. Completed checkpoints can be resumed from {work.resolve()}")
        ordered = [complete[q["query_id"]] for q in queries]
    memories = [row for scene in ordered for row in scene["memories"]]
    gold_rows = [scene["gold"] for scene in ordered]
    records = [Memory.from_dict(row) for row in memories]
    validate_memories(records)
    validate_dataset(records, queries, gold_rows)
    source_counts = Counter(label["source"] for scene in ordered for label in scene["gold"]["target_details"].values())
    groups = [Counter(r["source_ref"]["augmentation_group"] for r in s["memories"]
                      if r["source_ref"]["record_type"] == "synthetic") for s in ordered]
    report = {**signatures, "scenarios": len(queries), "memories": len(memories),
              "base_files_sha256": base_hashes,
              "memories_per_scenario": library_stats(memories), "gold_target_sources": dict(source_counts),
              "gold_targets": sum(source_counts.values()), "history_review_targets": source_counts.get("history_review", 0),
              "synthetic_counts": {key: dict(Counter(g[key] for g in groups)) for key in GENERATION_GROUPS},
              "history_unjudged": sum(len(row.get("unjudged_old_memory_ids", [])) for row in gold_rows),
              "human_reviewed": False}
    if args.inherit_v1:
        if dataset_hashes(args.inherit_v1, manifest=True) != prior_hashes or dataset_hashes(args.base) != base_hashes:
            raise ValueError("Dataset inputs changed during inheritance. Rerun against the intended snapshots.")
        report.update(expected_metadata)
    if args.resume_from and not args.inherit_v1:
        report["resumed_checkpoint_directories"] = [path.name for path in args.resume_from]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix="augment-") as stage:
        dataset = Path(stage) / "dataset"
        dataset.mkdir()
        for name, rows in (("memories", memories), ("queries", queries), ("gold", gold_rows)):
            write_jsonl(dataset / f"{name}.jsonl", rows)
        write_json(dataset / "manifest.json", report)
        dataset.rename(args.output.resolve())
    print(f"{args.output.resolve()} | {len(queries)} scenarios, {len(memories)} old memories | {dict(source_counts)}",
          flush=True)


if __name__ == "__main__":
    main()
