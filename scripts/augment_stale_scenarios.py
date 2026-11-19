"""Generate and independently review added memories within each STALE scene bank."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
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
from memory_retrieval.data import Memory, fingerprint, nonempty_text, read_jsonl, validate_memories, write_json, sha256_file, write_jsonl
from memory_retrieval.evaluation import validate_dataset
from scripts.convert_stale_scenarios import DATA_ROOT, library_stats, memory_row, text_key
from scripts.stale_augmentation_prompts import GENERATION, HISTORY_REVIEW, REVIEW


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


def review_contract(result, count, *, history=False):
    labels = result["targets" if history else "candidates"]
    ids = [row["id"] for row in labels]
    relations = TARGET_RELATIONS if history else REVIEW_RELATIONS
    if history:
        # Some gateways emit full objects for ID-only arrays. Keep the referenced ID.
        for key in ("uncertain", "discard"):
            result[key] = [item["id"] if isinstance(item, dict) else item for item in result[key]]
            ids.extend(result[key])
    elif len(ids) != count:
        raise ValueError("Review must cover all synthetic candidates")
    if any(type(item) is not int or not 0 <= item < count for item in ids):
        raise ValueError("Review IDs must match supplied integer IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Review IDs must be unique across labels")
    for row in labels:
        if row["relation"] not in relations:
            raise ValueError("Unrecognized review relation")
        nonempty_text(row["reason"], "reason")
    return result


def review_batch(client, payload, texts, *, history=False):
    key = "history" if history else "candidates"
    task, prompt = (("stale_background_review", HISTORY_REVIEW) if history
                    else ("stale_blind_review", REVIEW))
    try:
        return client.json_call(
            task, prompt, {**payload, key: [{"id": i, "text": text} for i, text in enumerate(texts)]},
            lambda result: review_contract(result, len(texts), history=history))
    except APIError as exc:
        retryable = re.search(r"HTTP (?:408|5\d\d)\b|超时|格式修复|截断", str(exc))
        if not retryable or not texts:
            raise
        if len(texts) == 1:
            if history:
                return {"targets": [], "uncertain": [0], "discard": []}
            return {"candidates": [{"id": 0, "relation": "uncertain", "reason": "Review unavailable after retries."}]}

    # Use smaller inputs for a failed review. The request IDs start at zero for
    # each batch and are mapped back to the original scene's positions.
    size = (len(texts) + 1) // 2
    merged = {name: [] for name in (("targets", "uncertain", "discard") if history else (key,))}
    for offset in range(0, len(texts), size):
        result = review_batch(client, payload, texts[offset:offset + size], history=history)
        for name in merged:
            for item in result[name]:
                merged[name].append({**item, "id": item["id"] + offset}
                                    if isinstance(item, dict) else item + offset)
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


def review_scene(settings, review_settings, query, bank, gold, rounds):
    scenario_id = query["scenario_id"]
    native, = [row for row in bank if row["source_ref"]["record_type"] == "M_old"]
    history = [row for row in bank if row is not native]
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

    background_review = review_batch(reviewer, payload, [item["content"] for item in history], history=True)
    target_details = dict(gold["target_details"])
    unjudged = set(background_review["uncertain"] + background_review["discard"])
    for label in background_review["targets"]:
        old = history[label["id"]]
        excerpt = text_key(label.get("old_excerpt") or "")
        if (len(excerpt.split()) < 3 or excerpt not in text_key(old["content"])
                or ADMITS_COMPATIBILITY.search(label["reason"])):
            unjudged.add(label["id"])
            continue
        target_details[old["memory_id"]] = {"source": "history_review", "relation": label["relation"],
                                            "reason": label["reason"]}
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
               "unjudged_old_memory_ids": [history[i]["memory_id"] for i in sorted(unjudged)]}
    return {"memories": sorted(memories, key=lambda row: row["memory_id"]), "gold": targets,
            "audit": {"generated": counts, "rounds": round_index + 1,
                      "rejected_synthetic_labels": dict(rejected_counts)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DATA_ROOT / "base")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "augmented")
    parser.add_argument("--work-dir", type=Path, default=DATA_ROOT / "work")
    parser.add_argument("--resume-from", type=Path, action="append", default=[],
                        help="Reuse completed per-scene checkpoints from a verified run with the same prompts and models")
    parser.add_argument("--env", type=Path, help="Author model environment; defaults to project .env")
    parser.add_argument("--review-env", type=Path, help="Optional separate model environment for review")
    parser.add_argument("--review-model", help="Optional model override for review")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--limit", type=int, help="Optional reproducible pilot sample size")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.rounds < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers, rounds and limit must be positive")
    if args.output.exists():
        raise ValueError(f"输出目录已存在：{args.output.resolve()}")
    queries = read_jsonl(args.base / "queries.jsonl")
    banks = defaultdict(list)
    for row in read_jsonl(args.base / "memories.jsonl"):
        banks[row["scenario_id"]].append(row)
    base_gold = {row["query_id"]: row for row in read_jsonl(args.base / "gold.jsonl")}
    if args.limit and args.limit < len(queries):
        queries = random.Random(20261119).sample(queries, args.limit)
    if not queries:
        raise ValueError("No base queries")
    settings = Settings.from_env(args.env)
    review_settings = Settings.from_env(args.review_env) if args.review_env else settings
    if args.review_model:
        review_settings = replace(review_settings, llm_model=args.review_model)
    signatures = {"generation_model": settings.llm_identity(), "review_model": review_settings.llm_identity(),
                  "generation_prompt_sha256": fingerprint(GENERATION), "review_prompt_sha256": fingerprint(REVIEW),
                  "history_prompt_sha256": fingerprint(HISTORY_REVIEW),
                  "script_sha256": sha256_file(Path(__file__))}
    work = args.work_dir / fingerprint(signatures)[:16]
    work.mkdir(parents=True, exist_ok=True)
    if any(not path.is_dir() for path in args.resume_from):
        raise ValueError("--resume-from must point to an existing checkpoint directory")
    settings = replace(settings, cache_path=args.work_dir / "api_cache.sqlite3")
    review_settings = replace(review_settings, cache_path=settings.cache_path,
                              max_retries=min(review_settings.max_retries, 1),
                              timeout=min(review_settings.timeout, 65))
    for item in (settings, review_settings):
        item.require(llm=True)

    def process(query):
        bank = banks[query["scenario_id"]]
        gold = base_gold[query["query_id"]]
        filename = fingerprint([query, bank, gold]) + ".json"
        checkpoint = next((path / filename for path in (work, *args.resume_from)
                           if (path / filename).is_file()), None)
        if checkpoint is not None:
            result = json.loads(checkpoint.read_text(encoding="utf-8"))
            validate_scene(query, bank, gold, result)
            return result
        result = review_scene(settings, review_settings, query, bank, gold, args.rounds)
        validate_scene(query, bank, gold, result)
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
                if isinstance(exc, APIError) and re.search(r"HTTP (?:401|402|404)\b", str(exc)):
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
    report = {**signatures, "scenarios": len(queries), "memories": len(memories),
              "base_files_sha256": {name: sha256_file(args.base / name)
                                    for name in ("memories.jsonl", "queries.jsonl", "gold.jsonl")},
              "memories_per_scenario": library_stats(memories), "gold_target_sources": dict(source_counts),
              "synthetic_counts": {key: dict(Counter(scene["audit"]["generated"][key] for scene in ordered))
                                   for key in GENERATION_GROUPS},
              "history_unjudged": sum(len(row["unjudged_old_memory_ids"]) for row in gold_rows),
              "human_reviewed": False}
    if args.resume_from:
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
