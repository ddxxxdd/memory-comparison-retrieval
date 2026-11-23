from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from memory_retrieval.config import Settings
from memory_retrieval.data import read_jsonl, sha256_file, write_json, write_jsonl
from scripts import augment_stale_scenarios as augment
from scripts import convert_stale_scenarios as convert
from scripts.export_stale_scenarios import export_scenarios
from scripts.audit_stale_v2 import audit


def sample():
    def user(text):
        return {"role": "user", "content": text}

    old = "I live in Seattle and work at the city library."
    new = "I just updated my address after settling into a flat in Austin."
    repeated = "I visit the local radio station every Tuesday afternoon"
    return {"uid": "case", "type": "T1", "M_old": old, "M_new": new, "relevant_session_index": [0, 2],
            "haystack_session": [
                [user("I bake a fresh loaf of bread every Sunday. " + repeated + ". "
                      "I also learned how to make fermented radishes last summer."),
                 user("My local grocery store delivers vegetables on Thursday nights."),
                 user("I use a red notebook to record cooking times."),
                 user("I keep my wooden rolling pin by the stove. My sister brings eggs from her farm."),
                 {"role": "assistant", "content": "I grow fresh lettuce at home every year."}],
                [user(repeated + "!"), user(old)],
                [user(new), user("I collect the letters from my new mailbox every morning.")]]}


def make_inputs(root):
    raw = sample()
    source = root / "source.json"
    write_json(source, [raw])
    base = root / "base"
    convert.main(["--version", "v2", "--source", str(source), "--output", str(base)])
    bank, query, gold = convert.convert_scenario(raw)
    prior = root / "v1-augmented"
    history = [r["memory_id"] for r in bank if r["source_ref"]["record_type"] == "history"]
    gold["target_details"][history[0]] = {"source": "history_review", "relation": "stale", "reason": "Legacy history label"}
    gold["conflict_old_memory_ids"].append(history[0])
    gold.update(review_status="model_reviewed", unjudged_old_memory_ids=[history[1]], annotation_note="Keep V1 metadata")
    for group in ("compatible", "implicit"):
        for i in range(3):
            row = convert.memory_row("case", f"My {group} example describes a separate routine number {i}.",
                                     "synthetic", augmentation_group=group)
            bank.append(row)
            if group == "implicit":
                mid = row["memory_id"]
                gold["conflict_old_memory_ids"].append(mid)
                gold["target_details"][mid] = {"source": "synthetic_review", "relation": "stale", "reason": "V1 synthetic label"}
    for name, rows in (("memories", bank), ("queries", [query]), ("gold", [gold])):
        write_jsonl(prior / f"{name}.jsonl", rows)
    write_json(prior / "manifest.json", {"generation_model": {"model": "legacy-generator"},
                                         "review_model": {"model": "legacy-synthetic-reviewer"}})
    return base, prior


class StaleV2Tests(unittest.TestCase):
    def test_all_matching_sentences_keep_v1_representative_and_temporal_boundary(self):
        raw = sample()
        old, query, gold = convert.convert_scenario(raw)
        bank, new_query, new_gold = convert.convert_scenario(raw, version="v2")
        self.assertEqual((query, gold), (new_query, new_gold))
        by_id = {r["memory_id"]: r for r in bank}
        self.assertTrue(all(by_id[r["memory_id"]] == r for r in old))
        repeated, = [r for r in bank if "radio station" in r["content"]]
        self.assertEqual((repeated["source_ref"]["session_index"], repeated["content"][-1]), (1, "!"))
        added = [r for r in bank if r not in old]
        self.assertEqual({r["content"] for r in added}, {
            "I also learned how to make fermented radishes last summer.",
            "I keep my wooden rolling pin by the stove.", "My sister brings eggs from her farm."})
        self.assertTrue(all(r["source_ref"]["session_index"] < 2 for r in bank))
        self.assertFalse(any("lettuce" in r["content"] for r in bank))

    def test_offline_inheritance_filters_only_history_gold_and_ignores_review_cache(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            base, prior = make_inputs(root)
            work = root / "work"
            write_json(work / "legacy" / "checkpoint.json", {"reviewed": 3, "target_details": {"unused": "Do not inherit"}})
            cache = work / "api_cache.sqlite3"
            cache.write_bytes(b"preserved legacy cache; must not be opened by the build")
            before_work = {p.relative_to(work): p.read_bytes() for p in work.rglob("*") if p.is_file()}
            before_v1 = {p.relative_to(prior): p.read_bytes() for p in prior.rglob("*") if p.is_file()}
            args = ["--base", str(base), "--inherit-v1", str(prior), "--work-dir", str(work),
                    "--review-env", str(root / "nonexistent-review.env"), "--review-model", "unused"]
            with patch.object(Settings, "from_env", side_effect=AssertionError("V2 accessed model settings")), patch.object(
                augment, "APIClient", side_effect=AssertionError("V2 created an API client")
            ), patch.object(augment, "generate_candidates", side_effect=AssertionError("V2 generated synthetic memories")):
                augment.main([*args, "--output", str(root / "augmented"), "--check-only"])
                augment.main([*args, "--output", str(root / "augmented")])
                augment.main([*args, "--output", str(root / "replay")])
                augment.main([*args, "--output", str(root / "augmented"), "--check-only"])
            self.assertEqual(before_work, {p.relative_to(work): p.read_bytes() for p in work.rglob("*") if p.is_file()})
            self.assertEqual(before_v1, {p.relative_to(prior): p.read_bytes() for p in prior.rglob("*") if p.is_file()})
            old_rows = read_jsonl(prior / "memories.jsonl")
            old_gold, = read_jsonl(prior / "gold.jsonl")
            added = [r for r in read_jsonl(base / "memories.jsonl") if r not in old_rows]
            output = root / "augmented"
            self.assertEqual(read_jsonl(output / "memories.jsonl"), sorted(old_rows + added, key=lambda r: r["memory_id"]))
            gold, = read_jsonl(output / "gold.jsonl")
            removed = {mid for mid, label in old_gold["target_details"].items() if label["source"] == "history_review"}
            expected = {**old_gold,
                        "conflict_old_memory_ids": [mid for mid in old_gold["conflict_old_memory_ids"] if mid not in removed],
                        "target_details": {mid: label for mid, label in old_gold["target_details"].items() if mid not in removed}}
            self.assertEqual(gold, expected)
            self.assertEqual(gold["unjudged_old_memory_ids"], old_gold["unjudged_old_memory_ids"])
            self.assertFalse(set(gold["unjudged_old_memory_ids"]) & removed)
            report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((report["removed_history_targets"], report["gold_targets"], report["inherited_synthetic"]), (1, 4, 6))
            self.assertEqual((report["model_calls"], report["reviewed_new_history"], report["added_history_targets"],
                              report["added_history_unjudged"], report["history_review_targets"], report["new_synthetic"]), (0, 0, 0, 0, 0, 0))
            self.assertEqual(report["gold_target_sources"], {"stale_native": 1, "synthetic_review": 3})
            for name in ("memories", "queries", "gold"):
                self.assertEqual((output / f"{name}.jsonl").read_bytes(), (root / "replay" / f"{name}.jsonl").read_bytes())
            folder = export_scenarios(output) / "T1" / "case"
            self.assertEqual(read_jsonl(folder / "gold.jsonl"), [gold])

    def test_changed_v1_source_rejected_before_any_model_calls(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            base, prior = make_inputs(Path(tmp))
            old = read_jsonl(prior / "memories.jsonl")
            old[0]["content"] += " Changed text."
            write_jsonl(prior / "memories.jsonl", old)
            with self.assertRaisesRegex(ValueError, "changed V1"):
                augment.main(["--base", str(base), "--inherit-v1", str(prior), "--check-only"])

    def test_v1_build_reviews_synthetic_compatible_and_implicit_without_history_calls(self):
        with TemporaryDirectory() as tmp:
            settings = Settings(llm_api_key="test", llm_base_url="https://example.test/v1", llm_model="reviewer",
                                cache_path=Path(tmp) / "cache.sqlite3")
            bank, query, gold = convert.convert_scenario(sample())
            calls = []

            def response(client, task, prompt, payload, validate):
                calls.append(task)
                self.assertNotIn("history", payload)
                if task == "stale_author":
                    return validate({group: [f"My {group} example describes a separate routine number {i}." for i in range(3)]
                                     for group in ("compatible", "implicit")})
                self.assertEqual(task, "stale_blind_review")
                return validate({"candidates": [{"id": row["id"], "relation": "stale" if "implicit" in row["text"]
                                                 else "compatible", "reason": "Synthetic review."}
                                                for row in payload["candidates"]]})

            with patch.object(augment.APIClient, "json_call", new=response):
                result = augment.review_scene(settings, settings, query, bank, gold, 1)
            self.assertEqual(calls, ["stale_author", "stale_blind_review"])
            self.assertEqual(result["gold"]["unjudged_old_memory_ids"], [])
            self.assertEqual(len(result["gold"]["conflict_old_memory_ids"]), 4)
            counts = augment.validate_scene(query, bank, gold, result)
            self.assertEqual(counts, {"compatible": 3, "implicit": 3})

    def test_export_rejects_reintroduced_history_target(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            base, prior = make_inputs(root)
            output = root / "augmented"
            augment.main(["--base", str(base), "--inherit-v1", str(prior), "--output", str(output)])
            old_gold = read_jsonl(prior / "gold.jsonl")
            write_jsonl(output / "gold.jsonl", old_gold)
            with self.assertRaisesRegex(ValueError, "only official or reviewed synthetic"):
                export_scenarios(output)


def fixture(root):
    base, prior = make_inputs(root)
    v1, v2 = root / "v1", root / "v2"
    convert.main(["--source", str(root / "source.json"), "--output", str(v1 / "base")])
    shutil.copytree(prior, v1 / "augmented")
    shutil.copytree(base, v2 / "base")
    snapshot = root / "snapshot.json"
    write_json(snapshot, {"v1_root": str(v1), "v1_files_sha256": {
        p.relative_to(v1).as_posix(): sha256_file(p) for p in v1.rglob("*") if p.is_file()}})
    return v1, v2, root / "source.json", snapshot


def build(v1, v2):
    with patch.object(Settings, "from_env", side_effect=AssertionError("Offline build accessed settings")), patch.object(
        augment, "APIClient", side_effect=AssertionError("Offline build used a model")
    ):
        augment.main(["--base", str(v2 / "base"), "--inherit-v1", str(v1 / "augmented"),
                      "--output", str(v2 / "augmented"), "--work-dir", str(v2 / "work")])
    for version in ("base", "augmented"):
        export_scenarios(v2 / version)


class StaleV2AuditTests(unittest.TestCase):
    def test_preflight_checks_all_raw_sentences_and_detects_v1_hash_changes(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            v1, v2, source, snapshot = fixture(Path(tmp))
            report = audit(v1, v2, source, snapshot=snapshot)
            self.assertEqual((report["scenarios"], report["new_history"], report["inherited_synthetic"]), (1, 3, 6))
            self.assertEqual((report["v1_targets"], report["removed_history_targets"], report["inherited_targets"]), (5, 1, 4))
            self.assertTrue(report["all_eligible_history_included"])
            with (v1 / "base" / "memories.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "V1 files changed"):
                audit(v1, v2, source, snapshot=snapshot)

    def test_preflight_rejects_added_history_at_write_session(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            v1, v2, source, snapshot = fixture(Path(tmp))
            old = read_jsonl(v1 / "base" / "memories.jsonl")
            rows = read_jsonl(v2 / "base" / "memories.jsonl")
            next(r for r in rows if r not in old)["source_ref"]["session_index"] = 2
            write_jsonl(v2 / "base" / "memories.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "pre-write"):
                audit(v1, v2, source, snapshot=snapshot)

    def test_final_audit_checks_gold_filter_exports_and_preserved_cache(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            v1, v2, source, snapshot = fixture(root)
            work = v2 / "work"
            work.mkdir()
            cache = work / "api_cache.sqlite3"
            cache.write_bytes(b"Retain this previous cache without reading its results")
            cache_snapshot = root / "cache_before.json"
            write_json(cache_snapshot, {"work_root": str(work), "files_sha256": {cache.name: sha256_file(cache)}})
            build(v1, v2)
            report = audit(v1, v2, source, snapshot=snapshot, final=True, preserved_work_snapshot=cache_snapshot)
            self.assertTrue(report["gold_matches_v1_after_history_filter"])
            self.assertTrue(report["legacy_pending_and_flags_preserved"])
            self.assertTrue(report["aggregates_match_scene_files"])
            self.assertEqual((report["model_calls"], report["history_review_targets"], report["final_targets"],
                              report["final_unjudged"], report["preserved_cache_and_checkpoint_files"]), (0, 0, 4, 1, 1))
            scene_file = v2 / "augmented" / "scenarios" / "T1" / "case" / "memories.jsonl"
            rows = read_jsonl(scene_file)
            rows[0]["content"] += " Changed export."
            write_jsonl(scene_file, rows)
            with self.assertRaisesRegex(ValueError, "scene file differs from aggregate"):
                audit(v1, v2, source, snapshot=snapshot, final=True)
            export_scenarios(v2 / "augmented")
            cache.write_bytes(b"Changed cache")
            with self.assertRaisesRegex(ValueError, "Preserved cache or checkpoint changed"):
                audit(v1, v2, source, snapshot=snapshot, final=True, preserved_work_snapshot=cache_snapshot)

    def test_final_audit_rejects_new_history_pending_and_reintroduced_history_gold(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            v1, v2, source, snapshot = fixture(Path(tmp))
            build(v1, v2)
            gold_path = v2 / "augmented" / "gold.jsonl"
            gold, = read_jsonl(gold_path)
            old_ids = {r["memory_id"] for r in read_jsonl(v1 / "augmented" / "memories.jsonl")}
            added = next(r for r in read_jsonl(v2 / "base" / "memories.jsonl") if r["memory_id"] not in old_ids)
            gold["unjudged_old_memory_ids"].append(added["memory_id"])
            write_jsonl(gold_path, [gold])
            with self.assertRaisesRegex(ValueError, "published records or labels"):
                audit(v1, v2, source, snapshot=snapshot, final=True)
            write_jsonl(gold_path, read_jsonl(v1 / "augmented" / "gold.jsonl"))
            with self.assertRaisesRegex(ValueError, "published records or labels"):
                audit(v1, v2, source, snapshot=snapshot, final=True)
