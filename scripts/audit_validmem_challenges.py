"""Audit frozen probe data/rankings and export evidence; never run or alter a retriever."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import math
import statistics

from validate_validmem_challenges import (
    ROOT, REPORT, SOURCE, METHODS, SEEDS, rows, read_json, dump, file_hash,
    verify_frozen, parse_fact, norm, digest,
)
from validmem_challenge_analysis import write_csv


EXPERIMENTS = (
    ("primary", "validmem_challenge_validation_v1", "validmem-challenge-validation-20261112", ""),
    ("attribute_specific", "validmem_attribute_controls_v1", "validmem-attribute-controls-20261112", "attribute_controls/"),
)


def condition(c):
    return (c["module"], c["variant"], c["added_distractors"], c["bank_size"], c["gold_count"], c["wording"])


def main():
    verify_frozen()
    source_m = {r["memory_id"]: r for r in rows(SOURCE / "memories.jsonl")}
    source_q = {r["query_id"]: r for r in rows(SOURCE / "queries.jsonl")}
    source_g = {r["query_id"]: r for r in rows(SOURCE / "gold.jsonl")}
    audit, overlap, rank_summary, ties, witnesses, cardinality_retained = {}, [], [], [], [], []
    for experiment, data_name, run_name, protocol_prefix in EXPERIMENTS:
        data, run = ROOT / "data" / data_name, ROOT / "runs/experiments" / run_name
        manifest, completion, preflight = [read_json(p) for p in
            (data / "manifest.json", run / "completion.json", run / "preflight.json")]
        assert completion["status"] == "complete"
        assert file_hash(data / "manifest.json") == completion["dataset_manifest_sha256"] == preflight["dataset_manifest_sha256"]
        assert all(file_hash(data / p) == h for p, h in manifest["files_sha256"].items())
        assert all(file_hash(run / p) == h for p, h in completion["result_hashes"].items())
        assert all(file_hash(ROOT / p) == h for p, h in read_json(data / "source_hashes.json").items())
        assert file_hash(REPORT / protocol_prefix / "protocol.zh.md") == preflight["protocol_sha256"]
        assert file_hash(run / "protocol.frozen.zh.md") == preflight["protocol_sha256"]
        builder = ROOT / "scripts" / ("validate_validmem_challenges.py" if experiment == "primary" else "validate_validmem_attribute_controls.py")
        assert file_hash(builder) == manifest["builder_sha256"]
        runtime = ROOT / "scripts/validmem_challenge_runtime.py"
        runtime_matches = file_hash(runtime) == completion["runtime_sha256"]
        if not runtime_matches:
            assert file_hash(run / "runtime_at_execution.py") == completion["runtime_sha256"]
        for encoding in ("qwen", "local"):
            assert file_hash(run / f"{encoding}_vectors.npz") == read_json(run / f"{encoding}_encoding.json")["vectors_sha256"]
        texts = {r["text_id"]: r["content"] for r in rows(data / "texts.jsonl")}
        memories = {r["memory_id"]: r for r in rows(data / "memories.jsonl")}
        banks = {r["bank_id"]: r["memory_ids"] for r in rows(data / "banks.jsonl")}
        cases = {r["case_id"]: r for r in rows(data / "cases.jsonl")}
        anchors = {r["anchor_id"]: r for r in rows(data / "anchors.jsonl")}
        results = {(r["case_id"], r["method"]): r for r in rows(run / "per_case.jsonl")}
        assert len(results) == 4 * len(cases) == completion["method_case_count"]
        if experiment == "primary":
            excluded = rows(data / "excluded_queries.jsonl")
            assert set(anchors).isdisjoint({r["query_id"] for r in excluded})
            assert set(anchors) | {r["query_id"] for r in excluded} == set(source_q)
            for aid, a in anchors.items():
                assert a["source_query"] == source_q[aid]["content"]
                assert a["source_target"] == source_m[a["source_target_id"]]["content"]
                assert a["query_fact"] == parse_fact(a["source_query"])
                assert set(a["source_gold_ids"]) == set(source_g[aid]["conflict_old_memory_ids"])
                for mid in a["source_gold_ids"]:
                    f = parse_fact(source_m[mid]["content"])
                    assert f["entity"] == a["query_fact"]["entity"] and f["attribute"] == a["query_fact"]["attribute"]
                    assert norm(f["value"]) != norm(a["query_fact"]["value"])
        by_anchor_seed = defaultdict(list)
        for c in cases.values():
            a = anchors[c["anchor_id"]]
            q = a["query_fact"]
            ids = banks[c["bank_id"]]
            assert len(ids) == len(set(ids)) == c["bank_size"]
            assert len(c["gold_ids"]) == c["gold_count"]
            gold = []
            for mid in ids:
                m = memories[mid]
                content = texts[m["text_id"]]
                if m["role"] == "source_background":
                    assert content == source_m[mid]["content"]
                    assert q["entity"].casefold() not in content.casefold()
                    continue
                f = m["fact"]
                attr = q["attribute"] if experiment == "primary" else a["recipe"]["slots"][0]
                if f["entity"] == q["entity"] and f["attribute"] == attr and m.get("style") != "mentioned" and norm(f["value"]) != norm(q["value"]):
                    gold.append(mid)
            assert set(gold) == set(c["gold_ids"]), c["case_id"]
            by_anchor_seed[c["anchor_id"], c["seed"]].append(c)
        paired_checks = Counter()
        for group in by_anchor_seed.values():
            if experiment == "primary":
                wording = [c for c in group if c["module"] == "wording"]
                assert len({c["bank_id"] for c in wording}) == 1
                paired_checks["identical_wording_bank"] += 1
                scale = sorted((c for c in group if c["module"] == "scale"), key=lambda c: c["bank_size"])
                assert set(banks[scale[0]["bank_id"]]) < set(banks[scale[1]["bank_id"]]) < set(banks[scale[2]["bank_id"]])
                paired_checks["nested_scale_bank"] += 1
                for d in (1, 5, 15):
                    pair = [c for c in group if c["module"] == "identity" and c["added_distractors"] == d]
                    bg = [{mid for mid in banks[c["bank_id"]] if mid in source_m} for c in pair]
                    assert len(pair) == 2 and bg[0] == bg[1]
                    names = [{memories[mid]["fact"]["entity"] for mid in banks[c["bank_id"]] if mid not in source_m} for c in pair]
                    assert names[0] == names[1] and pair[0]["gold_ids"] == pair[1]["gold_ids"]
                    paired_checks["matched_identity_pair"] += 1
                card = sorted((c for c in group if c["module"] == "cardinality"), key=lambda c: c["gold_count"])
                if card:
                    for before, after in zip(card, card[1:]):
                        assert set(before["gold_ids"]) < set(after["gold_ids"])
                    paired_checks["nested_distinct_gold"] += 1
                    retained = card[0]["gold_ids"][0]
                    for c in card:
                        assert len({norm(memories[mid]["fact"]["value"]) for mid in c["gold_ids"]}) == c["gold_count"]
                        for method in METHODS:
                            rr = results[c["case_id"], method]
                            cardinality_retained.append(dict(anchor_id=c["anchor_id"], seed=c["seed"], family=c["family"],
                                method=method, gold_count=c["gold_count"], retained_gold_id=retained,
                                retained_gold_rank=rr["gold_ranks"][retained]))
            else:
                pair = [c for c in group if c["variant"] in ("same_entity_other_attribute", "other_entity_other_attribute")]
                bg = [{mid for mid in banks[c["bank_id"]] if mid in source_m} for c in pair]
                assert len(pair) == 2 and bg[0] == bg[1] and pair[0]["gold_ids"] == pair[1]["gold_ids"]
                paired_checks["matched_attribute_pair"] += 1
        groups = defaultdict(list)
        for c in cases.values():
            groups[condition(c)].append(c)
        for key, group in groups.items():
            base = dict(zip(("module", "variant", "added_distractors", "bank_size", "gold_count", "wording"), key))
            for k in (1, 5, 10):
                failed = [c for c in group if all(any(v > k for v in results[c["case_id"], m]["gold_ranks"].values()) for m in METHODS)]
                missed = sum(sum(all(results[c["case_id"], m]["gold_ranks"][mid] > k for m in METHODS) for mid in c["gold_ids"]) for c in group)
                by_anchor = Counter(c["anchor_id"] for c in failed)
                overlap.append(dict(experiment=experiment, **base, k=k, cases=len(group), all_four_incomplete=len(failed),
                    all_four_incomplete_anchors=len(by_anchor), all_five_seeds_incomplete_anchors=sum(v == len(SEEDS) for v in by_anchor.values()),
                    same_pair_missed_all_four=missed))
            for method in METHODS:
                worst = [max(results[c["case_id"], method]["gold_ranks"].values()) for c in group]
                ranks = [r for c in group for r in results[c["case_id"], method]["gold_ranks"].values()]
                g = group[0]["gold_count"]
                row = dict(experiment=experiment, **base, method=method, cases=len(group),
                    minimum_full_budget_median=statistics.median(worst), minimum_full_budget_max=max(worst),
                    mean_overhead_over_gold_count=statistics.mean(v-g for v in worst),
                    recall_at_gold_count=sum(v <= g for v in ranks) / len(ranks))
                for low, high in ((1,1),(2,3),(4,5),(6,10),(11,20),(21,50),(51,1000)):
                    row[f"rank_{low}_{high}"] = sum(low <= r <= high for r in ranks)
                rank_summary.append(row)
        seen = set()
        boundary = defaultdict(Counter)
        with gzip.open(run / "rankings.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                cid, method = r["case_id"], r["method"]
                assert (cid, method) not in seen
                seen.add((cid, method))
                c, saved = cases[cid], results[cid, method]
                ids, scores = r["ranked_ids"], r["scores"]
                assert len(ids) == len(set(ids)) == len(scores) == c["bank_size"]
                assert set(ids) == set(banks[c["bank_id"]])
                assert all(math.isfinite(s) for s in scores)
                assert all(a >= b for a, b in zip(scores, scores[1:]))
                pos = {mid: i for i, mid in enumerate(ids)}
                assert {mid: pos[mid]+1 for mid in c["gold_ids"]} == saved["gold_ranks"]
                assert ids[0] == saved["top1"] and ids[:5] == saved["top5"] and ids[:10] == saved["top10"]
                for k in (1, 5, 10):
                    counter = boundary[condition(c), method, k]
                    counter["gold_pairs"] += len(c["gold_ids"])
                    for mid in c["gold_ids"]:
                        if pos[mid] >= k:
                            counter["missed_pairs"] += 1
                            counter["misses_with_score_tied_to_cutoff_1e12"] += abs(scores[pos[mid]] - scores[k-1]) <= 1e-12
        assert seen == set(results)
        for (key, method, k), v in boundary.items():
            ties.append(dict(experiment=experiment, **dict(zip(("module","variant","added_distractors","bank_size","gold_count","wording"),key)),
                             method=method,k=k,gold_pairs=v["gold_pairs"],missed_pairs=v["missed_pairs"],
                             misses_with_score_tied_to_cutoff_1e12=v["misses_with_score_tied_to_cutoff_1e12"]))
        # These are post-result illustrations, not the sample-selection population.
        selectors = ([
            ("identity_pressure", lambda c: c["module"] == "identity" and c["variant"] == "other_entity_query_value" and c["added_distractors"] == 15 and all(min(results[c["case_id"],m]["gold_ranks"].values()) > 5 for m in METHODS)),
            ("identity_small_budget_success", lambda c: c["module"] == "identity" and c["variant"] == "other_entity_query_value" and c["added_distractors"] == 5 and all(min(results[c["case_id"],m]["gold_ranks"].values()) <= 5 for m in METHODS)),
            ("chinese_configuration_gap", lambda c: c["module"] == "wording" and c["wording"] == "chinese" and c["family"] != "grocery" and min(results[c["case_id"],"sparsecl"]["gold_ranks"].values()) > 10 and min(results[c["case_id"],"dense"]["gold_ranks"].values()) <= 5),
            ("multiple_targets_without_capacity_deficit", lambda c: c["module"] == "cardinality" and c["gold_count"] == 4 and all(max(results[c["case_id"],m]["gold_ranks"].values()) > 5 for m in METHODS)),
            ("scale_bm25_sensitivity", lambda c: c["module"] == "scale" and c["bank_size"] == 1000 and min(results[c["case_id"],"bm25"]["gold_ranks"].values()) > 5 and min(results[c["case_id"],"dense"]["gold_ranks"].values()) <= 5),
            ("restatement_sparsecl_success", lambda c: c["module"] == "attribute" and c["variant"] == "same_value_restatement"),
        ] if experiment == "primary" else [
            ("event_attribute", lambda c: c["family"] == "event_registration" and c["variant"] == "same_entity_other_attribute"),
            ("adult_child_scope", lambda c: c["family"] == "adult_child_fee" and c["variant"] == "same_entity_other_attribute"),
        ])
        for label, select in selectors:
            eligible = [c for c in cases.values() if c["seed"] == 17 and select(c)]
            if not eligible:
                eligible = [c for c in cases.values() if select(c)]
            assert eligible, label
            c = min(eligible, key=lambda c: digest(c["case_id"]))
            peers = [x for x in by_anchor_seed[c["anchor_id"],c["seed"]] if x["module"] == c["module"]]
            def memory_text(mid):
                return {"id":mid, "text":texts[memories[mid]["text_id"]], "fact":memories[mid].get("fact")}
            witnesses.append(dict(label=label, experiment=experiment, selection="Post-result illustrative case; hash-selected within stated condition, seed 17 if available",
                case=c, anchor=anchors[c["anchor_id"]], query=texts[c["query_text_id"]],
                gold=[memory_text(mid) for mid in c["gold_ids"]],
                controlled_foreground=[memory_text(mid) for mid in banks[c["bank_id"]] if mid not in source_m],
                retrieval={m:{"gold_ranks":results[c["case_id"],m]["gold_ranks"],"top5":[memory_text(mid) for mid in results[c["case_id"],m]["top5"]]} for m in METHODS},
                paired_conditions=[{"case_id":x["case_id"],"variant":x["variant"],"bank_size":x["bank_size"],"gold_count":x["gold_count"],
                    "added_distractors":x["added_distractors"],"query":texts[x["query_text_id"]],
                    "ranks":{m:results[x["case_id"],m]["gold_ranks"] for m in METHODS}} for x in peers]))
        audit[experiment] = dict(cases=len(cases), rankings_checked=len(seen), paired_checks=dict(paired_checks),
            dataset_files_and_sources_unchanged=True, labels_recomputed_from_explicit_rules=True,
            complete_rankings_match_per_case=True, current_runtime_matches_executed=runtime_matches,
            execution_snapshot_verified=not runtime_matches,
            protocol_and_frozen_copy_match=True, encoder_vector_hashes_verified=True)
        if experiment == "primary":
            # Inspect the existing BM25 scoring formula on one fixed pair as N changes.
            from memory_retrieval.baselines import BM25, tokenize
            witness = next(w for w in witnesses if w["label"] == "scale_bm25_sensitivity")
            gold_id = witness["gold"][0]["id"]
            score_states = []
            for peer in witness["paired_conditions"]:
                c = cases[peer["case_id"]]
                ids = banks[c["bank_id"]]
                bm25 = BM25([texts[memories[mid]["text_id"]] for mid in ids])
                score_states.append((c, ids, bm25, bm25.score(witness["query"])))
            before, after = score_states[0], score_states[-1]
            eligible = [mid for mid in before[1] if mid != gold_id and mid not in source_m
                and before[3][before[1].index(mid)] < before[3][before[1].index(gold_id)]
                and after[3][after[1].index(mid)] > after[3][after[1].index(gold_id)]]
            assert eligible
            distractor_id = min(eligible, key=digest)
            evidence = []
            for c, ids, bm25, scores in score_states:
                terms = []
                for token in sorted(set(tokenize(witness["query"]))):
                    df = bm25.df.get(token, 0)
                    idf = math.log(1 + (len(ids)-df+0.5)/(df+0.5))
                    contributions = []
                    for mid in (gold_id, distractor_id):
                        i = ids.index(mid)
                        tf = bm25.docs[i].get(token, 0)
                        denominator = tf + 1.2*(1-0.75+0.75*bm25.lengths[i]/bm25.average)
                        contributions.append(idf*tf*2.2/denominator)
                    terms.append(dict(term=token,df=df,idf=idf,gold_contribution=contributions[0],distractor_contribution=contributions[1]))
                evidence.append(dict(case_id=c["case_id"],bank_size=len(ids),query=witness["query"],
                    gold_text=texts[memories[gold_id]["text_id"]],distractor_text=texts[memories[distractor_id]["text_id"]],
                    gold_score=float(scores[ids.index(gold_id)]),distractor_score=float(scores[ids.index(distractor_id)]),terms=terms))
            dump(REPORT / "scale_score_evidence.json", evidence)
    write_csv(REPORT / "common_failures.csv", overlap)
    write_csv(REPORT / "rank_distribution.csv", rank_summary)
    write_csv(REPORT / "tie_diagnostics.csv", ties)
    write_csv(REPORT / "cardinality_retained_target.csv", cardinality_retained)
    dump(REPORT / "case_evidence.json", witnesses)
    dump(REPORT / "audit.json", audit)
    lines = ["# 可追溯案例", "", "以下均为来源支持的受控反事实，按结果条件选作解释示例；不是自然发生频率估计，也不参与样本筛选。完整前五名、前景和配对结果见 case_evidence.json。", ""]
    for w in witnesses:
        c = w["case"]
        lines += [f"## {w['label']}", "", f"case `{c['case_id']}`；anchor `{c['anchor_id']}`；seed={c['seed']}；N={c['bank_size']}；G={c['gold_count']}。", "", f"查询：{w['query']}", ""]
        lines += [f"- 金标 `{m['id']}`：{m['text']}" for m in w["gold"]]
        lines += ["", "| 方法 | 金标排名 | 第一名 |", "|---|---|---|"]
        for m in METHODS:
            rr = w["retrieval"][m]
            lines.append(f"| {m} | {', '.join(str(v) for v in rr['gold_ranks'].values())} | {rr['top5'][0]['text']} |")
        lines += ["", "| 配对条件 | d | N | G | BM25 名次 | Dense 名次 | Hybrid 名次 | SparseCL 名次 |", "|---|---:|---:|---:|---|---|---|---|"]
        for p in w["paired_conditions"]:
            vals = [", ".join(str(v) for v in p["ranks"][m].values()) for m in METHODS]
            lines.append(f"| {p['variant']} | {p['added_distractors']} | {p['bank_size']} | {p['gold_count']} | " + " | ".join(vals) + " |")
        lines.append("")
    (REPORT / "cases.zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(audit)


if __name__ == "__main__":
    main()
