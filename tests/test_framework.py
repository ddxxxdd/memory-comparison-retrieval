from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from memory_retrieval.adapters import load_graph, load_text
from memory_retrieval.api import APIClient, APIError
from memory_retrieval.baselines import tokenize
from memory_retrieval.config import DESCRIPTION_DUPLICATE_THRESHOLD, MEMORY_TEMPLATE_THRESHOLD, Settings
from memory_retrieval.data import Memory, fingerprint, read_jsonl
from memory_retrieval.evaluation import evaluate, validate_gold
from memory_retrieval.index import MemoryIndex, _attach_entries, _fallback_descriptions, build_index
from memory_retrieval.pipeline import Retriever
from memory_retrieval.prompts import describe


ROOT = Path(__file__).resolve().parents[1]


def descriptions(content):
    found = []
    if "工作单位" in content or "跳槽" in content:
        found.append("小王的工作单位" if "小王" in content else "小林的工作单位")
    if "常住城市" in content:
        found.append("小林的常住城市")
    if "面试" in content:
        found.append("小林的面试安排")
    if "举办日期" in content:
        found.append("项目A第二期发布会的日期" if "第二期" in content else "项目A第一期发布会的日期")
    if "举办地点" in content:
        found.append("项目B第三期发布会的地点" if "项目B" in content else "项目A第一期发布会的地点")
    if "入职培训" in content:
        found.append("小林的入职培训日期")
    return found


def vector(text):
    dims = ["工作单位", "常住城市", "面试", "日期", "地点", "入职培训", "第二期", "小王", "项目B"]
    # 使用可预测的人工向量测试数据流，不假装验证真实语义模型。
    values = [float(word in text) for word in dims]
    return values + [0.05]


def new_items(needed, known, style="default", drift=""):
    """构造新增申请：每条未覆盖事项给出 value 与 description 两栏。"""
    items = []
    for description in needed:
        if description in known:
            continue
        items.append({"value": "测试桩取值", "description": description + drift})
    return items


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body, self.headers.get("Authorization")))
        if self.server.failures:
            self.server.failures -= 1
            self.send_response(500)
            self.end_headers()
            return
        if self.path.endswith("/embeddings"):
            values = [{"index": i, "embedding": vector(text)} for i, text in enumerate(body["input"])]
            if self.server.zero_vectors:
                values[0]["embedding"] = [0] * len(values[0]["embedding"])
            result = {"data": list(reversed(values)), "usage": {"prompt_tokens": 8, "total_tokens": 8}}
        else:
            user = next(m for m in body["messages"] if m["role"] == "user")
            payload = json.loads(user["content"])
            task = payload["task"]
            if task == "connection_check":
                answer = {"status": "ok"}
            elif task == "select_descriptions":
                needed = descriptions(payload["content"])
                known = {d["description"]: d["id"] for d in payload["catalog"]}
                answer = {"selected_ids": [known[d] for d in needed if d in known],
                          "new_items": new_items(needed, known)}
                if self.server.invalid_selection:
                    self.server.invalid_selection -= 1
                    answer["selected_ids"] = ["NONEXISTENT"]
            elif task == "batch_select_descriptions":
                known = {d["description"]: d["id"] for d in payload["catalog"]}
                results = []
                for item in payload["memories"]:
                    needed = descriptions(item["content"])
                    row = {"index": item["index"],
                           "selected_ids": [known[d] for d in needed if d in known],
                           "new_items": new_items(needed, known)}
                    results.append(row)
                if self.server.batch_drop:
                    self.server.batch_drop -= 1
                    results.pop()  # 故意漏掉一条记忆，用于验证漏项会被拦下
                if self.server.batch_empty and results:
                    self.server.batch_empty -= 1
                    results[-1]["selected_ids"] = []
                    results[-1]["new_items"] = []  # 故意返回空，用于验证兜底
                if self.server.batch_drift and results:
                    # 同一事项写出两种措辞：用于验证批内一致性校验与重写。
                    self.server.batch_drift -= 1
                    for row in results[1]["new_items"]:
                        row["description"] += "（措辞不同）"
                answer = {"results": results}
            elif task == "generate_hypotheses":
                answer = {"same_value": [payload["content"]],
                          "alternative": [payload["content"].replace("阿里", "腾讯")],
                          "subject_relations": [
                              {"memory_id": candidate["memory_id"], "comparisons": [{
                               "new_matter": "测试事项", "old_matter": "测试事项",
                               "matter_relation": "corresponding",
                               "new_subject": {"kind": "explicit", "identity": "新主体", "basis": "测试桩"},
                               "old_subject": {"kind": "explicit", "identity": "旧主体", "basis": "测试桩"},
                               "subject_relation": (
                                   self.server.subject_relation_override.get(candidate["memory_id"], "same")
                                   if isinstance(self.server.subject_relation_override, dict)
                                   else self.server.subject_relation_override or "same"),
                               "reason": "测试桩的预设主体关系"}]}
                              for candidate in payload["candidates"]]}
            elif task == "judge_conflict":
                new, old = payload["new_content"], payload["old_content"]
                conflict = (("小林" in new and "工作单位" in new and "2026年10月13日" in old
                             and "工作单位是腾讯" in old) or
                            ("项目A第一期" in new and "22日" in new and "项目A第一期" in old and "20日" in old))
                answer = {"conflict_label": "conflict" if conflict else "no_conflict", "reason": "测试桩的预设判定"}
            else:
                raise AssertionError(task)
            result = {"choices": [{"message": {"content": json.dumps(answer, ensure_ascii=False)}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        data = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.requests, self.server.failures = [], 0
        self.server.subject_relation_override = None
        self.server.invalid_selection, self.server.zero_vectors = 0, False
        self.server.batch_drop = 0
        self.server.batch_empty = 0
        self.server.batch_drift = 0
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.settings = Settings(llm_base_url=url, llm_api_key="fake-llm-secret", llm_model="mock-chat",
                                 embedding_base_url=url, embedding_api_key="fake-embedding-secret",
                                 embedding_model="mock-vector", cache_path=self.directory / "cache.sqlite3",
                                 embedding_batch_size=3, max_retries=1)
        self.client = APIClient(self.settings)

    def build(self, mode="independent", batch_size=None):
        return build_index(load_text(ROOT / "examples/memories.jsonl"), self.client,
                           self.directory / "index", mode, batch_size=batch_size)

    def test_complete_pipeline_freezes_old_library_and_reuses_exact_text(self):
        index = self.build()
        paths = [self.directory / "index/index.json", self.directory / "index/vectors.npz"]
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        retriever = Retriever(index, self.client)
        query = read_jsonl(ROOT / "examples/queries.jsonl")[2]
        result = retriever.query(query["query_id"], query["content"], judge=True)
        selected = result["selection"]
        originals = {c["id"]: c["description"] for c in index.catalog}
        self.assertEqual([originals[key] for key in selected["selected_ids"]], ["小林的常住城市"])
        self.assertEqual([item["description"] for item in selected["new_items"]], ["小林的入职培训日期"])
        self.assertEqual(selected["new_items"][0]["value"], "测试桩取值")
        self.assertEqual(result["selection_policy"], "similarity_threshold")
        self.assertEqual(result["similarity_threshold"], 0.7)
        self.assertIsNone(result["coarse_k"])
        self.assertIsNone(result["final_k"])
        self.assertTrue(all(c["coarse_score"] >= 0.7 for c in result["coarse_candidates"]))
        self.assertTrue(all(c["fine_score"] >= 0.7 for c in result["candidates"]))
        self.assertEqual(len({c["memory_id"] for c in result["coarse_candidates"]}),
                         len(result["coarse_candidates"]))
        self.assertTrue({c["memory_id"] for c in result["candidates"]} <= {c["memory_id"] for c in result["coarse_candidates"]})
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
        self.assertNotIn("小林的入职培训日期", [c["description"] for c in index.catalog])
        judges = [json.loads(body["messages"][1]["content"]) for path, body, _ in self.server.requests
                  if path.endswith("completions") and json.loads(body["messages"][1]["content"])["task"] == "judge_conflict"]
        self.assertEqual(set(judges[0]), {"task", "new_content", "old_content"})
        calls = len(self.server.requests)
        again = retriever.query(query["query_id"], query["content"], judge=True)
        self.assertEqual(calls, len(self.server.requests))
        self.assertGreater(again["usage"]["cache_hits"], 0)
        self.assertEqual(again["candidates"], result["candidates"])
        fixed_threshold = retriever.query(query["query_id"], query["content"],
                                          coarse_k=1, final_k=1)
        self.assertEqual(fixed_threshold["similarity_threshold"], 0.7)
        self.assertEqual(fixed_threshold["library_size"], len(index.memories))
        cache = self.settings.cache_path.read_bytes()
        self.assertNotIn(b"fake-llm-secret", cache)
        self.assertNotIn(b"fake-embedding-secret", cache)

    def missed_reuse_memories(self):
        """两条内容近重复的记忆，共用同一事项描述；逐批建库时后一条能看到前一条的条目。"""
        return [Memory("a", "项目A第一期发布会唯一的举办日期是2026年10月20日。", "ref"),
                Memory("b", "项目A第一期发布会唯一的举办日期是2026年10月22日。", "ref")]

    def test_values_are_audited_but_not_indexed_in_all_build_modes(self):
        memories = self.missed_reuse_memories()
        for mode in ("independent", "reuse", "batch_reuse"):
            with self.subTest(mode=mode):
                request_start = len(self.server.requests)
                index = build_index(memories, self.client, self.directory / f"values-{mode}",
                                    mode, batch_size=2 if mode == "batch_reuse" else None)
                # 测试桩取值仅检验字段流向，不代表语义模型生成效果。
                applications = [item for row in index.manifest["extraction"]
                                for item in row["new_items"]]
                self.assertTrue(applications)
                self.assertTrue(all(item["value"] == "测试桩取值" for item in applications))
                self.assertEqual([item["value"] for item in index.manifest["values"]["items"]],
                                 [item["value"] for item in applications])
                self.assertEqual(len(index.catalog), 1)
                self.assertEqual(index.catalog[0]["description"], "项目A第一期发布会的日期")
                self.assertNotIn("value", index.catalog[0])
                self.assertEqual([memory.content for memory in index.memories],
                                 [memory.content for memory in memories])
                for path, body, _ in self.server.requests[request_start:]:
                    if path.endswith("/embeddings"):
                        self.assertNotIn("测试桩取值", body["input"])
                    else:
                        payload = json.loads(body["messages"][1]["content"])
                        for entry in payload.get("catalog", []):
                            self.assertEqual(set(entry), {"id", "description"})


    def test_shared_entry_keeps_all_original_records_and_no_rerank_preserves_pool(self):
        index = self.build()
        entry = next(c for c in index.catalog if c["description"] == "项目A第一期发布会的日期")
        self.assertEqual(entry["memory_ids"], ["m5", "m6"])
        retriever = Retriever(index, self.client)
        q = read_jsonl(ROOT / "examples/queries.jsonl")[1]
        full = retriever.query(q["query_id"], q["content"])
        ablated = retriever.query(q["query_id"], q["content"], rerank=False)
        self.assertEqual(full["coarse_candidates"], ablated["coarse_candidates"])
        self.assertIsNone(ablated["hypotheses"])
        self.assertEqual(ablated["candidates"], ablated["coarse_candidates"])
        independent = retriever.query(q["query_id"], q["content"], method="independent", rerank=False)
        self.assertEqual(independent["selection"]["selected_ids"], [])

    def test_subject_reranking_excludes_incompatible_and_keeps_uncertain(self):
        index = self.build()
        query = read_jsonl(ROOT / "examples/queries.jsonl")[2]
        baseline = Retriever(index, self.client).query(query["query_id"], query["content"])
        self.assertTrue(baseline["coarse_candidates"])
        baseline_ids = [row["memory_id"] for row in baseline["candidates"]]

        self.server.subject_relation_override = "different"
        incompatible_client = APIClient(replace(
            self.settings, cache_path=self.directory / "incompatible-cache.sqlite3"))
        incompatible = Retriever(index, incompatible_client).query(
            query["query_id"], query["content"])
        self.assertEqual(incompatible["candidates"], [])
        self.assertEqual(incompatible["hypotheses"]["subject_relation_counts"]["different"],
                         len(incompatible["coarse_candidates"]))

        self.server.subject_relation_override = "uncertain"
        uncertain_client = APIClient(replace(
            self.settings, cache_path=self.directory / "uncertain-cache.sqlite3"))
        uncertain = Retriever(index, uncertain_client).query(query["query_id"], query["content"])
        self.assertEqual([row["memory_id"] for row in uncertain["candidates"]], baseline_ids)
        self.assertEqual(uncertain["hypotheses"]["subject_relation_counts"]["uncertain"],
                         len(uncertain["coarse_candidates"]))

    def test_personal_owner_metadata_filters_only_final_candidates(self):
        content = "2026年10月13日，小林唯一的工作单位是腾讯。"
        memories = [
            Memory("same-owner", content, {"scope": "personal", "user_id": "U1"}),
            Memory("other-owner", content, {"scope": "personal", "user_id": "U2"}),
            Memory("public", content, {"scope": "world", "user_id": None}),
            Memory("missing-user", content, {"scope": "personal"}),
            Memory("missing-scope", content, {"user_id": "U2"}),
        ]
        index = build_index(memories, self.client, self.directory / "metadata-index", "independent")
        retriever = Retriever(index, self.client)

        result = retriever.query(
            "metadata-query", content, method="dense", coarse_k=len(memories),
            final_k=len(memories), query_source_ref={"scope": "personal", "user_id": "U1"},
        )
        self.assertEqual({row["memory_id"] for row in result["coarse_candidates"]},
                         {memory.memory_id for memory in memories})
        self.assertEqual({row["memory_id"] for row in result["candidates"]},
                         {"same-owner", "public", "missing-user", "missing-scope"})
        self.assertEqual(result["metadata_excluded_memory_ids"], ["other-owner"])

        legacy_metadata = retriever.query(
            "legacy-metadata-query", content, method="dense", coarse_k=len(memories),
            final_k=len(memories), scope="personal", user_id="U1",
        )
        self.assertEqual({row["memory_id"] for row in legacy_metadata["coarse_candidates"]},
                         {memory.memory_id for memory in memories})
        self.assertEqual({row["memory_id"] for row in legacy_metadata["candidates"]},
                         {"same-owner", "public", "missing-user", "missing-scope"})

    def test_baseline_final_k_is_applied_after_metadata_filter(self):
        content = "2026年10月13日，小林唯一的工作单位是腾讯。"
        memories = [
            Memory("a-other-owner", content, {"scope": "personal", "user_id": "U2"}),
            Memory("b-same-owner", content, {"scope": "personal", "user_id": "U1"}),
        ]
        index = build_index(memories, self.client, self.directory / "metadata-final-k", "independent")

        for method in ("dense", "bm25", "hybrid"):
            result = Retriever(index, self.client).query(
                f"metadata-final-k-{method}", content, method=method,
                coarse_k=2, final_k=1,
                query_source_ref={"scope": "personal", "user_id": "U1"},
            )
            self.assertEqual([row["memory_id"] for row in result["coarse_candidates"]],
                             ["a-other-owner", "b-same-owner"])
            self.assertEqual([row["memory_id"] for row in result["candidates"]],
                             ["b-same-owner"])
            self.assertEqual(result["metadata_excluded_memory_ids"], ["a-other-owner"])

    def test_graph_and_text_are_equivalent_and_qualifiers_cannot_be_lost(self):
        text = load_text(ROOT / "examples/memories.jsonl")
        graph = load_graph(ROOT / "examples/graph.json")
        self.assertEqual([(m.memory_id, m.content) for m in text], [(m.memory_id, m.content) for m in graph])
        data = json.loads((ROOT / "examples/graph.json").read_text(encoding="utf-8"))
        result = describe(self.client, "小林的常住城市是北京。", [{"id": "D1", "description": "小林的常住城市"}])
        self.assertEqual(result["selected_ids"], ["D1"])
        self.assertEqual(result["new_items"], [])
        self.assertEqual(sum(e["status"] == "invalid_output" for e in self.client.events), 0)
        self.server.invalid_selection = 2
        with self.assertRaisesRegex(APIError, "一次格式修复"):
            describe(self.client, "小林的常住城市是上海。", [{"id": "D1", "description": "小林的常住城市"}])
        index = self.build()
        with self.assertRaisesRegex(ValueError, "向量服务"):
            index.check_embedding(replace(self.settings, embedding_model="different"))
        path = self.directory / "index/index.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["catalog"][0]["description"] = "被修改的描述"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "已改变"):
            MemoryIndex.load(self.directory / "index")

    def test_empty_descriptions_no_writeback_and_build_fails_without_entries(self):
        index = self.build()
        result = Retriever(index, self.client).query("empty", "没有明确可比较事项")
        self.assertEqual(result["status"], "no_descriptions")
        self.assertEqual(result["candidates"], [])
        with self.assertRaisesRegex(ValueError, "未提取到事项"):
            build_index([Memory("empty", "没有明确可比较事项", "ref")], self.client, self.directory / "bad-index")
        self.assertFalse((self.directory / "bad-index").exists())

    def test_baselines_return_budgeted_unique_records_and_bm25_makes_no_api_calls(self):
        self.assertEqual(tokenize("项目A第一期 the Event"), ["项", "目", "a", "第", "一", "期", "the", "event"])
        retriever = Retriever(self.build(), self.client)
        for method in ("bm25", "dense", "hybrid"):
            calls = len(self.server.requests)
            result = retriever.query("q", "项目A第一期发布会的举办日期", method=method, coarse_k=4, final_k=2)
            self.assertEqual(len(result["candidates"]), 2)
            self.assertEqual(len({c["memory_id"] for c in result["candidates"]}), 2)
            if method == "bm25":
                self.assertEqual(calls, len(self.server.requests))

    def test_cli_build_query_evaluate_and_empty_credentials(self):
        env_file = self.directory / "test.env"
        values = {
            "LLM_BASE_URL": self.settings.llm_base_url,
            "LLM_API_KEY": self.settings.llm_api_key, "LLM_MODEL": self.settings.llm_model,
            "EMBEDDING_BASE_URL": self.settings.embedding_base_url,
            "EMBEDDING_API_KEY": self.settings.embedding_api_key,
            "EMBEDDING_MODEL": self.settings.embedding_model,
            "CACHE_PATH": str(self.settings.cache_path), "COARSE_K": "5", "FINAL_K": "3",
        }
        env_file.write_text("\n".join(f"{key}={value}" for key, value in values.items()), encoding="utf-8-sig")
        env = {key: value for key, value in os.environ.items() if key not in values}
        env["PYTHONUTF8"] = "1"

        def run(*args):
            return subprocess.run([sys.executable, "-m", "memory_retrieval", "--env", str(env_file), *map(str, args)],
                                  cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)

        query_help = run("query", "--help")
        self.assertEqual(query_help.returncode, 0)
        self.assertNotIn("--similarity-threshold", query_help.stdout)
        index_dir = self.directory / "cli-index"
        output = self.directory / "result.jsonl"
        report = self.directory / "report.json"
        build = run("build", "--input", ROOT / "examples/memories.jsonl", "--output", index_dir)
        self.assertEqual(build.returncode, 0, build.stderr)
        query = run("query", "--index", index_dir, "--queries", ROOT / "examples/queries.jsonl",
                    "--output", output, "--judge")
        self.assertEqual(query.returncode, 0, query.stderr)
        self.assertEqual(len(read_jsonl(output)), 4)
        metric = run("evaluate", "--gold", ROOT / "examples/gold.jsonl", "--results", output, "--output", report)
        self.assertEqual(metric.returncode, 0, metric.stderr)
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["total_queries"], 4)
        env_file.write_text("LLM_BASE_URL=\nLLM_API_KEY=\nLLM_MODEL=\n", encoding="utf-8")
        calls = len(self.server.requests)
        missing = run("build", "--input", ROOT / "examples/memories.jsonl", "--output", self.directory / "not-built")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("LLM_BASE_URL", missing.stderr)
        self.assertEqual(calls, len(self.server.requests))

    def test_connection_script_uses_real_requests_and_checks_both_services(self):
        env_file = self.directory / "connection.env"
        values = {
            "LLM_BASE_URL": self.settings.llm_base_url,
            "LLM_API_KEY": self.settings.llm_api_key, "LLM_MODEL": self.settings.llm_model,
            "EMBEDDING_BASE_URL": self.settings.embedding_base_url,
            "EMBEDDING_API_KEY": self.settings.embedding_api_key,
            "EMBEDDING_MODEL": self.settings.embedding_model,
        }
        env = {key: value for key, value in os.environ.items() if key not in values}
        env_file.write_text("\n".join(f"{key}={value}" for key, value in values.items()), encoding="utf-8")

        def run():
            return subprocess.run([sys.executable, str(ROOT / "check_api_connection.py"), "--env", str(env_file)],
                                  cwd=self.directory, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)

        for _ in range(2):
            before = len(self.server.requests)
            result = run()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("[成功] LLM", result.stdout)
            self.assertIn("[成功] 向量模型", result.stdout)
            self.assertEqual(len(self.server.requests) - before, 2)
            self.assertNotIn(self.settings.llm_api_key, result.stdout)
        self.server.failures = 1
        before = len(self.server.requests)
        result = run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("[失败] LLM", result.stdout)
        self.assertIn("HTTP 500", result.stdout)
        self.assertIn("[成功] 向量模型", result.stdout)
        self.assertEqual(len(self.server.requests) - before, 2)

    def test_old_library_reuse_mode_and_duplicate_ids(self):
        index = self.build(mode="reuse")
        self.assertEqual(index.manifest["description_mode"], "reuse")
        self.assertTrue(index.manifest["extraction"][1]["selected_ids"])
        with self.assertRaisesRegex(ValueError, "唯一"):
            build_index([Memory("a", "x", "x"), Memory("a", "y", "y")], self.client, self.directory / "duplicate")

    def test_no_repair_is_respected_by_independent_and_reuse_builds(self):
        memories = load_text(ROOT / "examples/memories.jsonl")
        with patch("memory_retrieval.index.describe", wraps=describe) as describe_spy:
            for mode in ("independent", "reuse"):
                with self.subTest(mode=mode):
                    index = build_index(memories, self.client, self.directory / f"index-{mode}-no-repair",
                                        mode, repair_rounds=0)
                    calls = describe_spy.call_args_list
                    self.assertEqual(len(calls), len(memories))
                    self.assertTrue(all(call.kwargs["check"] is False for call in calls))
                    describe_spy.reset_mock()

    def test_quality_repair_remains_enabled_for_reuse_builds(self):
        memories = load_text(ROOT / "examples/memories.jsonl")
        with patch("memory_retrieval.index.describe", wraps=describe) as describe_spy:
            index = build_index(memories, self.client, self.directory / "index-reuse-repair",
                                "reuse", repair_rounds=1)
        self.assertTrue(all(call.kwargs["check"] is True for call in describe_spy.call_args_list))

    def test_batch_reuse_mode_covers_every_memory_and_rejects_missing_results(self):
        index = self.build(mode="batch_reuse", batch_size=2)
        self.assertEqual(index.manifest["description_mode"], "batch_reuse")
        self.assertEqual(index.manifest["batch_size"], 2)
        # 每条旧记忆都必须挂到至少一个目录条目上，不能有记忆被漏掉
        linked = {mid for entry in index.catalog for mid in entry["memory_ids"]}
        self.assertEqual(linked, {memory.memory_id for memory in index.memories})
        # 批式复用应恰好发出 ceil(n/batch_size) 次批量请求，且后一批能看到前一批的条目
        payloads = [json.loads(body["messages"][1]["content"]) for path, body, _ in self.server.requests
                    if path.endswith("completions")
                    and json.loads(body["messages"][1]["content"])["task"] == "batch_select_descriptions"]
        self.assertEqual(len(payloads), 4)
        self.assertEqual(payloads[0]["catalog"], [])
        self.assertTrue(any(call["catalog"] for call in payloads[1:]))
        # 某条记忆没有结果时整次建库失败，不静默丢数据（格式修复重试后仍无效则报 APIError）
        self.server.batch_drop = 2
        with self.assertRaisesRegex(APIError, "漏项"):
            build_index(load_text(ROOT / "examples/memories.jsonl"), self.client,
                        self.directory / "batch-bad", "batch_reuse", batch_size=3)
        self.assertFalse((self.directory / "batch-bad").exists())
        with self.assertRaisesRegex(ValueError, "batch_size"):
            build_index(load_text(ROOT / "examples/memories.jsonl"), self.client,
                        self.directory / "batch-bad2", "batch_reuse")

    def test_similar_content_from_different_subjects_is_not_forced_into_one_description(self):
        """主体不同、内容相似的记忆不能被强制统一成同一描述（否则主体区别被抹掉）。"""
        memories = [Memory("a", "项目A第一期发布会唯一的举办日期是2026年10月20日。",
                           {"scope": "personal", "user_id": "U1"}),
                    Memory("b", "项目A第一期发布会唯一的举办日期是2026年10月22日。",
                           {"scope": "personal", "user_id": "U2"})]
        self.server.batch_drift = 2
        index = build_index(memories, self.client, self.directory / "diagnostic", "batch_reuse",
                            batch_size=2)
        quality = index.manifest["quality"]
        # 相似度只留诊断，不再触发重写
        self.assertEqual(quality["repaired_batches"], 0)
        self.assertEqual(quality["remaining_by_rule"], {"same_event_split": 1})
        # 两个主体各自保留描述，没有被并成一条
        self.assertEqual(len(index.catalog), 2)

    def test_repair_note_does_not_ask_for_unifying_merely_similar_descriptions(self):
        """相似度诊断与缺失描述同时出现时：只为缺失重写，且提示词不得要求统一相似描述。"""
        memories = [Memory("a", "项目A第一期发布会唯一的举办日期是2026年10月20日。",
                           {"scope": "personal", "user_id": "U1"}),
                    Memory("b", "项目A第一期发布会唯一的举办日期是2026年10月22日。",
                           {"scope": "personal", "user_id": "U2"}),
                    Memory("c", "项目A第一期发布会唯一的举办日期是2026年10月24日。",
                           {"scope": "personal", "user_id": "U3"})]
        self.server.batch_drift = 1   # a/b 内容相似但描述不同 -> 只有诊断
        self.server.batch_empty = 1   # c 没有任何描述 -> 可判定违规，触发重写
        index = build_index(memories, self.client, self.directory / "repair-note",
                            "batch_reuse", batch_size=3)
        quality = index.manifest["quality"]
        self.assertEqual(quality["repaired_batches"], 1)
        self.assertEqual(quality["initial_by_rule"].get("missing"), 1)
        self.assertGreaterEqual(quality["initial_by_rule"].get("same_event_split", 0), 1)
        notes = []
        for path, body, _ in self.server.requests:
            if not path.endswith("completions"):
                continue
            messages = body.get("messages") or []
            if len(messages) >= 4 and json.loads(messages[1]["content"]).get("task") == "batch_select_descriptions":
                notes.append(messages[-1]["content"])
        self.assertTrue(notes, "应当发生一次重写")
        note = notes[-1]
        # 重写只针对可判定问题，不能借机要求把相似记录并成一份描述
        self.assertIn("只有主体、待比较事项和必要实例范围均一致时", note)
        self.assertIn("内容相似只能作为检查线索", note)
        self.assertNotIn("内容近重复的记忆属于同一事项", note)

    def test_stale_retrieval_text_index_is_rejected_by_vector_methods(self):
        """旧索引的旧记忆向量不带主体上下文；用它做向量相似度的方法必须拒绝加载。"""
        self.build()
        path = self.directory / "index/index.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.pop("index_id")
        meta.pop("retrieval_text_version")  # 复现改造前的索引
        meta["index_id"] = fingerprint(meta)
        path.write_text(json.dumps(meta), encoding="utf-8")
        stale = MemoryIndex.load(self.directory / "index")
        retriever = Retriever(stale, self.client)
        for method in ("dense", "hybrid", "proposed"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "检索文本表示已变更"):
                    retriever.query("q", "小林的常住城市是北京。", method=method)
        # bm25 不读旧记忆向量，检索文本由 manifest 现算，仍然可用
        self.assertEqual(retriever.query("q", "小林的常住城市", method="bm25")["status"], "ok")

    def test_merged_entries_stay_queryable_through_their_aliases(self):
        memories = [Memory("a", "项目A第一期发布会唯一的举办日期是2026年10月20日。", "ref"),
                    Memory("b", "项目A第一期发布会唯一的举办日期是2026年10月22日。", "ref")]
        # 重写轮也压不住同一种漂移：留下两套措辞，交给建库后的归并收成一个条目
        self.server.batch_drift = 2
        index = build_index(memories, self.client, self.directory / "index-merge", "batch_reuse",
                            batch_size=2, description_threshold=DESCRIPTION_DUPLICATE_THRESHOLD,
                            template_threshold=MEMORY_TEMPLATE_THRESHOLD)
        self.assertEqual(index.manifest["quality"]["remaining_by_rule"], {"same_event_split": 1})
        self.assertEqual(len(index.catalog), 1)
        self.assertEqual(len(index.catalog[0]["aliases"]), 2)
        self.assertEqual(len(index.description_vectors), 2)
        # 归并后两条记忆仍然都能被检索到（别名向量参与打分）
        result = Retriever(index, self.client).query("q", "项目A第一期发布会唯一的举办日期是2026年10月23日。")
        self.assertEqual({c["memory_id"] for c in result["coarse_candidates"]}, {"a", "b"})
        self.assertEqual({c["memory_id"] for c in result["candidates"]}, {"a", "b"})

    def test_batch_empty_result_is_repaired_before_falling_back(self):
        self.server.batch_empty = 1
        index = build_index(load_text(ROOT / "examples/memories.jsonl"), self.client,
                            self.directory / "index-fallback", "batch_reuse", batch_size=4)
        # 空结果先判为违规并重写一轮，这一轮就修好了，不需要退回单条抽取
        self.assertEqual(index.manifest["batch_fallbacks"], [])
        self.assertEqual(index.manifest["quality"]["initial_by_rule"], {"missing": 1})
        self.assertEqual(index.manifest["quality"]["remaining_by_rule"], {})
        self.assertEqual(index.manifest["quality"]["repaired_batches"], 1)
        linked = {mid for entry in index.catalog for mid in entry["memory_ids"]}
        self.assertEqual(linked, {memory.memory_id for memory in index.memories})

    def test_repair_round_still_empty_falls_back_so_nothing_is_lost(self):
        # 首轮和重写轮都返回空，才退回单条抽取；这条记忆仍然有检索入口
        self.server.batch_empty = 2
        index = build_index(load_text(ROOT / "examples/memories.jsonl"), self.client,
                            self.directory / "index-fallback2", "batch_reuse", batch_size=4)
        self.assertEqual(len(index.manifest["batch_fallbacks"]), 1)
        self.assertTrue(all(item["value"] == "测试桩取值"
                            for row in index.manifest["extraction"]
                            for item in row["new_items"]))
        # 兜底把这一行整个换掉了，原批次对它的 missing 已经失效，不能继续留在统计里
        self.assertEqual(index.manifest["quality"]["remaining_by_rule"], {})
        linked = {mid for entry in index.catalog for mid in entry["memory_ids"]}
        self.assertEqual(linked, {memory.memory_id for memory in index.memories})

    def test_fallback_does_not_force_quality_rewrite_when_repair_is_off(self):
        """兜底必须沿用 --no-repair 开关，不能自己多发一次重写调用、改变目录。"""
        catalog = [{"id": "D000001", "description": "项目A第一期发布会的日期", "memory_ids": ["m"]}]
        memory = Memory("x", "项目A第一期发布会的举办地点是B厅。", "ref")
        result = _fallback_descriptions(self.client, memory, catalog, repair=False)
        self.assertEqual(result["violations"], [])
        selects = [body for path, body, _ in self.server.requests
                   if path.endswith("completions")
                   and json.loads(body["messages"][1]["content"])["task"] == "select_descriptions"]
        self.assertEqual(len(selects), 1)

