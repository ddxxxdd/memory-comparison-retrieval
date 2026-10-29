import unittest
from unittest.mock import Mock

from memory_retrieval.pipeline import (
    _candidate_may_match, _comparison_subject_context, _validate_subject_relations,
)
from memory_retrieval.prompts import (HYPOTHESIS_PROMPT, REPAIR_NOTE,
                                      SUBJECT_RELATION_BATCH_SIZE, _new_items,
                                      _selected_ids, _string_list, hypotheses)


class SelectionIdTests(unittest.TestCase):
    def test_plain_ids_and_the_catalog_object_shape_are_both_accepted(self):
        # 模型偶尔照抄目录条目形状回 {"id": ...}；语义就是那个编号。
        self.assertEqual(_selected_ids(["D000001", "D000002"], "selected_ids"),
                         ["D000001", "D000002"])
        self.assertEqual(_selected_ids([{"id": "D000003"}], "selected_ids"), ["D000003"])
        self.assertEqual(_selected_ids(["D000004", {"id": "D000004"}], "selected_ids"), ["D000004"])

    def test_other_shapes_still_fail_so_the_rewrite_round_can_fix_them(self):
        for bad in ([{"description": "x"}], [{"id": ""}],
                    [{"id": "D000001", "description": "x"}], [None], [""]):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "selected_ids"):
                    _selected_ids(bad, "selected_ids")
        with self.assertRaisesRegex(ValueError, "必须是数组"):
            _selected_ids("D000001", "selected_ids")

    def test_object_id_shape_is_not_accepted_as_a_hypothesis(self):
        client = Mock()
        client.json_call.side_effect = lambda task, system, payload, validate: validate(
            {"same_value": [{"id": "D000001"}], "alternative": [], "subject_relations": []})
        with self.assertRaisesRegex(ValueError, "same_value"):
            hypotheses(client, "新记忆", ["已有事项"])


class RepairPromptTests(unittest.TestCase):
    def test_content_similarity_alone_does_not_define_the_same_matter(self):
        self.assertIn("主体、待比较事项和必要实例范围", REPAIR_NOTE)
        self.assertIn("内容相似只能作为检查线索", REPAIR_NOTE)
        self.assertNotIn("内容近重复的记忆属于同一事项", REPAIR_NOTE)


class SubjectRelationTests(unittest.TestCase):
    def test_multi_matter_relation_uses_conservative_summary(self):
        self.assertIn("至少一项事项配对的 matter_relation 为 uncertain", HYPOTHESIS_PROMPT)
        self.assertIn("所有对应事项的主体均明确不同", HYPOTHESIS_PROMPT)
        self.assertIn("subject_relation 必须为 uncertain", HYPOTHESIS_PROMPT)

    def test_large_candidate_lists_are_batched_and_relations_are_merged(self):
        candidates = [{"memory_id": f"M{i:06d}", "content": f"旧记忆 {i}"}
                      for i in range(2 * SUBJECT_RELATION_BATCH_SIZE + 1)]
        calls = []
        client = Mock()

        def respond(task, system, payload, validate):
            calls.append(payload)
            answer = {
                "same_value": ["同值改述"],
                "alternative": [],
                "subject_relations": [
                    {"memory_id": item["memory_id"], "comparisons": [{
                        "new_matter": "事项", "old_matter": "事项",
                        "matter_relation": "corresponding",
                        "new_subject": {"kind": "unknown", "identity": "", "basis": "测试桩"},
                        "old_subject": {"kind": "unknown", "identity": "", "basis": "测试桩"},
                        "subject_relation": "uncertain", "reason": "测试桩的不确定关系"}]}
                    for item in payload["candidates"]],
            }
            return validate(answer)

        client.json_call.side_effect = respond
        result = hypotheses(client, "新记忆", ["事项"], candidates)

        self.assertEqual([len(call["candidates"]) for call in calls],
                         [SUBJECT_RELATION_BATCH_SIZE, SUBJECT_RELATION_BATCH_SIZE, 1])
        self.assertEqual([item["memory_id"] for item in result["subject_relations"]],
                         [item["memory_id"] for item in candidates])
        self.assertEqual(result["subject_relation_counts"], {
            "same": 0, "different": 0, "uncertain": len(candidates)})
        self.assertEqual(result["probes"], ["新记忆", "同值改述"])

    def test_hypothesis_payload_keeps_subject_context_separate_from_content(self):
        calls = []
        client = Mock()

        def respond(task, system, payload, validate):
            calls.append(payload)
            return validate({"same_value": [], "alternative": [], "subject_relations": [{
                "memory_id": "old", "comparisons": [{
                    "new_matter": "个人偏好", "old_matter": "个人偏好",
                    "matter_relation": "corresponding",
                    "new_subject": {"kind": "self", "identity": "self", "basis": "正文偏好"},
                    "old_subject": {"kind": "self", "identity": "self", "basis": "正文偏好"},
                    "subject_relation": "uncertain", "reason": "待校验"}]}
            ]})

        client.json_call.side_effect = respond
        hypotheses(client, "Preferred grocery store is Walmart", ["个人偏好"],
                   [{"memory_id": "old", "content": "Preferred grocery store is Whole Foods",
                     "subject_context": "用户 B"}], subject_context="用户 A")

        self.assertEqual(calls[0]["content"], "Preferred grocery store is Walmart")
        self.assertEqual(calls[0]["subject_context"], "用户 A")
        self.assertEqual(calls[0]["candidates"][0]["content"],
                         "Preferred grocery store is Whole Foods")
        self.assertEqual(calls[0]["candidates"][0]["subject_context"], "用户 B")


class SubjectContextValidationTests(unittest.TestCase):
    @staticmethod
    def normalize(new_kind="self", old_kind="self", relation="different",
                  query_ref=None, old_ref=None):
        comparison = {
            "new_matter": "个人偏好", "old_matter": "个人偏好",
            "matter_relation": "corresponding",
            "new_subject": {"kind": new_kind,
                            "identity": "self" if new_kind == "self" else "",
                            "basis": "测试正文"},
            "old_subject": {"kind": old_kind,
                            "identity": "self" if old_kind == "self" else "",
                            "basis": "测试正文"},
            "subject_relation": relation, "reason": "测试桩关系",
        }
        result, issues = _validate_subject_relations(
            [{"memory_id": "old", "comparisons": [comparison]}], query_ref,
            [{"memory_id": "old", "source_ref": old_ref}],
        )
        return result[0]["comparisons"], issues

    def test_same_owner_metadata_overrides_model_subject_relation(self):
        personal_a = {"scope": "personal", "user_id": "A"}
        comparisons, issues = self.normalize(query_ref=personal_a, old_ref=personal_a)

        self.assertEqual(comparisons[0]["subject_relation"], "same")
        self.assertEqual(comparisons[0]["model_subject_relation"], "different")
        self.assertEqual(comparisons[0]["subject_relation_resolution"], "personal_owner_metadata")
        self.assertEqual(issues, [])
        self.assertTrue(_candidate_may_match(comparisons))

    def test_different_self_owners_are_resolved_from_metadata_but_left_for_final_filter(self):
        comparisons, issues = self.normalize(
            relation="uncertain", query_ref={"scope": "personal", "user_id": "A"},
            old_ref={"scope": "personal", "user_id": "B"})

        self.assertEqual(comparisons[0]["subject_relation"], "different")
        self.assertEqual(comparisons[0]["subject_relation_resolution"], "personal_owner_metadata")
        self.assertEqual(issues, [])
        self.assertTrue(_candidate_may_match(comparisons))

    def test_missing_context_keeps_self_relation_uncertain(self):
        comparisons, issues = self.normalize(
            query_ref=None, old_ref={"scope": "personal", "user_id": "B"})

        self.assertEqual(comparisons[0]["subject_relation"], "uncertain")
        self.assertEqual(issues, [])
        self.assertTrue(_candidate_may_match(comparisons))

    def test_unknown_subject_cannot_support_a_definitive_relation(self):
        personal_a = {"scope": "personal", "user_id": "A"}
        comparisons, issues = self.normalize(
            new_kind="unknown", old_kind="self", query_ref=personal_a, old_ref=personal_a)

        self.assertEqual(comparisons[0]["subject_relation"], "uncertain")
        self.assertEqual(len(issues), 1)
        self.assertTrue(_candidate_may_match(comparisons))

    def test_model_context_for_personal_memories_does_not_expose_user_ids(self):
        self.assertEqual(
            _comparison_subject_context({"scope": "personal", "user_id": "PRIVATE-ID"}),
            "个人记忆的记录者本人",
        )
        self.assertEqual(_comparison_subject_context({"scope": "world", "user_id": None}),
                         "公共事实")
        self.assertIsNone(_comparison_subject_context({"user_id": "PRIVATE-ID"}))


class NewItemShapeTests(unittest.TestCase):
    def test_only_value_and_description_are_allowed(self):
        self.assertEqual(_new_items([{"value": "12月10日", "description": "考试日期"}], "new_items"),
                         [{"value": "12月10日", "description": "考试日期"}])
        with self.assertRaisesRegex(ValueError, "未约定的字段"):
            _new_items([{"value": "v", "description": "d", "contrast_id": "D000001"}], "new_items")


if __name__ == "__main__":
    unittest.main()
