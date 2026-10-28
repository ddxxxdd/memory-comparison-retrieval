import unittest
from unittest.mock import Mock

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
        self.assertIn("至少有一项对应事项的事实主体 compatible", HYPOTHESIS_PROMPT)
        self.assertIn("整体就必须为 compatible", HYPOTHESIS_PROMPT)
        self.assertIn("没有 compatible 事项但有对应事项无法判断时，整体为 uncertain", HYPOTHESIS_PROMPT)
        self.assertIn("只有所有可对应事项都明确 incompatible 时", HYPOTHESIS_PROMPT)

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
                    {"memory_id": item["memory_id"], "subject_relation": "uncertain",
                     "reason": "测试桩的不确定关系"}
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
            "compatible": 0, "incompatible": 0, "uncertain": len(candidates)})
        self.assertEqual(result["probes"], ["新记忆", "同值改述"])


class NewItemShapeTests(unittest.TestCase):
    def test_only_value_and_description_are_allowed(self):
        self.assertEqual(_new_items([{"value": "12月10日", "description": "考试日期"}], "new_items"),
                         [{"value": "12月10日", "description": "考试日期"}])
        with self.assertRaisesRegex(ValueError, "未约定的字段"):
            _new_items([{"value": "v", "description": "d", "contrast_id": "D000001"}], "new_items")


if __name__ == "__main__":
    unittest.main()
