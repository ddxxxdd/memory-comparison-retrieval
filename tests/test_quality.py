import unittest

import numpy as np

from memory_retrieval import quality


def unit(rows):
    matrix = np.asarray(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class AuditTests(unittest.TestCase):
    CONTENT = "项目A第一期发布会唯一的举办日期是2026年10月20日。"

    def test_missing_degenerate_and_too_long_are_rejected(self):
        self.assertEqual([v["rule"] for v in quality.audit([self.CONTENT], [{"index": 0, "descriptions": []}])],
                         ["missing"])
        self.assertEqual([v["rule"] for v in quality.audit([self.CONTENT],
                                                           [{"index": 0, "descriptions": ["。，"]}])],
                         ["degenerate"])
        self.assertEqual([v["rule"] for v in quality.audit([self.CONTENT],
                                                           [{"index": 0, "descriptions": ["很长" * 120]}])],
                         ["too_long"])
        self.assertEqual(quality.audit([self.CONTENT], [{"index": 0, "descriptions": ["项目A第一期发布会日期"]}]), [])

    def test_near_duplicate_memories_must_share_a_description(self):
        contents = [self.CONTENT, "项目A第一期发布会唯一的举办日期是2026年10月22日。"]
        rows = [{"index": 0, "descriptions": ["模板日期与20号"]},
                {"index": 1, "descriptions": ["模板日期与22号"]}]
        vectors = unit([[1.0, 0.0], [0.95, 0.05]])
        # 20 号/22 号只写在描述里 -> 文本不同但内容近重复,记为诊断项
        violations = quality.audit(contents, rows, content_vectors=vectors, template_threshold=0.9)
        self.assertEqual([v["rule"] for v in violations], ["same_event_split"])
        self.assertEqual(violations[0]["index"], 0)
        self.assertEqual(violations[0]["other_index"], 1)
        # 相似度只作诊断：不能凭它要求模型把两条描述统一起来
        self.assertEqual(quality.enforceable(violations), [])
        self.assertEqual([v["rule"] for v in quality.enforceable(
            violations + [{"index": 0, "rule": "missing"}])], ["missing"])
        # 共用同一条描述就不是碎片
        rows[1]["descriptions"] = ["模板日期与20号"]
        self.assertEqual(quality.audit(contents, rows, content_vectors=vectors, template_threshold=0.9), [])
        # 相似度没到阈值就不判
        far = unit([[1.0, 0.0], [0.0, 1.0]])
        rows[1]["descriptions"] = ["模板日期与22号"]
        self.assertEqual(quality.audit(contents, rows, content_vectors=far, template_threshold=0.9), [])

    def test_summarize_counts_rules(self):
        summary = quality.summarize([{"rule": "missing"}, {"rule": "missing"}, {"rule": "too_long"}], 3)
        self.assertEqual(summary, {"memories": 3, "violations": 3, "by_rule": {"missing": 2, "too_long": 1}})


class MergeEntryTests(unittest.TestCase):
    def catalog(self):
        return [
            {"id": "D000001", "description": "项目A第一期发布会的日期", "memory_ids": ["m1", "m2"]},
            {"id": "D000002", "description": "项目A第一期发布会日期", "memory_ids": ["m3"]},
            {"id": "D000003", "description": "小林的常住城市", "memory_ids": ["m4"]},
        ]

    def merge(self, catalog, descriptions, contents, positions):
        return quality.merge_entries(catalog, unit(descriptions), unit(contents), positions,
                                     description_threshold=0.95, template_threshold=0.90)

    def test_same_wording_entries_collapse_and_keep_every_memory(self):
        catalog = self.catalog()
        merged, report = self.merge(catalog, [[1, 0, 0], [0.99, 0.14, 0], [0, 0, 1]],
                                    [[1, 0], [0.9, 0.1], [0.95, 0.05], [0, 1]],
                                    {"m1": 0, "m2": 1, "m3": 2, "m4": 3})
        self.assertEqual([entry["id"] for entry in merged], ["D000001", "D000002"])
        # 条目文本取复用最多的那一份：D000001 挂了 2 条记忆
        self.assertEqual(merged[0]["description"], "项目A第一期发布会的日期")
        self.assertEqual(merged[0]["memory_ids"], ["m1", "m2", "m3"])
        self.assertEqual(merged[1]["description"], "小林的常住城市")
        self.assertEqual(report["merged_count"], 1)
        self.assertEqual((report["entries_before"], report["entries_after"]), (3, 2))
        self.assertEqual(report["merged_groups"][0]["rules"], ["description_duplicate"])
        self.assertEqual({m for entry in merged for m in entry["memory_ids"]},
                         {"m1", "m2", "m3", "m4"})

    def test_merged_entries_keep_every_description_as_an_alias(self):
        # 归并只改目录呈现方式：被收掉的措辞必须仍作为别名参与打分，否则可达性会下降
        catalog = self.catalog()
        merged, _ = self.merge(catalog, [[1, 0, 0], [0.99, 0.14, 0], [0, 0, 1]],
                               [[1, 0], [0.9, 0.1], [0.95, 0.05], [0, 1]],
                               {"m1": 0, "m2": 1, "m3": 2, "m4": 3})
        aliases = sorted(row for entry in merged for row in entry["aliases"])
        self.assertEqual(aliases, [0, 1, 2])
        self.assertEqual(merged[0]["aliases"], [0, 1])
        self.assertEqual(merged[1]["aliases"], [2])
        # aliases 第一项必须是条目对外呈现的那一行（查询时要拿它取向量）
        self.assertEqual(merged[0]["description"], catalog[merged[0]["aliases"][0]]["description"])

    def test_canonical_alias_comes_first_even_when_it_is_the_second_entry(self):
        catalog = [{"id": "D000001", "description": "短描述", "memory_ids": ["m1"]},
                   {"id": "D000002", "description": "更常用的描述", "memory_ids": ["m2", "m3"]}]
        merged, _ = self.merge(catalog, [[1, 0], [0.99, 0.14]], [[1, 0], [1, 0.1]],
                               {"m1": 0, "m2": 1, "m3": 1})
        self.assertEqual(len(merged), 1)
        # 挂记忆更多的是第二条，因此它成为条目文本，并且排在 aliases 第一位
        self.assertEqual(merged[0]["description"], "更常用的描述")
        self.assertEqual(merged[0]["aliases"], [1, 0])

    def test_different_values_of_one_event_are_reconciled_by_their_memories(self):
        catalog = [{"id": "D000001", "description": "甲事项", "memory_ids": ["m1"]},
                   {"id": "D000002", "description": "乙事项", "memory_ids": ["m2"]}]
        merged, report = self.merge(catalog, [[1, 0], [0, 1]], [[1, 0], [1, 0.1]],
                                    {"m1": 0, "m2": 1})
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["memory_ids"], ["m1", "m2"])
        self.assertEqual(merged[0]["aliases"], [0, 1])
        self.assertEqual(report["merged_groups"][0]["rules"], ["memory_template"])

    def test_unrelated_entries_stay_separate(self):
        catalog = [{"id": "D000001", "description": "甲事项", "memory_ids": ["m1"]},
                   {"id": "D000002", "description": "乙事项", "memory_ids": ["m2"]}]
        merged, report = self.merge(catalog, [[1, 0], [0, 1]], [[1, 0], [0, 1]],
                                    {"m1": 0, "m2": 1})
        self.assertEqual([entry["id"] for entry in merged], ["D000001", "D000002"])
        self.assertEqual([entry["aliases"] for entry in merged], [[0], [1]])
        self.assertEqual(report["merged_count"], 0)
        self.assertEqual(report["links"], [])

    def test_shared_memory_alone_does_not_force_a_merge(self):
        # 同一条记忆同时提供"日期"和"地点"，两个条目共享它并不代表同一事项被拆开
        catalog = [{"id": "D000001", "description": "项目A第一期发布会的日期", "memory_ids": ["m5", "m6"]},
                   {"id": "D000002", "description": "项目A第一期发布会的地点", "memory_ids": ["m6"]}]
        merged, report = self.merge(catalog, [[1, 0], [0, 1]],
                                    [[1, 0], [0.5, 0.866]],
                                    {"m5": 0, "m6": 1})
        self.assertEqual(len(merged), 2)
        self.assertEqual(report["merged_count"], 0)

    def test_single_entry_is_returned_untouched(self):
        catalog = [{"id": "D000001", "description": "甲事项", "memory_ids": ["m1"]}]
        merged, report = self.merge(catalog, [[1, 0]], [[1, 0]], {"m1": 0})
        self.assertEqual(merged, [{"id": "D000001", "description": "甲事项",
                                   "memory_ids": ["m1"], "aliases": [0]}])
        self.assertEqual(report["entries_after"], 1)


if __name__ == "__main__":
    unittest.main()
