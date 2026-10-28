import json
import unittest
from unittest.mock import Mock

from memory_retrieval.index import value_audit
from memory_retrieval.prompts import _new_items, describe, describe_batch


def application(value="12月10日", description="2026年秋季微积分期末考试的日期"):
    return {"value": value, "description": description}


def client_returning(*outputs):
    responses = iter(outputs)
    client = Mock()
    client.json_call.side_effect = lambda task, system, payload, validate, **kwargs: validate(next(responses))
    return client


class ValueSeparationTests(unittest.TestCase):
    def test_missing_or_invalid_value_is_rejected(self):
        missing = application()
        del missing["value"]
        for item in [missing, *(application(value=value) for value in (None, "", "  ", 10, [], {}))]:
            with self.subTest(item=item):
                with self.assertRaisesRegex(ValueError, r"new_items\.value"):
                    _new_items([item], "new_items")

    def test_description_and_value_are_both_required_and_only_two_fields_allowed(self):
        self.assertEqual(_new_items([application()], "new_items"), [application()])
        extra = application()
        extra["contrast_id"] = "D000001"
        with self.assertRaisesRegex(ValueError, "未约定的字段"):
            _new_items([extra], "new_items")

    def test_batch_keeps_distinct_values_for_a_shared_description(self):
        client = client_returning({"results": [
            {"index": 0, "selected_ids": [], "new_items": [application()]},
            {"index": 1, "selected_ids": [], "new_items": [application(value="12月11日")]},
        ]})
        result = describe_batch(client, ["2026年秋季微积分期末考试在12月10日举行。",
                                         "2026年秋季微积分期末考试在12月11日举行。"], [])
        rows = result["results"]
        self.assertEqual(rows[0]["descriptions"], rows[1]["descriptions"])
        self.assertEqual([row["new_items"][0]["value"] for row in rows], ["12月10日", "12月11日"])
        self.assertEqual(result["violations"], [])
        self.assertEqual(client.json_call.call_count, 1)

    def test_selected_text_and_new_description_are_merged_without_duplicates(self):
        catalog = [{"id": "D000001", "description": "2026年秋季微积分期末考试的日期"}]
        client = client_returning({"selected_ids": ["D000001"], "new_items": [
            application(value="204教室", description="2026年秋季微积分期末考试的地点")]})
        row = describe(client, "2026年秋季微积分期末考试在12月10日于204教室举行。", catalog)
        self.assertEqual(row["descriptions"],
                         ["2026年秋季微积分期末考试的日期", "2026年秋季微积分期末考试的地点"])
        self.assertEqual([item["value"] for item in row["new_items"]], ["204教室"])
        self.assertEqual(row["violations"], [])

    def test_necessary_qualifiers_are_not_removed_by_string_overlap(self):
        item = application(value="2026", description="2026年项目的预算数额")
        client = client_returning({"selected_ids": [], "new_items": [item]})
        row = describe(client, "2026年项目的预算数额是2026。", [])
        self.assertEqual(row["descriptions"], ["2026年项目的预算数额"])
        self.assertEqual(row["new_items"][0]["value"], "2026")
        self.assertEqual(row["violations"], [])

    def test_quality_repair_echo_and_final_audit_keep_values(self):
        first = application(description="异常冗长描述" * 50)
        repaired = application()
        client = client_returning({"selected_ids": [], "new_items": [first]},
                                  {"selected_ids": [], "new_items": [repaired]})
        row = describe(client, "2026年秋季微积分期末考试在12月10日举行。", [], check=True)
        self.assertEqual(client.json_call.call_count, 2)
        echo = json.loads(client.json_call.call_args.kwargs["extra_messages"][0]["content"])
        self.assertEqual(echo["new_items"][0]["value"], "12月10日")
        self.assertEqual(row["new_items"][0]["value"], "12月10日")
        self.assertEqual(row["descriptions"], [repaired["description"]])
        self.assertEqual(row["violations"], [])

    def test_value_audit_records_value_and_description_together(self):
        summary = value_audit([{"memory_id": "m1", "selected_ids": ["D000001"],
                                "new_items": [application()]}])
        self.assertEqual(summary["selected"], 1)
        self.assertEqual(summary["new"], 1)
        self.assertEqual(summary["items"], [{"memory_id": "m1", "value": "12月10日",
                                             "description": "2026年秋季微积分期末考试的日期"}])


if __name__ == "__main__":
    unittest.main()
