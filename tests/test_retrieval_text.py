import unittest

from memory_retrieval.data import Memory, build_retrieval_text, subject_context


class RetrievalTextTests(unittest.TestCase):
    def test_personal_memory_carries_its_owner_as_anonymous_subject(self):
        text = build_retrieval_text("Preferred grocery store is Walmart",
                                    {"scope": "personal", "user_id": "TC-A-0002"})
        self.assertEqual(text, "主体上下文：用户 TC-A-0002。\n记忆内容：Preferred grocery store is Walmart")

    def test_world_memory_keeps_the_public_object_as_subject(self):
        text = build_retrieval_text("Costco closes at 9 PM", {"scope": "world", "user_id": None})
        self.assertEqual(text, "主体上下文：公共事实。\n记忆内容：Costco closes at 9 PM")

    def test_label_fields_never_enter_the_retrieval_text(self):
        # 冲突实体、属性、主题、来源角色都是评测标注，进入输入即等于给出答案。
        ref = {"scope": "world", "user_id": None, "topic": "TOPIC-MARKER",
               "conflict_entity": "ENTITY-MARKER", "conflict_attribute": "ATTR-MARKER",
               "source_role": "ROLE-MARKER"}
        content = "活动政策说明书：各活动过期后不再有效"
        text = build_retrieval_text(content, ref)
        for label in ("TOPIC-MARKER", "ENTITY-MARKER", "ATTR-MARKER", "ROLE-MARKER"):
            self.assertNotIn(label, text)
        self.assertEqual(text, f"主体上下文：公共事实。\n记忆内容：{content}")

    def test_missing_scope_leaves_the_content_untouched(self):
        self.assertEqual(build_retrieval_text("原始内容", {}), "原始内容")
        self.assertEqual(build_retrieval_text("原始内容", "legacy-ref"), "原始内容")
        self.assertIsNone(subject_context({}))

    def test_memory_exposes_the_same_text_used_at_query_time(self):
        memory = Memory("m1", "Preferred grocery store is Walmart",
                        {"scope": "personal", "user_id": "TC-A-0002"})
        self.assertEqual(memory.retrieval_text,
                         build_retrieval_text(memory.content, memory.source_ref))
        # 原文本身不被改写。
        self.assertEqual(memory.content, "Preferred grocery store is Walmart")


if __name__ == "__main__":
    unittest.main()
