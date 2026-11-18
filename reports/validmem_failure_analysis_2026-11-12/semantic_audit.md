# 语义审阅台账

逐条读取原文后的单助手审阅，不是独立人工标注。`text_supported_non_conflict` 指按本地 v7 定义、正文支持非冲突；不证明模型内部错误机制。`unresolved` 与 `possible_label_gap` 不计为已确认负例。排名顺序统一为 BM25 / Dense / Hybrid / SparseCL；`>50` 为截断，非第 51 名。

## 1. TC-A-0001 × M-A-0171

- 新记忆：Peter Evans的住家地址更新为北京市海淀区中关村大街101号
- 历史记忆：Frank Owen的住址是北京市海淀区中关村一街5号
- 排名：1 / 2 / 2 / 3
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 2. TC-A-0002 × M-A-0411

- 新记忆：Quinn Evans's preferred grocery store changed to Trader Joe's
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 3. TC-A-0005 × M-A-0390

- 新记忆：Tina Evans's preferred grocery store changed to Costco.
- 历史记忆：Chloe Garcia's preferred grocery store changed to Costco.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 4. TC-A-0006 × M-A-0059

- 新记忆：Uma Evans's preferred grocery store is Whole Foods
- 历史记忆：Mia Fisher's preferred grocery store is Whole Foods
- 排名：1 / 3 / 1 / 3
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 5. TC-A-0010 × M-A-0012

- 新记忆：Yara Evans changed preferred grocery store to Trader Joe's.
- 历史记忆：Uma Evans's preferred grocery store is Trader Joe's
- 排名：2 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 6. TC-A-0010 × M-A-0411

- 新记忆：Yara Evans changed preferred grocery store to Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 2 / 2 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 7. TC-A-0011 × M-A-0390

- 新记忆：Zack Evans changed preferred grocery store to Costco.
- 历史记忆：Chloe Garcia's preferred grocery store changed to Costco.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 8. TC-A-0014 × M-A-0012

- 新记忆：Chloe Evans changed preferred grocery store to Trader Joe's.
- 历史记忆：Uma Evans's preferred grocery store is Trader Joe's
- 排名：2 / 2 / 2 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 9. TC-A-0014 × M-A-0411

- 新记忆：Chloe Evans changed preferred grocery store to Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 1 / 1 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 10. TC-A-0015 × M-A-0411

- 新记忆：Derek Evans changed preferred grocery store to Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 11. TC-A-0018 × M-A-0411

- 新记忆：Carol Fisher changed their primary grocery store to Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 12. TC-A-0019 × M-A-0076

- 新记忆：David Fisher changed their gym membership to Elite Fitness.
- 历史记忆：Tina Fisher's preferred gym membership is with Fitness First.
- 排名：1 / 2 / 1 / 3
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 13. TC-A-0026 × M-A-0411

- 新记忆：Karen Fisher changed preferred grocery store to Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 14. TC-A-0027 × M-A-0411

- 新记忆：Sam Jensen changed preferred grocery store to Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 15. TC-A-0028 × M-A-0411

- 新记忆：Liam Fisher changed preferred grocery store to Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 2 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 16. TC-A-0028 × M-A-0441

- 新记忆：Liam Fisher changed preferred grocery store to Trader Joe's.
- 历史记忆：Liam Hughes's preferred grocery store is Trader Joe's.
- 排名：2 / 1 / 2 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 17. TC-A-0029 × M-A-0444

- 新记忆：Mia Fisher's preferred grocery store is Trader Joe's
- 历史记忆：Mia Hughes's preferred grocery store is Trader Joe's.
- 排名：1 / 2 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 18. TC-A-0033 × M-A-0411

- 新记忆：Peter Fisher updated preferred grocery store to Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 2 / 1 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 19. TC-A-0034 × M-A-0411

- 新记忆：Quinn Fisher changed preferred grocery store to Trader Joe's
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 20. TC-A-0035 × M-A-0390

- 新记忆：Rachel Fisher's default grocery store changed to Costco
- 历史记忆：Chloe Garcia's preferred grocery store changed to Costco.
- 排名：2 / 1 / 2 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 21. TC-A-0036 × M-A-0411

- 新记忆：Sam Fisher changed preferred grocery store to Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 22. TC-A-0040 × M-A-0001

- 新记忆：Henry Klein的住址更新为上海市徐汇区漕溪北路88号
- 历史记忆：Peter Evans的住家地址是上海市浦东新区花木路88号
- 排名：1 / 4 / 2 / 3
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 23. TC-A-0041 × M-A-0012

- 新记忆：Wendy Fisher's preferred grocery store is now Trader Joe's.
- 历史记忆：Uma Evans's preferred grocery store is Trader Joe's
- 排名：1 / 9 / 3 / 4
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 24. TC-A-0041 × M-A-0411

- 新记忆：Wendy Fisher's preferred grocery store is now Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：6 / 2 / 2 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 25. TC-A-0042 × M-A-0411

- 新记忆：Xavier Fisher changed preferred grocery store to Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 26. TC-A-0045 × M-A-0393

- 新记忆：Amber Fisher's preferred grocery store is GreenGrocers.
- 历史记忆：Derek Garcia's preferred grocery store is GreenGrocers.
- 排名：2 / 2 / 2 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 27. TC-A-0046 × M-A-0012

- 新记忆：Brian Fisher's preferred grocery store is Trader Joe's.
- 历史记忆：Uma Evans's preferred grocery store is Trader Joe's
- 排名：1 / 8 / 2 / 5
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 28. TC-A-0046 × M-A-0441

- 新记忆：Brian Fisher's preferred grocery store is Trader Joe's.
- 历史记忆：Liam Hughes's preferred grocery store is Trader Joe's.
- 排名：3 / 17 / 9 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 29. TC-A-0050 × M-A-0390

- 新记忆：Alice Garcia updated preferred grocery store to Costco.
- 历史记忆：Chloe Garcia's preferred grocery store changed to Costco.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 30. TC-A-0060 × M-A-0068

- 新记忆：Liam Chen changed preferred grocery store to Whole Foods
- 历史记忆：Peter Fisher changed preferred grocery store to Whole Foods.
- 排名：1 / 3 / 1 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 31. TC-A-0071 × M-A-0393

- 新记忆：Henry Garcia updated preferred grocery store to GreenGrocers.
- 历史记忆：Derek Garcia's preferred grocery store is GreenGrocers.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 32. TC-A-0073 × M-A-0001

- 新记忆：Frank Owen的住址更新为上海市浦东新区世纪大道2000号
- 历史记忆：Peter Evans的住家地址是上海市浦东新区花木路88号
- 排名：1 / 3 / 2 / 3
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 33. TC-A-0077 × M-A-0201

- 新记忆：Karen Garcia's latest blood pressure reading is 125/80.
- 历史记忆：Noah Garcia updated blood pressure reading is 125/80 mmHg.
- 排名：1 / 2 / 1 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 34. TC-A-0078 × M-A-0390

- 新记忆：Liam Garcia changed preferred grocery store to Costco
- 历史记忆：Chloe Garcia's preferred grocery store changed to Costco.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 35. TC-A-0081 × M-A-0068

- 新记忆：Brian Jensen updated preferred grocery store to Whole Foods.
- 历史记忆：Peter Fisher changed preferred grocery store to Whole Foods.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 36. TC-A-0082 × M-A-0068

- 新记忆：Ivy Klein changed preferred grocery store to Whole Foods.
- 历史记忆：Peter Fisher changed preferred grocery store to Whole Foods.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 37. TC-A-0085 × M-A-0338

- 新记忆：Peter Garcia's address updated to 456 Oak Avenue, Springfield.
- 历史记忆：Emily's address is 457 Oak Avenue.
- 排名：1 / 2 / 2 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 38. TC-A-0087 × M-A-0213

- 新记忆：Quinn Garcia's preferred grocery store updated to Costco
- 历史记忆：Quinn Garcia's recently purchased apples from Costco
- 排名：4 / 1 / 2 / 3
- 审阅范围：top1_non_gold
- 类型：属性不同或未断言待比较属性值；证据状态：`text_supported_non_conflict`。
- 依据：同一 Quinn Garcia；候选断言从 Costco 买过苹果，未断言其首选商店。购买记录与偏好可并存。

## 39. TC-A-0087 × M-A-0390

- 新记忆：Quinn Garcia's preferred grocery store updated to Costco
- 历史记忆：Chloe Garcia's preferred grocery store changed to Costco.
- 排名：1 / 2 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 40. TC-A-0100 × M-A-0268

- 新记忆：Emma Chen's team lunch rescheduled to Friday 1pm at Cafe Blue
- 历史记忆：Frank Chen's team lunch scheduled on Friday at 1pm at Cafe Blue.
- 排名：1 / 2 / 2 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 41. TC-A-0121 × M-A-0338

- 新记忆：Lily's home address is 456 Oak Avenue.
- 历史记忆：Emily's address is 457 Oak Avenue.
- 排名：1 / 3 / 2 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 42. TC-A-0123 × M-A-0312

- 新记忆：Patient John is allergic only to penicillin.
- 历史记忆：Patient John is allergic only to penicillin.
- 排名：1 / 1 / 1 / 1000
- 审阅范围：top1_non_gold
- 类型：同主体同事项同值；证据状态：`text_supported_non_conflict`。
- 依据：新旧正文完全相同，均为 Patient John 只对 penicillin 过敏；v7 同值不计冲突。

## 43. TC-A-0123 × M-A-0314

- 新记忆：Patient John is allergic only to penicillin.
- 历史记忆：Patient Jane is allergic only to penicillin.
- 排名：3 / 3 / 3 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 44. TC-A-0128 × M-A-0338

- 新记忆：David's home address is 456 Oak Avenue.
- 历史记忆：Emily's address is 457 Oak Avenue.
- 排名：1 / 3 / 2 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 45. TC-A-0129 × M-A-0091

- 新记忆：Emma's favorite coffee shop is Brew Haven.
- 历史记忆：Zack Fisher's favorite coffee shop is 'Coffee Haven'.
- 排名：1 / 3 / 2 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 46. TC-A-0132 × M-A-0338

- 新记忆：Emma's address is 456 Oak Avenue.
- 历史记忆：Emily's address is 457 Oak Avenue.
- 排名：1 / 1 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 47. TC-A-0137 × M-A-0311

- 新记忆：Patient Mara Voss is allergic only to penicillin.
- 历史记忆：Patient Jane is allergic only to penicillin.
- 排名：2 / 3 / 1 / 3
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 48. TC-A-0137 × M-A-0352

- 新记忆：Patient Mara Voss is allergic only to penicillin.
- 历史记忆：Patient Jane Doe is allergic only to penicillin.
- 排名：5 / 1 / 3 / 4
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 49. TC-A-0140 × M-A-0361

- 新记忆：Sam Patel的张伟的微信号是 zhangwei_new2023
- 历史记忆：Sam Patel的张薇的微信号是 zhangwei_001
- 排名：2 / 2 / 2 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：外层 Sam Patel 相同，但查询所指张伟、候选所指张薇不同；需要核对嵌套对象身份。

## 50. TC-A-0141 × M-A-0364

- 新记忆：Sam Quinn's dr. Smith's office hours are Wednesday 2-4 PM.
- 历史记忆：Sam Quinn's dr. Jones' office hours are Wednesday 2-4 PM.
- 排名：1 / 2 / 2 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：外层 Sam Quinn 相同，但查询所指 dr. Smith、候选所指 dr. Jones 不同；相同办公时间不建立同一医生身份。

## 51. TC-A-0146 × M-A-0426

- 新记忆：Zack Garcia's preferred coffee brand has changed to Blue Bottle.
- 历史记忆：Grace Hughes's favorite coffee brand changed to Blue Bottle.
- 排名：2 / 1 / 1 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 52. TC-A-0149 × M-A-0408

- 新记忆：Brian Garcia recommended textbook for Physics is 'University Physics by Young and Freedman'.
- 历史记忆：Bob Hughes's recommended textbook for Physics 101 is 'University Physics' by Young and Freedman.
- 排名：1 / 19 / 8 / 3
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 53. TC-A-0150 × M-A-0012

- 新记忆：Chloe Garcia's preferred grocery store is now Trader Joe's.
- 历史记忆：Uma Evans's preferred grocery store is Trader Joe's
- 排名：1 / 4 / 1 / 5
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 54. TC-A-0150 × M-A-0411

- 新记忆：Chloe Garcia's preferred grocery store is now Trader Joe's.
- 历史记忆：Derek Chen's preferred grocery store changed to Trader Joe's.
- 排名：6 / 1 / 2 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 55. TC-A-0156 × M-A-0387

- 新记忆：Bob Hughes's recommended textbook for Physics 101 is 'Physics for Scientists and Engineers' by Serway and Jewett.
- 历史记忆：Brian Garcia recommended textbook for Physics is 'Physics for Scientists and Engineers by Serway'.
- 排名：1 / 22 / 11 / 4
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 56. TC-A-0157 × M-A-0059

- 新记忆：Derek Chen's preferred grocery store is now Whole Foods.
- 历史记忆：Mia Fisher's preferred grocery store is Whole Foods
- 排名：1 / 5 / 2 / 5
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 57. TC-A-0157 × M-A-0447

- 新记忆：Derek Chen's preferred grocery store is now Whole Foods.
- 历史记忆：Victor Quinn's preferred grocery store is Whole Foods.
- 排名：2 / 3 / 1 / 3
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 58. TC-A-0159 × M-A-0059

- 新记忆：David Hughes's preferred grocery store is Whole Foods.
- 历史记忆：Mia Fisher's preferred grocery store is Whole Foods
- 排名：1 / 6 / 4 / 7
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 59. TC-A-0159 × M-A-0447

- 新记忆：David Hughes's preferred grocery store is Whole Foods.
- 历史记忆：Victor Quinn's preferred grocery store is Whole Foods.
- 排名：2 / 3 / 1 / 4
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 60. TC-A-0161 × M-A-0426

- 新记忆：Frank Hughes's preferred coffee brand is Blue Bottle.
- 历史记忆：Grace Hughes's favorite coffee brand changed to Blue Bottle.
- 排名：1 / 1 / 1 / 2
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 61. TC-A-0167 × M-A-0059

- 新记忆：Liam Hughes's preferred grocery store is Whole Foods.
- 历史记忆：Mia Fisher's preferred grocery store is Whole Foods
- 排名：1 / 8 / 4 / 7
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 62. TC-A-0167 × M-A-0447

- 新记忆：Liam Hughes's preferred grocery store is Whole Foods.
- 历史记忆：Victor Quinn's preferred grocery store is Whole Foods.
- 排名：2 / 3 / 1 / 3
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 63. TC-A-0168 × M-A-0059

- 新记忆：Mia Hughes's preferred grocery store is Whole Foods.
- 历史记忆：Mia Fisher's preferred grocery store is Whole Foods
- 排名：1 / 2 / 1 / 1
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 64. TC-A-0169 × M-A-0012

- 新记忆：Victor Quinn's preferred grocery store is Trader Joe's.
- 历史记忆：Uma Evans's preferred grocery store is Trader Joe's
- 排名：1 / 6 / 2 / 6
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 65. TC-A-0169 × M-A-0417

- 新记忆：Victor Quinn's preferred grocery store is Trader Joe's.
- 历史记忆：David Hughes's preferred grocery store is Trader Joe's.
- 排名：2 / 4 / 1 / 4
- 审阅范围：top1_non_gold
- 类型：具名主体/所指人物不同；证据状态：`text_supported_non_conflict`。
- 依据：逐例核对正文：查询与候选的具名主体或所指人物不同；按 v7 不同姓名不同主体的规则，不构成当前查询的同主体冲突。

## 66. TW-0002 × M-A-0449

- 新记忆：Biology textbook edition is now 5th Edition
- 历史记忆：Noah Hughes's recommended textbook for Biology 101 is 'Introduction to Biology, 3rd Edition'.
- 排名：5 / >50 / 18 / 26
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：候选为 Noah Hughes 的 Biology 101 指定书目；查询未给出个人、课程编号或书名，不能确认版本是否对齐。

## 67. TW-0002 × M-A-0450

- 新记忆：Biology textbook edition is now 5th Edition
- 历史记忆：Noah Hughes's recommended textbook for Biology 101 is 'Biology: Concepts and Connections, 8th Edition'.
- 排名：6 / >50 / 16 / 28
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：候选为 Noah Hughes 的 Biology 101 指定书目；查询未给出个人、课程编号或书名，不能确认版本是否对齐。

## 68. TW-0002 × M-B-0535

- 新记忆：Biology textbook edition is now 5th Edition
- 历史记忆：Recommended textbook for biology is 'Principles of Biology'.
- 排名：7 / 14 / 4 / 11
- 审阅范围：top10_failed_query_non_gold
- 类型：属性不同或未断言待比较属性值；证据状态：`text_supported_non_conflict`。
- 依据：候选只推荐书名 Principles of Biology，没有版本号；查询只更新 Biology 教材版本，两者未给出不相容的同一属性值。

## 69. TW-0002 × M-B-0542

- 新记忆：Biology textbook edition is now 5th Edition
- 历史记忆：The preferred textbook for the math course is 'Advanced Algebra, 5th Edition'.
- 排名：3 / 36 / 9 / 13
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：数学课程教材与查询的 Biology 教材不是同一学科对象。

## 70. TW-0002 × M-C-0081

- 新记忆：Biology textbook edition is now 5th Edition
- 历史记忆：Recommended textbook for Chemistry course: 'Organic Chemistry Principles 5th Edition'.
- 排名：4 / 40 / 10 / 8
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：Chemistry 课程教材与 Biology 教材不同；5th Edition 相同不构成事项相同。

## 71. TW-0002 × M-C-0122

- 新记忆：Biology textbook edition is now 5th Edition
- 历史记忆：New Chemistry textbook recommendation: 'Essential Chemistry Principles, 5th Edition'.
- 排名：2 / 33 / 8 / 7
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：Chemistry 教材与 Biology 教材不同；5th Edition 相同不构成事项相同。

## 72. TW-0002 × M-C-0153

- 新记忆：Biology textbook edition is now 5th Edition
- 历史记忆：Recommended textbook for calculus: 'Advanced Calculus, 5th Ed.'
- 排名：9 / 43 / 15 / 12
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：calculus 教材与 Biology 教材不同；5th Ed. 相同不构成事项相同。

## 73. TW-0002 × M-C-0190

- 新记忆：Biology textbook edition is now 5th Edition
- 历史记忆：Physics 102 now uses Edition 6 instead of Edition 4.
- 排名：8 / 18 / 7 / 9
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：Physics 102 教材与 Biology 教材不同。

## 74. TW-0005 × M-C-0078

- 新记忆：Recommended dosage for Medication A is now 100mg daily
- 历史记忆：Recommended dosage for Medication B was 100mg daily.
- 排名：2 / 4 / 3 / 1
- 审阅范围：top1_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：查询为 Medication A，候选为 Medication B；100mg daily 相同不能建立同一药物对象。

## 75. TW-0009 × M-C-0187

- 新记忆：Park entry fee is now $15 per person
- 历史记忆：Park entry fee increased to $15 starting today.
- 排名：2 / 1 / 2 / 1
- 审阅范围：top1_non_gold
- 类型：同主体同事项同值；证据状态：`text_supported_non_conflict`。
- 依据：两条均给出当前公园门票 15 美元；per person 与 starting today 无相反断言，按现有标签同值。

## 76. TW-0011 × M-B-0142

- 新记忆：History study guide is now valid until day_90
- 历史记忆：Midterm study guide valid until day_90
- 排名：1 / 3 / 2 / 2
- 审阅范围：top1_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：History study guide 与 Midterm study guide 是否为同一份资料没有正文证据；不能仅以名称不同断言不冲突。

## 77. TW-0013 × M-C-0065

- 新记忆：Recommended textbook for physics is now 'Physics Concepts, 2nd Edition'
- 历史记忆：Recommended textbook for calculus is 'Calculus Fundamentals, 2nd Edition'.
- 排名：5 / >50 / 23 / 38
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：calculus 与 physics 教材属于不同学科；2nd Edition 相同。

## 78. TW-0013 × M-C-0126

- 新记忆：Recommended textbook for physics is now 'Physics Concepts, 2nd Edition'
- 历史记忆：Recommended textbook edition: Biology 2nd Edition
- 排名：4 / 45 / 18 / 31
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：Biology 与 physics 教材属于不同学科；2nd Edition 相同。

## 79. TW-0018 × M-A-0235

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：Sam Chen's parent-teacher meeting rescheduled to Friday 9am in Hall C.
- 排名：21 / 12 / 10 / 10
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：Sam Chen 的家长教师会议与无此人物/会议限定的社区活动是不同的明示事项。

## 80. TW-0018 × M-B-0020

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：The neighborhood cleanup event was scheduled to happen.
- 排名：8 / 4 / 4 / 9
- 审阅范围：top10_failed_query_non_gold
- 类型：属性不同或未断言待比较属性值；证据状态：`text_supported_non_conflict`。
- 依据：候选仅称社区清洁活动曾被安排，未给出活动日期；即使是同一活动也未断言与 day_35 不同的时间。

## 81. TW-0018 × M-B-0034

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：Weekend 10% discount promotion active from day 60 to day 62.
- 排名：10 / 41 / 16 / 60
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：周末折扣的生效时间与社区活动举办时间属于不同事项。

## 82. TW-0018 × M-B-0048

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：Community garden cleanup scheduled for Saturday
- 排名：30 / 16 / 15 / 5
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：社区花园清洁可能是社区活动的具体实例，但正文没有实例对齐依据。

## 83. TW-0018 × M-B-0077

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：The neighborhood recycling drive scheduled for the weekend of day 15.
- 排名：5 / 9 / 6 / 14
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：neighborhood recycling drive 与 generic community event 是否为同一活动无法确认。

## 84. TW-0018 × M-B-0107

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：Yoga class rescheduled to day 22
- 排名：6 / 7 / 5 / 6
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：Yoga class 与 generic community event 是否指同一活动缺乏正文限定；不武断归为已确认负例。

## 85. TW-0018 × M-B-0108

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：Community event guidelines for day 15 schedule
- 排名：2 / 2 / 2 / 2
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：guidelines for day 15 schedule 可能只是指南主题，也可能间接断言活动在 day 15；文本不足以确认非冲突。source_ref 标为 guidelines/other 不是语义证明。

## 86. TW-0018 × M-B-0121

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：Community park maintenance scheduled for Monday to Wednesday.
- 排名：19 / 8 / 9 / 15
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：公园维护时间与社区活动举办时间是不同明示事项。

## 87. TW-0018 × M-B-0189

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：Biology quiz rescheduled to day_130
- 排名：7 / 10 / 8 / 7
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：Biology quiz 与社区活动为不同明示事项。

## 88. TW-0018 × M-B-0196

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：Student club meeting scheduled on day_55
- 排名：>50 / 5 / 27 / 8
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：Student club meeting 与 generic community event 是否同一活动缺乏实例依据。

## 89. TW-0018 × M-B-0199

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：History essay deadline is day_35.
- 排名：9 / 31 / 13 / 18
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：历史论文截止日期与社区活动举办日期是不同事项；day_35 字面相同。

## 90. TW-0018 × M-B-0207

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：Community event registration closed on day_45
- 排名：3 / 3 / 3 / 3
- 审阅范围：top10_failed_query_non_gold
- 类型：属性不同或未断言待比较属性值；证据状态：`text_supported_non_conflict`。
- 依据：候选为报名关闭日期，查询为活动举办日期；即使活动相同，这两个日期也可不同。

## 91. TW-0018 × M-C-0119

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：Community yoga class rescheduled to 8 AM every Saturday.
- 排名：4 / 36 / 12 / 4
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：Community yoga class 是否就是查询的 community event 无法确认，周六循环时间与 day_35 也缺乏对应日历。

## 92. TW-0018 × M-C-0230

- 新记忆：Community event rescheduled to the weekend of day_35
- 历史记忆：The exam date is rescheduled to May 22nd due to holiday conflicts.
- 排名：11 / 6 / 7 / 21
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：考试日期与社区活动举办日期是不同明示事项。

## 93. TW-0020 × M-B-0144

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Extra credit assignment available until day_70
- 排名：>50 / 8 / 36 / 15
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：额外学分作业未写课程或具体实例，也未明确是提交截止还是可获取期限；不能确认与 history essay 的关系。

## 94. TW-0020 × M-B-0156

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Chemistry project submission deadline was day_90.
- 排名：8 / 44 / 15 / 26
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：Chemistry project 与 History essay 是不同学科事项。

## 95. TW-0020 × M-B-0171

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Physics homework submission deadline is day_90.
- 排名：9 / 32 / 13 / 27
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：Physics homework 与 History essay 是不同学科事项。

## 96. TW-0020 × M-B-0172

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Math assignment submission deadline was day_60.
- 排名：10 / 49 / 17 / 24
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：Math assignment 与 History essay 是不同学科事项。

## 97. TW-0020 × M-B-0184

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Essay submission deadline for Literature class is on day_50.
- 排名：6 / 31 / 10 / 6
- 审阅范围：top10_failed_query_non_gold
- 类型：事项对象、学科或明示事件不同；证据状态：`text_supported_non_conflict`。
- 依据：Literature class essay 与 History essay 是不同学科事项。

## 98. TW-0020 × M-B-0194

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：History assignment essay due on day_50
- 排名：12 / 3 / 5 / 9
- 审阅范围：top10_failed_query_non_gold
- 类型：疑似金标遗漏；证据状态：`possible_label_gap`。
- 依据：History assignment essay due on day_50 与查询的 History essay submission deadline day_75 直接对齐；现有金标还接受 history research paper 等改述。此对未入金标且没有逐对审计理由，疑似遗漏，仍需维护者确认。

## 99. TW-0020 × M-B-0201

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：History project guidelines available until day_25
- 排名：50 / 15 / 25 / 5
- 审阅范围：top10_failed_query_non_gold
- 类型：属性不同或未断言待比较属性值；证据状态：`text_supported_non_conflict`。
- 依据：候选为 History project 指南可用期限，查询为论文提交截止；两种属性可并存。

## 100. TW-0020 × M-B-0214

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Group project deadline extended to day_130.
- 排名：5 / 14 / 6 / 20
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：Group project 没有课程或项目标识，不能确认是否与 History essay 同一实例。

## 101. TW-0020 × M-B-0343

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Research paper submission window has been extended to 40 days.
- 排名：11 / 9 / 8 / 12
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：Research paper 未指明学科/实例；40 days 是窗口长度，缺少起点，不能直接和 day_75 比较。

## 102. TW-0020 × M-B-0375

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Research paper submission deadline set to end on day 50.
- 排名：7 / 19 / 9 / 17
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：Research paper 未指明学科或实例，无法从正文确认是否就是 History essay。

## 103. TW-0020 × M-B-0376

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Research paper submission deadline has been extended to day 55.
- 排名：3 / 5 / 4 / 3
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：Research paper 未指明学科或实例，无法从正文确认是否就是 History essay。

## 104. TW-0020 × M-B-0391

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Research paper submission deadline is on day 100.
- 排名：13 / 37 / 16 / 10
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：Research paper 未指明学科或实例，无法从正文确认是否就是 History essay。

## 105. TW-0020 × M-B-0392

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Research paper submission deadline was extended to day 90.
- 排名：2 / 6 / 3 / 4
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：Research paper 未指明学科或实例，无法从正文确认是否就是 History essay。

## 106. TW-0020 × M-B-0433

- 新记忆：History essay submission deadline extended to day_75
- 历史记忆：Research project extended for submissions until day 120.
- 排名：33 / 7 / 12 / 11
- 审阅范围：top10_failed_query_non_gold
- 类型：实例范围或断言含义不明；证据状态：`unresolved`。
- 依据：Research project 未指明学科或实例，无法从正文确认是否覆盖 History essay。
