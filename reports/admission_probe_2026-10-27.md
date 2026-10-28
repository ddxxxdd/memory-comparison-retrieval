# 新增准入合约对照（validmem-v7）

- 固定目录：765 条，来自 `C:\Users\dzp\project\paper\memory-comparison-retrieval\runs\indexes\validmem-v7`（未改动）
- 固定批次：5 条，与建库相同的批量合约
- 模型：`deepseek-v4.1-flash`，temperature=0.0
- 样例取自该索引已核实的漏复用记录，另加 2 条按人工预期应为新事项的控制样本
- **这只是机制诊断**：程序能保证输出合约，不能保证模型给出的差异确实成立

## 目录前提待人工核对

控制样本的读数以目录前提成立为条件；以下未确认的前提不能视为已验证。

- 无法自动确认控制样本 control-new-attribute 所依赖的事项不存在于目录，有限关键词未命中；需要人工核对目录前提。
- 无法自动确认控制样本 control-new-instance 所依赖的事项不存在于目录，有限关键词未命中；需要人工核对目录前提。

## M-C-0066　期望 reuse(D000057)

已核实：目录里早有「微积分课程推荐的教材」

> Recommended textbook for calculus is 'Advanced Calculus Principles, 3rd Edition'.

| 合约 | 输出 | 读数 |
|---|---|---|
| 旧 | selected=[]；new=["微积分课程推荐的教材 'Advanced Calculus Principles, 3rd Edition'"] | 新增（漏复用） |
| 新 | selected=[]；「微积分课程推荐的教材」对照 D000057 differs=[] 出口=contrast_reused | 命中预期ID |

- 差异理由：已有项问的是微积分课程推荐的教材，本条给出具体教材名称与版次，属同一事项。
- 实际描述入口：['微积分课程推荐的教材']

## M-C-0024　期望 reuse(D000061)

已核实：目录里早有「化学课程推荐的教材」

> Chemistry textbook recommended: 'Organic Chemistry' (9th edition).

| 合约 | 输出 | 读数 |
|---|---|---|
| 旧 | selected=[]；new=["化学课程推荐的教材 'Organic Chemistry'（第9版）"] | 新增（漏复用） |
| 新 | selected=[]；「化学课程推荐的教材」对照 D000061 differs=[] 出口=contrast_reused | 命中预期ID |

- 差异理由：已有项问的是化学课程推荐的教材，本条给出具体教材名称与版次，属同一事项。
- 实际描述入口：['化学课程推荐的教材']

## M-B-0139　期望 reuse(D000225)

已核实换措辞：本条英文写法对「数学作业的截止日期」

> Homework submission deadline for Math assignment is on day_50.

| 合约 | 输出 | 读数 |
|---|---|---|
| 旧 | selected=['D000376']；new=[] | 复用了其他旧项 |
| 新 | selected=['D000376']；（无新增） | 复用了其他旧项 |

- 实际描述入口：['Homework submission deadline for Math assignment']

## control-new-attribute　期望 new

对照：目录里有「数学作业的截止日期」和「数学考试的时间安排」，但没有微积分考试地点

> The calculus exam room is Room 204 on day_40.

| 合约 | 输出 | 读数 |
|---|---|---|
| 旧 | selected=[]；new=['微积分考试的地点（Room 204）与日期（day_40）'] | 新增 |
| 新 | selected=[]；「微积分考试的地点」对照 D000026 differs=['attribute'] 出口=new | 新增 |

- 差异理由：已有项问的是微积分考试的时间安排，本条问的是考试地点。
- 实际描述入口：['微积分考试的地点']

## control-new-instance　期望 new

对照：物理课期末考试地点

> Physics course final exam location is Room 204.

| 合约 | 输出 | 读数 |
|---|---|---|
| 旧 | selected=[]；new=['物理课程期末考试的地点（Room 204）'] | 新增 |
| 新 | selected=[]；「物理课程期末考试的地点」对照 D000074 differs=['attribute'] 出口=new | 新增 |

- 差异理由：已有项问的是期末考试日期，本条问的是物理课程期末考试地点。
- 实际描述入口：['物理课程期末考试的地点']

## 汇总

| 样例 | 期望 | 旧合约读数 | 新合约读数 |
|---|---|---|---|
| M-C-0066 | reuse | 新增（漏复用） | 命中预期ID |
| M-C-0024 | reuse | 新增（漏复用） | 命中预期ID |
| M-B-0139 | reuse | 复用了其他旧项 | 复用了其他旧项 |
| control-new-attribute | new | 新增 | 新增 |
| control-new-instance | new | 新增 | 新增 |

样本 5 条，只能说明这一批的可见行为，不能推断 765 项目录的整体效果；
新合约的差异理由由同一模型生成，程序不校验其真假。
「未新增（存在复用）」只表示这一条没有产生新增、却复用了别的旧事项；多事项记忆里也可能是遗漏了应新增的事项，不能读作该事项被合并。
