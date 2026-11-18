"""Repair the uploaded single-pair pilot and export the project's conflict-only schema.

The semantic decisions below are an assistant's content review, not an automatic
semantic proof or a human audit. No retrieval scores are used by this builder.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_retrieval.adapters import load_text
from memory_retrieval.data import build_retrieval_text, load_queries, read_jsonl
from memory_retrieval.evaluation import validate_gold

DATASET_ID = "single_pair_implicit_v2"
OUTPUT = ROOT / "data" / DATASET_ID
ANCHOR = "叙述时间参照：2026年11月17日。\n"
SOURCE_NAMES = {
    "queries": "single_pair_implicit_queries (1).jsonl",
    "memories": "single_pair_memory_bank (1).jsonl",
}

# Each entry records the semantic body, before the identical clock context is added.
QUERY_EDITS = {
    "q_hc_02": (
        "我一直有严重的晕车毛病，这辈子从没去驾校学过，也没有任何机动车驾驶资格；截至今天，我从未亲自驾驶过机动车。",
        "无驾照仍可能违法驾车；补充从未实际驾驶的事实，冲突不再依赖合法性推断。"),
    "q_hc_03": (
        "因为严重过敏性鼻炎，我租的公寓合同里严禁猫狗进入室内；我搬进来以后，家里从未养过或让人带进过任何活体宠物。",
        "避免把人、昆虫等都包括在活体动物中；以实际未有宠物入室为冲突依据，而非租约禁止。"),
    "q_hc_04": (
        "我个人非常排斥封闭系统，目前我自己的办公电脑、手机和平板安装并运行的操作系统全部都是安卓或 Linux，没有使用其他操作系统。",
        "明确比较的是实际运行的操作系统；设备品牌、硬件和不带系统的配件不作为冲突依据。"),
    "q_ic_01": (
        "我的病历上明确写着对青霉素类药物严重过敏；近一个月，我实际上一次都没有服用过这类抗生素。",
        "禁用与误服可同时成立；改为实际未服用的行为陈述，保留药物类别推理。"),
    "q_ic_02": (
        "医生要求我这半年戒断所有含咖啡因的饮品，我也确实一次都没有喝过这类饮品，日常主要喝白开水和无咖啡因花果茶。",
        "补充实际遵守的事实；不再用只喝两种饮品额外排斥柠檬水等相容记录。"),
    "q_ic_03": (
        "我确诊了乳糜泻，这半年一直严格执行无麸质饮食，任何含麸质的食品都没有吃过。",
        "将核对目标明确为是否摄入麸质；具体小麦食品到麸质仍需单步类别推理。"),
    "q_ic_05": (
        "为了改善失眠，从今年十月一日到今天，我在每晚入睡前两小时都没有看过任何发光电子屏幕，卧室里也没有放过带屏幕的设备。",
        "将这季度明确为已经发生的时间区间和实际行为，避免未来时段或仅表达打算。"),
    "q_bs_01": (
        "我今年一直患有严重的幽闭恐惧症。从今年一月一日到今天，我一次都没有独自进入过地下封闭空间，即使赶时间也会避开这类地方。",
        "删除只坐地面公交的过宽限制，允许出租车、自驾、骑车等地面出行；以实际未独自进入地下为准。"),
    "q_bs_02": (
        "上周五我把摔碎的手机送去官方售后返厂维修了。从那天到今天，每次外出我都没有携带或借用过能联网的设备，也从未在室外操作过这类设备。",
        "排除借用他人设备的替代解释；限制外出时的实际行为，允许室内台式机、设备所有权与历史使用记录。"),
    "q_bs_03": (
        "今年十月一日至八日这八天假期里，我全程待在成都的父母家，期间一步都没有离开过成都市。",
        "原文未说明老家城市，可能就在上海；明确成都，并将八天定义为个人假期而非官方节假日长度。"),
    "q_bs_04": (
        "十一月十二日我左脚踝韧带撕裂打上了石膏。从受伤那天到今天，我的左脚实际上一次都没有着地承重。",
        "医嘱不等于遵守；改为实际未承重，并固定受伤日期，使更早的游泳、散步等记录可相容。"),
}

MEMORY_EDITS = {
    "m_0017": ("以前朋友开车带我出门时，我一般都坐在副驾驶帮他看手机导航。", "将手机导航描述限定为过去的常态，不断言送修期间在室外使用设备。"),
    "m_0023": ("左脚受伤以前，平时出门买菜我都是步行或者骑共享单车，十分钟就能到家。", "明确步行、骑车发生在脚伤前，消除跨组冲突。"),
    "m_0027": ("去年年底我从救助站领回了一只三个月大的柯基幼犬，一直养在我现在租的这套公寓里，每天下班都陪它在客厅玩半小时。", "明确犬是活体宠物且在同一套公寓，排除异地寄养或其他客厅。"),
    "m_0030": ("左脚受伤以前，我每天下班在小区楼下散步时，经常停下来远远看一眼别人家的柴犬。", "将持续散步明确限定到脚伤前。"),
    "m_0036": ("房东人挺好说话，唯一底线就是绝对不能在公寓室内饲养任何活体宠物。", "去掉任何活体动物的字面过宽问题。"),
    "m_0040": ("上个月我刚把自己的主力开发机换成了搭载 M4 芯片的 MacBook Pro，现在每天都直接运行它自带的 macOS 写代码，用着很顺手。", "MacBook 硬件不保证使用 macOS；补明本人当前实际运行的系统。"),
    "m_0042": ("手机送修以前，我出门会带两部安卓手机，一台主力机，一台专门用来测试脚本。", "避免与近五天外出不带联网设备发生跨组冲突。"),
    "m_0045": ("我对随身数码设备的续航要求很高，送修前那台安卓旗舰充一次电能撑一整天。", "不再断言送修期间手机仍在手头正常使用。"),
    "m_0050": ("十一月十一日白天，我在实验室把安卓手机的开发者模式打开，顺手抓了个包看接口延迟。", "将具体手机操作置于送修前，消除白天指代漂移。"),
    "m_0053": ("上周二晚上我买票进了体育中心室内恒温池，在没有浮具也无人托举的情况下独立连续蛙泳了一千五百米。", "这周二晚上相对11月17日尚未发生；改为上周二，早于脚伤，并排除依靠浮具或他人完成。"),
    "m_0054": ("上周二晚上我陪室友去体育中心室内恒温池，自己全程坐在岸边看书等他。", "将同组泳池事件统一为上周二，避免未来事件。"),
    "m_0058": ("上周二晚上我在健身房练了四十分钟上肢力量，出了一身透汗特别舒服。", "统一该组事件日期，落在脚伤之前。"),
    "m_0061": ("左脚受伤以前，我平时锻炼主要靠室内划船机和拉伸垫，完全不沾深水项目。", "明确常规锻炼为受伤前的习惯。"),
    "m_0063": ("上周二晚上的风挺凉快，我在公园塑胶道上散步走了四十分钟。", "将具体步行置于11月10日，早于11月12日脚伤。"),
    "m_0079": ("最近每天早上到工位后，我第一件事就是去楼下买来并喝完一杯用普通非脱因咖啡豆萃取的双份浓缩冰美式提神。", "购买不等于喝过；补明实际饮用及非脱因原料。"),
    "m_0083": ("楼下饮品店新出了一款纯燕麦特饮，店员说原料是无麸质认证的燕麦，里面也不含任何咖啡因。", "避免燕麦制品的麸质污染疑点跨组干扰，同时未声称本人购买或饮用。"),
    "m_0092": ("昨天中午在街角面馆，我吃完了一大碗手工油泼扯面和两个烤烧饼；面和饼都用普通小麦粉制作，配料只有小麦粉、水、盐和植物油。", "明确实际吃下含小麦的食品，排除无麸质替代；同时明确纯植物配料，避免额外违反全素查询。"),
    "m_0100": ("外食聚餐时我都会提前跟服务员沟通，别用小麦面粉勾芡，要用确认无麸质的玉米淀粉。", "普通淀粉不必然含麸质；修正原文对淀粉的错误泛化。"),
    "m_0101": ("昨天晚上自己在厨房只用纯玉米面、清水和酵母蒸了一锅发糕，口感粗糙但吃着特别安心。", "明确配料，避免默认混入小麦粉或动物配料的跨组歧义。"),
    "m_0105": ("上个月我用自己那张有透支额度的银联白金卡在免税店刷了六笔分期消费，款项由发卡银行先垫付，我从下个月起再按月偿还账单。", "白金卡名称不等于信用卡；明示透支和延后还款，且与所有消费当场结清直接不相容。"),
    "m_0118": ("今年十月二十日晚，我在自己的卧室躺上床后，用亮着屏幕的手机看了一个小时搞笑短视频，看完立刻关屏入睡。", "明确实际看发光屏幕与入睡紧邻；限定到10月20日，避免同时成为11月16日整夜睡眠或手机送修查询的额外冲突。"),
    "m_0120": ("这季度每晚九点半一到，我就把笔记本留在客厅书桌上充电；手机送修前也一直留在客厅，从不带进卧室。", "明确手机充电是送修前的习惯。"),
    "m_0121": ("最近晚上准备上床之前，我有时在客厅听半小时没有显示屏的收音机，听完才进卧室关灯睡觉。", "熄灯后继续听广播与从关灯起全程熟睡可能冲突；改为上床前、不看屏幕的活动。"),
    "m_0123": ("今年九月底我就把卧室里带液晶屏的电子闹钟撤掉了，十月以来只留一个机械指针小闹钟。", "明确撤除发生在季度禁屏窗口之前。"),
    "m_0127": ("我买了一个普通纸质日历放在床头，每天睡前翻一页记录作息。", "移除纸质日历自带定时功能这一不合理细节。"),
    "m_0131": ("昨天早高峰为了赶时间，我一个人乘电梯到负二层的封闭站台，坐了四十五分钟地铁二号线去市中心。", "地铁可能行驶在地面或高架；补明本人独自进入负二层封闭站台。"),
    "m_0133": ("朋友约我坐地铁二号线去市中心看展，我解释说自己有幽闭恐惧症，要改走地面的公交路线。", "保留地下空间约束，去掉与地面出租车和自驾不相容的唯一交通方式断言。"),
    "m_0134": ("今年出门只要遇到必须走地下通道的路口，我宁可多绕两百米，沿地面的无障碍路线通过。", "通过路线可以步行或使用轮椅，不再把伤后也必须步行写进干扰记录。"),
    "m_0140": ("今年因为出门经常坐地面公交，我把学校周边十几条公交线路的停靠站背得滚瓜烂熟。", "经常乘公交可与其他地面交通并存，删除不必要的只坐限定。"),
    "m_0141": ("虽然不敢独自进入地下封闭空间，但只要在开阔的公园广场休息，我就完全没问题。", "保留开阔与封闭的对照，避免伤后步行歧义。"),
    "m_0143": ("周末陪朋友去地下一层超市买东西，我一直待在地面入口等他，自己没有下楼。", "明确扶梯口位于地面，排除已到地下再等人的解释。"),
    "m_0144": ("昨天下午两点我在公园长椅上亲自操作一部联网手机，扫码租了辆共享单车，还在地图上查了路线。", "明确本人在室外实际操作联网设备，不依赖手机是否属于本人。"),
    "m_0153": ("昨天下午路过公园门口的共享单车停放点，因为没手机扫码，我就改坐地面公交回家。", "删除伤后只能步行回家的额外冲突；公交可使用轮椅。"),
    "m_0156": ("昨天在公园休息时，我看到好几个路人都在低头看手机地图找出口。", "他人使用设备与本人禁用相容；休息不暗示伤后步行。"),
    "m_0161": ("十月九日，也就是我八天假期结束后的第一天，我才从成都父母家出发坐高铁返回学校。", "原10月8日返校可能与八天未离开成都冲突；明确离开在窗口之后。"),
    "m_0164": ("十月长假期间成都老家天气特别凉爽，我陪父母去了成都市内的一个生态园摘苹果。", "明确活动仍在成都范围内。"),
    "m_0167": ("在老家陪父母那几天，每天傍晚全家一起在院子里吃水果、聊天，不看电视。", "删除每晚看电视到十点的睡前屏幕歧义。"),
    "m_0170": ("前天傍晚我换上跑鞋去学校塑胶操场，双脚交替落地蹬跑，连续慢跑了五公里，跑完腿还有点酸。", "明确实际双脚着地，排除单脚或辅助设备的替代解释。"),
    "m_0177": ("昨天去医院做了复查，医生说脚踝恢复得还行，但提醒我继续保护伤处。", "不凭普通拍片直接断定韧带恢复，移除不必要的医疗细节。"),
    "m_0212": ("公交实体卡昨天刚充值了一百块钱，这几天我都直接用实体卡刷公交，不用手机 NFC。", "消除手机送修期间仍用手机NFC乘车的跨组歧义。"),
    "m_0215": ("左脚受伤以前，周末去市区图书馆我通常坐地面公交直达，下车走三百米就是正门。", "明确具体步行习惯在脚伤之前。"),
    "m_0222": ("上个月实验室同门过生日，大家一起去学校旁边的小公园散步聊天庆祝。", "将散步庆祝明确到脚伤之前。"),
}

# Gold and negative decisions come from reading each query against the whole bank.
# Annotation explanations stay outside the three model-input files.
REVIEWS = {
    "q_hc_01": ("m_0001", "过去半年未摄入动物源性食品", "本人上周五实际吃下大虾和黄油，两者均为动物源性食品，事件落在半年窗口内。", "未陈述本人在过去半年吃下动物源性食品；植物食品、偏好、他人行为或更早饮食均可与查询共存。"),
    "q_hc_02": ("m_0014", "截至参照日从未实际驾驶机动车", "本人上周末独自驾驶手动挡轿车，与从未亲自驾驶机动车直接冲突；是否有驾照不是判断依据。", "未陈述本人实际驾驶机动车；乘车、看别人练车、步行、骑自行车、浏览租车价格都不等于本人驾车。"),
    "q_hc_03": ("m_0027", "搬入现租公寓后从未有活体宠物入室", "柯基幼犬是活体宠物，明确一直养在同一现租公寓，构成直接反例。", "未陈述活体宠物进入或被养在该公寓；植物、视频、邻居动物、室外观察和一般生活事件不构成反例。"),
    "q_hc_04": ("m_0040", "当前个人电脑手机平板仅运行安卓或Linux", "本人自己的主力电脑目前每天实际运行macOS；macOS不属于安卓或Linux。", "未陈述本人当前的电脑手机平板运行安卓或Linux以外的系统；设备品牌、应用、外设或他人设备不能单独推出该结论。"),
    "q_hc_05": ("m_0053", "截至参照日仍不会基本浮水换气", "本人上周二无浮具、无人托举独立连续蛙泳1500米，体现自主浮水和换气能力，与仍完全不会不相容。", "未陈述本人会自主浮水换气或已独立游泳；岸边等人、浅水踩浪、其他训练、害怕深水及他人活动都可相容。"),
    "q_ic_01": ("m_0066", "近一个月实际未服用任何青霉素类抗生素", "本人前天实际吞服阿莫西林；阿莫西林属于青霉素类抗生素，落在近一个月窗口。", "未陈述本人在近一个月服用青霉素类抗生素；药箱存药、其他药物、就医、过敏史、他人用药及用药建议都不等于本人服用。"),
    "q_ic_02": ("m_0079", "过去半年实际未饮用含咖啡因饮品", "本人最近每天实际喝完普通非脱因咖啡豆制作的冰美式；该咖啡含咖啡因，与实际零摄入冲突。", "未陈述本人在过去半年喝过含咖啡因饮品；更早提神饮料、他人饮品、闻气味及明确无咖啡因饮品均不是反例。"),
    "q_ic_03": ("m_0092", "过去半年实际未吃含麸质食品", "本人昨天吃完普通小麦粉制作的面和饼；普通小麦粉含麸质，与无麸质零摄入直接冲突。", "未陈述本人在过去半年实际吃过含麸质食品；明确无麸质食品、过去吃过、转送点心、路过面馆和营养建议都不构成反例。"),
    "q_ic_04": ("m_0105", "从未使用信用卡且消费当场结清", "本人用有透支额度的卡消费，由发卡银行垫付并于以后还账单，与不用信用卡及当场结清不相容。", "未陈述本人使用信用卡或延后偿还消费欠款；借记卡、现金、支付渠道、储蓄、他人的信用卡行为及推销电话均可相容。"),
    "q_ic_05": ("m_0118", "10月1日至11月17日睡前两小时无屏幕且卧室无屏幕设备", "10月20日晚本人在卧室用亮屏手机看视频一小时并立即入睡，两项事实都落在查询禁止发生的实际时间空间范围内。", "未陈述在指定窗口的卧室放屏幕或本人睡前两小时看屏幕；白天办公、客厅设备、无屏广播、先前撤除设备及无屏活动都可相容。"),
    "q_bs_01": ("m_0131", "本年从未独自进入地下封闭空间", "本人昨天独自到负二层的封闭站台；负二层在地下，直接违反从未独自进入的事实陈述。", "未陈述本人本年独自进入地下封闭空间；地面交通、地面入口等候、开阔场所、他人乘地铁和建议计划都可相容。"),
    "q_bs_02": ("m_0144", "11月13日至17日外出未携带借用或操作联网设备", "本人11月16日14时在公园亲自操作联网手机；无论所有权如何，都在时间和场所窗口内直接构成反例。", "未陈述本人11月13日至17日外出时携带借用或操作联网设备；室内电脑、存储在设备内的信息、送修前使用、他人手机及实体卡均可相容。"),
    "q_bs_03": ("m_0157", "10月1日至8日始终在成都未离开", "10月3日晚本人亲自在上海外滩和南京路，上海不在成都市，时间重叠且地点互斥。", "未陈述本人10月1日至8日身处成都以外；假期后返校、前年旅游、他人照片、成都本地活动及其他日期事件均可相容。"),
    "q_bs_04": ("m_0170", "11月12日至17日左脚从未着地承重", "11月15日本人双脚交替落地蹬跑5公里，必然包含左脚着地承重，直接反驳该实际行为陈述。", "未陈述本人11月12日至17日左脚着地承重；更早散步锻炼、坐轮椅或乘车、上肢动作、未指定步行的到访及医嘱都可相容。"),
    "q_bs_05": ("m_0183", "11月16日22时至17日08时持续熟睡", "11月17日02:30至06:00本人在实验室主动修复代码并提交，位于连续睡眠窗口内，二者不能同时为真。", "未陈述本人在11月16日22时至17日08时保持清醒；其他夜晚活动、睡前准备、后台自动测试和醒后工作均可相容。"),
}

# These are the relevant non-target cases explicitly reconsidered in the second pass.
NEGATIVE_DETAILS = {
    "q_hc_01": {
        "m_0009": "骨头汤明确发生在改为全素之前，不在过去半年的禁食窗口。",
        "m_0092": "普通小麦粉、水、盐、植物油均不是动物源性配料；该记录只违反无麸质查询，不违反全素查询。",
        "m_0101": "发糕明确只用玉米面、水和酵母，没有动物配料。",
    },
    "q_hc_02": {
        "m_0021": "明确是室友驾驶、本人旁观。",
        "m_0131": "乘地铁不等于驾驶地铁。",
        "m_0144": "共享单车不是机动车，而且扫码租车不声称本人驾驶机动车。",
    },
    "q_hc_03": {
        "m_0028": "绿萝与虎皮兰是植物，不是活体宠物。",
        "m_0035": "猫在小区门口，由邻居牵着；未进入本人公寓。",
        "m_0039": "带入客厅的是多肉植物。",
    },
    "q_hc_04": {
        "m_0044": "调试的是师弟的电脑，且Python不限定操作系统。",
        "m_0144": "手机扫码与地图使用不限定安卓以外的系统。",
        "m_0199": "VS Code、虚拟环境、Git可在Linux运行；没有系统不相容陈述。",
    },
    "q_hc_05": {
        "m_0054": "本人在岸边等室友，未游泳。",
        "m_0057": "浅水踩浪不意味着会自主浮水换气。",
        "m_0064": "浅池弯腰埋头后憋气难受，与不会换气相容。",
    },
    "q_ic_01": {
        "m_0069": "药箱列举的物品不是本人服用青霉素的记录。",
        "m_0071": "医嘱避开该类药，与本人未服用相容。",
        "m_0074": "用药主体是室友，而非本人。",
    },
    "q_ic_02": {
        "m_0083": "仅介绍店里的饮品，且明确不含咖啡因。",
        "m_0085": "提神饮料属于戒掉以前；当前行为是午睡。",
        "m_0091": "白开水加柠檬没有声明加入咖啡因。",
    },
    "q_ic_03": {
        "m_0001": "大虾和黄油不因是动物源性就含麸质；未列出含小麦等配料，不能额外假定加入。",
        "m_0083": "燕麦明确为无麸质认证，且没有本人饮用行为。",
        "m_0099": "含小麦点心转送他人，没有吃下。",
        "m_0100": "使用的是已确认无麸质的玉米淀粉。",
        "m_0102": "吃烧饼明确是以前，现在已戒掉。",
    },
    "q_ic_04": {
        "m_0108": "借记卡扫码当场付清，与查询一致。",
        "m_0110": "分期消费的主体是前面的顾客。",
        "m_0112": "使用多张信用卡的是室友。",
        "m_0144": "扫码只表明支付方式，资金可来自借记卡余额，不推出信用卡或借贷。",
        "m_0211": "存入备用金是储蓄行为，未涉及透支消费。",
    },
    "q_ic_05": {
        "m_0120": "设备留在客厅而非卧室，充电不等于本人睡前看屏幕。",
        "m_0121": "收音机无显示屏；活动在客厅、上床之前。",
        "m_0123": "电子闹钟已于9月底撤走，早于10月1日窗口。",
        "m_0124": "视频观看发生在白天工位午休时，不是晚间入睡前。",
        "m_0130": "平板在客厅定时锁屏，不表明睡前观看或进入卧室。",
        "m_0144": "手机使用发生在下午2时室外。",
        "m_0167": "已明确改为傍晚在院子吃水果聊天，不看电视。",
        "m_0183": "凌晨在实验室工作到6时，没有声称何时入睡；可以8时之后再睡，不能补推睡前两小时看屏幕。",
        "m_0220": "微信视频的时刻和设备地点未限定为卧室或睡前，不能额外推定违规。",
    },
    "q_bs_01": {
        "m_0014": "沿滨海公路驾驶，没有进入地下封闭空间的陈述；查询不再只准公交。",
        "m_0139": "地面出租车与地下空间约束相容。",
        "m_0143": "明确始终在地面入口，自己没有下楼。",
        "m_0161": "高铁行程未陈述地下封闭空间；不能凭交通工具名称推定进入地下。",
        "m_0213": "带顶棚的连廊未被陈述为地下空间。",
    },
    "q_bs_02": {
        "m_0004": "收藏内容存在手机中，不等于送修期间在室外操作或携带手机。",
        "m_0017": "导航是以前的乘车习惯。",
        "m_0042": "外出带手机明确发生在送修前。",
        "m_0045": "描述送修前的续航。",
        "m_0050": "操作发生在11月11日，早于11月13日送修。",
        "m_0118": "手机观看发生在10月20日，且在室内。",
        "m_0120": "手机充电限定送修前，笔记本放客厅不违反外出限定。",
        "m_0138": "导航软件已下载并设置，不表明近五天出门实际操作。",
        "m_0145": "借用电脑发生在室内；查询允许室内电脑。",
        "m_0154": "台式机在实验室室内使用，离校后断网。",
        "m_0212": "使用公交实体卡，明确不用手机NFC。",
        "m_0220": "微信视频未指明室外或移动设备，可在室内用电脑。",
    },
    "q_bs_03": {
        "m_0159": "外滩照片由室友发送，不证明本人在上海。",
        "m_0161": "离开发生在10月9日，已超出1日至8日的窗口。",
        "m_0164": "生态园明确在成都市内。",
        "m_0165": "本人游上海是前年夏天，与今年10月窗口不同。",
    },
    "q_bs_04": {
        "m_0014": "驾车未声明左脚在地面承受体重；不能把离合器操作或医学禁忌替代落地承重标签。",
        "m_0023": "步行与骑车明确是受伤前。",
        "m_0030": "楼下散步明确是受伤前。",
        "m_0053": "游泳发生在上周二11月10日，早于11月12日受伤。",
        "m_0058": "上肢训练发生在11月10日，早于受伤。",
        "m_0061": "划船机锻炼明确是受伤前。",
        "m_0063": "11月10日步行在受伤前。",
        "m_0131": "乘电梯、地铁可使用轮椅，文本不要求左脚落地承重。",
        "m_0144": "亲自扫码租车与查地图没有声明已骑行，更未声明左脚落地承重。",
        "m_0153": "已改坐地面公交，不再断言伤后步行回家。",
        "m_0157": "上海游览发生在10月3日，早于脚伤。",
        "m_0172": "本人坐轮椅，由室友推行，未承重。",
        "m_0215": "下车步行明确发生在受伤前。",
        "m_0219": "在院子晒太阳可坐轮椅，未陈述左脚承重。",
        "m_0222": "公园散步是上个月，早于11月12日脚伤。",
    },
    "q_bs_05": {
        "m_0118": "亮屏观看已限定10月20日晚，与11月16日晚睡眠窗口不重叠。",
        "m_0121": "听广播在上床关灯之前，没有声称关灯后仍清醒。",
        "m_0129": "入睡事件在前天晚上11月15日，不是昨晚。",
        "m_0184": "熬夜是11月15日凌晨，随后11月16日晚睡觉，时间相容。",
        "m_0188": "运行的是自动测试；8时醒后查看，不要求睡眠期间本人清醒。",
        "m_0194": "服务器脚本后台运行不意味着本人醒着操作。",
    },
}

FACT_SOURCES = [
    {"query_id": "q_ic_01", "fact": "阿莫西林属于青霉素类抗生素。", "url": "https://medlineplus.gov/druginfo/meds/a685001.html", "checked_on": "2026-11-18"},
    {"query_id": "q_ic_02", "fact": "普通咖啡是咖啡因来源。", "url": "https://www.fda.gov/consumers/consumer-updates/spilling-beans-how-much-caffeine-too-much", "checked_on": "2026-11-18"},
    {"query_id": "q_ic_03", "fact": "小麦天然含有麸质。", "url": "https://www.niddk.nih.gov/health-information/digestive-diseases/celiac-disease/treatment", "checked_on": "2026-11-18"},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_rows(path, values):
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")


def validate(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifact_sha256"].items():
        assert sha(directory / name) == digest, f"Hash mismatch: {name}"
    memories = load_text(directory / "memories.jsonl")
    queries = load_queries(directory / "queries.jsonl")
    mids = {m.memory_id for m in memories}
    qids = {q["query_id"] for q in queries}
    gold = validate_gold(read_jsonl(directory / "gold.jsonl"), mids)
    assert len(memories) == 225 and len(queries) == len(gold) == 15
    assert qids == {g["query_id"] for g in gold}
    assert len({m.content for m in memories}) == 225
    assert len({q["content"] for q in queries}) == 15
    assert not {m.content for m in memories} & {q["content"] for q in queries}
    assert all(m.retrieval_text == m.content for m in memories)
    assert all(build_retrieval_text(q["content"], q.get("source_ref")) == q["content"] for q in queries)
    assert all(m.content.startswith(ANCHOR) for m in memories)
    assert all(q["content"].startswith(ANCHOR) for q in queries)
    assert all(set(m) == {"memory_id", "content", "source_ref"} for m in read_jsonl(directory / "memories.jsonl"))
    assert all(set(q) == {"query_id", "content"} for q in queries)
    pairs = read_jsonl(directory / "audit_pairs.jsonl")
    by_pair = {(p["query_id"], p["memory_id"]): p for p in pairs}
    assert len(pairs) == len(by_pair) == 3375
    assert set(by_pair) == {(qid, mid) for qid in qids for mid in mids}
    mm = {m.memory_id: m for m in memories}
    qm = {q["query_id"]: q for q in queries}
    for p in pairs:
        assert p["label"] in {"conflict", "no_conflict"}
        assert p["query_content_sha256"] == hashlib.sha256(qm[p["query_id"]]["content"].encode()).hexdigest()
        assert p["memory_content_sha256"] == hashlib.sha256(mm[p["memory_id"]].content.encode()).hexdigest()
        assert p["rationale"] and p["reviewer"] == "assistant"
        assert p["human_reviewed"] is False
    for g in gold:
        positives = {mid for (qid, mid), p in by_pair.items() if qid == g["query_id"] and p["label"] == "conflict"}
        assert positives == set(g["conflict_old_memory_ids"]) and len(positives) == 1
        assert g["fully_reviewed"] is True and g["human_reviewed"] is False
    source_queries = read_jsonl(directory / "source" / SOURCE_NAMES["queries"])
    metadata = {q["query_id"]: q for q in read_jsonl(directory / "query_metadata.jsonl")}
    assert Counter(m["category"] for m in metadata.values()) == {
        "hyponym_commonsense": 5, "implicit_constraint": 5, "behavior_state_mutex": 5}
    for q in source_queries:
        assert len(q["distractor_ids"]) == 12
        assert set(q["distractor_ids"]) <= mids
        assert all(by_pair[(q["query_id"], mid)]["label"] == "no_conflict" for mid in q["distractor_ids"])
    for qid, details in NEGATIVE_DETAILS.items():
        for mid in details:
            assert by_pair[(qid, mid)]["label"] == "no_conflict"
            assert by_pair[(qid, mid)]["review_basis"] == "explicit_near_miss_review"
    return {
        "passed": True, "queries": 15, "memories": 225, "reviewed_pairs": 3375,
        "conflict_pairs": sum(p["label"] == "conflict" for p in pairs),
        "no_conflict_pairs": sum(p["label"] == "no_conflict" for p in pairs),
        "uncertain_pairs": 0, "conflicts_per_query": 1,
        "query_body_repairs": len(QUERY_EDITS), "memory_body_repairs": len(MEMORY_EDITS),
        "retrieval_text_equals_content": True, "label_fields_in_model_input": False,
        "source_snapshots_and_artifact_hashes_verified": True,
        "project_loaders_and_gold_validator_passed": True,
        "semantic_review": "single assistant full-bank content review; not an automatic proof or independent human audit",
        "human_reviewed": False, "evaluation_use": "exploratory",
    }


def build(source_dir, output):
    if output.exists():
        raise ValueError(f"Output exists; use --validate-only or a new path: {output}")
    sources = {key: source_dir / name for key, name in SOURCE_NAMES.items()}
    original_q = read_jsonl(sources["queries"])
    original_m = read_jsonl(sources["memories"])
    assert len(original_q) == 15 and len(original_m) == 225
    assert {q["query_id"] for q in original_q} == set(REVIEWS)
    assert set(QUERY_EDITS) <= {q["query_id"] for q in original_q}
    assert set(MEMORY_EDITS) <= {m["memory_id"] for m in original_m}
    output.mkdir(parents=True)
    (output / "source").mkdir()
    for path in sources.values():
        shutil.copyfile(path, output / "source" / path.name)

    repaired_q, repaired_m, changes = [], [], []
    for rows, id_field, text_field, edits, result in (
        (original_q, "query_id", "query_text", QUERY_EDITS, repaired_q),
        (original_m, "memory_id", "text", MEMORY_EDITS, repaired_m),
    ):
        for row in rows:
            rid = row[id_field]
            body = row[text_field]
            if rid in edits:
                body, reason = edits[rid]
                assert body != row[text_field]
                changes.append({"record_type": id_field.removesuffix("_id"), "record_id": rid,
                                "before": row[text_field], "after": body, "reason": reason})
            new = {id_field: rid, "content": ANCHOR + body}
            if id_field == "memory_id":
                new["source_ref"] = {"dataset": DATASET_ID,
                    "file": "source/" + SOURCE_NAMES["memories"], "source_memory_id": rid}
            result.append(new)

    gold, audit, metadata = [], [], []
    for q, source_q in zip(repaired_q, original_q):
        qid = q["query_id"]
        target, scope, positive_reason, negative_reason = REVIEWS[qid]
        assert target == source_q["gold_memory_id"]
        metadata.append({"query_id": qid, "category": source_q["category"],
            "domain": source_q["domain"], "subject": "user_self", "claim_scope": scope,
            "reference_date": "2026-11-17", "authored_distractor_ids": source_q["distractor_ids"],
            "reviewer": "assistant", "human_reviewed": False})
        for m in repaired_m:
            mid = m["memory_id"]
            positive = mid == target
            detail = NEGATIVE_DETAILS.get(qid, {}).get(mid)
            assert not (positive and detail)
            reason = positive_reason if positive else (detail or negative_reason)
            audit.append({"query_id": qid, "memory_id": mid,
                "label": "conflict" if positive else "no_conflict", "rationale": reason,
                "review_basis": "direct_pair_contradiction" if positive else
                                "explicit_near_miss_review" if detail else "query_claim_not_negated_by_memory",
                "query_content_sha256": hashlib.sha256(q["content"].encode()).hexdigest(),
                "memory_content_sha256": hashlib.sha256(m["content"].encode()).hexdigest(),
                "reviewer": "assistant", "human_reviewed": False})
        gold.append({"query_id": qid, "conflict_old_memory_ids": [target],
            "fully_reviewed": True, "human_reviewed": False,
            "label_source": "assistant_full_bank_pair_review_v2"})

    save_rows(output / "memories.jsonl", repaired_m)
    save_rows(output / "queries.jsonl", repaired_q)
    save_rows(output / "gold.jsonl", gold)
    save_rows(output / "audit_pairs.jsonl", audit)
    save_rows(output / "query_metadata.jsonl", metadata)
    save_rows(output / "repairs.jsonl", changes)
    save_json(output / "fact_sources.json", FACT_SOURCES)
    review_lines = ["# 15条查询的全库核对记录", "", "核对者为同一个助手；不是独立人工标注。每条查询都与同一225条旧库逐对核对，"
                    "标签不依赖其他旧记忆的真实性或联合推理。", ""]
    mmap = {m["memory_id"]: m for m in repaired_m}
    for q in repaired_q:
        qid = q["query_id"]
        mid, scope, reason, negative = REVIEWS[qid]
        review_lines += [f"## {qid}", "", q["content"].replace("\n", " "), "",
                         f"冲突目标 {mid}：{mmap[mid]['content'].replace(chr(10), ' ')}", "", reason,
                         "", "其余224条：" + negative, ""]
        for near, detail in NEGATIVE_DETAILS.get(qid, {}).items():
            review_lines.append(f"- {near}：{detail}")
        review_lines.append("")
    (output / "review.zh.md").write_text("\n".join(review_lines), encoding="utf-8")

    readme = f"""# 单条隐式冲突诊断集 v2

从用户最后上传的两份 `(1).jsonl` 修订版修复而来。15条查询（三类各5条）共用225条冻结旧记忆；
每条查询恰有1条可独立成立的冲突目标、12条原配相容干扰；另外30条为背景。
所有225条记忆都参与每条查询的搜索，不能按原始角色、分组、领域或金标筛库。

## 三个主文件

| 文件 | 行数 | 项目输入字段 |
|---|---:|---|
| memories.jsonl | 225 | memory_id、content、source_ref |
| queries.jsonl | 15 | query_id、content |
| gold.jsonl | 15 | query_id、conflict_old_memory_ids、fully_reviewed、human_reviewed、label_source |

`content` 是唯一语义检索正文。记忆的 `source_ref` 仅是可追溯来源，不含主体过滤字段或标注；
已核验项目的 `retrieval_text` 与 `content` 相等。角色、类别、原分组、目标和理由都不拼入输入。

## 语义和时间

冲突指同一主体在重叠的时间、条件、场所下，两条事实不能同时为真。
只给出禁令、医嘱、无资格或偏好，不能据此否定实际发生的违规行为。
每一对只使用该查询、该旧记忆和普通概念知识，不组合其他旧记忆补前提。
`no_conflict` 表示该对事实可同时为真，不代表无关、蕴含或已经证实为真。
对未写出的设备、原料、地点和事件，不擅自补事实。

所有正文统一附上“叙述时间参照：2026年11月17日。”，固定相对日期，且在检索器只读正文时仍有效。
这是共同叙述时钟，不是写入时间；今天=11月17日、昨天=11月16日、前天=11月15日、
上周五=11月13日、上周二=11月10日。脚伤日明确为11月12日，手机送修日为11月13日。
旧组中未来的“这周二晚上”已改为上周二；所有查询独立评测，不写回或改变旧库。

## 修复与审阅

修复了 {len(QUERY_EDITS)} 条查询正文和 {len(MEMORY_EDITS)} 条记忆正文，并给全部240条输入补了相同的时间参照。
逐条前后文本及原因见 repairs.jsonl；完整来源快照保存在 source/，字节级哈希保存在 manifest.json。
查询类别等辅助字段在 query_metadata.jsonl；3375个查询—旧记忆配对的标签、理由和正文哈希在 audit_pairs.jsonl。
疑难配对的二次核对说明见 review.zh.md；成分类别依据见 fact_sources.json。

全库助手审阅完成，因此 `fully_reviewed=true`；没有独立人工核验，因此 `human_reviewed=false`。
这个标记如实区分核对覆盖范围与核对者，不能把程序通过当成逻辑正确性的自动证明。

## 使用和版本边界

这是已查看过的合成探索诊断集，不是未接触的正式测试集；本次修复只依据语义与时间，未按检索分数筛选、删例或调难度。
共享旧库包含相互不一致的陈述（例如目标与相容改述），它是逐对检索诊断库，不宣称恢复真实一致的个人时间线。
单条冲突对允许借助普通类别知识，但不允许多条旧记忆组合成冲突。
本版正文已变化，上一版的4/15等基线数字和旧向量不能直接用于本版，需要重新建索引和评测。
此次交付只修复和转换数据，没有运行新的效果实验。

项目入口校验（项目根目录）：

```powershell
.\\.venv\\Scripts\\python.exe -m memory_retrieval validate-data --memories data\\{DATASET_ID}\\memories.jsonl --queries data\\{DATASET_ID}\\queries.jsonl --gold data\\{DATASET_ID}\\gold.jsonl
.\\.venv\\Scripts\\python.exe scripts\\build_single_pair_implicit_v2.py --validate-only
```

后续探索性基线入口可以使用 `scripts/run_experiment.py --dataset data/{DATASET_ID} --limit 0 --evaluation-split exploratory --arms bm25 dense hybrid`。
只有金标和核对记录可提供给评估器；`source/`、`audit_pairs.jsonl`、`query_metadata.jsonl` 等辅助文件不送给检索器。
"""
    (output / "README.zh.md").write_text(readme, encoding="utf-8")
    artifacts = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = {
        "dataset_id": DATASET_ID, "created_on": "2026-11-18", "timezone": "Asia/Shanghai",
        "reference_date": "2026-11-17", "evaluation_use": "exploratory",
        "query_count": 15, "memory_count": 225, "pair_count": 3375,
        "conflict_pair_count": 15, "nonconflict_pair_count": 3360,
        "expected_conflicts_per_query": 1, "content_prefix": ANCHOR,
        "source_inputs": {key: {"name": p.name, "sha256": sha(p)} for key, p in sources.items()},
        "semantic_body_edits": {"queries": len(QUERY_EDITS), "memories": len(MEMORY_EDITS)},
        "review": {"coverage": "full query-memory Cartesian product", "reviewer": "assistant",
                   "human_reviewed": False, "retrieval_scores_used_for_repair": False},
        "builder": {"path": "scripts/build_single_pair_implicit_v2.py", "sha256": sha(Path(__file__))},
        "artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in artifacts},
    }
    save_json(output / "manifest.json", manifest)
    result = validate(output)
    save_json(output / "validation.json", result)
    return result


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = validate(args.output) if args.validate_only else build(args.source_dir, args.output)
    print(json.dumps({"output": str(args.output.resolve()), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
