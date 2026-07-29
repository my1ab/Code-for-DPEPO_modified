"""
Search（QA 检索）任务测试样例数据

数据来源：test_bert_similarity/验证轨迹search/ 下的真实轨迹
（bamboogle / hotpotqa / musique / nq / popqa / triviaqa / 2wikimultihopqa）

动作类型（源自 get_actions 字段，已统一规范化）：
- search[<query>]   : 检索动作（<search>...</search>）
- answer[<answer>]  : 终止作答动作（<answer>...</answer>）
- null              : 该环境本轮不动作

包含以下数据：
- SEARCH_PAIRS: Search 任务动作测试对（带期望结果标注）
  结构: (action_a, action_b, should_match, category)
- SEARCH_TEXTS: Search 任务动作列表（用于构建相似度矩阵）

构建原则（对齐 test_data.py 的 WebShop 设计）：
1. 从 get_actions 中提取动作，组成对
2. 按动作类型分别处理（search / answer / null / 跨类型）
3. 同类型变体（同义查询、词序重排、大小写差异、引号差异等）
   排版在同一个区域内，便于阈值分析
4. 仅供 test_bert_similarity.py 中 import 引用（暂不使用）

表述来源说明：
- 主体动作表述 优先采用参考验证文件中的真实表述
  （如 search[head of NASA during Apollo 11] / search[Andy Murray family brother] /
    search[Marty's girlfriend Back to the Future actress] / answer[Thomas O. Paine] 等）
- 大小写变体、短查询、单复数对照、不同实体反例等为构造项
  （真实轨迹中一个查询通常只有一种表述，需构造变体测试阈值边界）
  构造项在注释中标注「构造对照」
"""

# ── Search 任务动作测试对 ──────────────────────────────
# 基于验证轨迹search/ 真实动作模式设计
# 每对标注期望结果，用于验证阈值合理性
# (action_a, action_b, should_match, category)
SEARCH_PAIRS = [
    # ================================================================
    # 一、search 动作区
    # ================================================================

    # === 完全相同（应判为重复）===
    ("search[head of NASA during Apollo 11]", "search[head of NASA during Apollo 11]", True, "完全相同-search"),
    ("search[The Terminal director]", "search[The Terminal director]", True, "完全相同-search"),
    ("search[Andy Murray brother]", "search[Andy Murray brother]", True, "完全相同-search"),
    ("search[author of The Jungle]", "search[author of The Jungle]", True, "完全相同-search"),

    # === 大小写差异（同一查询的不同大小写，真实数据中最常见的变体）===
    # 句首字母大小写差异
    ("search[The Terminal director]", "search[the terminal director]", True, "大小写-The Terminal"),
    ("search[The Believers composer]", "search[the believers composer]", True, "大小写-The Believers"),
    ("search[The Last Days of Pompeii producer]", "search[the last days of pompeii producer]", True, "大小写-The Last Days"),
    # 专有名词大小写差异
    ("search[NASA administrator Apollo 11]", "search[nasa administrator apollo 11]", True, "大小写-NASA Apollo"),
    ("search[Britney Spears birthplace city state]", "search[britney spears birthplace city state]", True, "大小写-Britney Spears"),
    ("search[Andy Murray brother]", "search[andy murray brother]", True, "大小写-Andy Murray"),
    # 标题/电影名大小写（真实表述带引号，构造对照为纯小写）
    ("search[\"Before Midnight\" screenwriter Ethan Hawke]", "search[before midnight ethan hawke screenwriter]", True, "大小写-Before Midnight"),
    ("search[Marty's girlfriend Back to the Future actress]", "search[marty's girlfriend back to the future actress]", True, "大小写-Back to the Future"),

    # === 同义查询-句式重排（同一检索意图的不同表达，应判为相同）===
    # head of NASA / NASA administrator / who was in charge of NASA（同义改写，均源自真实轨迹）
    ("search[head of NASA during Apollo 11]", "search[NASA administrator Apollo 11]", True, "同义-NASA head改写"),
    ("search[head of NASA during Apollo 11]", "search[who was in charge of NASA during Apollo 11]", True, "同义-NASA head疑问句改写"),
    ("search[head of NASA during Apollo 11]", "search[Apollo 11 mission commander NASA]", False, "不同-NASA head vs commander"),
    # director / who directed / director of（同义动词改写，均源自真实轨迹）
    ("search[The Terminal director]", "search[who directed The Terminal]", True, "同义-director改写"),
    ("search[The Terminal director]", "search[The Terminal 2004 director]", True, "同义-director+年份限定"),
    # author（The Arrangement 用真实带引号表述，与 The Jungle 是不同实体）
    ("search[\"The Arrangement\" author]", "search[author of The Arrangement]", True, "同义-The Arrangement author改写"),
    # producer / who produced（同义改写，源自真实轨迹）
    ("search[The Last Days of Pompeii producer]", "search[who produced The Last Days of Pompeii]", True, "同义-producer改写"),
    # composer / who composed（同义改写，源自真实轨迹）
    ("search[The Believers composer]", "search[\"The Believers\" 1987 film composer]", True, "同义-composer改写"),
    # screenwriter / screenwriter of / writer（同义改写，均源自真实轨迹）
    ("search[screenwriter of Before Midnight]", "search[screenwriter Before Midnight]", True, "同义-screenwriter改写"),
    ("search[screenwriter of Before Midnight]", "search[writer Before Midnight Before Sunrise After Midnight]", True, "同义-screenwriter=writer"),
    # birthplace / born in / where was ... born（同义改写，源自真实轨迹）
    ("search[Britney Spears birthplace city state]", "search[Britney Spears born in what city]", True, "同义-birthplace改写"),
    ("search[Britney Spears birthplace city state]", "search[Britney Spears born in what city state]", True, "同义-birthplace改写2"),
    # brother / sibling / family（同义改写，均源自真实轨迹）
    ("search[Andy Murray brother]", "search[Andy Murray family brother]", True, "同义-brother+family"),
    ("search[Andy Murray brother]", "search[What is Andy Murray's brother]", True, "同义-brother疑问句改写"),

    # === 同义查询-词序重排（关键词集合相同或相近，顺序不同，应判为相同）===
    # Curious fragrance/cologne singer（真实轨迹中的同义变体）
    ("search[Curious women's fragrance singer]", "search[Curious women's cologne singer]", True, "同义-Curious fragrance vs cologne"),
    # Team USA Olympics（真实轨迹中的词序变体）
    ("search[Team USA starting pitchers 2000 Summer Olympics]", "search[2000 Summer Olympics baseball USA starting pitchers]", True, "词序-Team USA重排"),
    ("search[Team USA starting pitchers 2000 Summer Olympics]", "search[Baseball at 2000 Summer Olympics USA starting pitchers list]", True, "词序-Team USA重排2"),
    # BBC Sports Personality（真实轨迹中的词序变体）
    ("search[BBC Sports Personality of the Year awards winner most wins]", "search[player with most BBC Sports Personality of the Year awards]", True, "词序-BBC奖项重排"),
    ("search[BBC Sports Personality of the Year awards winner most wins]", "search[who won the most BBC Sports Personality of the Year]", True, "词序-BBC奖项重排2"),
    # Oak Beach（真实轨迹中的词序变体）
    ("search[Oak Beach New York between which island]", "search[Island between Oak Beach and Great South Bay New York]", True, "词序-Oak Beach重排"),

    # === 引号差异（查询词加引号 vs 不加引号，检索意图相同，均源自真实轨迹）===
    ("search[\"Curious\" fragrance singer]", "search[Curious women's fragrance singer]", True, "引号-Curious加引号"),
    ("search[\"The Arrangement\" author]", "search[author of The Arrangement]", True, "引号-The Arrangement加引号"),
    ("search[\"Umchabezi River\" mouth]", "search[Umchabezi River mouth]", True, "引号-Umchabezi River加引号"),
    ("search[\"Before Midnight\" screenwriter Ethan Hawke]", "search[Before Midnight Ethan Hawke screenwriter]", True, "引号-Before Midnight加引号"),

    # === 单复数/形态变化（应判为相同，复数为真实轨迹表述，单数为构造对照）===
    ("search[Team USA starting pitchers 2000 Summer Olympics]", "search[Team USA starting pitcher 2000 Summer Olympics]", True, "单复数-pitchers vs pitcher"),
    ("search[BBC Sports Personality of the Year awards winner most wins]", "search[BBC Sports Personality of the Year award winner most wins]", True, "单复数-awards vs award"),
    ("search[Great South Bay between which islands]", "search[Great South Bay between which island]", True, "单复数-islands vs island"),
    ("search[Murray brothers tennis]", "search[Murray brother tennis]", True, "单复数-brothers vs brother"),

    # === 短查询（前缀占比大，测试前缀剥离效果）===
    ("search[mattress]", "search[mattress]", True, "短查询-完全相同"),
    ("search[keywords]", "search[keywords]", True, "短查询-占位符相同"),
    ("search[Andy Murray]", "search[Andy Murray]", True, "短查询-人名相同"),
    # 短查询增加修饰词视为不同（A是B的子集）
    ("search[Andy Murray]", "search[Andy Murray brother]", False, "不同-短查询增加brother"),
    ("search[Britney Spears]", "search[Britney Spears birthplace]", False, "不同-短查询增加birthplace"),
    ("search[The Terminal]", "search[The Terminal director]", False, "不同-短查询增加director"),

    # === 不同查询意图（同一领域不同实体，不应判为重复）===
    # 不同电影
    ("search[The Terminal director]", "search[The Last Days of Pompeii producer]", False, "不同-The Terminal vs Last Days of Pompeii"),
    ("search[screenwriter of Before Midnight]", "search[Marty's girlfriend Back to the Future actress]", False, "不同-Before Midnight vs Back to the Future"),
    # 不同人物
    ("search[Andy Murray brother]", "search[Britney Spears birthplace city state]", False, "不同-Andy Murray vs Britney Spears"),
    ("search[author of The Jungle]", "search[\"The Arrangement\" author]", False, "不同-The Jungle vs The Arrangement"),
    # 不同属性（同一实体不同查询维度）
    ("search[Andy Murray brother]", "search[Andy Murray birthplace]", False, "不同-Andy Murray brother vs birthplace"),
    ("search[Britney Spears birthplace city state]", "search[Britney Spears fragrance]", False, "不同-Britney Spears birthplace vs fragrance"),
    # 不同机构/赛事
    ("search[head of NASA during Apollo 11]", "search[BBC Sports Personality of the Year awards winner most wins]", False, "不同-NASA vs BBC"),
    ("search[Team USA starting pitchers 2000 Summer Olympics]", "search[Oak Beach New York between which island]", False, "不同-奥运会 vs Oak Beach"),

    # === 1词差异-不同实体（验证阈值下方的区分能力）===
    # 兄弟 vs 姐妹
    ("search[Andy Murray brother]", "search[Andy Murray sister]", False, "不同-brother vs sister"),
    # 出生地 vs 国籍
    ("search[Britney Spears birthplace city state]", "search[Britney Spears nationality country]", False, "不同-birthplace vs nationality"),
    # 导演 vs 制片人
    ("search[The Terminal director]", "search[The Terminal producer]", False, "不同-director vs producer"),
    # 作者 vs 插画师（插画师为构造对照，真实轨迹无此查询）
    ("search[author of The Jungle]", "search[illustrator of The Jungle]", False, "不同-author vs illustrator"),
    # composer vs lyricist（lyricist 为构造对照）
    ("search[The Believers composer]", "search[The Believers lyricist]", False, "不同-composer vs lyricist"),

    # === 错别字/拼写差异（真实数据中存在）===
    # Andrey vs Andy（真实轨迹中的误拼变体）
    ("search[Andy Murray brother]", "search[Andrey Murray brother tennis]", False, "不同-Andy vs Andrey误拼"),
    # Oakh Beach vs Oak Beach（真实轨迹中的多字母误拼）
    ("search[Oak Beach New York between which island]", "search[Oakh Beach New York location]", False, "不同-Oak Beach vs Oakh Beach"),
    # moon landingNASA（真实轨迹中存在的缺空格变体）
    ("search[moon landing NASA administrator]", "search[moon landingNASA administrator]", True, "同义-缺空格moon landingNASA"),

    # ================================================================
    # 二、answer 动作区
    # ================================================================

    # === 完全相同（应判为重复）===
    ("answer[Thomas O. Paine]", "answer[Thomas O. Paine]", True, "完全相同-answer"),
    ("answer[Steven Spielberg]", "answer[Steven Spielberg]", True, "完全相同-answer"),
    ("answer[Upton Sinclair]", "answer[Upton Sinclair]", True, "完全相同-answer"),
    ("answer[Elia Kazan]", "answer[Elia Kazan]", True, "完全相同-answer"),
    ("answer[Long Island]", "answer[Long Island]", True, "完全相同-answer"),
    ("answer[Jamie Murray]", "answer[Jamie Murray]", True, "完全相同-answer"),

    # === 大小写差异（应判为相同）===
    ("answer[Steven Spielberg]", "answer[steven spielberg]", True, "大小写-answer Spielberg"),
    ("answer[Thomas O. Paine]", "answer[thomas o. paine]", True, "大小写-answer Paine"),
    ("answer[Upton Sinclair]", "answer[upton sinclair]", True, "大小写-answer Sinclair"),
    ("answer[Long Island]", "answer[long island]", True, "大小写-answer Long Island"),

    # === 同义答案-全称 vs 简称（应判为相同）===
    # 全称 vs 简称
    ("answer[George IV of the United Kingdom]", "answer[George IV]", True, "同义-全称 vs 简称 George IV"),
    ("answer[King George IV]", "answer[George IV]", True, "同义-King George IV vs George IV"),
    # 全名 vs 姓
    ("answer[Thomas O. Paine]", "answer[Thomas Paine]", True, "同义-Thomas O. Paine vs Thomas Paine"),
    ("answer[Upton Sinclair]", "answer[Sinclair]", True, "同义-Upton Sinclair vs Sinclair"),

    # === 同义答案-同义表达（应判为相同）===
    # city, state 组合（McComb, Mississippi vs McComb Mississippi）
    ("answer[McComb, Mississippi]", "answer[McComb Mississippi]", True, "同义-逗号差异 McComb"),
    ("answer[McComb, Mississippi]", "answer[McComb, MS]", True, "同义-全称 vs 缩写 Mississippi MS"),
    # 拼写变体
    ("answer[Steven Spielberg]", "answer[Stephen Spielberg]", True, "同义-Steven vs Stephen Spielberg"),
    # 带/不带中间名
    ("answer[Claudia Wells]", "answer[Claudia Grace Wells]", True, "同义-Claudia Wells vs Claudia Grace Wells"),

    # === 不同答案（同一问题不同答案，不应判为重复）===
    # 不同人物
    ("answer[Thomas O. Paine]", "answer[Steven Spielberg]", False, "不同-answer Paine vs Spielberg"),
    ("answer[Upton Sinclair]", "answer[Elia Kazan]", False, "不同-answer Sinclair vs Kazan"),
    ("answer[Jamie Murray]", "answer[Andy Murray]", False, "不同-answer Jamie vs Andy Murray"),
    # 不同地点
    ("answer[Long Island]", "answer[McComb, Mississippi]", False, "不同-answer Long Island vs McComb"),
    # 相近人名（不是同一人）
    ("answer[Thomas O. Paine]", "answer[Thomas Paine]", True, "同义-Thomas O. Paine vs Thomas Paine（同义区域）"),
    ("answer[Andy Murray]", "answer[Jamie Murray]", False, "不同-answer Andy vs Jamie Murray"),
    # 姓相同名不同
    ("answer[Upton Sinclair]", "answer[Christine Sinclair]", False, "不同-Upton Sinclair vs Christine Sinclair"),

    # ================================================================
    # 三、null 动作区
    # ================================================================

    # === 完全相同（应判为重复）===
    ("null", "null", True, "null完全相同"),

    # === null vs 有效动作（不应判为重复）===
    ("null", "search[head of NASA during Apollo 11]", False, "null vs 有效搜索"),
    ("null", "answer[Thomas O. Paine]", False, "null vs 有效答案"),
    ("null", "search[Andy Murray brother]", False, "null vs 有效搜索2"),
    ("null", "answer[Steven Spielberg]", False, "null vs 有效答案2"),

    # ================================================================
    # 四、跨类型区（search vs answer vs null，不应判为重复）
    # ================================================================

    # === search vs answer（即使语义相关，动作类型不同也不应判为重复）===
    ("search[head of NASA during Apollo 11]", "answer[Thomas O. Paine]", False, "跨类型-search vs answer"),
    ("search[The Terminal director]", "answer[Steven Spielberg]", False, "跨类型-search vs answer"),
    ("search[author of The Jungle]", "answer[Upton Sinclair]", False, "跨类型-search vs answer"),
    ("search[Andy Murray brother]", "answer[Jamie Murray]", False, "跨类型-search vs answer"),

    # === answer vs search（顺序反之）===
    ("answer[Thomas O. Paine]", "search[head of NASA during Apollo 11]", False, "跨类型-answer vs search"),
    ("answer[Steven Spielberg]", "search[The Terminal director]", False, "跨类型-answer vs search"),

    # === search vs null / answer vs null ===
    ("search[head of NASA during Apollo 11]", "null", False, "跨类型-search vs null"),
    ("answer[Thomas O. Paine]", "null", False, "跨类型-answer vs null"),
]

# Search 任务动作列表，用于构建相似度矩阵
SEARCH_TEXTS = [
    # search 动作-完全相同/大小写/同义改写代表性样本（均源自真实轨迹）
    "search[head of NASA during Apollo 11]",
    "search[NASA administrator Apollo 11]",
    "search[who was in charge of NASA during Apollo 11]",
    "search[Apollo 11 mission commander NASA]",
    "search[The Terminal director]",
    "search[who directed The Terminal]",
    "search[The Terminal 2004 director]",
    "search[author of The Jungle]",
    "search[\"The Arrangement\" author]",
    "search[author of The Arrangement]",
    "search[The Last Days of Pompeii producer]",
    "search[who produced The Last Days of Pompeii]",
    "search[The Believers composer]",
    "search[\"The Believers\" 1987 film composer]",
    "search[screenwriter of Before Midnight]",
    "search[screenwriter Before Midnight]",
    "search[writer Before Midnight Before Sunrise After Midnight]",
    "search[Marty's girlfriend Back to the Future actress]",
    "search[Britney Spears birthplace city state]",
    "search[Britney Spears born in what city]",
    "search[Andy Murray brother]",
    "search[Andy Murray family brother]",
    "search[What is Andy Murray's brother]",
    # search 动作-词序重排（均源自真实轨迹）
    "search[Curious women's fragrance singer]",
    "search[Curious women's cologne singer]",
    "search[\"Curious\" fragrance singer]",
    "search[Team USA starting pitchers 2000 Summer Olympics]",
    "search[2000 Summer Olympics baseball USA starting pitchers]",
    "search[Baseball at 2000 Summer Olympics USA starting pitchers list]",
    "search[BBC Sports Personality of the Year awards winner most wins]",
    "search[player with most BBC Sports Personality of the Year awards]",
    "search[who won the most BBC Sports Personality of the Year]",
    "search[Oak Beach New York between which island]",
    "search[Island between Oak Beach and Great South Bay New York]",
    # search 动作-引号差异（真实带引号表述）
    "search[\"The Arrangement\" author]",
    "search[author of The Arrangement]",
    "search[\"Umchabezi River\" mouth]",
    "search[Umchabezi River mouth]",
    "search[\"Before Midnight\" screenwriter Ethan Hawke]",
    "search[Before Midnight Ethan Hawke screenwriter]",
    # search 动作-单复数（构造对照，真实轨迹多带复数形式）
    "search[Team USA starting pitchers 2000 Summer Olympics]",
    "search[Great South Bay between which islands]",
    "search[Great South Bay between which island]",
    "search[Murray brothers tennis]",
    # search 动作-短查询（构造对照，测试前缀剥离效果）
    "search[mattress]",
    "search[keywords]",
    "search[Andy Murray]",
    "search[Britney Spears]",
    "search[The Terminal]",
    # search 动作-错别字/拼写差异（源自真实轨迹）
    "search[Andrey Murray brother tennis]",
    "search[Oakh Beach New York location]",
    "search[moon landing NASA administrator]",
    "search[moon landingNASA administrator]",
    # answer 动作代表性样本（均源自真实轨迹）
    "answer[Thomas O. Paine]",
    "answer[thomas o. paine]",
    "answer[Thomas Paine]",
    "answer[Steven Spielberg]",
    "answer[steven spielberg]",
    "answer[Stephen Spielberg]",
    "answer[Upton Sinclair]",
    "answer[Sinclair]",
    "answer[Elia Kazan]",
    "answer[George IV of the United Kingdom]",
    "answer[George IV]",
    "answer[King George IV]",
    "answer[Long Island]",
    "answer[long island]",
    "answer[McComb, Mississippi]",
    "answer[McComb Mississippi]",
    "answer[McComb, MS]",
    "answer[Jamie Murray]",
    "answer[Andy Murray]",
    "answer[Claudia Wells]",
    "answer[Claudia Grace Wells]",
    # null 动作
    "null",
]
