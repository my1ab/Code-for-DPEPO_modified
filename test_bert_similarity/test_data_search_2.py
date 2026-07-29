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
    # 一、search 动作区（所有 search 动作按同义类型合并组织）
    # ================================================================

    # === 完全相同（应判为重复）===
    ("search[head of NASA during Apollo 11]", "search[head of NASA during Apollo 11]", True, "完全相同-search"),
    ("search[The Terminal director]", "search[The Terminal director]", True, "完全相同-search"),
    ("search[Andy Murray brother]", "search[Andy Murray brother]", True, "完全相同-search"),
    ("search[author of The Jungle]", "search[author of The Jungle]", True, "完全相同-search"),
    ("search[American Hustle director]", "search[American Hustle director]", True, "完全相同-American Hustle"),
    ("search[Beaches 1988 film director]", "search[Beaches 1988 film director]", True, "完全相同-Beaches 1988"),
    ("search[Battle of Tarawa date]", "search[Battle of Tarawa date]", True, "完全相同-Battle of Tarawa"),
    ("search[Alicia Keys education university]", "search[Alicia Keys education university]", True, "完全相同-Alicia Keys"),
    ("search[Alessandro Allori Bronzino adoption]", "search[Alessandro Allori Bronzino adoption]", True, "完全相同-Alessandro Allori"),
    ("search[Alexandros Matsas birthplace]", "search[Alexandros Matsas birthplace]", True, "完全相同-Alexandros Matsas"),

    # === 大小写差异 ===
    ("search[The Terminal director]", "search[the terminal director]", True, "大小写-The Terminal"),
    ("search[The Believers composer]", "search[the believers composer]", True, "大小写-The Believers"),
    ("search[The Last Days of Pompeii producer]", "search[the last days of pompeii producer]", True, "大小写-The Last Days"),
    ("search[NASA administrator Apollo 11]", "search[nasa administrator apollo 11]", True, "大小写-NASA Apollo"),
    ("search[Britney Spears birthplace city state]", "search[britney spears birthplace city state]", True, "大小写-Britney Spears"),
    ("search[Andy Murray brother]", "search[andy murray brother]", True, "大小写-Andy Murray"),
    ("search[\"Before Midnight\" screenwriter Ethan Hawke]", "search[before midnight ethan hawke screenwriter]", True, "大小写-Before Midnight"),
    ("search[Marty's girlfriend Back to the Future actress]", "search[marty's girlfriend back to the future actress]", True, "大小写-Back to the Future"),
    ("search[American Hustle director]", "search[american hustle director]", True, "大小写-American Hustle"),
    ("search[Beaches 1988 film director]", "search[beaches 1988 film director]", True, "大小写-Beaches 1988"),
    ("search[Alessandro Allori Bronzino adoption]", "search[alessandro allori bronzino adoption]", True, "大小写-Alessandro Allori"),
    ("search[Alexandros Matsas birthplace]", "search[alexandros matsas birthplace]", True, "大小写-Alexandros Matsas"),
    ("search[Battle of Tarawa date]", "search[battle of tarawa date]", True, "大小写-Battle of Tarawa"),

    # # === 同义词改写（不同词汇/句式表达同一检索意图）===
    # # head of NASA / NASA administrator / who was in charge of NASA（同义改写，均源自真实轨迹）
    # ("search[head of NASA during Apollo 11]", "search[NASA administrator Apollo 11]", True, "同义-NASA head改写"),  # 同义改写
    # ("search[head of NASA during Apollo 11]", "search[who was in charge of NASA during Apollo 11]", True, "同义-NASA head疑问句改写"),  # 同义改写
    # # director / who directed / director of（同义改写，均源自真实轨迹）
    # ("search[The Terminal director]", "search[who directed The Terminal]", True, "同义-director改写"),  # 同义改写
    # ("search[The Terminal director]", "search[The Terminal 2004 director]", True, "同义-director+年份限定"),
    # # author（同义改写，源自真实轨迹）
    # ("search[\"The Arrangement\" author]", "search[author of The Arrangement]", True, "同义-The Arrangement author改写"),  # 同义改写
    # # producer / who produced（同义改写，源自真实轨迹）
    # ("search[The Last Days of Pompeii producer]", "search[who produced The Last Days of Pompeii]", True, "同义-producer改写"),  # 同义改写
    # # composer / who composed（同义改写，源自真实轨迹）
    # ("search[The Believers composer]", "search[\"The Believers\" 1987 film composer]", True, "同义-composer改写"),  # 同义改写
    # # screenwriter / screenwriter of / writer（同义改写，均源自真实轨迹）
    # ("search[screenwriter of Before Midnight]", "search[screenwriter Before Midnight]", True, "同义-screenwriter改写"),  # 同义改写
    # ("search[screenwriter of Before Midnight]", "search[writer Before Midnight Before Sunrise After Midnight]", True, "同义-screenwriter=writer"),
    # # birthplace / born in / where was ... born（同义改写，源自真实轨迹）
    # ("search[Britney Spears birthplace city state]", "search[Britney Spears born in what city]", True, "同义-birthplace改写"),  # 同义改写
    # ("search[Britney Spears birthplace city state]", "search[Britney Spears born in what city state]", True, "同义-birthplace改写2"),  # 同义改写
    # # brother / sibling / family（同义改写，均源自真实轨迹）
    # ("search[Andy Murray brother]", "search[Andy Murray family brother]", True, "同义-brother+family"),
    # ("search[Andy Murray brother]", "search[What is Andy Murray's brother]", True, "同义-brother疑问句改写"),  # 同义改写
    # # fragrance vs cologne（同义改写，源自真实轨迹）
    # ("search[Curious women's fragrance singer]", "search[Curious women's cologne singer]", True, "同义-Curious fragrance vs cologne"),
    # # Alexandros Matsas: birthplace / born / city / footballer birth（真实轨迹四种同义改写）
    # ("search[Alexandros Matsas birthplace]", "search[Alexandros Matsas born]", True, "同义-Matsas birthplace vs born"),
    # ("search[Alexandros Matsas birthplace]", "search[Alexandros Matsas city]", True, "同义-Matsas birthplace vs city"),
    # ("search[Alexandros Matsas born]", "search[Alexandros Matsas footballer birth]", True, "同义-Matsas born vs footballer birth"),
    # # Alicia Keys: education university / performed studies / studied at（真实轨迹三种同义改写）
    # ("search[Alicia Keys education university]", "search[Alicia Keys performed studies]", True, "同义-Alicia Keys education vs studies"),
    # ("search[Alicia Keys education university]", "search[Alicia Keys studied at]", True, "同义-Alicia Keys education vs studied at"),
    # ("search[Alicia Keys education university]", "search[Alicia Keys singer education background]", True, "同义-Alicia Keys education vs background"),
    # # Beaches 1988: film director / Garry Marshall（同义改写，前者描述目标，后者直接给出导演名）
    # ("search[Beaches 1988 film director]", "search[Beaches 1988 Garry Marshall]", True, "同义-Beaches director vs Garry Marshall"),
    # # American Hustle: director / film director（同义改写）
    # ("search[American Hustle director]", "search[American Hustle film director]", True, "同义-American Hustle director改写"),  # 同义改写
    # # Battle of Tarawa: date / 1943（同义改写）
    # ("search[Battle of Tarawa date]", "search[Battle of Tarawa 1943]", True, "同义-Tarawa date vs 1943"),
    # # Alessandro Allori: adoption / uncle death（同一关系不同角度的改写）
    # ("search[Alessandro Allori Bronzino adoption]", "search[Alessandro Allori Bronzino uncle death]", True, "同义-Allori adoption vs uncle death"),
    # ("search[Alessandro Allori Bronzino adoption]", "search[Alessandro Allori uncle Bronzino death]", True, "同义-Allori adoption vs uncle death词序"),
    # # Came Home: father / father of / filly father（同义改写）
    # ("search[\"Came Home\" father]", "search[\"Came Home\" father of]", True, "同义-Came Home father vs father of"),
    # ("search[\"Came Home\" father]", "search[\"Came Home\" filly father]", True, "同义-Came Home father vs filly father"),
    # ("search[\"Came Home\" father]", "search[Came Home father]", True, "同义-Came Home father引号差异"),
    # # Bronzino: uncle / adopted name（同义改写）
    # ("search[\"Bronzino\" \"uncle\" painting]", "search[\"Bronzino\" adopted name painter]", True, "同义-Bronzino uncle vs adoption"),
    # # Bob Boles: opera / opera character / opera role（同义改写）
    # ("search[\"Bob Boles\" opera]", "search[\"Bob Boles\" opera character]", True, "同义-Bob Boles opera vs character"),
    # ("search[\"Bob Boles\" opera]", "search[\"Bob Boles\" opera role]", True, "同义-Bob Boles opera vs role"),
    # ("search[\"Bob Boles\" opera]", "search[Bob Boles character opera]", True, "同义-Bob Boles opera引号差异"),
    # ("search[Bob Boles opera]", "search[Bob Boles role in opera]", True, "同义-Bob Boles opera vs role in opera"),
    # # Bankim Chandra: brother / sibling / siblings（同义改写）
    # ("search[\"Bankim Chandra Chatterjee\" brother]", "search[Bankim Chandra Chattopadhyay brother]", True, "同义-Chatterjee brother姓氏变体"),
    # ("search[Bankim Chandra Chattopadhyay brother]", "search[Bankim Chandra Chattopadhyay sibling]", True, "同义-Chattopadhyay brother vs sibling"),
    # ("search[Bankim Chandra Chattopadhyay sibling]", "search[Bankim Chandra Chattopadhyay siblings]", True, "同义-Chattopadhyay sibling vs siblings单复数"),
    # # Brownsville Illinois: 陈述句 vs 疑问句（同义改写）
    # ("search[Brownsville Illinois capital became capital of Illinois 1839]", "search[when did Brownsville become the capital of Illinois]", True, "同义-Brownsville陈述句vs疑问句"),

    # === 词序重排（关键词集合相同，语序不同）===
    ("search[Team USA starting pitchers 2000 Summer Olympics]", "search[2000 Summer Olympics baseball USA starting pitchers]", True, "词序-Team USA重排"),
    ("search[Team USA starting pitchers 2000 Summer Olympics]", "search[Baseball at 2000 Summer Olympics USA starting pitchers list]", True, "词序-Team USA重排2"),
    ("search[BBC Sports Personality of the Year awards winner most wins]", "search[player with most BBC Sports Personality of the Year awards]", True, "词序-BBC奖项重排"),
    ("search[BBC Sports Personality of the Year awards winner most wins]", "search[who won the most BBC Sports Personality of the Year]", True, "词序-BBC奖项重排2"),
    ("search[Oak Beach New York between which island]", "search[Island between Oak Beach and Great South Bay New York]", True, "词序-Oak Beach重排"),
    ("search[Alessandro Allori Bronzino uncle death]", "search[Alessandro Allori uncle Bronzino death]", True, "词序-Allori uncle Bronzino重排"),
    ("search[\"Bob Boles\" \"opera\" cast]", "search[\"Bob Boles\" cast \"opera\"]", True, "词序-Bob Boles opera cast重排"),
    ("search[Ned Keene Bob Boles opera]", "search[opera Bob Boles Ned Keene]", True, "词序-Ned Keene Bob Boles重排"),
    ("search[Ned Keene Bob Boles opera together]", "search[Ned Keene Bob Boles together opera]", True, "词序-Ned Keene together重排"),

    # === 引号差异（查询词加引号 vs 不加引号，检索意图相同）===
    ("search[\"Curious\" fragrance singer]", "search[Curious women's fragrance singer]", True, "引号-Curious加引号"),
    ("search[\"The Arrangement\" author]", "search[author of The Arrangement]", True, "引号-The Arrangement加引号"),
    ("search[\"Umchabezi River\" mouth]", "search[Umchabezi River mouth]", True, "引号-Umchabezi River加引号"),
    ("search[\"Before Midnight\" screenwriter Ethan Hawke]", "search[Before Midnight Ethan Hawke screenwriter]", True, "引号-Before Midnight加引号"),
    ("search[\"Bankim Chandra Chatterjee\" brother]", "search[Bankim Chandra Chatterjee brother]", True, "引号-Bankim Chandra加引号"),
    ("search[\"Came Home\" father]", "search[Came Home father Gone West]", True, "引号-Came Home father加引号"),
    ("search[\"Came Home\" \"father\" \"Gone West\"]", "search[\"Came Home\" \"father\" Gone West]", True, "引号-Came Home Gone West引号差异"),
    ("search[\"Bob Boles\" opera]", "search[Bob Boles opera]", True, "引号-Bob Boles加引号"),
    ("search[\"Bronzino\" \"uncle\" painting]", "search[Bronzino uncle painting]", True, "引号-Bronzino uncle加引号"),

    # === 空格差异（缺空格/多空格）===
    ("search[moon landing NASA administrator]", "search[moon landingNASA administrator]", True, "空格-moon landingNASA缺空格"),

    # === 单复数（同一查询的单复数变体）===
    ("search[Team USA starting pitchers 2000 Summer Olympics]", "search[Team USA starting pitcher 2000 Summer Olympics]", True, "单复数-pitchers vs pitcher"),
    ("search[BBC Sports Personality of the Year awards winner most wins]", "search[BBC Sports Personality of the Year award winner most wins]", True, "单复数-awards vs award"),
    ("search[Great South Bay between which islands]", "search[Great South Bay between which island]", True, "单复数-islands vs island"),
    ("search[Murray brothers tennis]", "search[Murray brother tennis]", True, "单复数-brothers vs brother"),

    # === 短查询 ===
    ("search[mattress]", "search[mattress]", True, "短查询-完全相同"),
    ("search[keywords]", "search[keywords]", True, "短查询-占位符相同"),
    ("search[Andy Murray]", "search[Andy Murray]", True, "短查询-人名相同"),
    ("search[American Hustle]", "search[American Hustle]", True, "短查询-American Hustle相同"),
    ("search[Battle of Tarawa]", "search[Battle of Tarawa]", True, "短查询-Battle of Tarawa相同"),
    # 短查询增加修饰词视为不同（A是B的子集）
    ("search[Andy Murray]", "search[Andy Murray brother]", False, "不同-短查询增加brother"),
    ("search[Britney Spears]", "search[Britney Spears birthplace]", False, "不同-短查询增加birthplace"),
    ("search[The Terminal]", "search[The Terminal director]", False, "不同-短查询增加director"),
    ("search[American Hustle]", "search[American Hustle director]", False, "不同-短查询American Hustle增加director"),
    ("search[Battle of Tarawa]", "search[Battle of Tarawa date]", False, "不同-短查询Tarawa增加date"),
    ("search[Battle of Tarawa]", "search[Battle of Tarawa 1943]", False, "不同-短查询Tarawa增加1943"),

    # === 拼写/错别字 ===
    ("search[Andy Murray brother]", "search[Andrey Murray brother tennis]", False, "不同-Andy vs Andrey误拼"),
    ("search[Oak Beach New York between which island]", "search[Oakh Beach New York location]", False, "不同-Oak Beach vs Oakh Beach"),

    # === 不同查询意图（反例：不同实体/属性/领域，不应判为重复）===
    # 不同电影/不同导演
    ("search[The Terminal director]", "search[The Last Days of Pompeii producer]", False, "不同-The Terminal vs Last Days of Pompeii"),
    ("search[screenwriter of Before Midnight]", "search[Marty's girlfriend Back to the Future actress]", False, "不同-Before Midnight vs Back to the Future"),
    ("search[American Hustle director]", "search[Beaches 1988 film director]", False, "不同-American Hustle vs Beaches"),
    ("search[American Hustle director]", "search[The Terminal director]", False, "不同-American Hustle vs The Terminal"),
    # 不同人物
    ("search[Andy Murray brother]", "search[Britney Spears birthplace city state]", False, "不同-Andy Murray vs Britney Spears"),
    ("search[author of The Jungle]", "search[\"The Arrangement\" author]", False, "不同-The Jungle vs The Arrangement"),
    ("search[Alexandros Matsas birthplace]", "search[Alicia Keys education university]", False, "不同-Matsas vs Alicia Keys"),
    ("search[Alessandro Allori Bronzino adoption]", "search[Alexandros Matsas birthplace]", False, "不同-Allori vs Matsas"),
    # 不同属性（同一实体不同查询维度）
    ("search[Andy Murray brother]", "search[Andy Murray birthplace]", False, "不同-Andy Murray brother vs birthplace"),
    ("search[Britney Spears birthplace city state]", "search[Britney Spears fragrance]", False, "不同-Britney Spears birthplace vs fragrance"),
    ("search[Beaches 1988 film director]", "search[Beaches 1988 film director nationality]", False, "不同-Beaches director vs nationality"),
    ("search[Beaches 1988 film director]", "search[Beaches 1988 film director American]", False, "不同-Beaches director vs American"),
    ("search[Beaches 1988 film director]", "search[Beaches 1988 film director British]", False, "不同-Beaches director vs British"),
    # 不同机构/赛事/事件
    ("search[head of NASA during Apollo 11]", "search[BBC Sports Personality of the Year awards winner most wins]", False, "不同-NASA vs BBC"),
    ("search[Team USA starting pitchers 2000 Summer Olympics]", "search[Oak Beach New York between which island]", False, "不同-奥运会 vs Oak Beach"),
    ("search[Battle of Tarawa date]", "search[Battle of the Ch'ongch'on River 1945]", False, "不同-Tarawa vs Chongchon River"),
    # 同一实体不同关系
    ("search[\"Bankim Chandra Chatterjee\" brother]", "search[Bankim Chandra Chattopadhyay sister]", False, "不同-Chatterjee brother vs sister"),
    ("search[\"Bankim Chandra Chatterjee\" brother]", "search[who is the wife of Bankim Chandra Chattopadhyay]", False, "不同-Chatterjee brother vs wife"),
    # 1词差异-不同意图
    ("search[Andy Murray brother]", "search[Andy Murray sister]", False, "不同-brother vs sister"),
    ("search[Britney Spears birthplace city state]", "search[Britney Spears nationality country]", False, "不同-birthplace vs nationality"),
    ("search[The Terminal director]", "search[The Terminal producer]", False, "不同-director vs producer"),
    ("search[author of The Jungle]", "search[illustrator of The Jungle]", False, "不同-author vs illustrator"),
    ("search[The Believers composer]", "search[The Believers lyricist]", False, "不同-composer vs lyricist"),
    ("search[\"Bob Boles\" opera]", "search[Bob Boles opera roles other than La boheme]", False, "不同-Bob Boles opera vs roles excluding La boheme"),
    ("search[Albany International Airport location]", "search[Albany International Airport town]", False, "不同-Albany location vs town"),
    ("search[Albany International Airport location]", "search[Albany International Airport town nearly Albany]", False, "不同-Albany location vs town nearly"),
    ("search[\"Bronzino\" \"uncle\" painting]", "search[Bronzino name change uncle death]", False, "不同-Bronzino uncle vs name change"),
    ("search[\"Bronzino\" \"uncle\" painting]", "search[Bronzino painter 1535-1607 uncle]", False, "不同-Bronzino uncle vs painter年份"),
    ("search[\"Came Home\" father]", "search[\"Came Home\" misspelled]", False, "不同-Came Home father vs misspelled"),
    ("search[\"Came Home\" father]", "search[Came Home horse father]", False, "不同-Came Home father vs horse father"),
    ("search[\"Came Home\" father]", "search[Came Home movie father]", False, "不同-Came Home father vs movie father"),
    ("search[\"Came Home\" father]", "search[Came Home paternal father]", False, "不同-Came Home father vs paternal father"),
    ("search[head of NASA during Apollo 11]", "search[Apollo 11 mission commander NASA]", False, "不同-NASA head vs commander"),

    # ================================================================
    # 二、answer 动作区（所有 answer 动作按同义类型合并组织）
    # ================================================================

    # === 完全相同（应判为重复）===
    ("answer[Thomas O. Paine]", "answer[Thomas O. Paine]", True, "完全相同-answer"),
    ("answer[Steven Spielberg]", "answer[Steven Spielberg]", True, "完全相同-answer"),
    ("answer[Upton Sinclair]", "answer[Upton Sinclair]", True, "完全相同-answer"),
    ("answer[Elia Kazan]", "answer[Elia Kazan]", True, "完全相同-answer"),
    ("answer[Long Island]", "answer[Long Island]", True, "完全相同-answer"),
    ("answer[Jamie Murray]", "answer[Jamie Murray]", True, "完全相同-answer"),
    ("answer[Alessandro Allori]", "answer[Alessandro Allori]", True, "完全相同-Alessandro Allori"),
    ("answer[Antonio Vivaldi]", "answer[Antonio Vivaldi]", True, "完全相同-Antonio Vivaldi"),
    ("answer[Anthony Hopkins]", "answer[Anthony Hopkins]", True, "完全相同-Anthony Hopkins"),
    ("answer[Bryan Cranston]", "answer[Bryan Cranston]", True, "完全相同-Bryan Cranston"),
    ("answer[Baton Rouge, Louisiana]", "answer[Baton Rouge, Louisiana]", True, "完全相同-Baton Rouge"),
    ("answer[Columbia University]", "answer[Columbia University]", True, "完全相同-Columbia University"),

    # === 大小写差异 ===
    ("answer[Steven Spielberg]", "answer[steven spielberg]", True, "大小写-answer Spielberg"),
    ("answer[Thomas O. Paine]", "answer[thomas o. paine]", True, "大小写-answer Paine"),
    ("answer[Upton Sinclair]", "answer[upton sinclair]", True, "大小写-answer Sinclair"),
    ("answer[Long Island]", "answer[long island]", True, "大小写-answer Long Island"),
    ("answer[Alessandro Allori]", "answer[alessandro allori]", True, "大小写-Alessandro Allori"),
    ("answer[Antonio Vivaldi]", "answer[antonio vivaldi]", True, "大小写-Antonio Vivaldi"),
    ("answer[Anthony Hopkins]", "answer[anthony hopkins]", True, "大小写-Anthony Hopkins"),
    ("answer[Bryan Cranston]", "answer[bryan cranston]", True, "大小写-Bryan Cranston"),
    ("answer[Columbia University]", "answer[columbia university]", True, "大小写-Columbia University"),

    # === 同义表达（全称 vs 简称、逗号差异、拼写变体）（环境端 answer 完全匹配，同义无效，应判为不同）===
    # 全称 vs 简称
    ("answer[George IV of the United Kingdom]", "answer[George IV]", False, "同义-全称 vs 简称 George IV"),
    ("answer[King George IV]", "answer[George IV]", False, "同义-King George IV vs George IV"),
    # 全名 vs 姓
    ("answer[Thomas O. Paine]", "answer[Thomas Paine]", False, "同义-Thomas O. Paine vs Thomas Paine"),
    ("answer[Upton Sinclair]", "answer[Sinclair]", False, "同义-Upton Sinclair vs Sinclair"),
    # city, state 组合（逗号/缩写差异）
    ("answer[McComb, Mississippi]", "answer[McComb Mississippi]", False, "同义-逗号差异 McComb"),
    ("answer[McComb, Mississippi]", "answer[McComb, MS]", False, "同义-全称 vs 缩写 Mississippi MS"),
    ("answer[Baton Rouge, Louisiana]", "answer[Baton Rouge Louisiana]", False, "同义-Baton Rouge逗号差异"),
    ("answer[Baton Rouge, Louisiana]", "answer[Baton Rouge, LA]", False, "同义-Baton Rouge全称vs缩写"),
    # 拼写变体/中间名
    ("answer[Steven Spielberg]", "answer[Stephen Spielberg]", False, "同义-Steven vs Stephen Spielberg"),
    ("answer[Claudia Wells]", "answer[Claudia Grace Wells]", False, "同义-Claudia Wells vs Claudia Grace Wells"),

    # === 不同答案（同一问题不同答案，不应判为重复）===
    ("answer[Thomas O. Paine]", "answer[Steven Spielberg]", False, "不同-answer Paine vs Spielberg"),
    ("answer[Upton Sinclair]", "answer[Elia Kazan]", False, "不同-answer Sinclair vs Kazan"),
    ("answer[Jamie Murray]", "answer[Andy Murray]", False, "不同-answer Jamie vs Andy Murray"),
    ("answer[Long Island]", "answer[McComb, Mississippi]", False, "不同-answer Long Island vs McComb"),
    ("answer[Thomas O. Paine]", "answer[Thomas Paine]", False, "同义-Thomas O. Paine vs Thomas Paine（同义区域）"),
    ("answer[Andy Murray]", "answer[Jamie Murray]", False, "不同-answer Andy vs Jamie Murray"),
    ("answer[Upton Sinclair]", "answer[Christine Sinclair]", False, "不同-Upton Sinclair vs Christine Sinclair"),
    ("answer[Alessandro Allori]", "answer[Antonio Vivaldi]", False, "不同-Allori vs Vivaldi"),
    ("answer[Anthony Hopkins]", "answer[Bryan Cranston]", False, "不同-Hopkins vs Cranston"),
    ("answer[Baton Rouge, Louisiana]", "answer[Columbia University]", False, "不同-Baton Rouge vs Columbia"),
    ("answer[Claudia Wells]", "answer[Alessandro Allori]", False, "不同-Claudia Wells vs Allori"),

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

    # === 真实轨迹扩充-跨类型（真实轨迹中的 search/answer 对）===
    ("search[American Hustle director]", "answer[David O. Russell]", False, "跨类型-search American Hustle vs answer"),
    ("search[Beaches 1988 film director]", "answer[Garry Marshall]", False, "跨类型-search Beaches vs answer"),
    ("search[Battle of Tarawa date]", "answer[1943]", False, "跨类型-search Tarawa vs answer"),
    ("search[Alessandro Allori Bronzino adoption]", "answer[Alessandro Allori]", False, "跨类型-search Allori vs answer"),
    ("search[Alexandros Matsas birthplace]", "answer[Athens]", False, "跨类型-search Matsas vs answer"),
    ("search[Alicia Keys education university]", "answer[Columbia University]", False, "跨类型-search Alicia Keys vs answer"),
    ("answer[Antonio Vivaldi]", "search[Alessandro Allori Bronzino adoption]", False, "跨类型-answer Vivaldi vs search Allori"),
    ("answer[Bryan Cranston]", "search[American Hustle director]", False, "跨类型-answer Cranston vs search American Hustle"),
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
    # search 动作-真实轨迹扩充样本
    "search[American Hustle director]",
    "search[American Hustle film director]",
    "search[Beaches 1988 film director]",
    "search[Beaches 1988 Garry Marshall]",
    "search[Battle of Tarawa date]",
    "search[Battle of Tarawa 1943]",
    "search[Alicia Keys education university]",
    "search[Alicia Keys performed studies]",
    "search[Alicia Keys studied at]",
    "search[Alessandro Allori Bronzino adoption]",
    "search[Alessandro Allori Bronzino uncle death]",
    "search[Alessandro Allori uncle Bronzino death]",
    "search[Alexandros Matsas birthplace]",
    "search[Alexandros Matsas born]",
    "search[Alexandros Matsas city]",
    "search[Alexandros Matsas footballer birth]",
    "search[\"Bankim Chandra Chatterjee\" brother]",
    "search[Bankim Chandra Chattopadhyay brother]",
    "search[Bankim Chandra Chattopadhyay sibling]",
    "search[Bankim Chandra Chattopadhyay sister]",
    "search[\"Bob Boles\" opera]",
    "search[\"Bob Boles\" opera character]",
    "search[\"Bob Boles\" opera role]",
    "search[Bob Boles role in opera]",
    "search[\"Came Home\" father]",
    "search[\"Came Home\" father of]",
    "search[\"Came Home\" filly father]",
    "search[\"Bronzino\" \"uncle\" painting]",
    "search[\"Bronzino\" adopted name painter]",
    "search[Ned Keene Bob Boles opera]",
    "search[opera Bob Boles Ned Keene]",
    "search[Albany International Airport location]",
    "search[Albany International Airport town]",
    "search[Brownsville Illinois capital became capital of Illinois 1839]",
    "search[when did Brownsville become the capital of Illinois]",
    # answer 动作-真实轨迹扩充样本
    "answer[Alessandro Allori]",
    "answer[Antonio Vivaldi]",
    "answer[Anthony Hopkins]",
    "answer[Bryan Cranston]",
    "answer[Baton Rouge, Louisiana]",
    "answer[Baton Rouge Louisiana]",
    "answer[Baton Rouge, LA]",
    "answer[Columbia University]",
    "answer[Garry Marshall]",
    "answer[David O. Russell]",
]
