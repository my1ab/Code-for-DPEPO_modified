"""
Search（QA 检索）任务测试样例数据

数据来源：test_bert_similarity/验证轨迹search/ 下的真实轨迹
（bamboogle / hotpotqa / musique / nq / popqa / triviaqa / 2wikimultihopqa）

动作类型（与 get_actions 字段格式一致）：
- <search>query</search>   : 检索动作
- <answer>answer</answer>  : 终止作答动作
- null                     : 该环境本轮不动作

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
  （如 <search>head of NASA during Apollo 11</search> / <search>Andy Murray family brother</search> /
    <search>Marty's girlfriend Back to the Future actress</search> / <answer>Thomas O. Paine</answer> 等）
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
    ("<search>head of NASA during Apollo 11</search>", "<search>head of NASA during Apollo 11</search>", True, "完全相同-search"),
    ("<search>The Terminal director</search>", "<search>The Terminal director</search>", True, "完全相同-search"),
    ("<search>Andy Murray brother</search>", "<search>Andy Murray brother</search>", True, "完全相同-search"),
    ("<search>author of The Jungle</search>", "<search>author of The Jungle</search>", True, "完全相同-search"),

    # === 大小写差异（同一查询的不同大小写，真实数据中最常见的变体）===
    # 句首字母大小写差异
    ("<search>The Terminal director</search>", "<search>the terminal director</search>", True, "大小写-The Terminal"),
    ("<search>The Believers composer</search>", "<search>the believers composer</search>", True, "大小写-The Believers"),
    ("<search>The Last Days of Pompeii producer</search>", "<search>the last days of pompeii producer</search>", True, "大小写-The Last Days"),
    # 专有名词大小写差异
    ("<search>NASA administrator Apollo 11</search>", "<search>nasa administrator apollo 11</search>", True, "大小写-NASA Apollo"),
    ("<search>Britney Spears birthplace city state</search>", "<search>britney spears birthplace city state</search>", True, "大小写-Britney Spears"),
    ("<search>Andy Murray brother</search>", "<search>andy murray brother</search>", True, "大小写-Andy Murray"),
    # 标题/电影名大小写（真实表述带引号，构造对照为纯小写）
    ("<search>\"Before Midnight\" screenwriter Ethan Hawke</search>", "<search>before midnight ethan hawke screenwriter</search>", True, "大小写-Before Midnight"),
    ("<search>Marty's girlfriend Back to the Future actress</search>", "<search>marty's girlfriend back to the future actress</search>", True, "大小写-Back to the Future"),

    # === 同义查询-句式重排（同一检索意图的不同表达，应判为相同）===
    # head of NASA / NASA administrator / who was in charge of NASA（同义改写，均源自真实轨迹）
    ("<search>head of NASA during Apollo 11</search>", "<search>NASA administrator Apollo 11</search>", True, "同义-NASA head改写"),
    ("<search>head of NASA during Apollo 11</search>", "<search>who was in charge of NASA during Apollo 11</search>", True, "同义-NASA head疑问句改写"),
    ("<search>head of NASA during Apollo 11</search>", "<search>Apollo 11 mission commander NASA</search>", False, "不同-NASA head vs commander"),
    # director / who directed / director of（同义动词改写，均源自真实轨迹）
    ("<search>The Terminal director</search>", "<search>who directed The Terminal</search>", True, "同义-director改写"),
    ("<search>The Terminal director</search>", "<search>The Terminal 2004 director</search>", True, "同义-director+年份限定"),
    # author（The Arrangement 用真实带引号表述，与 The Jungle 是不同实体）
    ("<search>\"The Arrangement\" author</search>", "<search>author of The Arrangement</search>", True, "同义-The Arrangement author改写"),
    # producer / who produced（同义改写，源自真实轨迹）
    ("<search>The Last Days of Pompeii producer</search>", "<search>who produced The Last Days of Pompeii</search>", True, "同义-producer改写"),
    # composer / who composed（同义改写，源自真实轨迹）
    ("<search>The Believers composer</search>", "<search>\"The Believers\" 1987 film composer</search>", True, "同义-composer改写"),
    # screenwriter / screenwriter of / writer（同义改写，均源自真实轨迹）
    ("<search>screenwriter of Before Midnight</search>", "<search>screenwriter Before Midnight</search>", True, "同义-screenwriter改写"),
    ("<search>screenwriter of Before Midnight</search>", "<search>writer Before Midnight Before Sunrise After Midnight</search>", True, "同义-screenwriter=writer"),
    # birthplace / born in / where was ... born（同义改写，源自真实轨迹）
    ("<search>Britney Spears birthplace city state</search>", "<search>Britney Spears born in what city</search>", True, "同义-birthplace改写"),
    ("<search>Britney Spears birthplace city state</search>", "<search>Britney Spears born in what city state</search>", True, "同义-birthplace改写2"),
    # brother / sibling / family（同义改写，均源自真实轨迹）
    ("<search>Andy Murray brother</search>", "<search>Andy Murray family brother</search>", True, "同义-brother+family"),
    ("<search>Andy Murray brother</search>", "<search>What is Andy Murray's brother</search>", True, "同义-brother疑问句改写"),

    # === 同义查询-词序重排（关键词集合相同或相近，顺序不同，应判为相同）===
    # Curious fragrance/cologne singer（真实轨迹中的同义变体）
    ("<search>Curious women's fragrance singer</search>", "<search>Curious women's cologne singer</search>", True, "同义-Curious fragrance vs cologne"),
    # Team USA Olympics（真实轨迹中的词序变体）
    ("<search>Team USA starting pitchers 2000 Summer Olympics</search>", "<search>2000 Summer Olympics baseball USA starting pitchers</search>", True, "词序-Team USA重排"),
    ("<search>Team USA starting pitchers 2000 Summer Olympics</search>", "<search>Baseball at 2000 Summer Olympics USA starting pitchers list</search>", True, "词序-Team USA重排2"),
    # BBC Sports Personality（真实轨迹中的词序变体）
    ("<search>BBC Sports Personality of the Year awards winner most wins</search>", "<search>player with most BBC Sports Personality of the Year awards</search>", True, "词序-BBC奖项重排"),
    ("<search>BBC Sports Personality of the Year awards winner most wins</search>", "<search>who won the most BBC Sports Personality of the Year</search>", True, "词序-BBC奖项重排2"),
    # Oak Beach（真实轨迹中的词序变体）
    ("<search>Oak Beach New York between which island</search>", "<search>Island between Oak Beach and Great South Bay New York</search>", True, "词序-Oak Beach重排"),

    # === 引号差异（查询词加引号 vs 不加引号，检索意图相同，均源自真实轨迹）===
    ("<search>\"Curious\" fragrance singer</search>", "<search>Curious women's fragrance singer</search>", True, "引号-Curious加引号"),
    ("<search>\"The Arrangement\" author</search>", "<search>author of The Arrangement</search>", True, "引号-The Arrangement加引号"),
    ("<search>\"Umchabezi River\" mouth</search>", "<search>Umchabezi River mouth</search>", True, "引号-Umchabezi River加引号"),
    ("<search>\"Before Midnight\" screenwriter Ethan Hawke</search>", "<search>Before Midnight Ethan Hawke screenwriter</search>", True, "引号-Before Midnight加引号"),

    # === 单复数/形态变化（应判为相同，复数为真实轨迹表述，单数为构造对照）===
    ("<search>Team USA starting pitchers 2000 Summer Olympics</search>", "<search>Team USA starting pitcher 2000 Summer Olympics</search>", True, "单复数-pitchers vs pitcher"),
    ("<search>BBC Sports Personality of the Year awards winner most wins</search>", "<search>BBC Sports Personality of the Year award winner most wins</search>", True, "单复数-awards vs award"),
    ("<search>Great South Bay between which islands</search>", "<search>Great South Bay between which island</search>", True, "单复数-islands vs island"),
    ("<search>Murray brothers tennis</search>", "<search>Murray brother tennis</search>", True, "单复数-brothers vs brother"),

    # === 短查询（前缀占比大，测试前缀剥离效果）===
    ("<search>mattress</search>", "<search>mattress</search>", True, "短查询-完全相同"),
    ("<search>keywords</search>", "<search>keywords</search>", True, "短查询-占位符相同"),
    ("<search>Andy Murray</search>", "<search>Andy Murray</search>", True, "短查询-人名相同"),
    # 短查询增加修饰词视为不同（A是B的子集）
    ("<search>Andy Murray</search>", "<search>Andy Murray brother</search>", False, "不同-短查询增加brother"),
    ("<search>Britney Spears</search>", "<search>Britney Spears birthplace</search>", False, "不同-短查询增加birthplace"),
    ("<search>The Terminal</search>", "<search>The Terminal director</search>", False, "不同-短查询增加director"),

    # === 不同查询意图（同一领域不同实体，不应判为重复）===
    # 不同电影
    ("<search>The Terminal director</search>", "<search>The Last Days of Pompeii producer</search>", False, "不同-The Terminal vs Last Days of Pompeii"),
    ("<search>screenwriter of Before Midnight</search>", "<search>Marty's girlfriend Back to the Future actress</search>", False, "不同-Before Midnight vs Back to the Future"),
    # 不同人物
    ("<search>Andy Murray brother</search>", "<search>Britney Spears birthplace city state</search>", False, "不同-Andy Murray vs Britney Spears"),
    ("<search>author of The Jungle</search>", "<search>\"The Arrangement\" author</search>", False, "不同-The Jungle vs The Arrangement"),
    # 不同属性（同一实体不同查询维度）
    ("<search>Andy Murray brother</search>", "<search>Andy Murray birthplace</search>", False, "不同-Andy Murray brother vs birthplace"),
    ("<search>Britney Spears birthplace city state</search>", "<search>Britney Spears fragrance</search>", False, "不同-Britney Spears birthplace vs fragrance"),
    # 不同机构/赛事
    ("<search>head of NASA during Apollo 11</search>", "<search>BBC Sports Personality of the Year awards winner most wins</search>", False, "不同-NASA vs BBC"),
    ("<search>Team USA starting pitchers 2000 Summer Olympics</search>", "<search>Oak Beach New York between which island</search>", False, "不同-奥运会 vs Oak Beach"),

    # === 1词差异-不同实体（验证阈值下方的区分能力）===
    # 兄弟 vs 姐妹
    ("<search>Andy Murray brother</search>", "<search>Andy Murray sister</search>", False, "不同-brother vs sister"),
    # 出生地 vs 国籍
    ("<search>Britney Spears birthplace city state</search>", "<search>Britney Spears nationality country</search>", False, "不同-birthplace vs nationality"),
    # 导演 vs 制片人
    ("<search>The Terminal director</search>", "<search>The Terminal producer</search>", False, "不同-director vs producer"),
    # 作者 vs 插画师（插画师为构造对照，真实轨迹无此查询）
    ("<search>author of The Jungle</search>", "<search>illustrator of The Jungle</search>", False, "不同-author vs illustrator"),
    # composer vs lyricist（lyricist 为构造对照）
    ("<search>The Believers composer</search>", "<search>The Believers lyricist</search>", False, "不同-composer vs lyricist"),

    # === 错别字/拼写差异（真实数据中存在）===
    # Andrey vs Andy（真实轨迹中的误拼变体）
    ("<search>Andy Murray brother</search>", "<search>Andrey Murray brother tennis</search>", False, "不同-Andy vs Andrey误拼"),
    # Oakh Beach vs Oak Beach（真实轨迹中的多字母误拼）
    ("<search>Oak Beach New York between which island</search>", "<search>Oakh Beach New York location</search>", False, "不同-Oak Beach vs Oakh Beach"),
    # moon landingNASA（真实轨迹中存在的缺空格变体）
    ("<search>moon landing NASA administrator</search>", "<search>moon landingNASA administrator</search>", True, "同义-缺空格moon landingNASA"),

    # ================================================================
    # 一-B、search 动作区-真实轨迹扩充
    # 以下对均直接引用自 验证轨迹search/ 下 *_success.json
    # 中 assistant turn 的 <search>...</search> 动作
    # ================================================================

    # === 完全相同（真实轨迹直接引用，应判为重复）===
    ("<search>American Hustle director</search>", "<search>American Hustle director</search>", True, "完全相同-American Hustle"),
    ("<search>Beaches 1988 film director</search>", "<search>Beaches 1988 film director</search>", True, "完全相同-Beaches 1988"),
    ("<search>Battle of Tarawa date</search>", "<search>Battle of Tarawa date</search>", True, "完全相同-Battle of Tarawa"),
    ("<search>Alicia Keys education university</search>", "<search>Alicia Keys education university</search>", True, "完全相同-Alicia Keys"),
    ("<search>Alessandro Allori Bronzino adoption</search>", "<search>Alessandro Allori Bronzino adoption</search>", True, "完全相同-Alessandro Allori"),
    ("<search>Alexandros Matsas birthplace</search>", "<search>Alexandros Matsas birthplace</search>", True, "完全相同-Alexandros Matsas"),

    # === 大小写差异（真实轨迹中的查询做大小写变换）===
    ("<search>American Hustle director</search>", "<search>american hustle director</search>", True, "大小写-American Hustle"),
    ("<search>Beaches 1988 film director</search>", "<search>beaches 1988 film director</search>", True, "大小写-Beaches 1988"),
    ("<search>Alessandro Allori Bronzino adoption</search>", "<search>alessandro allori bronzino adoption</search>", True, "大小写-Alessandro Allori"),
    ("<search>Alexandros Matsas birthplace</search>", "<search>alexandros matsas birthplace</search>", True, "大小写-Alexandros Matsas"),
    ("<search>Battle of Tarawa date</search>", "<search>battle of tarawa date</search>", True, "大小写-Battle of Tarawa"),

    # === 同义查询-同义改写（同一检索意图，不同表达方式，均源自真实轨迹）===
    # Alexandros Matsas: birthplace / born / city / footballer birth（真实轨迹四种同义改写）
    ("<search>Alexandros Matsas birthplace</search>", "<search>Alexandros Matsas born</search>", True, "同义-Matsas birthplace vs born"),
    ("<search>Alexandros Matsas birthplace</search>", "<search>Alexandros Matsas city</search>", True, "同义-Matsas birthplace vs city"),
    ("<search>Alexandros Matsas born</search>", "<search>Alexandros Matsas footballer birth</search>", True, "同义-Matsas born vs footballer birth"),
    # Alicia Keys: education university / performed studies / studied at（真实轨迹三种同义改写）
    ("<search>Alicia Keys education university</search>", "<search>Alicia Keys performed studies</search>", True, "同义-Alicia Keys education vs studies"),
    ("<search>Alicia Keys education university</search>", "<search>Alicia Keys studied at</search>", True, "同义-Alicia Keys education vs studied at"),
    ("<search>Alicia Keys education university</search>", "<search>Alicia Keys singer education background</search>", True, "同义-Alicia Keys education vs background"),
    # Beaches 1988: film director / Garry Marshall（真实轨迹中同义改写，前者描述搜索目标，后者直接给出导演名）
    ("<search>Beaches 1988 film director</search>", "<search>Beaches 1988 Garry Marshall</search>", True, "同义-Beaches director vs Garry Marshall"),
    # American Hustle: director / film director（真实轨迹中存在的同义改写）
    ("<search>American Hustle director</search>", "<search>American Hustle film director</search>", True, "同义-American Hustle director改写"),
    # Battle of Tarawa: date / 1943（真实轨迹中同义改写，后者直接给出年份）
    ("<search>Battle of Tarawa date</search>", "<search>Battle of Tarawa 1943</search>", True, "同义-Tarawa date vs 1943"),
    # Alessandro Allori: adoption / uncle death（真实轨迹中同一关系不同角度的改写）
    ("<search>Alessandro Allori Bronzino adoption</search>", "<search>Alessandro Allori Bronzino uncle death</search>", True, "同义-Allori adoption vs uncle death"),
    ("<search>Alessandro Allori Bronzino adoption</search>", "<search>Alessandro Allori uncle Bronzino death</search>", True, "同义-Allori adoption vs uncle death词序"),
    # Came Home: father / father of / filly father（真实轨迹中同义改写）
    ("<search>\"Came Home\" father</search>", "<search>\"Came Home\" father of</search>", True, "同义-Came Home father vs father of"),
    ("<search>\"Came Home\" father</search>", "<search>\"Came Home\" filly father</search>", True, "同义-Came Home father vs filly father"),
    ("<search>\"Came Home\" father</search>", "<search>Came Home father</search>", True, "同义-Came Home father引号差异"),
    # Bronzino: uncle / adopted name（真实轨迹中同义改写）
    ("<search>\"Bronzino\" \"uncle\" painting</search>", "<search>\"Bronzino\" adopted name painter</search>", True, "同义-Bronzino uncle vs adoption"),
    # Bob Boles: opera / opera character / opera role（真实轨迹中同义改写）
    ("<search>\"Bob Boles\" opera</search>", "<search>\"Bob Boles\" opera character</search>", True, "同义-Bob Boles opera vs character"),
    ("<search>\"Bob Boles\" opera</search>", "<search>\"Bob Boles\" opera role</search>", True, "同义-Bob Boles opera vs role"),
    ("<search>\"Bob Boles\" opera</search>", "<search>Bob Boles character opera</search>", True, "同义-Bob Boles opera引号差异"),
    ("<search>Bob Boles opera</search>", "<search>Bob Boles role in opera</search>", True, "同义-Bob Boles opera vs role in opera"),
    # Bankim Chandra: brother / sibling / siblings（真实轨迹中同义改写）
    ("<search>\"Bankim Chandra Chatterjee\" brother</search>", "<search>Bankim Chandra Chattopadhyay brother</search>", True, "同义-Chatterjee brother姓氏变体"),
    ("<search>Bankim Chandra Chattopadhyay brother</search>", "<search>Bankim Chandra Chattopadhyay sibling</search>", True, "同义-Chattopadhyay brother vs sibling"),
    ("<search>Bankim Chandra Chattopadhyay sibling</search>", "<search>Bankim Chandra Chattopadhyay siblings</search>", True, "同义-Chattopadhyay sibling vs siblings单复数"),

    # === 同义查询-词序重排（真实轨迹中同一查询的词序变体）===
    # Alessandro Allori: 两种词序（真实轨迹中直接引用）
    ("<search>Alessandro Allori Bronzino uncle death</search>", "<search>Alessandro Allori uncle Bronzino death</search>", True, "词序-Allori uncle Bronzino重排"),
    # Bob Boles: "opera" cast / cast "opera"（真实轨迹中词序重排）
    ("<search>\"Bob Boles\" \"opera\" cast</search>", "<search>\"Bob Boles\" cast \"opera\"</search>", True, "词序-Bob Boles opera cast重排"),
    # Ned Keene Bob Boles: 两种排列顺序（真实轨迹中直接引用）
    ("<search>Ned Keene Bob Boles opera</search>", "<search>opera Bob Boles Ned Keene</search>", True, "词序-Ned Keene Bob Boles重排"),
    ("<search>Ned Keene Bob Boles opera together</search>", "<search>Ned Keene Bob Boles together opera</search>", True, "词序-Ned Keene together重排"),
    # Brownsville Illinois: 两种句式（真实轨迹中同义改写）
    ("<search>Brownsville Illinois capital became capital of Illinois 1839</search>", "<search>when did Brownsville become the capital of Illinois</search>", True, "同义-Brownsville陈述句vs疑问句"),

    # === 引号差异（真实轨迹中同一查询带引号 vs 不带引号变体）===
    ("<search>\"Bankim Chandra Chatterjee\" brother</search>", "<search>Bankim Chandra Chatterjee brother</search>", True, "引号-Bankim Chandra加引号"),
    ("<search>\"Came Home\" father</search>", "<search>Came Home father Gone West</search>", True, "引号-Came Home father加引号"),
    ("<search>\"Came Home\" \"father\" \"Gone West\"</search>", "<search>\"Came Home\" \"father\" Gone West</search>", True, "引号-Came Home Gone West引号差异"),
    ("<search>\"Bob Boles\" opera</search>", "<search>Bob Boles opera</search>", True, "引号-Bob Boles加引号"),
    ("<search>\"Bronzino\" \"uncle\" painting</search>", "<search>Bronzino uncle painting</search>", True, "引号-Bronzino uncle加引号"),

    # === 短查询（真实轨迹中的短查询变体）===
    ("<search>American Hustle</search>", "<search>American Hustle</search>", True, "短查询-American Hustle相同"),
    ("<search>Battle of Tarawa</search>", "<search>Battle of Tarawa</search>", True, "短查询-Battle of Tarawa相同"),
    # 短查询增加修饰词视为不同（真实轨迹中的变体）
    ("<search>American Hustle</search>", "<search>American Hustle director</search>", False, "不同-短查询American Hustle增加director"),
    ("<search>Battle of Tarawa</search>", "<search>Battle of Tarawa date</search>", False, "不同-短查询Tarawa增加date"),
    ("<search>Battle of Tarawa</search>", "<search>Battle of Tarawa 1943</search>", False, "不同-短查询Tarawa增加1943"),

    # === 不同查询意图-不同实体（真实轨迹中不同的搜索目标，不应判为重复）===
    # 不同电影/不同导演
    ("<search>American Hustle director</search>", "<search>Beaches 1988 film director</search>", False, "不同-American Hustle vs Beaches"),
    ("<search>American Hustle director</search>", "<search>The Terminal director</search>", False, "不同-American Hustle vs The Terminal"),
    # 不同人物
    ("<search>Alexandros Matsas birthplace</search>", "<search>Alicia Keys education university</search>", False, "不同-Matsas vs Alicia Keys"),
    ("<search>Alessandro Allori Bronzino adoption</search>", "<search>Alexandros Matsas birthplace</search>", False, "不同-Allori vs Matsas"),
    # 不同事件
    ("<search>Battle of Tarawa date</search>", "<search>Battle of the Ch'ongch'on River 1945</search>", False, "不同-Tarawa vs Chongchon River"),
    # 不同关系（同一实体不同查询维度）
    ("<search>\"Bankim Chandra Chatterjee\" brother</search>", "<search>Bankim Chandra Chattopadhyay sister</search>", False, "不同-Chatterjee brother vs sister"),
    ("<search>\"Bankim Chandra Chatterjee\" brother</search>", "<search>who is the wife of Bankim Chandra Chattopadhyay</search>", False, "不同-Chatterjee brother vs wife"),
    ("<search>\"Came Home\" father</search>", "<search>\"Came Home\" misspelled</search>", False, "不同-Came Home father vs misspelled"),
    # Beaches 1988: director vs nationality（真实轨迹中不同查询维度）
    ("<search>Beaches 1988 film director</search>", "<search>Beaches 1988 film director nationality</search>", False, "不同-Beaches director vs nationality"),
    ("<search>Beaches 1988 film director</search>", "<search>Beaches 1988 film director American</search>", False, "不同-Beaches director vs American"),
    ("<search>Beaches 1988 film director</search>", "<search>Beaches 1988 film director British</search>", False, "不同-Beaches director vs British"),

    # === 1词差异-不同实体（真实轨迹中的近义不同查询，不应判为重复）===
    # Bob Boles: opera vs opera roles other than La boheme（不同查询意图）
    ("<search>\"Bob Boles\" opera</search>", "<search>Bob Boles opera roles other than La boheme</search>", False, "不同-Bob Boles opera vs roles excluding La boheme"),
    # Albany: location vs town（真实轨迹中不同的查询维度）
    ("<search>Albany International Airport location</search>", "<search>Albany International Airport town</search>", False, "不同-Albany location vs town"),
    ("<search>Albany International Airport location</search>", "<search>Albany International Airport town nearly Albany</search>", False, "不同-Albany location vs town nearly"),
    # Bronzino: uncle vs name change（真实轨迹中不同的查询维度）
    ("<search>\"Bronzino\" \"uncle\" painting</search>", "<search>Bronzino name change uncle death</search>", False, "不同-Bronzino uncle vs name change"),
    ("<search>\"Bronzino\" \"uncle\" painting</search>", "<search>Bronzino painter 1535-1607 uncle</search>", False, "不同-Bronzino uncle vs painter年份"),
    # Came Home: father vs horse father vs movie father（真实轨迹中不同领域歧义）
    ("<search>\"Came Home\" father</search>", "<search>Came Home horse father</search>", False, "不同-Came Home father vs horse father"),
    ("<search>\"Came Home\" father</search>", "<search>Came Home movie father</search>", False, "不同-Came Home father vs movie father"),
    ("<search>\"Came Home\" father</search>", "<search>Came Home paternal father</search>", False, "不同-Came Home father vs paternal father"),

    # ================================================================
    # 二、answer 动作区
    # ================================================================

    # === 完全相同（应判为重复）===
    ("<answer>Thomas O. Paine</answer>", "<answer>Thomas O. Paine</answer>", True, "完全相同-answer"),
    ("<answer>Steven Spielberg</answer>", "<answer>Steven Spielberg</answer>", True, "完全相同-answer"),
    ("<answer>Upton Sinclair</answer>", "<answer>Upton Sinclair</answer>", True, "完全相同-answer"),
    ("<answer>Elia Kazan</answer>", "<answer>Elia Kazan</answer>", True, "完全相同-answer"),
    ("<answer>Long Island</answer>", "<answer>Long Island</answer>", True, "完全相同-answer"),
    ("<answer>Jamie Murray</answer>", "<answer>Jamie Murray</answer>", True, "完全相同-answer"),

    # === 大小写差异（应判为相同）===
    ("<answer>Steven Spielberg</answer>", "<answer>steven spielberg</answer>", True, "大小写-answer Spielberg"),
    ("<answer>Thomas O. Paine</answer>", "<answer>thomas o. paine</answer>", True, "大小写-answer Paine"),
    ("<answer>Upton Sinclair</answer>", "<answer>upton sinclair</answer>", True, "大小写-answer Sinclair"),
    ("<answer>Long Island</answer>", "<answer>long island</answer>", True, "大小写-answer Long Island"),

    # === 同义答案-全称 vs 简称（环境端 answer 完全匹配，同义无效，应判为不同）===
    # 全称 vs 简称
    ("<answer>George IV of the United Kingdom</answer>", "<answer>George IV</answer>", False, "同义-全称 vs 简称 George IV"),
    ("<answer>King George IV</answer>", "<answer>George IV</answer>", False, "同义-King George IV vs George IV"),
    # 全名 vs 姓
    ("<answer>Thomas O. Paine</answer>", "<answer>Thomas Paine</answer>", False, "同义-Thomas O. Paine vs Thomas Paine"),
    ("<answer>Upton Sinclair</answer>", "<answer>Sinclair</answer>", False, "同义-Upton Sinclair vs Sinclair"),

    # === 同义答案-同义表达（环境端 answer 完全匹配，同义无效，应判为不同）===
    # city, state 组合（McComb, Mississippi vs McComb Mississippi）
    ("<answer>McComb, Mississippi</answer>", "<answer>McComb Mississippi</answer>", False, "同义-逗号差异 McComb"),
    ("<answer>McComb, Mississippi</answer>", "<answer>McComb, MS</answer>", False, "同义-全称 vs 缩写 Mississippi MS"),
    # 拼写变体
    ("<answer>Steven Spielberg</answer>", "<answer>Stephen Spielberg</answer>", False, "同义-Steven vs Stephen Spielberg"),
    # 带/不带中间名
    ("<answer>Claudia Wells</answer>", "<answer>Claudia Grace Wells</answer>", False, "同义-Claudia Wells vs Claudia Grace Wells"),

    # === 不同答案（同一问题不同答案，不应判为重复）===
    # 不同人物
    ("<answer>Thomas O. Paine</answer>", "<answer>Steven Spielberg</answer>", False, "不同-answer Paine vs Spielberg"),
    ("<answer>Upton Sinclair</answer>", "<answer>Elia Kazan</answer>", False, "不同-answer Sinclair vs Kazan"),
    ("<answer>Jamie Murray</answer>", "<answer>Andy Murray</answer>", False, "不同-answer Jamie vs Andy Murray"),
    # 不同地点
    ("<answer>Long Island</answer>", "<answer>McComb, Mississippi</answer>", False, "不同-answer Long Island vs McComb"),
    # 相近人名（不是同一人）
    ("<answer>Thomas O. Paine</answer>", "<answer>Thomas Paine</answer>", False, "同义-Thomas O. Paine vs Thomas Paine（同义区域）"),
    ("<answer>Andy Murray</answer>", "<answer>Jamie Murray</answer>", False, "不同-answer Andy vs Jamie Murray"),
    # 姓相同名不同
    ("<answer>Upton Sinclair</answer>", "<answer>Christine Sinclair</answer>", False, "不同-Upton Sinclair vs Christine Sinclair"),

    # === 真实轨迹扩充-answer 完全相同（均源自真实轨迹）===
    ("<answer>Alessandro Allori</answer>", "<answer>Alessandro Allori</answer>", True, "完全相同-Alessandro Allori"),
    ("<answer>Antonio Vivaldi</answer>", "<answer>Antonio Vivaldi</answer>", True, "完全相同-Antonio Vivaldi"),
    ("<answer>Anthony Hopkins</answer>", "<answer>Anthony Hopkins</answer>", True, "完全相同-Anthony Hopkins"),
    ("<answer>Bryan Cranston</answer>", "<answer>Bryan Cranston</answer>", True, "完全相同-Bryan Cranston"),
    ("<answer>Baton Rouge, Louisiana</answer>", "<answer>Baton Rouge, Louisiana</answer>", True, "完全相同-Baton Rouge"),
    ("<answer>Columbia University</answer>", "<answer>Columbia University</answer>", True, "完全相同-Columbia University"),

    # === 真实轨迹扩充-answer 大小写差异 ===
    ("<answer>Alessandro Allori</answer>", "<answer>alessandro allori</answer>", True, "大小写-Alessandro Allori"),
    ("<answer>Antonio Vivaldi</answer>", "<answer>antonio vivaldi</answer>", True, "大小写-Antonio Vivaldi"),
    ("<answer>Anthony Hopkins</answer>", "<answer>anthony hopkins</answer>", True, "大小写-Anthony Hopkins"),
    ("<answer>Bryan Cranston</answer>", "<answer>bryan cranston</answer>", True, "大小写-Bryan Cranston"),
    ("<answer>Columbia University</answer>", "<answer>columbia university</answer>", True, "大小写-Columbia University"),

    # === 真实轨迹扩充-answer 同义表达（全称 vs 简称、逗号差异）（环境端 answer 完全匹配，同义无效，应判为不同）===
    ("<answer>Baton Rouge, Louisiana</answer>", "<answer>Baton Rouge Louisiana</answer>", False, "同义-Baton Rouge逗号差异"),
    ("<answer>Baton Rouge, Louisiana</answer>", "<answer>Baton Rouge, LA</answer>", False, "同义-Baton Rouge全称vs缩写"),

    # === 真实轨迹扩充-answer 不同答案 ===
    ("<answer>Alessandro Allori</answer>", "<answer>Antonio Vivaldi</answer>", False, "不同-Allori vs Vivaldi"),
    ("<answer>Anthony Hopkins</answer>", "<answer>Bryan Cranston</answer>", False, "不同-Hopkins vs Cranston"),
    ("<answer>Baton Rouge, Louisiana</answer>", "<answer>Columbia University</answer>", False, "不同-Baton Rouge vs Columbia"),
    ("<answer>Claudia Wells</answer>", "<answer>Alessandro Allori</answer>", False, "不同-Claudia Wells vs Allori"),

    # ================================================================
    # 三、null 动作区
    # ================================================================

    # === 完全相同（应判为重复）===
    ("null", "null", True, "null完全相同"),

    # === null vs 有效动作（不应判为重复）===
    ("null", "<search>head of NASA during Apollo 11</search>", False, "null vs 有效搜索"),
    ("null", "<answer>Thomas O. Paine</answer>", False, "null vs 有效答案"),
    ("null", "<search>Andy Murray brother</search>", False, "null vs 有效搜索2"),
    ("null", "<answer>Steven Spielberg</answer>", False, "null vs 有效答案2"),

    # ================================================================
    # 四、跨类型区（search vs answer vs null，不应判为重复）
    # ================================================================

    # === search vs answer（即使语义相关，动作类型不同也不应判为重复）===
    ("<search>head of NASA during Apollo 11</search>", "<answer>Thomas O. Paine</answer>", False, "跨类型-search vs answer"),
    ("<search>The Terminal director</search>", "<answer>Steven Spielberg</answer>", False, "跨类型-search vs answer"),
    ("<search>author of The Jungle</search>", "<answer>Upton Sinclair</answer>", False, "跨类型-search vs answer"),
    ("<search>Andy Murray brother</search>", "<answer>Jamie Murray</answer>", False, "跨类型-search vs answer"),

    # === answer vs search（顺序反之）===
    ("<answer>Thomas O. Paine</answer>", "<search>head of NASA during Apollo 11</search>", False, "跨类型-answer vs search"),
    ("<answer>Steven Spielberg</answer>", "<search>The Terminal director</search>", False, "跨类型-answer vs search"),

    # === search vs null / answer vs null ===
    ("<search>head of NASA during Apollo 11</search>", "null", False, "跨类型-search vs null"),
    ("<answer>Thomas O. Paine</answer>", "null", False, "跨类型-answer vs null"),

    # === 真实轨迹扩充-跨类型（真实轨迹中的 search/answer 对）===
    ("<search>American Hustle director</search>", "<answer>David O. Russell</answer>", False, "跨类型-search American Hustle vs answer"),
    ("<search>Beaches 1988 film director</search>", "<answer>Garry Marshall</answer>", False, "跨类型-search Beaches vs answer"),
    ("<search>Battle of Tarawa date</search>", "<answer>1943</answer>", False, "跨类型-search Tarawa vs answer"),
    ("<search>Alessandro Allori Bronzino adoption</search>", "<answer>Alessandro Allori</answer>", False, "跨类型-search Allori vs answer"),
    ("<search>Alexandros Matsas birthplace</search>", "<answer>Athens</answer>", False, "跨类型-search Matsas vs answer"),
    ("<search>Alicia Keys education university</search>", "<answer>Columbia University</answer>", False, "跨类型-search Alicia Keys vs answer"),
    ("<answer>Antonio Vivaldi</answer>", "<search>Alessandro Allori Bronzino adoption</search>", False, "跨类型-answer Vivaldi vs search Allori"),
    ("<answer>Bryan Cranston</answer>", "<search>American Hustle director</search>", False, "跨类型-answer Cranston vs search American Hustle"),
]

# Search 任务动作列表，用于构建相似度矩阵
SEARCH_TEXTS = [
    # search 动作-完全相同/大小写/同义改写代表性样本（均源自真实轨迹）
    "<search>head of NASA during Apollo 11</search>",
    "<search>NASA administrator Apollo 11</search>",
    "<search>who was in charge of NASA during Apollo 11</search>",
    "<search>Apollo 11 mission commander NASA</search>",
    "<search>The Terminal director</search>",
    "<search>who directed The Terminal</search>",
    "<search>The Terminal 2004 director</search>",
    "<search>author of The Jungle</search>",
    "<search>\"The Arrangement\" author</search>",
    "<search>author of The Arrangement</search>",
    "<search>The Last Days of Pompeii producer</search>",
    "<search>who produced The Last Days of Pompeii</search>",
    "<search>The Believers composer</search>",
    "<search>\"The Believers\" 1987 film composer</search>",
    "<search>screenwriter of Before Midnight</search>",
    "<search>screenwriter Before Midnight</search>",
    "<search>writer Before Midnight Before Sunrise After Midnight</search>",
    "<search>Marty's girlfriend Back to the Future actress</search>",
    "<search>Britney Spears birthplace city state</search>",
    "<search>Britney Spears born in what city</search>",
    "<search>Andy Murray brother</search>",
    "<search>Andy Murray family brother</search>",
    "<search>What is Andy Murray's brother</search>",
    # search 动作-词序重排（均源自真实轨迹）
    "<search>Curious women's fragrance singer</search>",
    "<search>Curious women's cologne singer</search>",
    "<search>\"Curious\" fragrance singer</search>",
    "<search>Team USA starting pitchers 2000 Summer Olympics</search>",
    "<search>2000 Summer Olympics baseball USA starting pitchers</search>",
    "<search>Baseball at 2000 Summer Olympics USA starting pitchers list</search>",
    "<search>BBC Sports Personality of the Year awards winner most wins</search>",
    "<search>player with most BBC Sports Personality of the Year awards</search>",
    "<search>who won the most BBC Sports Personality of the Year</search>",
    "<search>Oak Beach New York between which island</search>",
    "<search>Island between Oak Beach and Great South Bay New York</search>",
    # search 动作-引号差异（真实带引号表述）
    "<search>\"The Arrangement\" author</search>",
    "<search>author of The Arrangement</search>",
    "<search>\"Umchabezi River\" mouth</search>",
    "<search>Umchabezi River mouth</search>",
    "<search>\"Before Midnight\" screenwriter Ethan Hawke</search>",
    "<search>Before Midnight Ethan Hawke screenwriter</search>",
    # search 动作-单复数（构造对照，真实轨迹多带复数形式）
    "<search>Team USA starting pitchers 2000 Summer Olympics</search>",
    "<search>Great South Bay between which islands</search>",
    "<search>Great South Bay between which island</search>",
    "<search>Murray brothers tennis</search>",
    # search 动作-短查询（构造对照，测试前缀剥离效果）
    "<search>mattress</search>",
    "<search>keywords</search>",
    "<search>Andy Murray</search>",
    "<search>Britney Spears</search>",
    "<search>The Terminal</search>",
    # search 动作-错别字/拼写差异（源自真实轨迹）
    "<search>Andrey Murray brother tennis</search>",
    "<search>Oakh Beach New York location</search>",
    "<search>moon landing NASA administrator</search>",
    "<search>moon landingNASA administrator</search>",
    # answer 动作代表性样本（均源自真实轨迹）
    "<answer>Thomas O. Paine</answer>",
    "<answer>thomas o. paine</answer>",
    "<answer>Thomas Paine</answer>",
    "<answer>Steven Spielberg</answer>",
    "<answer>steven spielberg</answer>",
    "<answer>Stephen Spielberg</answer>",
    "<answer>Upton Sinclair</answer>",
    "<answer>Sinclair</answer>",
    "<answer>Elia Kazan</answer>",
    "<answer>George IV of the United Kingdom</answer>",
    "<answer>George IV</answer>",
    "<answer>King George IV</answer>",
    "<answer>Long Island</answer>",
    "<answer>long island</answer>",
    "<answer>McComb, Mississippi</answer>",
    "<answer>McComb Mississippi</answer>",
    "<answer>McComb, MS</answer>",
    "<answer>Jamie Murray</answer>",
    "<answer>Andy Murray</answer>",
    "<answer>Claudia Wells</answer>",
    "<answer>Claudia Grace Wells</answer>",
    # null 动作
    "null",
    # search 动作-真实轨迹扩充样本
    "<search>American Hustle director</search>",
    "<search>American Hustle film director</search>",
    "<search>Beaches 1988 film director</search>",
    "<search>Beaches 1988 Garry Marshall</search>",
    "<search>Battle of Tarawa date</search>",
    "<search>Battle of Tarawa 1943</search>",
    "<search>Alicia Keys education university</search>",
    "<search>Alicia Keys performed studies</search>",
    "<search>Alicia Keys studied at</search>",
    "<search>Alessandro Allori Bronzino adoption</search>",
    "<search>Alessandro Allori Bronzino uncle death</search>",
    "<search>Alessandro Allori uncle Bronzino death</search>",
    "<search>Alexandros Matsas birthplace</search>",
    "<search>Alexandros Matsas born</search>",
    "<search>Alexandros Matsas city</search>",
    "<search>Alexandros Matsas footballer birth</search>",
    "<search>\"Bankim Chandra Chatterjee\" brother</search>",
    "<search>Bankim Chandra Chattopadhyay brother</search>",
    "<search>Bankim Chandra Chattopadhyay sibling</search>",
    "<search>Bankim Chandra Chattopadhyay sister</search>",
    "<search>\"Bob Boles\" opera</search>",
    "<search>\"Bob Boles\" opera character</search>",
    "<search>\"Bob Boles\" opera role</search>",
    "<search>Bob Boles role in opera</search>",
    "<search>\"Came Home\" father</search>",
    "<search>\"Came Home\" father of</search>",
    "<search>\"Came Home\" filly father</search>",
    "<search>\"Bronzino\" \"uncle\" painting</search>",
    "<search>\"Bronzino\" adopted name painter</search>",
    "<search>Ned Keene Bob Boles opera</search>",
    "<search>opera Bob Boles Ned Keene</search>",
    "<search>Albany International Airport location</search>",
    "<search>Albany International Airport town</search>",
    "<search>Brownsville Illinois capital became capital of Illinois 1839</search>",
    "<search>when did Brownsville become the capital of Illinois</search>",
    # answer 动作-真实轨迹扩充样本
    "<answer>Alessandro Allori</answer>",
    "<answer>Antonio Vivaldi</answer>",
    "<answer>Anthony Hopkins</answer>",
    "<answer>Bryan Cranston</answer>",
    "<answer>Baton Rouge, Louisiana</answer>",
    "<answer>Baton Rouge Louisiana</answer>",
    "<answer>Baton Rouge, LA</answer>",
    "<answer>Columbia University</answer>",
    "<answer>Garry Marshall</answer>",
    "<answer>David O. Russell</answer>",
]
