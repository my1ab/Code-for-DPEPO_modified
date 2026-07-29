"""
测试样例数据

包含以下数据：
- TEXT_PAIRS: 通用英文文本对（用于基础嵌入相似度验证）
- TEXTS: 单独的文本列表（用于批量嵌入构建相似度矩阵）
- WEBSHOP_PAIRS: WebShop 动作测试对（带期望结果标注）
- WEBSHOP_TEXTS: WebShop 动作列表（用于构建相似度矩阵）
"""

# 测试用的文本对（通用英文句子）
TEXT_PAIRS = [
    ("Machine learning is a branch of artificial intelligence", "Deep learning is a subset of machine learning"),
    ("The weather is beautiful today", "The weather is beautiful today"),
    ("Cats are adorable animals", "Dogs are man's best friend"),
    ("Python is a popular programming language", "Java is also a programming language"),
    ("I like eating apples", "I like eating apples"),
]

# 单独的文本列表，用于批量嵌入后构建相似度矩阵
TEXTS = [
    "Artificial intelligence is transforming the world",
    "Machine learning enables computers to learn from data",
    "Deep learning has achieved great success in image recognition",
    "Natural language processing is an important area of AI",
    "The lunch today was delicious",
    "I love programming in Python",
]

# ── WebShop 动作测试对 ──────────────────────────────────
# 基于 WebShop_test_seed_42_0_99.json 真实动作模式设计
# 每对标注期望结果，用于验证阈值合理性
# (action_a, action_b, should_match, category)
WEBSHOP_PAIRS = [
    # === 完全相同（应判为重复）===
    ("search[men's dress shirt short sleeve classic fit cotton spandex]", "search[men's dress shirt short sleeve classic fit cotton spandex]", True, "完全相同"),
    ("click[Buy Now]", "click[Buy Now]", True, "完全相同"),
    ("click[back to search]", "click[back to search]", True, "完全相同"),

    # === 大小写差异（真实数据中最常见的变体）===
    ("click[back to search]", "click[Back to Search]", True, "大小写-click导航"),
    ("click[buy now]", "click[Buy Now]", True, "大小写-click购买"),
    ("click[Description]", "click[description]", True, "大小写-click描述"),
    ("click[Next >]", "click[next >]", True, "大小写-click翻页"),
    ("search[Van Heusen men's classic fit short sleeve dress shirt]", "search[van heusen men's classic fit short sleeve dress shirt]", True, "大小写-品牌名"),
    ("search[Haggar men's classic fit short sleeve dress shirt]", "search[haggar men's classic fit short sleeve dress shirt]", True, "大小写-品牌名"),
    ("search[3X-large]", "search[3x-large]", True, "大小写-尺码"),

    # === 词序差异（真实数据中大量存在，同一关键词集合的不同排列）===
    ("search[men's dress shirt short sleeve classic fit cotton spandex]", "search[men's classic fit dress shirt short sleeve cotton spandex]", True, "词序-重排"),
    ("search[men's dress shirt short sleeve classic fit cotton spandex]", "search[men's dress shirt cotton spandex short sleeve classic fit]", True, "词序-重排"),
    ("search[men's dress shirt short sleeve classic fit cotton spandex]", "search[classic fit short sleeve dress shirt men's cotton spandex]", True, "词序-完全打乱"),
    ("search[brown heather shirt xx-large needle sleeve classic fit women]", "search[heather brown classic fit needle sleeve xx-large women shirt]", True, "词序-完全打乱"),
    ("search[yellow henley shirt men's large]", "search[henley yellow shirt large men's]", True, "词序-短查询"),

    # 增加限定词视为不同（A是B的子集）
    ("search[men's short sleeve dress shirt]", "search[men's short sleeve button down dress shirt]", False, "不同-增加限定词button down"),

    # === 更具体（添加尺码视为不同）===
    ("search[men's short sleeve dress shirt stretch cotton classic fit]", "search[men's short sleeve dress shirt stretch cotton classic fit x-large]", False, "不同-添加尺码x-large"),
    # 添加属性（如 machine wash）视为不同搜索意图
    ("search[men's dress shirt short sleeve classic fit cotton spandex]", "search[men's dress shirt short sleeve classic fit cotton spandex machine wash]", False, "不同-添加属性machine wash"),

    # === 不同搜索意图（不应判为重复）===
    ("search[men's dress shirt short sleeve classic fit cotton spandex]", "search[men's dress shirt long sleeve classic fit cotton spandex]", False, "不同-短袖vs长袖"),
    ("search[men's dress shirt short sleeve classic fit]", "search[women's dress shirt short sleeve classic fit]", False, "不同-男装vs女装"),
    ("search[men's dress shirt short sleeve classic fit]", "search[men's dress shirt short sleeve slim fit]", False, "不同-classic vs slim fit"),
    ("search[needle sleeve dress shirt classic fit men]", "search[needle sleeve dress shirt classic fit women]", False, "不同-男vs女"),
    ("search[yellow henley shirt men's large]", "search[blue henley shirt men's large]", False, "不同-颜色"),
    ("search[white table lamp living room under 40]", "search[white table lamp bedroom under 40]", False, "不同-客厅vs卧室"),
    ("search[body lotion sensitive skin paraben free]", "search[face cream sensitive skin paraben free]", False, "不同-身体乳vs面霜"),

    # === click 目标：导航类（不应判为重复）===
    ("click[Buy Now]", "click[back to search]", False, "不同-click导航"),
    ("click[Buy Now]", "click[Description]", False, "不同-click导航"),
    ("click[Buy Now]", "click[Features]", False, "不同-click导航"),
    ("click[Buy Now]", "click[Reviews]", False, "不同-click导航"),
    ("click[Next >]", "click[< Prev]", False, "不同-click翻页方向"),
    ("click[back to search]", "click[Description]", False, "不同-click导航"),

    # === click 目标：ASIN 产品 ID（不应判为重复，除非完全相同）===
    ("click[B07F2G93BJ]", "click[B09QQP3356]", False, "不同-ASIN"),
    ("click[B07F2G93BJ]", "click[B07FKGQKZ1]", False, "不同-ASIN"),

    # === click 目标：尺码/颜色选项（不应判为重复）===
    ("click[small]", "click[medium]", False, "不同-尺码"),
    ("click[large]", "click[xx-large]", False, "不同-尺码"),
    ("click[black]", "click[blue]", False, "不同-颜色"),
    ("click[heather grey]", "click[black]", False, "不同-颜色"),


    # === 跨动作类型（不应判为重复）===
    ("search[men's dress shirt short sleeve classic fit]", "click[Buy Now]", False, "跨类型-search vs click"),
    ("search[men's dress shirt short sleeve classic fit]", "click[back to search]", False, "跨类型-search vs click"),
    ("search[yellow henley shirt men's large]", "click[B07F2G93BJ]", False, "跨类型-search vs ASIN"),

    # === 词序重排-更多真实变体（源自 WebShop_test_seed_42_0_99.json）===
    # 9-token 完全打乱
    ("search[brown heather classic fit shirt women xx-large needle sleeve]", "search[women brown heather classic fit shirt xx-large needle sleeve]", True, "词序-9token重排"),
    # 8-token 含品牌名重排
    ("search[Van Heusen men's classic fit short sleeve dress shirt stretch]", "search[van heusen men's classic fit stretch short sleeve dress shirt]", True, "词序-含品牌名重排"),
    # 8-token button down 重排
    ("search[men's classic fit cotton spandex short sleeve button down]", "search[cotton spandex men's classic fit short sleeve button down]", True, "词序-buttondown重排"),
    # 8-token needle sleeve 重排
    ("search[needle sleeve classic fit dress shirt youth small pink]", "search[youth classic fit dress shirt pink small needle sleeve]", True, "词序-needle sleeve重排"),

    # === 大小写-更多品牌名 ===
    ("search[Calvin Klein men's short sleeve dress shirt cotton spandex]", "search[calvin klein men's short sleeve dress shirt cotton spandex]", True, "大小写-Calvin Klein"),
    ("search[Haggar classic fit short sleeve dress shirt cotton spandex]", "search[haggar classic fit short sleeve dress shirt cotton spandex]", True, "大小写-Haggar"),
    ("search[Geoffrey Beene men's short sleeve dress shirt stretch]", "search[geoffrey beene men's short sleeve dress shirt stretch]", True, "大小写-Geoffrey Beene"),

    # === 短查询（前缀占比大，测试前缀剥离效果）===
    ("search[mattress]", "search[mattress]", True, "短查询-完全相同"),
    ("search[keywords]", "search[keywords]", True, "短查询-占位符相同"),
    ("search[mattress]", "search[double sided mattress]", False, "短查询-不同产品"),
    ("search[keywords]", "search[mattress]", False, "短查询-占位符vs实际"),

    # === 异常-ASIN 作为 search（真实数据中存在）===
    ("search[B07FKGQKZ1]", "search[B07FKGQKZ1]", True, "异常-ASIN搜索相同"),
    ("search[B07FKGQKZ1]", "search[B09CQ45ZRB]", False, "异常-ASIN搜索不同"),

    # === search 与 click 同文本（back to search 既是 search 又是 click）===
    ("search[back to search]", "click[back to search]", False, "跨类型-同文本search vs click"),

    # === 不同-搜索类别差异 ===
    ("search[men's short sleeve dress shirt cotton spandex]", "search[men's short sleeve henley cotton]", False, "不同-dress shirt vs henley"),
    ("search[men's short sleeve dress shirt cotton spandex]", "search[women's long sleeve sweater polyester]", False, "不同-dress shirt vs sweater"),

    # === 不同-尺码变体（search 中）===
    ("search[men's dress shirt cotton spandex classic fit short sleeve 3xl]", "search[men's dress shirt cotton spandex classic fit short sleeve x-large]", False, "不同-search尺码3xl vs x-large"),

    # === null 动作 ===
    ("null", "null", True, "null完全相同"),
    ("null", "search[men's dress shirt short sleeve classic fit]", False, "null vs 有效搜索"),
    ("null", "click[Buy Now]", False, "null vs 有效点击"),

    # ================================================================
    # 以下样本源自 WebShop_test_seed_42_100_199.json（2026-07-18 新增）
    # 重点覆盖阈值附近的 search 相似样本
    # ================================================================

    # === 词序重排-长查询（10~13 token，测试嵌入对长文本词序的稳定性）===
    # 12-token 含尺码+属性，完全打乱
    ("search[haggar men's classic fit short sleeve dress shirt cotton spandex black x-large]", "search[haggar men's classic fit short sleeve dress shirt cotton spandex x-large black]", True, "词序-12token尺码换序"),
    # 12-token 含属性+颜色
    ("search[men's classic fit short sleeve dress shirt black x-large cotton spandex machine wash]", "search[men's classic fit short sleeve dress shirt cotton spandex black x-large machine wash]", True, "词序-12token属性换序"),
    # 13-token 超长查询重排
    ("search[men's classic fit short sleeve dress shirt black x-large cotton spandex machine wash]", "search[men's classic fit short sleeve dress shirt cotton spandex black x-large machine wash]", True, "词序-13token超长重排"),
    # 11-token 含品牌名+尺码
    ("search[van heusen men's classic fit short sleeve dress shirt 3xl tall black]", "search[van heusen men's classic fit short sleeve dress shirt black 3xl tall]", True, "词序-11token品牌+尺码换序"),
    # 10-token 含品牌名+尺码+属性
    ("search[van heusen classic fit short sleeve dress shirt cotton spandex]", "search[van heusen dress shirt cotton spandex short sleeve classic fit]", True, "词序-10token品牌重排"),
    # 10-token Calvin Klein
    ("search[calvin klein men's short sleeve dress shirt classic fit stretch]", "search[calvin klein men's short sleeve dress shirt stretch classic fit]", True, "词序-10token品牌+属性换序"),
    # 10-token 含颜色+尺码
    ("search[men's dress shirt classic fit cotton spandex short sleeve 3x]", "search[men's short sleeve dress shirt classic fit cotton spandex 3x]", True, "词序-10token尺码换序"),
    # 9-token 含品牌名 Haggar 重排
    ("search[haggar men's classic fit short sleeve dress shirt cotton spandex]", "search[haggar men's short sleeve dress shirt classic fit cotton spandex]", True, "词序-9tokenHaggar重排"),
    # 9-token Geoffrey Beene 重排
    ("search[geoffrey beene dress shirt cotton spandex short sleeve]", "search[geoffrey beene short sleeve dress shirt cotton spandex]", True, "词序-9tokenGeoffrey重排"),
    # 8-token 含颜色+尺码
    ("search[men's dress shirt short sleeve black x-large]", "search[men's short sleeve dress shirt x-large black]", True, "词序-8token颜色尺码换序"),
    # 8-token polyester heather 重排
    ("search[polyester heather classic fit small women cranberry]", "search[polyester heather cranberry classic fit women small]", True, "词序-8token颜色换序"),
    # 7-token 极短词序重排
    ("search[cotton spandex dress shirt men's]", "search[men's dress shirt cotton spandex]", True, "词序-7token短重排"),
    # 6-token 短查询词序重排
    ("search[cotton spandex men's dress shirt]", "search[men's dress shirt cotton spandex]", True, "词序-6token短重排"),
    # 5-token 超短词序重排
    ("search[orbit green pants women x-small]", "search[orbit green women pants x-small]", True, "词序-5token超短重排"),
    # 4-token 极短词序重排（前缀占比最大）
    ("search[silver men t-shirt xx-large]", "search[xx-large t-shirt silver men]", True, "词序-4token极短重排"),

    # 单复数 shirt/shirts（真实数据中存在，应判为相同）
    ("search[classic fit short sleeve dress shirts men cotton spandex]", "search[classic fit short sleeve dress shirt men cotton spandex]", True, "单复数-复数shirts vs shirt"),

    # === 1词差异-不同产品（验证阈值下方的区分能力）===
    # 颜色替换：navy vs cranberry
    ("search[polyester heather classic fit small women cranberry]", "search[polyester heather classic fit small women navy]", False, "不同-颜色cranberry vs navy"),
    # 尺码替换：6x vs 3x
    ("search[haggar men's short sleeve classic fit dress shirt 6x]", "search[haggar men's short sleeve classic fit dress shirt 3x]", False, "不同-尺码6x vs 3x"),
    # 性别替换：toddler vs women
    ("search[toddler shirt navy heather classic fit 4t]", "search[women shirt 4t heather navy classic fit]", False, "不同-toddler vs women"),
    # 品牌替换：Gildan vs 无品牌
    ("search[Gildan men's classic fit shirt purple heather]", "search[men's heather shirt purple classic fit x-small]", False, "不同-Gildan vs 无品牌"),
    # 产品类别替换：women vs pants
    ("search[nylon spandex activewear women black]", "search[nylon spandex activewear pants black]", False, "不同-women vs pants"),
    # 属性替换：white vs poly
    ("search[men's dress shirt classic fit short sleeve white cotton spandex]", "search[men's short sleeve dress shirt cotton poly spandex classic fit]", False, "不同-white vs poly"),

    # ================================================================
    # 以下样本源自验证输出 1.5B_epoch5_hislen8_test（2026-07-18 新增）
    # 补充非衬衫类别的词序重排和不同产品对
    # 验证集高频类别：家具/家居、鞋类、食品、电子
    # ================================================================

    # === 家具/家居类-词序重排（验证集最高频非衬衫类）===
    # 5词 bed frame 词序完全打乱（验证集9变体）
    ("search[wood frame bed charcoal twin]", "search[charcoal twin wood bed frame]", True, "词序-家具5token重排"),
    # 5词 box spring 重排
    ("search[box spring twin wood charcoal]", "search[charcoal twin box spring wood]", True, "词序-boxspring5token重排"),
    # 5词 platform bed 重排
    ("search[twin charcoal wood platform bed]", "search[charcoal twin wood platform bed]", True, "词序-platform5token重排"),
    # 6词 furniture set 重排
    ("search[furniture set taupe rectangular under 70]", "search[rectangular furniture set taupe under 70]", True, "词序-家具set6token重排"),
    # 7词 sofa table 重排
    ("search[walnut sofa table easy clean solid wood]", "search[easy clean solid wood sofa table walnut]", True, "词序-sofatable7token重排"),

    # === 鞋类-词序重排（验证集高频非衬衫类）===
    # 6词 loafers 重排
    ("search[men's loafers slip resistant rubber outsole]", "search[slip resistant men's loafers rubber outsole]", True, "词序-loafers6token重排"),
    # 7词 pumps 重排
    ("search[pumps womens black rubber sole closed toe]", "search[womens black pumps closed toe rubber sole]", True, "词序-pumps7token重排"),
    # 6词 boots 重排
    ("search[men's boots moisture wicking size 12]", "search[men's boots size 12 moisture wicking]", True, "词序-boots6token重排"),

    # === 食品类-词序重排（验证集高频非衬衫类）===
    # 6词 keto 重排
    ("search[dark chocolate keto milk non gmo]", "search[non gmo keto milk dark chocolate]", True, "词序-keto6token重排"),
    # 5词 protein 重排
    ("search[gluten free honey cinnamon protein]", "search[honey cinnamon protein gluten free]", True, "词序-protein5token重排"),

    # === 服装类-词序重排（非衬衫）===
    # 6词 jeans 重排
    ("search[charcoal dust jeans straight leg men's]", "search[charcoal dust men's jeans straight leg]", True, "词序-jeans6token重排"),
    # 6词 hoodie 重排
    ("search[machine wash hoodie white medium women]", "search[women hoodie white medium machine wash]", True, "词序-hoodie6token重排"),

    # === 家具类-1词差异不同产品（验证集真实对）===
    # 办公椅颜色替换：black vs navy
    ("search[height adjustable office chair black under 130]", "search[height adjustable office chair navy under 130]", False, "不同-办公椅black vs navy"),
    # 办公椅颜色替换：black vs grey
    ("search[height adjustable office chair black under 130]", "search[office chair height adjustable grey under 130]", False, "不同-办公椅black vs grey"),
    # 镜子类型替换：decorative vs accent
    ("search[decorative mirror narrow bronze]", "search[accent bronze narrow mirror]", False, "不同-镜子decorative vs accent"),
    # bed 类型替换：frame vs spring
    ("search[twin woodbed frame charcoal]", "search[twin woodbed spring charcoal]", False, "不同-woodbed frame vs spring"),
    # bed 类型替换：storage vs solid
    ("search[storage platform bed charcoal twin wood]", "search[solid wood twin platform bed charcoal]", False, "不同-platform storage vs solid"),

    # === 跨类别-不同产品（验证嵌入对完全不同产品的区分能力）===
    # 家具 vs 服装
    ("search[wood frame bed charcoal twin]", "search[men's dress shirt short sleeve classic fit cotton spandex]", False, "不同-床架 vs 衬衫"),
    # 鞋类 vs 家具
    ("search[men's loafers slip resistant rubber outsole]", "search[height adjustable office chair black under 130]", False, "不同-乐福鞋 vs 办公椅"),
    # 食品 vs 服装
    ("search[dark chocolate keto milk non gmo]", "search[charcoal dust jeans straight leg men's]", False, "不同-巧克力 vs 牛仔裤"),

    # ================================================================
    # 以下样本源自 学习轨迹+验证轨迹 综合分析（2026-07-18 新增）
    # 重点覆盖：
    #   1. 非衬衫类别的阈值边界样本（1词差异，sim 0.925~0.968）
    #   2. 非衬衫类别的词序重排（方案 B+E 跨类别验证）
    #   3. 单复数/形态变化的阈值紧贴样本
    #   4. 非衬衫类别的同义/近义替换
    # ================================================================

    # === 家具类-1词差异（阈值边界，sim 预期 0.92~0.97）===
    # 床类型：frame vs spring（词集不同，意图不同）
    ("search[twin wood bed frame charcoal]", "search[twin wood bed spring charcoal]", False, "不同-家具frame vs spring"),
    # 床材质：wood vs metal
    ("search[twin platform bed frame charcoal solid wood]", "search[twin platform bed frame charcoal metal]", False, "不同-家具wood vs metal"),
    # 尺寸：twin vs full
    ("search[wood platform bed frame charcoal twin]", "search[wood platform bed frame charcoal full]", False, "不同-家具twin vs full"),
    # 颜色：charcoal vs walnut
    ("search[twin wood bed frame charcoal]", "search[twin wood bed frame walnut]", False, "不同-家具charcoal vs walnut"),
    # 储物类型：storage vs standard
    ("search[storage platform bed charcoal twin wood]", "search[standard platform bed charcoal twin wood]", False, "不同-家具storage vs standard"),

    # === 家具类-词序重排（验证方案 B+E）===
    # 床头柜 6-token 重排
    ("search[nightstand solid wood charcoal 2 drawer]", "search[2 drawer charcoal solid wood nightstand]", True, "词序-家具nightstand重排"),
    # 梳妆台 7-token 重排
    ("search[dresser white 6 drawer solid wood modern]", "search[modern white solid wood dresser 6 drawer]", True, "词序-家具dresser重排"),
    # 书柜 6-token 重排
    ("search[bookshelf 5 shelf walnut engineered wood]", "search[walnut engineered wood bookshelf 5 shelf]", True, "词序-家具bookshelf重排"),

    # === 灯具类-1词差异（阈值边界）===
    # 灯类型：pendant vs chandelier
    ("search[glass shade pendant kitchen island living room]", "search[glass shade chandelier kitchen island living room]", False, "不同-灯具pendant vs chandelier"),
    # 材质：glass vs metal shade
    ("search[glass shade pendant light kitchen island]", "search[metal shade pendant light kitchen island]", False, "不同-灯具glass vs metal"),
    # 颜色：white vs black
    ("search[white table lamp living room under 40]", "search[black table lamp living room under 40]", False, "不同-灯具white vs black"),
    # 房间：living room vs bedroom
    ("search[white table lamp living room under 40]", "search[white table lamp bedroom under 40]", False, "不同-灯具客厅vs卧室"),

    # === 灯具类-词序重排 ===
    ("search[glass shade pendant kitchen island living room]", "search[kitchen island pendant glass shade living room]", True, "词序-灯具pendant重排"),
    ("search[semi flush mount chandelier easy clean under 80]", "search[easy clean semi flush mount chandelier under 80]", True, "词序-灯具chandelier重排"),

    # === 家居装饰类-1词差异（阈值边界）===
    # 地毯：round vs rectangular
    ("search[round area rug black gray 3x5 under 70]", "search[rectangular area rug black gray 3x5 under 70]", False, "不同-家居round vs rectangular"),
    # 颜色：black gray vs beige
    ("search[round area rug black gray 3x5 under 70]", "search[round area rug beige 3x5 under 70]", False, "不同-家居black gray vs beige"),
    # 窗帘颜色：charcoal vs white
    ("search[charcoal grey curtain 52 wide 45 long machine washable]", "search[white curtain 52 wide 45 long machine washable]", False, "不同-家居charcoal vs white"),
    # 抱枕形状：square vs lumbar
    ("search[24 inch square decorative pillow double sided]", "search[24 inch lumbar decorative pillow double sided]", False, "不同-家居square vs lumbar"),
    # 挂画风格：modern vs traditional
    ("search[ready hang wall art modern living room 24x36]", "search[ready hang wall art traditional living room 24x36]", False, "不同-家居modern vs traditional"),

    # === 家居装饰类-词序重排 ===
    ("search[round area rug black gray 3x5 under 70]", "search[black gray round area rug 3x5 under 70]", True, "词序-家居rug重排"),
    ("search[charcoal grey curtain 52 wide 45 long machine washable]", "search[machine washable charcoal grey curtain 52 wide 45 long]", True, "词序-家居curtain重排"),

    # === 鞋类-1词差异（阈值边界）===
    # 尺码：size 12 vs size 10
    ("search[men's boots moisture wicking size 12]", "search[men's boots moisture wicking size 10]", False, "不同-鞋类size 12 vs 10"),
    # 颜色：black vs brown
    ("search[men's loafers slip resistant rubber sole black]", "search[men's loafers slip resistant rubber sole brown]", False, "不同-鞋类black vs brown"),
    # 类型：loafers vs oxfords
    ("search[men's slip resistant rubber sole loafers black]", "search[men's slip resistant rubber sole oxfords black]", False, "不同-鞋类loafers vs oxfords"),
    # 材质：rubber vs leather sole
    ("search[men's loafers slip resistant rubber outsole]", "search[men's loafers slip resistant leather outsole]", False, "不同-鞋类rubber vs leather"),
    # 性别：men's vs women's
    ("search[men's loafers slip resistant rubber outsole black]", "search[women's loafers slip resistant rubber outsole black]", False, "不同-鞋类men vs women"),

    # === 鞋类-词序重排 ===
    ("search[pumps womens black rubber sole closed toe size 7]", "search[womens black pumps closed toe rubber sole size 7]", True, "词序-鞋类pumps重排"),
    ("search[men's boots moisture wicking size 12 waterproof]", "search[waterproof men's boots size 12 moisture wicking]", True, "词序-鞋类boots重排"),

    # === 食品类-1词差异（阈值边界）===
    # 口味：dark chocolate vs milk chocolate
    ("search[dark chocolate keto milk non gmo protein bar]", "search[milk chocolate keto milk non gmo protein bar]", False, "不同-食品dark vs milk chocolate"),
    # 类型：protein bar vs protein shake
    ("search[gluten free honey cinnamon protein bar 12 pack]", "search[gluten free honey cinnamon protein shake 12 pack]", False, "不同-食品bar vs shake"),
    # 品牌：wasabi peas
    ("search[wasabi peas gluten free non gmo 4 oz]", "search[wasabi peas organic gluten free 4 oz]", False, "不同-食品non gmo vs organic"),
    # 甜味剂：honey vs stevia
    ("search[gluten free honey cinnamon protein bar]", "search[gluten free stevia cinnamon protein bar]", False, "不同-食品honey vs stevia"),
    # 坚果：almond vs peanut
    ("search[keto almond butter protein bar gluten free]", "search[keto peanut butter protein bar gluten free]", False, "不同-食品almond vs peanut"),

    # === 食品类-词序重排 ===
    ("search[gluten free honey cinnamon protein bar 12 pack]", "search[honey cinnamon protein bar gluten free 12 pack]", True, "词序-食品protein重排"),
    ("search[wasabi peas gluten free non gmo 4 oz]", "search[non gmo gluten free wasabi peas 4 oz]", True, "词序-食品wasabi重排"),

    # === 个护类-1词差异（阈值边界）===
    # 产品类型：body lotion vs face cream
    ("search[body lotion sensitive skin paraben free 4.2 oz]", "search[face cream sensitive skin paraben free 4.2 oz]", False, "不同-个护lotion vs cream"),
    # 品牌：CeraVe vs Cetaphil
    ("search[cerave body cream sensitive skin 4.2 ounce]", "search[cetaphil body cream sensitive skin 4.2 ounce]", False, "不同-个护CeraVe vs Cetaphil"),
    # 功能：moisturizing vs anti-aging
    ("search[body lotion moisturizing sensitive skin paraben free]", "search[body lotion anti-aging sensitive skin paraben free]", False, "不同-个护moisturizing vs anti-aging"),
    # 规格：4.2 oz vs 8 oz
    ("search[cerave body cream sensitive skin 4.2 ounce]", "search[cerave body cream sensitive skin 8 ounce]", False, "不同-个护4.2oz vs 8oz"),
    # 敏感 vs 普通
    ("search[body lotion sensitive skin paraben free fragrance free]", "search[body lotion normal skin paraben free fragrance free]", False, "不同-个护sensitive vs normal"),

    # === 个护类-词序重排 ===
    ("search[cerave body cream sensitive skin 4.2 ounce fragrance free]", "search[fragrance free cerave body cream 4.2 ounce sensitive skin]", True, "词序-个护cerave重排"),
    ("search[body lotion sensitive skin paraben free fragrance free]", "search[paraben free fragrance free body lotion sensitive skin]", True, "词序-个护lotion重排"),

    # === 电子类-1词差异（阈值边界）===
    # 产品：charger vs cable
    ("search[iphone wireless charger quick release hands free orbiter]", "search[iphone usb cable quick release hands free orbiter]", False, "不同-电子charger vs cable"),
    # 型号：iPhone vs Android
    ("search[iphone wireless charger quick release hands free dash]", "search[android wireless charger quick release hands free dash]", False, "不同-电子iphone vs android"),
    # 颜色：black vs white
    ("search[wireless charging pad fast charge black qi compatible]", "search[wireless charging pad fast charge white qi compatible]", False, "不同-电子black vs white"),
    # 充电速度：fast vs standard
    ("search[wireless charging pad fast charge qi compatible 15w]", "search[wireless charging pad standard charge qi compatible 5w]", False, "不同-电子fast vs standard"),
    # 类型：screen protector vs case
    ("search[tempered glass screen protector iphone 14 pro max]", "search[silicone case iphone 14 pro max drop protection]", False, "不同-电子protector vs case"),

    # === 电子类-词序重排 ===
    ("search[tempered glass screen protector iphone 14 pro max]", "search[iphone 14 pro max tempered glass screen protector]", True, "词序-电子screenprotector重排"),
    ("search[wireless charging pad fast charge qi compatible black]", "search[black qi compatible fast charge wireless charging pad]", True, "词序-电子charger重排"),
    ("search[quick release hands free iphone wireless charging orbiter dash]", "search[iphone wireless charging orbiter dash quick release hands free]", True, "词序-电子orbiter重排"),

    # === 裤装类-1词差异（阈值边界）===
    # 颜色：black vs gray
    ("search[men's polyester spandex jogger pants xx-large drawstring black]", "search[men's polyester spandex jogger pants xx-large drawstring gray]", False, "不同-裤装black vs gray"),
    # 尺码：xx-large vs medium
    ("search[men's polyester spandex jogger pants xx-large drawstring]", "search[men's polyester spandex jogger pants medium drawstring]", False, "不同-裤装xx-large vs medium"),
    # 类型：jogger vs chino
    ("search[men's polyester spandex jogger pants xx-large drawstring]", "search[men's polyester spandex chino pants xx-large drawstring]", False, "不同-裤装jogger vs chino"),
    # 闭合方式：drawstring vs elastic
    ("search[men's athletic shorts drawstring elastic waist gym]", "search[men's athletic shorts elastic waistband gym]", False, "不同-裤装drawstring vs elastic"),
    # 性别：men's vs women's
    ("search[men's moisture wicking shorts elastic waistband polyester]", "search[women's moisture wicking shorts elastic waistband polyester]", False, "不同-裤装men vs women"),

    # === 裤装类-词序重排 ===
    ("search[men's polyester spandex jogger pants xx-large drawstring black]", "search[black men's polyester spandex jogger pants drawstring xx-large]", True, "词序-裤装jogger重排"),
    ("search[orbit green women pants x-small elastic waist]", "search[women pants orbit green x-small elastic waist]", True, "词序-裤装orbit重排"),

    # === 上衣类-1词差异（阈值边界，非衬衫）===
    # 颜色：purple vs navy
    ("search[Hanes men's classic fit t-shirt purple heather x-small]", "search[Hanes men's classic fit t-shirt navy heather x-small]", False, "不同-上衣purple vs navy"),
    # 尺码：x-small vs xx-large
    ("search[Hanes men's classic fit t-shirt purple heather x-small]", "search[Hanes men's classic fit t-shirt purple heather xx-large]", False, "不同-上衣x-small vs xx-large"),
    # 品牌：Hanes vs Gildan
    ("search[Hanes men's classic fit t-shirt purple heather x-small]", "search[Gildan men's classic fit t-shirt purple heather x-small]", False, "不同-上衣Hanes vs Gildan"),
    # 类型：t-shirt vs polo
    ("search[men's classic fit t-shirt purple heather cotton x-small]", "search[men's classic fit polo purple heather cotton x-small]", False, "不同-上衣tshirt vs polo"),
    # 材质：cotton vs polyester
    ("search[men's classic fit t-shirt purple heather cotton x-small]", "search[men's classic fit t-shirt purple heather polyester x-small]", False, "不同-上衣cotton vs polyester"),

    # === 上衣类-词序重排 ===
    ("search[Hanes men's classic fit t-shirt purple heather x-small cotton]", "search[cotton Hanes men's purple heather classic fit t-shirt x-small]", True, "词序-上衣tshirt重排"),
    ("search[women hoodie white medium machine wash fleece lined]", "search[fleece lined women hoodie white medium machine wash]", True, "词序-上衣hoodie重排"),

    # === 单复数/形态变化-阈值紧贴样本 ===
    # shirt vs shirts（应判为相同）
    ("search[men's dress shirt cotton spandex classic fit short sleeve]", "search[men's dress shirts cotton spandex classic fit short sleeve]", True, "单复数-shirt vs shirts"),
    # shoe vs shoes
    ("search[men's running shoe breathable mesh size 10]", "search[men's running shoes breathable mesh size 10]", True, "单复数-shoe vs shoes"),
    # pant vs pants
    ("search[women's yoga pant high waist stretch black]", "search[women's yoga pants high waist stretch black]", True, "单复数-pant vs pants"),
    # sock vs socks
    ("search[men's athletic sock moisture wicking cotton 6 pack]", "search[men's athletic socks moisture wicking cotton 6 pack]", True, "单复数-sock vs socks"),
    # curtain vs curtains
    ("search[charcoal grey curtain 52 wide 45 long blackout]", "search[charcoal grey curtains 52 wide 45 long blackout]", True, "单复数-curtain vs curtains"),
    # pillow vs pillows
    ("search[decorative throw pillow 18x18 machine washable]", "search[decorative throw pillows 18x18 machine washable]", True, "单复数-pillow vs pillows"),

    # === 短查询增加修饰词 ===
    # 增加修饰词视为不同（A是B的真子集）
    ("search[wireless charging pad]", "search[wireless charging pad fast charge]", False, "不同-短查询增加fast charge"),
    ("search[glass screen protector]", "search[glass screen protector tempered]", False, "不同-短查询增加tempered"),
    ("search[body lotion sensitive]", "search[body lotion sensitive fragrance free]", False, "不同-短查询增加fragrance"),
    ("search[protein bar gluten free]", "search[protein bar gluten free 12 pack]", False, "不同-短查询增加12 pack"),

    # ================================================================
    # 以下为同义替换统一区域
    # 包含所有同义/近义替换条目（search内容同义、click同义）
    # 分为两组：应判为相同（True）和不应判为相同（False）
    # ================================================================

    # === 同义替换-应判为相同（True）===
    # cotton spandex ↔ stretch（核心面料同义对，真实数据高频）
    ("search[men's dress shirt cotton spandex classic fit short sleeve]", "search[men's dress shirt stretch cotton classic fit short sleeve]", True, "同义-cotton spandex=stretch"),
    ("search[arrow classic fit short sleeve dress shirt cotton spandex]", "search[arrow classic fit short sleeve dress shirt stretch cotton]", True, "同义-cotton spandex=stretch cotton"),
    # 带品牌名 + cotton spandex = stretch（2词->1词映射，词集匹配不可解）
    ("search[Van Heusen men's classic fit short sleeve dress shirt cotton spandex]", "search[Van Heusen men's classic fit short sleeve dress shirt stretch]", True, "同义-品牌+cotton spandex=stretch"),
    # 带尺码（A和B都含black，纯spandex->stretch替换）
    ("search[men's short sleeve button down shirt 3xl tall black cotton spandex]", "search[men's short sleeve button down shirt 3xl tall black stretch cotton]", True, "同义-带尺码cotton spandex=stretch"),
    # 不带品牌
    ("search[men's button down shirt cotton spandex classic fit short sleeve]", "search[men's button down shirt stretch cotton classic fit short sleeve]", True, "同义-button down cotton spandex=stretch"),

    # click 目标：同义购买按钮
    ("click[Buy Now]", "click[buy now]", True, "同义-购买按钮"),
    ("click[Back to Search]", "click[back to search]", True, "同义-返回搜索"),

    # 灯具-同义/词序重排
    ("search[ceiling light semi flush mount easy clean]", "search[semi flush mount ceiling light easy clean]", True, "同义-灯具ceiling light重排"),

    # 短查询同义替换（词集完全不同，但语义相同，搜索结果一致）
    ("search[sofa bed]", "search[sleeper sofa]", True, "同义-短查询sofa bed=sleeper sofa"),
    ("search[area rug]", "search[floor rug]", True, "同义-短查询area rug=floor rug"),

    # 跨类别同义替换（词集不同，语义相同，搜索结果一致）
    # 家具：sofa vs couch
    ("search[sofa bed sleeper charcoal memory foam]", "search[couch bed sleeper charcoal memory foam]", True, "同义-sofa=couch"),
    # 电子：wireless charger vs charging pad
    ("search[wireless charger fast charge qi compatible 15w]", "search[wireless charging pad fast charge qi compatible 15w]", True, "同义-charger=charging pad"),
    # 鞋类：sneakers vs athletic shoes
    ("search[men's sneakers breathable mesh running size 10]", "search[men's athletic shoes breathable mesh running size 10]", True, "同义-sneakers=athletic shoes"),
    # 个护：moisturizer vs lotion
    ("search[face moisturizer sensitive skin fragrance free 4 oz]", "search[face lotion sensitive skin fragrance free 4 oz]", True, "同义-moisturizer=lotion"),
    # 家居：rug vs mat（厨房场景下搜索结果高度重叠）
    ("search[kitchen rug non slip washable 3x5]", "search[kitchen mat non slip washable 3x5]", True, "同义-kitchen rug=mat"),
    # 床上用品：comforter vs duvet（"down comforter"和"down duvet"是同一产品）
    ("search[down comforter queen size all season machine washable]", "search[down duvet queen size all season machine washable]", True, "同义-comforter=duvet"),

    # === 同义替换-不应判为相同（False）===
    # 带颜色变化（white->coral，不是纯同义）
    ("search[men's dress shirt classic fit short sleeve white cotton spandex]", "search[men's dress shirt classic fit short sleeve stretch cotton coral]", False, "不同-同义替换+颜色white vs coral"),
    # oxford 是 dress shirt 的子类型（面料/领型不同），非同义
    ("search[men's short sleeve dress shirt]", "search[men's short sleeve oxford shirt]", False, "不同-dress shirt vs oxford shirt"),
    # desk lamp 偏任务照明，table lamp 偏装饰照明，用途不同
    ("search[desk lamp]", "search[table lamp]", False, "不同-desk lamp vs table lamp"),
    # protein bar 关注蛋白质，energy bar 关注能量/碳水，不同产品类别
    ("search[protein bar gluten free chocolate 12 pack]", "search[energy bar gluten free chocolate 12 pack]", False, "不同-protein bar vs energy bar"),
]

# WebShop 动作列表，用于构建相似度矩阵
WEBSHOP_TEXTS = [
    "search[men's dress shirt short sleeve classic fit cotton spandex]",
    "search[men's classic fit dress shirt short sleeve cotton spandex]",
    "search[men's dress shirt long sleeve classic fit cotton spandex]",
    "search[women's dress shirt short sleeve classic fit cotton spandex]",
    "search[yellow henley shirt men's large]",
    "search[blue henley shirt men's large]",
    "search[Van Heusen men's classic fit short sleeve dress shirt stretch]",
    "search[van heusen men's classic fit stretch short sleeve dress shirt]",
    # 新增：源自 WebShop_test_seed_42_100_199.json 的样本
    "search[haggar men's classic fit short sleeve dress shirt cotton spandex black x-large]",
    "search[haggar men's classic fit short sleeve dress shirt cotton spandex x-large black]",
    "search[arrow classic fit short sleeve dress shirt cotton spandex]",
    "search[arrow classic fit short sleeve dress shirt stretch cotton]",
    "search[men's button down shirt cotton spandex classic fit short sleeve]",
    "search[men's button down shirt stretch cotton classic fit short sleeve]",
    "search[cotton spandex dress shirt men's]",
    "search[men's dress shirt cotton spandex]",
    "search[silver men t-shirt xx-large]",
    "search[xx-large t-shirt silver men]",
    "search[polyester heather classic fit small women cranberry]",
    "search[polyester heather classic fit small women navy]",
    # 新增：源自验证输出的非衬衫类样本
    "search[wood frame bed charcoal twin]",
    "search[charcoal twin wood bed frame]",
    "search[box spring twin wood charcoal]",
    "search[charcoal twin box spring wood]",
    "search[men's loafers slip resistant rubber outsole]",
    "search[slip resistant men's loafers rubber outsole]",
    "search[dark chocolate keto milk non gmo]",
    "search[non gmo keto milk dark chocolate]",
    "search[charcoal dust jeans straight leg men's]",
    "search[charcoal dust men's jeans straight leg]",
    "search[height adjustable office chair black under 130]",
    "search[height adjustable office chair navy under 130]",
    "search[decorative mirror narrow bronze]",
    "search[accent bronze narrow mirror]",
    "search[mattress]",
    "search[keywords]",
    "search[B07FKGQKZ1]",
    "click[Buy Now]",
    "click[buy now]",
    "click[back to search]",
    "click[Back to Search]",
    "click[Description]",
    "click[small]",
    "click[medium]",
    "click[B07F2G93BJ]",
    "null",
    # 新增：非衬衫类代表性样本（家具/灯具/个护/食品/电子）
    "search[twin wood bed frame charcoal]",
    "search[twin wood bed spring charcoal]",
    "search[glass shade pendant kitchen island living room]",
    "search[glass shade chandelier kitchen island living room]",
    "search[round area rug black gray 3x5 under 70]",
    "search[black gray round area rug 3x5 under 70]",
    "search[body lotion sensitive skin paraben free 4.2 oz]",
    "search[face cream sensitive skin paraben free 4.2 oz]",
    "search[cerave body cream sensitive skin 4.2 ounce]",
    "search[cetaphil body cream sensitive skin 4.2 ounce]",
    "search[gluten free honey cinnamon protein bar 12 pack]",
    "search[honey cinnamon protein bar gluten free 12 pack]",
    "search[tempered glass screen protector iphone 14 pro max]",
    "search[iphone 14 pro max tempered glass screen protector]",
    "search[men's polyester spandex jogger pants xx-large drawstring black]",
    "search[black men's polyester spandex jogger pants drawstring xx-large]",
    "search[Hanes men's classic fit t-shirt purple heather x-small]",
    "search[Hanes men's classic fit t-shirt navy heather x-small]",
    "search[men's sneakers breathable mesh running size 10]",
    "search[men's athletic shoes breathable mesh running size 10]",
]
