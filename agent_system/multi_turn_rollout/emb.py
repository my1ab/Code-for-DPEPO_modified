"""
嵌入相似度模块。

提供 EmbeddingManager 类，用于将 calculate_penalties 中的
精确字符串匹配替换为语义相似度加权计数。

相似度计算方案同步自 test_bert_similarity/test_bert_similarity.py（方案 B + E）：
  - click: 大小写不敏感精确匹配，返回 0.0 或 1.0
  - search: 词集匹配优先（方案 E），嵌入兜底，≥ 阈值返回线性放大值，否则 0.0
  - 跨类型: 返回 0.0
  - null: null==null 返回 1.0，否则 0.0
  - 其他(unknown): 精确匹配（已转小写）

全局开关和配置变量定义在 rollout_loop_parallel_webshop.py 中:
    USE_EMBEDDING_SIMILARITY, EMBEDDING_MODEL_PATH,
    EMBEDDING_SIMILARITY_THRESHOLD, EMBEDDING_GPU_MEMORY_UTILIZATION

设备选择：
    EmbeddingManager(device=None) 时默认 CPU 推理
    手动指定 device="cuda" 或 "cuda:N" 可切换到 GPU 推理
    EMBEDDING_GPU_MEMORY_UTILIZATION 仅在 GPU 模式下作为显存使用比例参考
"""

import math
import os
import re


# 使用 transformers 在 CPU 上推理，绕过 Ray actor 的 GPU 可见性问题。
# bge-large-en-v1.5 是小模型（~1.3GB），CPU 推理足够快。
import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingManager:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path="emb_models/bge-large-en-v1.5",
                 gpu_memory_utilization=0.1,
                 similarity_threshold=0.9,
                 device='cpu',
                 task_type='webshop'):
        """初始化嵌入模型。

        参数：
            model_path: 嵌入模型本地目录路径
            gpu_memory_utilization: GPU 显存使用比例（仅在 device='cuda' 时参考，
                当前不直接控制 transformers 的显存分配，仅保留为兼容参数）
            similarity_threshold: search 嵌入相似度阈值
            device: 推理设备。
                - "cpu"（默认）：CPU 推理，绕过 Ray actor GPU 可见性问题
                - "cuda" 或 "cuda:N"：GPU 推理（手动指定，需确保进程有 GPU 可见性）
            task_type: 任务类型，决定 action_similarity 的分类型策略
                - "webshop"：click 精确匹配 + search 词集匹配+嵌入兜底（方案 B+E）
                - "search"：search/answer 词集匹配+嵌入兜底，null 精确匹配
                其他值退化为 webshop 行为
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        # ── 设备解析：默认 CPU，手动传入 device 时使用指定设备 ──
        self.gpu_memory_utilization = gpu_memory_utilization  # 保留为兼容字段
        if device == "cpu":
            self.device = torch.device("cpu")
            _device_desc = "CPU（默认，绕过 Ray actor GPU 可见性问题）"
        else:
            self.device = torch.device(device)
            _device_desc = f"GPU ({device})（手动指定）"
        print(f"[EmbeddingManager] 推理设备: {_device_desc}, gpu_memory_utilization={gpu_memory_utilization}（仅 GPU 模式参考）")

        # ── 本地路径校验 + 禁用远端下载 ──
        model_path_abs = os.path.abspath(model_path)
        if not os.path.isdir(model_path_abs):
            raise RuntimeError(
                f"[EmbeddingManager] 本地嵌入模型目录不存在: {model_path!r} (abs: {model_path_abs})\n"
                f"请将 bge-large-en-v1.5 模型放置到该目录，或修改 EMBEDDING_MODEL_PATH。\n"
                f"禁止从远端下载——已设置为仅本地加载模式。"
            )
        print(f"[EmbeddingManager] 本地模型校验通过: {model_path_abs}")

        # 禁用 HuggingFace Hub 远端下载
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

        try:
            # 不再强制 CPU，按 self.device 加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_abs, trust_remote_code=False, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                model_path_abs, trust_remote_code=False, local_files_only=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[EmbeddingManager] 模型已加载到 {self.device}")
        except Exception as e:
            raise RuntimeError(
                f"[EmbeddingManager] 加载本地嵌入模型失败: {model_path_abs}\n"
                f"  原始异常: {type(e).__name__}: {e}\n"
                f"  已禁用远端下载，不会自动从 HuggingFace 下载。请检查本地模型文件是否完整。"
            ) from e
        # search 嵌入相似度阈值（仅 search 动作的嵌入兜底使用）
        self.threshold = similarity_threshold
        self._cache = {}
        self._initialized = True

        # 任务类型与对应的分类型相似度处理器
        # webshop: click 精确匹配 + search 词集匹配+嵌入兜底（方案 B+E）
        # search: search/answer 词集匹配+嵌入兜底，null 精确匹配
        self.task_type = task_type if task_type in _TASK_HANDLERS else 'webshop'
        self._handler = _TASK_HANDLERS[self.task_type]

    def embed(self, text):
        """获取单个文本的嵌入向量（带缓存）。"""
        if text in self._cache:
            return self._cache[text]
        emb = self._encode([text])[0]
        self._cache[text] = emb
        return emb

    def embed_batch(self, texts):
        """批量嵌入，未缓存的批量请求，已缓存的直接返回。"""
        results = [None] * len(texts)
        uncached_indices, uncached_texts = [], []
        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        if uncached_texts:
            embeddings = self._encode(uncached_texts)
            for idx, emb in zip(uncached_indices, embeddings):
                self._cache[texts[idx]] = emb
                results[idx] = emb
        return results

    def _encode(self, texts):
        """使用 transformers 模型编码文本，返回嵌入向量列表。

        使用 CLS pooling + 归一化，与 bge-large-en-v1.5 官方用法和 vLLM 的
        PoolerConfig(pooling_type='CLS', normalize=True) 一致。
        """
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # CLS pooling：取 last_hidden_state 的第一个 token（[CLS]）并归一化
        cls_emb = outputs.last_hidden_state[:, 0, :]
        norm = cls_emb.norm(dim=1, keepdim=True)
        cls_emb = cls_emb / norm.clamp(min=1e-9)
        return cls_emb.cpu().tolist()

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """计算两个向量的余弦相似度。"""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def similarity(self, text_a, text_b):
        """返回两个文本的余弦相似度 [0, 1]。精确匹配快速路径。"""
        if text_a == text_b:
            return 1.0
        return self.cosine_similarity(self.embed(text_a), self.embed(text_b))

    def action_similarity(self, action_a, action_b):
        """分类型相似度计算，委托给 task_type 对应的处理器。

        通过构造时传入的 task_type 选择计算策略：
          - "webshop"：click 精确匹配 + search 词集匹配+嵌入兜底（方案 B+E）
          - "search"：search 词集匹配+嵌入兜底，answer 完全匹配，null 精确匹配
          - "answer"：等同 search 任务（answer 为终止动作，走完全匹配）

        所有处理器的公共约定：
          - 跨动作类型：返回 0.0
          - null vs null：返回 1.0；null vs 有效动作：返回 0.0
          - search 内容：词集相同返回 1.0；否则走嵌入，< 阈值返回 0.0，
            >= 阈值按线性放大返回 (sim - threshold) / (1 - threshold)
          - answer 内容：完全匹配（已统一小写）返回 1.0 或 0.0
          - click 目标：大小写不敏感精确匹配（仅 webshop 任务）
        """
        return self._handler(self, action_a, action_b)

    def weighted_match(self, action_a, action_b):
        """加权匹配：委托给 action_similarity（按 task_type 分类型策略）。"""
        return self.action_similarity(action_a, action_b)

    def raw_similarity(self, action_a, action_b):
        """返回未归一化的原始余弦相似度（阈值过滤前、线性放大前）。

        与 action_similarity 的差异：
          - 不做阈值过滤：即使 cosine 低于 threshold 也返回真实值
          - 不做线性放大：返回 [0, 1] 的原始 cosine，而非放大后的值
          - 精确匹配/词集匹配：返回 1.0（与 action_similarity 一致）
          - click/unknown/跨类型：与 action_similarity 一致（1.0/0.0），
            note 标注来源，方便日志区分

        用于测试与日志观察，不参与训练时的相似度计算。

        Returns:
            (raw_sim, note): raw_sim 为 float；note 标注来源：
              "exact" / "wordset" / "embedding" / "click" / "null" /
              "cross-type" / "unknown"
        """
        # null 处理
        if action_a == "null" or action_b == "null":
            return (1.0, "null") if action_a == action_b else (0.0, "null")

        action_a_lower = action_a.lower()
        action_b_lower = action_b.lower()

        type_a, content_a = _parse_action(action_a_lower)
        type_b, content_b = _parse_action(action_b_lower)

        # 不同动作类型 -> 不相似
        if type_a != type_b:
            return (0.0, "cross-type")

        # search 内容：返回阈值过滤前的原始 cosine
        # （webshop/search 任务中 search 走此分支；answer 走精确匹配，见下方 unknown 分支）
        if type_a in _SEARCH_ACTION_TYPES:
            if content_a == content_b:
                return (1.0, "exact")
            if set(content_a.split()) == set(content_b.split()):
                return (1.0, "wordset")
            sim = self.cosine_similarity(self.embed(content_a), self.embed(content_b))
            return (sim, "embedding")

        # webshop 任务的 click：精确匹配（已转小写）
        if type_a == "click":
            return (1.0 if content_a == content_b else 0.0, "click")

        # 其他类型（unknown / search 任务中误入的 click 等）：精确匹配（已转小写）
        return (1.0 if action_a_lower == action_b_lower else 0.0, "unknown")


# ───────────────────────────────────────────────────────────────────────
# 分任务相似度策略处理器（方案 B + E）
# ───────────────────────────────────────────────────────────────────────
# 各处理器签名统一为 (emb_mgr, action_a, action_b) -> float，值域 [0, 1]。
# 通过模块级注册表 _TASK_HANDLERS 在 EmbeddingManager 构造时选取。

# Search 任务的动作类型列表（search / answer），用于 _search_task_handler
_SEARCH_ACTION_TYPES = ("search",)

# # WebShop 任务的动作类型列表（click / search），用于 _webshop_task_handler
# _WEBSHOP_ACTION_TYPES = ("click", "search")


def _parse_action(action_lower):
    """从已转小写的动作字符串中解析出 (type, content)。

    支持两种动作格式：
      - 方括号格式（WebShop）：search[...], click[...], answer[...]
      - 尖括号格式（Search）：<search>...</search>, <answer>...</answer>, <click>...</click>

    解析逻辑：
      - "null" -> ("null", "")
      - "search[...]" / "<search>...</search>" -> ("search", "...")
      - "answer[...]" / "<answer>...</answer>" -> ("answer", "...")
      - "click[...]" / "<click>...</click>" -> ("click", "...")
      - 其他 -> ("unknown", action_lower)

    数据来源验证：
      - WebShop 轨迹（验证轨迹/1.5B_epoch3.5_hislen8_test_v2.json）使用方括号格式
      - Search 轨迹（验证轨迹search/search_coldstart_*.json）使用尖括号格式
      - RL 训练 sample JSON 中 action_dict 保持与各任务 coldstart 轨迹一致的格式
    """
    # 尖括号格式：<search>...</search>, <answer>...</answer>, <click>...</click>
    # 针对search
    xml_match = re.match(r'^<(search|answer|click)>(.*)</\1>$', action_lower, re.DOTALL)
    if xml_match:
        type_str = xml_match.group(1)
        content = xml_match.group(2).strip()
        return type_str, content

    # 方括号格式：search[...], answer[...], click[...]
    # 针对webshop
    if "[" not in action_lower:
        return "unknown", action_lower
    idx = action_lower.index("[")
    type_str = action_lower[:idx]
    # 去掉 "<type>[" 前缀和尾部的 "]"；若以 "]" 结尾则剥离
    content = action_lower[idx + 1:]
    if content.endswith("]"):
        content = content[:-1]
    return type_str, content


def _text_content_similarity(emb_mgr, content_a, content_b):
    """search/answer 文本的词集匹配 + 嵌入兜底（方案 E + 线性放大）。

    - 精确匹配：返回 1.0
    - 词集相同：返回 1.0（方案 E：词序重排视为重复）
    - 嵌入相似度 >= 阈值：返回 (sim - threshold) / (1 - threshold)（线性放大）
    - 嵌入相似度 < 阈值：返回 0.0
    """
    # 精确匹配快速路径
    if content_a == content_b:
        return 1.0

    # 方案 E：词集匹配，完全相同词集的不同排列 -> 一定是重复
    if set(content_a.split()) == set(content_b.split()):
        return 1.0

    # 嵌入语义相似度兜底
    sim = emb_mgr.cosine_similarity(emb_mgr.embed(content_a), emb_mgr.embed(content_b))
    # 阈值过滤 + 范围放大：
    #   1) 低于阈值 self.threshold 的相似度贡献为 0
    #   2) 通过阈值的相似度从 [threshold, 1.0] 线性放大到 [0, 1]
    if sim < emb_mgr.threshold:
        return 0.0
    return (sim - emb_mgr.threshold) / (1.0 - emb_mgr.threshold)
    # return sim


def _webshop_task_handler(emb_mgr, action_a, action_b):
    """WebShop 任务分类型相似度计算（方案 B + E）。

    - click: 大小写不敏感精确匹配，返回 0.0 或 1.0
    - search: 词集匹配优先（方案 E），嵌入兜底 + 线性放大
    - 跨类型: 返回 0.0
    - null: null==null 返回 1.0，否则 0.0
    - 其他(unknown): 精确匹配（已转小写）返回 1.0 或 0.0
    """
    # null 处理
    if action_a == "null" or action_b == "null":
        return 1.0 if action_a == action_b else 0.0

    action_a_lower = action_a.lower()
    action_b_lower = action_b.lower()

    type_a, content_a = _parse_action(action_a_lower)
    type_b, content_b = _parse_action(action_b_lower)

    # 不同动作类型 -> 不相似
    if type_a != type_b:
        return 0.0

    if type_a == "click":
        # click 按键：精确匹配（已转小写）
        return 1.0 if content_a == content_b else 0.0

    if type_a == "search":
        # search 内容：词集匹配 + 嵌入兜底（线性放大）
        return _text_content_similarity(emb_mgr, content_a, content_b)

    # 其他类型（unknown）：精确匹配（已转小写）
    return 1.0 if action_a_lower == action_b_lower else 0.0


def _search_task_handler(emb_mgr, action_a, action_b):
    """Search（QA 检索）任务分类型相似度计算（方案 B + E）。

    与 WebShop 的差异：Search 任务没有 click 动作，终止动作为 answer[...]。
    - search: 词集匹配优先（方案 E），嵌入兜底 + 线性放大
    - answer: 完全匹配（已统一小写），返回 0.0 或 1.0
    - 跨类型: 返回 0.0
    - null: null==null 返回 1.0，否则 0.0
    - 其他(unknown): 精确匹配（已转小写）返回 1.0 或 0.0
    """
    # null 处理
    if action_a == "null" or action_b == "null":
        return 1.0 if action_a == action_b else 0.0

    action_a_lower = action_a.lower()
    action_b_lower = action_b.lower()

    type_a, content_a = _parse_action(action_a_lower)
    type_b, content_b = _parse_action(action_b_lower)

    # 不同动作类型 -> 不相似
    if type_a != type_b:
        return 0.0

    if type_a == "search":
        # search 内容：词集匹配 + 嵌入兜底（线性放大）
        return _text_content_similarity(emb_mgr, content_a, content_b)

    if type_a == "answer":
        # answer 内容：完全匹配（已统一小写）
        return 1.0 if content_a == content_b else 0.0

    # 其他类型（unknown / 误入的 click 等）：精确匹配（已转小写）
    return 1.0 if action_a_lower == action_b_lower else 0.0


# 任务类型 -> 处理器注册表
_TASK_HANDLERS = {
    'webshop': _webshop_task_handler,
    'search': _search_task_handler,
}
