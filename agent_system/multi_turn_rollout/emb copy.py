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

    def __init__(self, model_path="models/bge-large-en-v1.5",
                 gpu_memory_utilization=0.1,
                 similarity_threshold=0.975,
                 device='cpu'):
        """初始化嵌入模型。

        参数：
            model_path: 嵌入模型本地目录路径
            gpu_memory_utilization: GPU 显存使用比例（仅在 device='cuda' 时参考，
                当前不直接控制 transformers 的显存分配，仅保留为兼容参数）
            similarity_threshold: search 嵌入相似度阈值
            device: 推理设备。
                - "cpu"（默认）：CPU 推理，绕过 Ray actor GPU 可见性问题
                - "cuda" 或 "cuda:N"：GPU 推理（手动指定，需确保进程有 GPU 可见性）
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
        """方案 B + E：分类型相似度计算（同步自 test_bert_similarity.py）。

        - click: 大小写不敏感精确匹配，返回 0.0 或 1.0
        - search: 词集匹配优先（方案 E），嵌入兜底，≥ 阈值返回线性放大值，否则 0.0
        - 跨类型: 返回 0.0
        - null: null==null 返回 1.0，否则 0.0
        - 其他(unknown): 精确匹配（已转小写）返回 1.0 或 0.0

        方案 E：在嵌入计算之前，先对 search 内容做词集精确匹配，
        捕获所有"词序重排"变体，避免嵌入模型对词序不稳定导致的漏检。
        """
        # null 处理
        if action_a == "null" or action_b == "null":
            return 1.0 if action_a == action_b else 0.0

        action_a_lower = action_a.lower()
        action_b_lower = action_b.lower()

        # 提取动作类型
        type_a = action_a_lower.split("[")[0] if "[" in action_a_lower else "unknown"
        type_b = action_b_lower.split("[")[0] if "[" in action_b_lower else "unknown"

        # 不同动作类型 -> 不相似
        if type_a != type_b:
            return 0.0

        if type_a == "click":
            # click 按键：精确匹配（已转小写）
            target_a = action_a_lower[6:-1]
            target_b = action_b_lower[6:-1]
            return 1.0 if target_a == target_b else 0.0

        if type_a == "search":
            # search 内容（使用小写版本，与 test_bert_similarity.py 一致）
            content_a = action_a_lower[7:-1]
            content_b = action_b_lower[7:-1]

            # 精确匹配快速路径
            if content_a == content_b:
                return 1.0

            # 方案 E：词集匹配
            # 完全相同词集的不同排列 -> 一定是重复
            words_a = set(content_a.split())
            words_b = set(content_b.split())
            if words_a == words_b:
                return 1.0

            # 嵌入语义相似度兜底
            sim = self.cosine_similarity(self.embed(content_a), self.embed(content_b))
            # 阈值过滤 + 范围放大：
            #   1) 低于阈值 self.threshold 的相似度贡献为 0
            #   2) 通过阈值的相似度从 [self.threshold, 1.0] 线性放大到 [0, 1]
            if sim < self.threshold:
                return 0.0
            return (sim - self.threshold) / (1.0 - self.threshold)

        # 其他类型（unknown）：精确匹配（已转小写）
        return 1.0 if action_a_lower == action_b_lower else 0.0

    def weighted_match(self, action_a, action_b):
        """加权匹配（方案 B + E）：使用分类型相似度计算。

        click: 精确匹配（大小写不敏感）
        search: 词集匹配 + 嵌入兜底（阈值过滤 + 范围放大）
        """
        return self.action_similarity(action_a, action_b)
