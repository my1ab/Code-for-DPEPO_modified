# 嵌入模型（Embedding）理论知识汇总

> 基于 vLLM 的 BERT-based 嵌入模型（如 BAAI/bge-base-en-v1.5）实践总结
>
> 参考文档: https://docs.vllm.ai/en/latest/models/pooling_models/embed/

---

## 目录

1. [嵌入模型概述](#1-嵌入模型概述)
2. [完整处理流程](#2-完整处理流程)
3. [Tokenization 阶段](#3-tokenization-阶段)
4. [Embedding Lookup 阶段](#4-embedding-lookup-阶段)
5. [BERT 编码阶段](#5-bert-编码阶段)
6. [Pooling 机制](#6-pooling-机制)
7. [归一化（Normalize）](#7-归一化normalize)
8. [相似度计算方法](#8-相似度计算方法)
9. [变长序列的处理](#9-变长序列的处理)
10. [Matryoshka 嵌入](#10-matryoshka-嵌入)
11. [BERT 嵌入模型规模对比](#11-bert-嵌入模型规模对比)
12. [vLLM 相关](#12-vllm-相关)

---

## 1. 嵌入模型概述

嵌入模型是一种将非结构化数据（文本、图像、音频等）转化为**结构化数值向量**（嵌入向量）的机器学习模型。

### 核心概念

| 概念 | 说明 |
|---|---|
| **序列嵌入** | 为整个输入序列生成 1 个嵌入向量 |
| **Token 嵌入** | 为序列中每个 token 生成嵌入向量 |
| **输出维度** | 由模型隐藏层宽度决定，与输入长度无关 |

### vLLM API 摘要

- **池化任务**: `embed`
- **离线 API**: `LLM.embed(...)`, `LLM.encode(..., pooling_task="embed")`, `LLM.score(...)`
- **在线 API**: OpenAI 兼容 (`/v1/embeddings`), Cohere Embed API (`/v2/embed`), Pooling API (`/pooling`)

---

## 2. 完整处理流程

```
文本输入
  │
  ├─ 1. Tokenization（分词）     "machine learning" → ["machine", "learning"]
  │                                 → [5025, 4083]（token ID 序列）
  │
  ├─ 2. Embedding Lookup（查表）  每个 token ID → 768 维向量
  │                                 N 个 token → N × 768 矩阵
  │
  ├─ 3. BERT 编码（12 层 Transformer）
  │     每层通过自注意力机制让 token 间互相交互
  │     输入: N × 768  →  输出: N × 768
  │
  ├─ 4. Pooling（池化）          N × 768 → 1 × 768（提取句子级表示）
  │
  ├─ 5. Normalize（归一化）      向量长度归一化为 1
  │
  └─ 最终嵌入向量 (768 维)
```

### 各阶段变量类型与维度

| 阶段 | 每个 token 的变量数 | 类型 | 范围 |
|---|---|---|---|
| Tokenization | **1** | 离散（整数 token ID） | [0, 30521]（词表大小 30,522） |
| Embedding 查表 | **768** | 连续（float16） | [-∞, +∞] |
| BERT 编码后 | **768** | 连续（float16） | [-∞, +∞] |
| 最终序列嵌入 | **768**（整个序列） | 连续（float16） | [-1, 1]（归一化后） |

---

## 3. Tokenization 阶段

### 机制

- 分词器: **WordPiece**（BERT 标准）
- 词表大小: **30,522**
- 每个 token 被映射为 **1 个离散变量**（整数 ID），信息量约 $\log_2(30522) \approx 14.9$ bits

### 示例

```
"Machine learning is a branch of artificial intelligence"
  ↓ Tokenizer
["[CLS]", "machine", "learning", "is", "a", "branch", "of", "artificial", "intelligence", "[SEP]"]
  ↓ 映射为 ID
[101, 5025, 4083, 2003, 1037, 3817, 1997, 7520, 4385, 102]
```

### 特殊 Token

| Token | 作用 |
|---|---|
| `[CLS]` | 放在序列开头，用于 CLS Pooling 提取句子级表示 |
| `[SEP]` | 放在序列末尾或句子间，标记分隔 |

---

## 4. Embedding Lookup 阶段

### 机制

每个 token ID 通过查表（Embedding 矩阵）映射为一个 **768 维连续向量**：

```
token ID 5025 → [0.012, -0.034, 0.078, ..., 0.056]   # 768 个 float
```

- Embedding 矩阵大小: 30,522 × 768（词表大小 × 隐藏层宽度）
- 这一层是可训练的，在预训练中学习

### 输出

```
N 个 token → N × 768 矩阵
```

---

## 5. BERT 编码阶段

### 机制

通过 **12 层 Transformer Encoder** 进行上下文编码：

- 每层包含: 多头自注意力（Multi-Head Self-Attention） + 前馈网络（FFN） + 残差连接 + LayerNorm
- token 间通过自注意力机制**互相交互**，每个 token 的表示融入了上下文信息

### 输入输出

```
输入:  N × 768
输出:  N × 768   (维度不变，但每个 token 的向量已被上下文更新)
```

### `[CLS]` token 的特殊作用

`[CLS]` 在 12 层 Transformer 中通过自注意力机制逐步聚合了整个序列的信息：

```
第 1 层: [CLS] 初步关注到相邻词
第 2 层: [CLS] 通过第1层信息，间接关注到更远的词
  ...
第 12 层: [CLS] 已聚合整个序列的语义信息
```

注意力计算公式：

$$\text{Attention}(\text{CLS}) = \sum_{i=1}^{N} \alpha_i \cdot h_i$$

其中 $\alpha_i$ 是 `[CLS]` 对第 $i$ 个 token 的注意力权重。

---

## 6. Pooling 机制

Pooling 的作用是将 **N 个 token 的变长隐状态序列** 压缩为 **1 个固定维度的向量**。

### 三种主流 Pooling 方式

#### CLS Pooling（bge-base-en-v1.5 使用）

直接取 `[CLS]` token 的输出向量：

```
BERT 输出 (11×768):
         维0    维1    ...  维767
[CLS]  → [0.12, -0.05, ..., 0.08]  ← 取这一行 (768维)
machine→ [0.44,  0.21, ..., 0.15]
learning→[0.09, -0.30, ..., 0.22]
  ...

最终嵌入 = [0.12, -0.05, ..., 0.08]  (768维)
```

- **本质**: 不做矩阵压缩，只取第 0 行。压缩发生在 BERT 内部的注意力机制中
- **优点**: 简单高效，`[CLS]` 在训练时已通过注意力"汇总"全局信息
- **缺点**: 所有信息压缩到一个 token，可能丢失细节

#### MEAN Pooling（平均池化）

对所有 token 的隐状态取逐维平均值：

```
最终嵌入 = (h₀ + h₁ + h₂ + ... + h_N) / N   (768维)
```

- **优点**: 充分利用所有 token 的信息，通常效果更好
- **缺点**: 计算量略大

#### MAX Pooling（最大池化）

对每个维度取所有 token 中的最大值：

```
对第 i 维: max(h₀[i], h₁[i], h₂[i], ..., h_N[i])
```

- **优点**: 捕捉最显著的特征
- **缺点**: 容易受噪声影响

### 对比总结

| Pooling | 向量维度 | 是否考虑所有 token | 计算量 | 典型适用场景 |
|---|---|---|---|---|
| CLS | 768 | ❌ 只用 [CLS] | $O(1)$ | BERT 原生训练方式 |
| MEAN | 768 | ✅ 平均所有 token | $O(N)$ | sentence-transformers 默认 |
| MAX | 768 | ✅ 取每维最大值 | $O(N)$ | 较少使用 |

> **注意**: pooling 类型应与模型训练方式匹配。切换 pooling 类型可能导致效果变差。

---

## 7. 归一化（Normalize）

Pooling 之后还有一个归一化步骤（当 `normalize=True` 时）：

$$\vec{e}_{\text{final}} = \frac{\vec{e}_{\text{pool}}}{\|\vec{e}_{\text{pool}}\|}$$

归一化后向量长度为 1，此时：

$$\cos(\vec{A}, \vec{B}) = \vec{A} \cdot \vec{B}$$

余弦相似度等于点积，计算更高效。

---

## 8. 相似度计算方法

### 方法 1: LLM.score（内部计算）

```python
(output,) = llm.score(text_a, text_b)
score = output.outputs.score   # 直接返回相似度标量
```

- vLLM 内部先分别嵌入两个文本，再计算余弦相似度
- 返回一个标量分数
- **不能**获取原始嵌入向量

### 方法 2: LLM.embed + 手动余弦

```python
(out_a,) = llm.embed(text_a)
(out_b,) = llm.embed(text_b)
sim = cosine_similarity(out_a.outputs.embedding, out_b.outputs.embedding)
```

- 先获取嵌入向量，再手动计算余弦相似度
- **可以**获取原始嵌入向量
- 适合批量计算和构建向量库

### 余弦相似度公式

$$\cos(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \cdot \|\vec{B}\|} = \frac{\sum_{i=1}^{d} A_i B_i}{\sqrt{\sum_{i=1}^{d} A_i^2} \cdot \sqrt{\sum_{i=1}^{d} B_i^2}}$$

### 两种方法对比

| | LLM.score | LLM.embed + 手动计算 |
|---|---|---|
| API 调用次数 | 1 次 | 2 次 |
| 能否获取向量 | ❌ | ✅ |
| 批量效率 | 高（内部优化） | 低（逐个调用） |
| 结果（normalize=True 时） | **完全一致** | **完全一致** |

### 结果一致的条件

当 `normalize=True` 时，嵌入向量已是单位向量，两种方法数学上等价：

$$\text{LLM.score}(A, B) = \cos(\text{embed}(A), \text{embed}(B)) = \text{embed}(A) \cdot \text{embed}(B)$$

### 可能导致结果不一致的情况

| 情况 | 原因 |
|---|---|
| 关闭归一化 (`use_activation=False`) | `LLM.score` 内部仍归一化，手动计算若未归一化则不同 |
| 使用不同 pooling 类型 | 两种方法用了不同 pooling |
| Matryoshka 降维 | 两方法用了不同 `dimensions` 参数 |
| 数值精度 | fp16 vs fp32 计算的微小差异 |

### 选型建议

| 场景 | 推荐方法 |
|---|---|
| 只需要相似度分数 | **LLM.score**（更简洁高效） |
| 需要保存/复用嵌入向量（如构建向量库） | **LLM.embed**（可拿到原始向量） |
| 批量计算 N×N 相似度矩阵 | **LLM.embed**（先批量 embed 再两两计算，避免重复推理） |

---

## 9. 变长序列的处理

### 核心机制

无论输入文本有多少个 token，经过 Pooling 后输出**永远是固定维度**（768 维），因此不同长度的文本可以直接比较。

### 示例

```
文本A: "Machine learning is a branch of artificial intelligence" (10 个词)
  → [CLS] machine learning is a branch of artificial intelligence [SEP]
  → 11 个 token → 11 × 768 → CLS Pooling → 1 × 768

文本B: "Deep learning is a subset of machine learning" (8 个词)
  → [CLS] deep learning is a subset of machine learning [SEP]
  → 10 个 token → 10 × 768 → CLS Pooling → 1 × 768

相似度 = cos(768维, 768维) = 0.8046  ✅ 可以计算
```

### 各阶段维度变化

| 步骤 | 文本A | 文本B |
|---|---|---|
| Tokenizer | 11 个 token | 10 个 token |
| Embedding | 11×768 | 10×768 |
| BERT 编码 | 11×768 | 10×768 |
| **CLS Pooling** | **768** | **768** |

> CLS token 就像"摘要"，无论会议纪要有多少页，最终都提取出一份固定格式的摘要，可以直接比较。

---

## 10. Matryoshka 嵌入

### 概念

Matryoshka Representation Learning (MRL) 是一种训练技术，允许用户在**性能和成本之间权衡**——截取嵌入向量的前 N 维仍能保持较好的语义表示。

### 工作原理

```
完整嵌入 (768维): [v₀, v₁, v₂, ..., v₇₆₇]
                         ↓ 截取前 N 维
截断嵌入 (128维): [v₀, v₁, v₂, ..., v₁₂₇]  ← 仍可用，精度略降
```

类比俄罗斯套娃（Matryoshka）：外层套娃包含内层，每层都是完整的。

### 使用方式

**离线**:
```python
outputs = llm.embed(
    ["Follow the white rabbit."],
    pooling_params=PoolingParams(dimensions=32),
)
```

**在线**:
```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "Follow the white rabbit.", "model": "...", "dimensions": 32}'
```

### 支持与不支持

| 模型 | 支持 Matryoshka | 原因 |
|---|---|---|
| `BAAI/bge-base-en-v1.5` | ❌ | 训练时未使用 MRL |
| `Snowflake/snowflake-arctic-embed-m-v1.5` | ✅ | 原生支持 |
| `jinaai/jina-embeddings-v3` | ✅ | 原生支持 |

### 强制开启（不推荐）

```python
llm = LLM(
    model=local_model_path,
    runner="pooling",
    hf_overrides={"is_matryoshka": True},  # 强制开启
)
```

> **警告**: 未经过 MRL 训练的模型强制开启后，降维效果会很差。

---

## 11. BERT 嵌入模型规模对比

### 显存估算公式

$$\text{显存} \approx \underbrace{\text{参数量} \times 2 \text{ bytes}}_{\text{权重 (fp16)}} + \underbrace{\text{KV Cache} + \text{激活值}}_{\text{约 30\%-60\% 额外开销}}$$

### 英文模型规模表

| 模型 | 参数量 | 嵌入维度 | 最大序列长度 | fp16 权重 | 预估显存 |
|---|---|---|---|---|---|
| `Snowflake/snowflake-arctic-embed-xs` | ~22M | 384 | 512 | ~44 MB | ~0.3-0.5 GB |
| `sentence-transformers/all-MiniLM-L6-v2` | ~23M | 384 | 256 | ~46 MB | ~0.3-0.5 GB |
| `BAAI/bge-small-en-v1.5` | ~33M | 384 | 512 | ~66 MB | ~0.4-0.7 GB |
| `intfloat/e5-small-v2` | ~33M | 384 | 512 | ~66 MB | ~0.4-0.7 GB |
| `BAAI/bge-base-en-v1.5` ← 当前使用 | ~109M | 768 | 512 | ~218 MB | ~0.8-1.5 GB |
| `intfloat/e5-base-v2` | ~109M | 768 | 512 | ~218 MB | ~0.8-1.5 GB |
| `Snowflake/snowflake-arctic-embed-m-v1.5` ✨MRL | ~109M | 768 | 512 | ~218 MB | ~0.8-1.5 GB |
| `Snowflake/snowflake-arctic-embed-m-v2.0` | ~109M | 768 | 8192 | ~218 MB | ~1.0-1.8 GB |
| `sentence-transformers/all-mpnet-base-v2` | ~110M | 768 | 384 | ~220 MB | ~0.8-1.5 GB |
| `nomic-ai/nomic-embed-text-v1` | ~137M | 768 | 2048 | ~274 MB | ~1.0-1.8 GB |
| `BAAI/bge-large-en-v1.5` | ~335M | 1024 | 512 | ~670 MB | ~1.5-3.0 GB |
| `intfloat/e5-large-v2` | ~335M | 1024 | 512 | ~670 MB | ~1.5-3.0 GB |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | ~335M | 1024 | 8192 | ~670 MB | ~2.0-3.5 GB |

### 选型建议

| 场景 | 推荐模型 | 理由 |
|---|---|---|
| 资源受限 / 快速测试 | `snowflake-arctic-embed-xs` (~22M) | 最小，显存 <0.5GB |
| 通用英文嵌入 | `BAAI/bge-base-en-v1.5` (~109M) | 性价比最高 |
| 需要 Matryoshka 降维 | `snowflake-arctic-embed-m-v1.5` (~109M) | 原生支持维度截断 |
| 长文本处理 | `snowflake-arctic-embed-m-v2.0` (~109M) | 支持 8192 token |
| 追求最高精度 | `BAAI/bge-large-en-v1.5` (~335M) | MTEB 英文榜领先 |

---

## 12. vLLM 相关

### 平台检测

vLLM 在 `import` 时自动检测硬件平台：

```
INFO 07-15 21:35:51 [__init__.py:216] Automatically detected platform cuda.
```

检测顺序: CUDA → ROCm → XPU → CPU

### gpu_memory_utilization 参数

vLLM 默认占用 GPU **90%** 显存。当 GPU 被其他进程占用时需降低：

```python
llm = LLM(
    model=local_model_path,
    runner="pooling",
    gpu_memory_utilization=0.3,  # 降低到 30%
)
```

### 进度条禁用

vLLM 内部使用 `tqdm` 显示进度条（输出到 stderr）。通过环境变量禁用：

```python
os.environ["TQDM_DISABLE"] = "1"  # 必须在 import vllm 之前设置
from vllm import LLM, PoolingParams
```

### PoolerConfig 配置

从日志中可查看模型的 pooler 配置：

```
pooler_config=PoolerConfig(
    pooling_type='CLS',    # CLS / MEAN / MAX
    normalize=True,        # 是否归一化
    dimensions=None,       # Matryoshka 降维维度
)
```

### 支持的 Pooling 参数

```python
PoolingParams(
    use_activation: bool | None = None,   # 启用/禁用归一化
    dimensions: int | None = None,         # Matryoshka 输出维度
)
```

### 已移除功能

- `normalize` 已从 `PoolingParams` 中移除，改用 `use_activation`

---

## 附：当前测试环境配置

| 项目 | 值 |
|---|---|
| 模型 | `BAAI/bge-base-en-v1.5` |
| 架构 | BertModel (BERT-base) |
| 参数量 | ~109M |
| 嵌入维度 | 768 |
| 最大序列长度 | 512 |
| Pooling 类型 | CLS |
| 归一化 | True |
| 精度 | fp16 |
| 模型显存占用 | ~0.21 GiB |
| GPU | NVIDIA L40 (卡7) |
| gpu_memory_utilization | 0.3 |
| vLLM 版本 | v0.11.0 |
