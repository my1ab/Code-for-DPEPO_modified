# 数据集索引范围 (Row Index Ranges)

数据来源: `PeterJinGo/nq_hotpotqa_train` (HuggingFace)
预处理脚本: `examples/data_preprocess/preprocess_search_r1_dataset.py`
处理后路径: `~/data/searchR1_processed_direct/`

---

## test.parquet

总行数: **51,713** ✅ 已验证

文件内按 **`nq`→`triviaqa`→`popqa`→`hotpotqa`→`2wikimultihopqa`→`musique`→`bamboogle`** 顺序排列。

| data_source | 论文标记 | 行索引范围 | 行数 | 类型 | 边界验证（末条 → 下一条首条） |
|:-----------|:-------:|:---------:|:----:|:----:|:--------------------------|
| nq | NQ† | `0` ~ `3,609` | 3,610 | in-domain | ✅ row 3609 [`nq`] → row 3610 [`triviaqa`] |
| triviaqa | TriviaQA⋆ | `3,610` ~ `14,922` | 11,313 | out-of-domain | ✅ row 14922 [`triviaqa`] → row 14923 [`popqa`] |
| popqa | PopQA⋆ | `14,923` ~ `29,189` | 14,267 | out-of-domain | ✅ row 29189 [`popqa`] → row 29190 [`hotpotqa`] |
| hotpotqa | HotpotQA† | `29,190` ~ `36,594` | 7,405 | in-domain | ✅ row 36594 [`hotpotqa`] → row 36595 [`2wikimultihopqa`] |
| 2wikimultihopqa | 2Wiki⋆ | `36,595` ~ `49,170` | 12,576 | out-of-domain | ✅ row 49170 [`2wikimultihopqa`] → row 49171 [`musique`] |
| musique | MuSiQue⋆ | `49,171` ~ `51,587` | 2,417 | out-of-domain | ✅ row 51587 [`musique`] → row 51588 [`bamboogle`] |
| bamboogle | Bamboogle⋆ | `51,588` ~ `51,712` | 125 | out-of-domain | ✅ 末行 51712，无后续 |

### 常用分片配置

```python
# ── 全覆盖 7 个数据集（各取前 100 条）──
slice_ranges = [
    (0, 99),           # nq 前 100
    (3610, 3709),      # triviaqa 前 100
    (14923, 15022),    # popqa 前 100
    (29190, 29289),    # hotpotqa 前 100
    (36595, 36694),    # 2wikimultihopqa 前 100
    (49171, 49270),    # musique 前 100
    (51588, 51687),    # bamboogle 前 100（但总共仅 125 条）
]

# ── 只测 in-domain (NQ + HotpotQA) ──
slice_ranges = [
    (0, 3609),         # 全部 NQ
    (29190, 36594),    # 全部 HotpotQA
]

# ── 只测 out-of-domain ──
slice_ranges = [
    (3610, 14922),     # triviaqa
    (14923, 29189),    # popqa
    (36595, 49170),    # 2wikimultihopqa
    (49171, 51587),    # musique
    (51588, 51712),    # bamboogle
]
```

---

## train.parquet

总行数: **169,615** ✅ 已验证

文件内按 **`nq`→`hotpotqa`** 顺序排列。

| data_source | 论文标记 | 行索引范围 | 行数 | 说明 | 边界验证 |
|:-----------|:-------:|:---------:|:----:|:----|:--------:|
| nq | NQ† | `0` ~ `79,167` | 79,168 | 训练集 | ✅ row 79167 [`nq`] → row 79168 [`hotpotqa`] |
| hotpotqa | HotpotQA† | `79,168` ~ `169,614` | 90,447 | 训练集 | ✅ 末行 169614，无后续 |

### 论文明确提到的训练配置

论文原文中明确提及的训练参数**仅有以下信息**：

> "We set the **train data size to 256** and use a group size of 5." —— Appendix E.1（仅 Search-Augmented QA）
> "each for **150 iterations**" (ALFWorld/WebShop) / "each for **200 iterations**" (Search-Augmented QA) —— Computing Details
> "Results are averaged over **3 random seeds**" —— Table 1 注释

**论文明确说 VS 论文没说的：**

| 信息 | 论文是否说明 |
|:----|:-----------:|
| train_data_size=256, group_size=5, 200 iterations (Search-QA) | ✅ 说了 |
| 150 iterations (ALFWorld/WebShop), 3 random seeds | ✅ 说了 |
| 使用 Qwen2.5-3B/7B（Search）, 1.5B/7B（ALFWorld/WebShop） | ✅ 说了 |
| **batch size 是多少** | ❌ **没说** |
| **seed 具体值** | ❌ **没说** |
| **shuffle / RandomSampler 方式** | ❌ **没说** |
| **test 集是否用全量** | ❌ **没说** |
| **total_epochs** | ❌ **没说** |
| **验证频率 (test_freq)** | ❌ **没说** |

> ⚠️ **Search-Augmented QA 没有使用 1.5B 模型**。论文中 1.5B 仅用于 ALFWorld 和 WebShop 实验（Table 1）。

### 论文未说的内容（来自代码，仅供参考）

以下信息论文**均未提及**，仅从代码 `run_search.sh` / `ray_trainer.py` / `create_rl_sampler()` 中推断：

```python
# create_rl_sampler() — 论文没说 shuffle 方式和 seed
if data_config.shuffle:
    sampler = RandomSampler(data_source=dataset, generator=torch.Generator().manual_seed(1))
else:
    sampler = SequentialSampler(data_source=dataset)
```

若采用 `RandomSampler(seed=1)` 从全量 train.parquet 中随机采样，每 iteration 取 256 条，
200 iterations 共 51,200 条。按整体数据集比例估算各数据集的期望采样数：

| data_source | 整体条数 | 整体占比 | 200×256=51,200 条期望采样 |
|:-----------|:-------:|:-------:|:------------------------:|
| nq | 79,168 | 46.7% | ~23,900 条 |
| hotpotqa | 90,447 | 53.3% | ~27,300 条 |

**代码 seed=1 模拟：取 500 条时各 data_source 分布**

训练集 `RandomSampler(seed=1)` 取 500 条：

| data_source | 条数 | 占比 |
|:-----------|:---:|:----:|
| nq | 228 | 45.6% |
| hotpotqa | 272 | 54.4% |
| **合计** | **500** | **100%** |

验证集 `SequentialSampler` 取前 500 条（按文件顺序截取，非随机）：

| data_source | 条数 | 占比 |
|:-----------|:---:|:----:|
| nq | 500 | 100% |
| **合计** | **500** | **100%** |

> ⚠️ 验证集按字母序排列，前 500 条全部为 nq（row 0~499）。若需覆盖率 7 个数据集，应使用随机采样或手动指定各数据集范围。

**验证集若改用 `RandomSampler(seed=1)` 取 500 条的分布对比：**

| data_source | Sequential 条数 | Sequential 占比 | Random 条数 | Random 占比 |
|:-----------|:--------------:|:--------------:|:-----------:|:-----------:|
| nq | 500 | 100.0% | 42 | 8.4% |
| triviaqa | 0 | 0.0% | 114 | 22.8% |
| popqa | 0 | 0.0% | 123 | 24.6% |
| hotpotqa | 0 | 0.0% | 73 | 14.6% |
| 2wikimultihopqa | 0 | 0.0% | 134 | 26.8% |
| musique | 0 | 0.0% | 13 | 2.6% |
| bamboogle | 0 | 0.0% | 1 | 0.2% |
| **合计** | **500** | **100%** | **500** | **100%** |

```python
# ray_trainer.py — 论文没说 test 是否为全量
self.val_dataloader = StatefulDataLoader(
    dataset=self.val_dataset,      # 全部 test.parquet (51,713 条)
    batch_size=val_batch_size,     # val_data_size=512, 但这是 batch size
    shuffle=False,                 # 顺序采样
    drop_last=False,               # 保留尾批 → 遍历全部
)
```

> **结论：论文从未说明验证是否使用全量测试集、batch size 具体值、seed 和 shuffle 方式。** 上述代码仅为开源仓库的实现方式，不能等同于论文的实验设置。

---

## 论文 Table 2 对应关系

| 论文名称 | data_source | 训练/评估 | test.parquet 范围 |
|:-------:|:-----------|:--------:|:----------------:|
| NQ† | `nq` | 训练 + in-domain 评估 | 0 ~ 3,609 |
| HotpotQA† | `hotpotqa` | 训练 + in-domain 评估 | 29,190 ~ 36,594 |
| TriviaQA⋆ | `triviaqa` | out-of-domain 评估 | 3,610 ~ 14,922 |
| PopQA⋆ | `popqa` | out-of-domain 评估 | 14,923 ~ 29,189 |
| 2Wiki⋆ | `2wikimultihopqa` | out-of-domain 评估 | 36,595 ~ 49,170 |
| MuSiQue⋆ | `musique` | out-of-domain 评估 | 49,171 ~ 51,587 |
| Bamboogle⋆ | `bamboogle` | out-of-domain 评估 | 51,588 ~ 51,712 |

> 注: `†` = in-domain (训练集中也包含), `⋆` = out-of-domain (仅评估)

---

> 验证脚本: `coldstart_test_search/verify_boundaries.py`
> 预处理脚本 (`preprocess_search_r1_dataset.py`) **不改变数据行顺序**，仅逐行转换格式。
