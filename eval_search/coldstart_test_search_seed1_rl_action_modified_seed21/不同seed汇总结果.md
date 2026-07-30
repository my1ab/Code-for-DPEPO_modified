# 不同 Seed 汇总结果

三个模型（使用相同权重，不同推理 seed）：
- **emb_3.5epoch_without_rl** (SFT only, 无RL): `/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-10500`
- **emb_080** (RL, emb b080): `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs1/search_emb_b080_bs1/global_step_500/merged`
- **noemb** (RL, 无emb): `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs1/search_noemb_bs1/global_step_500/merged`

---

## 第 1 章：Seed21 结果

| 数据集 | emb_3.5epoch_without_rl | emb_080 | noemb |
|--------|:-----------------------:|:-------:|:-----:|
| 2wikimultihopqa_12576 | 71/200 = **35.50%** | 73/200 = **36.50%** | 74/200 = **37.00%** |
| bamboogle_125 | 37/125 = **29.60%** | 35/125 = **28.00%** | 35/125 = **28.00%** |
| hotpotqa_7405 | 72/200 = **36.00%** | 69/200 = **34.50%** | 75/200 = **37.50%** |
| musique_2417 | 22/200 = **11.00%** | 22/200 = **11.00%** | 22/200 = **11.00%** |
| nq_3610 | 75/200 = **37.50%** | 76/200 = **38.00%** | 82/200 = **41.00%** |
| popqa_14267 | 81/200 = **40.50%** | 84/200 = **42.00%** | 85/200 = **42.50%** |
| triviaqa_11313 | 116/200 = **58.00%** | 116/200 = **58.00%** | 117/200 = **58.50%** |
| **总成功率** | **474/1325 = 35.77%** | **475/1325 = 35.85%** | **490/1325 = 36.98%** |

**说明：** 输出文件夹位于 `coldstart_test_search_seed1_rl_action_modified_seed21/` 下，对应子文件夹 `test_emb_3.5epoch_without_rl/`、`test_emb_080/`、`test_noemb/`。

---

## 第 2 章：Seed11 结果

| 数据集 | emb_3.5epoch_without_rl | emb_080 | noemb |
|--------|:-----------------------:|:-------:|:-----:|
| 2wikimultihopqa_12576 | 62/200 = **31.00%** | 66/200 = **33.00%** | 53/200 = **26.50%** |
| bamboogle_125 | 43/125 = **34.40%** | 29/125 = **23.20%** | 38/125 = **30.40%** |
| hotpotqa_7405 | 66/200 = **33.00%** | 73/200 = **36.50%** | 65/200 = **32.50%** |
| musique_2417 | 24/200 = **12.00%** | 26/200 = **13.00%** | 31/200 = **15.50%** |
| nq_3610 | 73/200 = **36.50%** | 67/200 = **33.50%** | 61/200 = **30.50%** |
| popqa_14267 | 79/200 = **39.50%** | 76/200 = **38.00%** | 75/200 = **37.50%** |
| triviaqa_11313 | 105/200 = **52.50%** | 114/200 = **57.00%** | 103/200 = **51.50%** |
| **总成功率** | **452/1325 = 34.11%** | **451/1325 = 34.04%** | **426/1325 = 32.15%** |

**说明：** 输出文件夹位于 `coldstart_test_search_seed1_rl_action_modified_seed11/` 下，对应子文件夹 `test_emb_3.5epoch_without_rl/`、`test_emb_080/`、`test_noemb/`。

---

## 第 3 章：Seed1 结果

| 数据集 | emb_3.5epoch_without_rl | emb_080 | noemb |
|--------|:-----------------------:|:-------:|:-----:|
| 2wikimultihopqa_12576 | 75/200 = **37.50%** | 78/200 = **39.00%** | 68/200 = **34.00%** |
| bamboogle_125 | 35/125 = **28.00%** | 36/125 = **28.80%** | 35/125 = **28.00%** |
| hotpotqa_7405 | 73/200 = **36.50%** | 71/200 = **35.50%** | 71/200 = **35.50%** |
| musique_2417 | 27/200 = **13.50%** | 27/200 = **13.50%** | 24/200 = **12.00%** |
| nq_3610 | 70/200 = **35.00%** | 75/200 = **37.50%** | 74/200 = **37.00%** |
| popqa_14267 | 87/200 = **43.50%** | 83/200 = **41.50%** | 85/200 = **42.50%** |
| triviaqa_11313 | 119/200 = **59.50%** | 129/200 = **64.50%** | 122/200 = **61.00%** |
| **总成功率** | **486/1325 = 36.68%** | **499/1325 = 37.66%** | **479/1325 = 36.15%** |

**说明：** 输出文件夹位于 `coldstart_test_search_seed1_rl_action_modified/` 下。其中 emb_3.5epoch_without_rl 分两个子文件夹运行：
- `test_emb_3.5epoch_without_rl-1/`：hotpotqa_7405, nq_3610, popqa_14267, triviaqa_11313（349/800 = 43.62%）
- `test_emb_3.5epoch_without_rl-2/`：2wikimultihopqa_12576, bamboogle_125, musique_2417（137/525 = 26.10%）
- `test_emb_080/`、`test_noemb/` 各为完整文件夹。

---

## 第 4 章：汇总成功率结果比较

### 4.1 各模型在三种 Seed 下的总成功率

| 模型 | Seed1 | Seed11 | Seed21 | 三 Seed 平均 |
|------|:-----:|:------:|:------:|:------------:|
| **emb_3.5epoch_without_rl** (SFT) | 486/1325 = 36.68% | 452/1325 = 34.11% | 474/1325 = 35.77% | 1412/3975 = **35.52%** |
| **emb_080** (RL, emb b080) | 499/1325 = 37.66% | 451/1325 = 34.04% | 475/1325 = 35.85% | 1425/3975 = **35.85%** |
| **noemb** (RL, 无emb) | 479/1325 = 36.15% | 426/1325 = 32.15% | 490/1325 = 36.98% | 1395/3975 = **35.09%** |

### 4.2 模型表现排名（按三 Seed 平均总成功率）

| 排名 | 模型 | 三 Seed 平均成功率 | 最好表现 | 最差表现 | 波动范围 |
|:----:|------|:------------------:|:--------:|:--------:|:--------:|
| **1** | **emb_080** (RL, emb b080) | **35.85%** | 37.66% (seed1) | 34.04% (seed11) | 3.62% |
| **2** | **emb_3.5epoch_without_rl** (SFT) | **35.52%** | 36.68% (seed1) | 34.11% (seed11) | 2.57% |
| **3** | **noemb** (RL, 无emb) | **35.09%** | 36.98% (seed21) | 32.15% (seed11) | 4.83% |

### 4.3 各数据集上的 3 Seed 平均成功率

| 数据集 | emb_3.5epoch_without_rl | emb_080 | noemb | 最佳模型 |
|--------|:-----------------------:|:-------:|:-----:|:--------:|
| 2wikimultihopqa_12576 | 34.67% | **36.17%** | 32.50% | emb_080 |
| bamboogle_125 | **30.67%** | 26.67% | 28.80% | emb_3.5epoch_without_rl |
| hotpotqa_7405 | 35.17% | **35.50%** | 35.17% | emb_080 |
| musique_2417 | 12.17% | 12.50% | **12.83%** | noemb |
| nq_3610 | **36.33%** | **36.33%** | 36.17% | emb_3.5epoch_without_rl / emb_080 |
| popqa_14267 | **41.17%** | 40.50% | 40.83% | emb_3.5epoch_without_rl |
| triviaqa_11313 | 56.67% | **59.83%** | 57.00% | emb_080 |

### 4.4 结论

1. **emb_080（带 emb 的 RL 模型）平均总成功率最高（35.85%），在 7 个数据集中取得 3 个第一**（2wikimultihopqa、hotpotqa、triviaqa），是综合表现最优的模型。
2. **emb_3.5epoch_without_rl（纯 SFT 模型）平均总成功率次之（35.52%），取得 2 个第一**（bamboogle、popqa），且 seed 间波动最小（2.57%），稳定性最好。
3. **noemb（无 emb 的 RL 模型）平均总成功率最低（35.09%），仅在 musique 上取得第一**，且 seed 间波动最大（4.83%），对 seed 敏感，稳定性最差。
4. 在 **triviaqa_11313** 上，emb_080 的 3 seed 平均显著领先（59.83% vs 57.00% / 56.67%），是 emb_080 的核心优势数据集。
5. 在 **musique_2417** 上，三个模型均表现很差（~12%），是该 benchmark 的难点，模型间差异不显著。
6. **emb_080 相比 emb_3.5epoch_without_rl 平均提升 +0.33%**，相比 noemb 平均提升 +0.76%，说明 emb 训练对 RL 有正向贡献，但提升幅度有限。
7. **Seed 对结果影响显著**：同一模型在不同 seed 下总成功率可相差 2.5–4.8 个百分点，建议在比较模型时参考多个 seed 的平均值而非单次结果。

---

## 第 5 章：脚本验证——模型权重与输出文件夹对应关系审查

### 5.1 三种模型权重

| 模型 | 权重路径 |
|------|----------|
| **emb_3.5epoch_without_rl** (SFT) | `/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-10500` |
| **emb_080** (RL) | `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs1/search_emb_b080_bs1/global_step_500/merged` |
| **noemb** (RL) | `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs1/search_noemb_bs1/global_step_500/merged` |

### 5.2 审查范围

审查了 `结果目标文件夹.txt` 中指定的 3 个脚本所在文件夹下的全部 shell 脚本，共 9 个有效脚本（3 seed × 3 模型，seed1 的 without_rl 拆分为 2 个脚本但属于同一实验）：

**9 个有效脚本完整路径：**

| # | Seed | 模型 | 脚本完整路径 |
|:-:|:----:|------|-------------|
| 1 | seed21 | without_rl | `coldstart_test_search_seed1_rl_action_modified_seed21/coldstart_search_local_3.5epoch_without_rl.sh` |
| 2 | seed21 | emb_080 | `coldstart_test_search_seed1_rl_action_modified_seed21/coldstart_search_local_3.5epoch_rl_multithread_080.sh` |
| 3 | seed21 | noemb | `coldstart_test_search_seed1_rl_action_modified_seed21/coldstart_search_local_3.5epoch_rl_multithread_noemb.sh` |
| 4 | seed11 | without_rl | `coldstart_test_search_seed1_rl_action_modified_seed11/coldstart_search_local_3.5epoch_without_rl.sh` |
| 5 | seed11 | emb_080 | `coldstart_test_search_seed1_rl_action_modified_seed11/coldstart_search_local_3.5epoch_rl_multithread_080.sh` |
| 6 | seed11 | noemb | `coldstart_test_search_seed1_rl_action_modified_seed11/coldstart_search_local_3.5epoch_rl_multithread_noemb.sh` |
| 7 | seed1 | without_rl | `coldstart_test_search_seed1_rl_action_modified/coldstart_search_local_3.5epoch_without_rl-1.sh` + `coldstart_search_local_3.5epoch_without_rl-2.sh` |
| 8 | seed1 | emb_080 | `coldstart_test_search_seed1_rl_action_modified/coldstart_search_local_3.5epoch_rl_multithread_080.sh` |
| 9 | seed1 | noemb | `coldstart_test_search_seed1_rl_action_modified/coldstart_search_local_3.5epoch_rl_multithread_noemb.sh` |

> 注：seed1 的 without_rl 拆分为 `-1.sh`（前 4 个数据集）和 `-2.sh`（后 3 个数据集）两个脚本并行运行，使用同一模型权重，合并后等价于 1 个实验，故有效脚本计为 9 个。

### 5.3 脚本中的 `--model` 与 `--json_output_dir` 对应表

| Seed | 模型 | 脚本中的 `--model` | 脚本中的 `--json_output_dir` | `--seed` |
|------|------|----------|----------|:------:|
| seed1 | without_rl | `.../checkpoint-10500` | `test_emb_3.5epoch_without_rl-1` + `-2` | 未传参（脚本内 `seed=1`） |
| seed1 | emb_080 | `.../search_emb_b080_bs1/global_step_500/merged` | `test_emb_080` | 未传参（脚本内 `seed=1`） |
| seed1 | noemb | `.../search_noemb_bs1/global_step_500/merged` | `test_noemb` | 未传参（脚本内 `seed=1`） |
| seed11 | without_rl | `.../checkpoint-10500` | `test_emb_3.5epoch_without_rl` | `--seed 11` |
| seed11 | emb_080 | `.../search_emb_b080_bs1/global_step_500/merged` | `test_emb_080` | `--seed 11` |
| seed11 | noemb | `.../search_noemb_bs1/global_step_500/merged` | `test_noemb` | `--seed 11` |
| seed21 | without_rl | `.../checkpoint-10500` | `test_emb_3.5epoch_without_rl` | `--seed 21` |
| seed21 | emb_080 | `.../search_emb_b080_bs1/global_step_500/merged` | `test_emb_080` | `--seed 21` |
| seed21 | noemb | `.../search_noemb_bs1/global_step_500/merged` | `test_noemb` | `--seed 21` |

### 5.4 日志验证

通过日志中的 `[DEBUG] ckpt_path`、`Trajectories saved to`、`Sampling mode: random (seed=X)` 三个关键信息验证实际运行路径与脚本配置一致：

| Seed | 模型 | 日志文件 | 模型路径 ✅ | 输出目录 ✅ | Seed ✅ |
|------|------|---------|:---:|:---:|:---:|
| seed1 | without_rl-1 | `test_search_3.5epoch_without_rl-1_20260726_091718.log` | checkpoint-10500 | `test_emb_3.5epoch_without_rl-1/` | seed=1 |
| seed1 | without_rl-2 | `test_search_3.5epoch_without_rl-2_20260726_091807.log` | checkpoint-10500 | `test_emb_3.5epoch_without_rl-2/` | seed=1 |
| seed1 | emb_080 | `test_search_080_20260724_235903.log` | search_emb_b080_bs1 | `test_emb_080/` | seed=1 |
| seed1 | noemb | `test_search_noemb_20260724_030631.log` | search_noemb_bs1 | `test_noemb/` | seed=1 |
| seed11 | without_rl | `test_search_3.5epoch_without_rl_20260726_170759.log` | checkpoint-10500 | `test_emb_3.5epoch_without_rl/` | seed=11 |
| seed11 | emb_080 | `test_search_080_20260726_171224.log` | search_emb_b080_bs1 | `test_emb_080/` | seed=11 |
| seed11 | noemb | `test_search_noemb_20260726_170829.log` | search_noemb_bs1 | `test_noemb/` | seed=11 |
| seed21 | without_rl | `test_search_3.5epoch_without_rl_20260727_044234.log` | checkpoint-10500 | `test_emb_3.5epoch_without_rl/` | seed=21 |
| seed21 | emb_080 | `test_search_080_20260727_044206.log` | search_emb_b080_bs1 | `test_emb_080/` | seed=21 |
| seed21 | noemb | `test_search_noemb_20260727_044217.log` | search_noemb_bs1 | `test_noemb/` | seed=21 |

### 5.5 补充说明

- **seed1 的 without_rl 拆分为两个脚本**（`-1` 跑前 4 个数据集：nq/triviaqa/popqa/hotpotqa，`-2` 跑后 3 个数据集：2wikimultihopqa/musique/bamboogle），但使用同一模型权重（checkpoint-10500），合并后等价于一个完整实验。
- **seed1 脚本中未传 `--seed` 参数**，但 Python 脚本内硬编码 `seed = 1`，日志确认 `Sampling mode: random (seed=1)`，与预期一致。
- **seed11 和 seed21 脚本显式传递 `--seed 11` / `--seed 21`**，日志确认一致。

### 5.6 审查结论

**3 种权重 × 3 个 seed = 9 个实验组合（seed1 的 without_rl 拆分为 2 个子文件夹，共 10 个输出文件夹），模型路径与输出文件夹全部正确对应，无错误。** 脚本配置与日志记录完全一致。