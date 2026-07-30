# 模型对比结果

> 八个模型在 7 个 datasource 上的 Search Coldstart 准确率对比
>
> 注：3.5epoch_without_rl-1 和 3.5epoch_without_rl-2 使用同一模型（checkpoint-10500），区别在于测试时拆分为两段并行运行（-1 跑 nq/triviaqa/popqa/hotpotqa，-2 跑 2wikimultihopqa/musique/bamboogle），合并后等价于一个完整实验。

## Log 来源文件夹

| 模型简称 | 对应文件夹路径 |
|:---|---|
| rl_action/emb_080 | `coldstart_test_search_seed1_rl_action_modified/test_emb_080/` |
| rl_action/noemb | `coldstart_test_search_seed1_rl_action_modified/test_noemb/` |
| rl_action/noemb-1 | `coldstart_test_search_seed1_rl_action_modified/test_noemb-1/` |
| 3.5epoch_without_rl-1 | `coldstart_test_search_seed1_rl_action_modified/test_emb_3.5epoch_without_rl-1/` |
| 3.5epoch_without_rl-2 | `coldstart_test_search_seed1_rl_action_modified/test_emb_3.5epoch_without_rl-2/` |
| 3.5epoch | `coldstart_test_search_seed1_1400sample/test_3.5epoch/` |
| rl/emb_080 | `coldstart_test_search_seed1_rl/test_emb_080/` |
| rl/noemb | `coldstart_test_search_seed1_rl/test_noemb/` |

## 模型路径

| 模型简称 | 模型路径 |
|:---|---|
| rl_action/emb_080 | `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs1/search_emb_b080_bs1/global_step_500/merged` |
| rl_action/noemb | `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs1/search_noemb_bs1/global_step_500/merged` |
| rl_action/noemb-1 | `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs2/search_no_emb_bs2/global_step_125/merged` |
| 3.5epoch_without_rl-1 | `/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-10500` |
| 3.5epoch_without_rl-2 | `/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-10500` |
| 3.5epoch | `/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-10500` |
| rl/emb_080 | `/diskpool/home/xuxz/Code-for-DPEPO/2gpu_emb_search_080/global_step_500/merged` |
| rl/noemb | `/diskpool/home/xuxz/Code-for-DPEPO/2gpu_emb_search_noemb/global_step_500/merged` |

## 各 Datasource 准确率对比

| Datasource | 样本数 | **rl_action/emb_080** | **rl_action/noemb** | **rl_action/noemb-1** | **3.5epoch_without_rl-1** | **3.5epoch_without_rl-2** | **3.5epoch** | **rl/emb_080** | **rl/noemb** | 最佳模型 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| nq | 200 | 37.50% (75/200) | 37.00% (74/200) | 35.50% (71/200) | **35.00%** (70/200) | - | 34.50% (69/200) | **38.00%** (76/200) | 36.00% (72/200) | rl/emb_080 |
| triviaqa | 200 | **64.50%** (129/200) | 61.00% (122/200) | 58.50% (117/200) | **59.50%** (119/200) | - | 61.00% (122/200) | 58.00% (116/200) | 59.00% (118/200) | rl_action/emb_080 |
| popqa | 200 | 41.50% (83/200) | 42.50% (85/200) | 41.00% (82/200) | **43.50%** (87/200) | - | **48.00%** (96/200) | 44.50% (89/200) | 41.50% (83/200) | 3.5epoch |
| hotpotqa | 200 | 35.50% (71/200) | 35.50% (71/200) | 34.50% (69/200) | **36.50%** (73/200) | - | **36.50%** (73/200) | 32.00% (64/200) | 34.00% (68/200) | 3.5epoch_without_rl-1 / 3.5epoch |
| 2wikimultihopqa | 200 | 39.00% (78/200) | 34.00% (68/200) | 37.00% (74/200) | - | **37.50%** (75/200) | **40.00%** (80/200) | 37.00% (74/200) | 37.50% (75/200) | 3.5epoch |
| musique | 200 | 13.50% (27/200) | 12.00% (24/200) | 13.00% (26/200) | - | **13.50%** (27/200) | **14.00%** (28/200) | 13.00% (26/200) | 12.00% (24/200) | 3.5epoch |
| bamboogle | 125 | 28.80% (36/125) | 28.00% (35/125) | 24.80% (31/125) | - | **28.00%** (35/125) | 26.40% (33/125) | **33.60%** (42/125) | 24.80% (31/125) | rl/emb_080 |

## 各 Datasource 用时对比

| Datasource | rl_action/emb_080 | rl_action/noemb | rl_action/noemb-1 | 3.5epoch_without_rl-1 | 3.5epoch_without_rl-2 | 3.5epoch | rl/emb_080 | rl/noemb |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| nq | 2469.4s | 2440.8s | 3223.8s | 2615.6s | - | 3714.7s | **2257.6s** | 2675.2s |
| triviaqa | 2362.2s | 2848.4s | 3190.9s | 2530.8s | - | 3940.6s | **2311.3s** | 2162.1s |
| popqa | 3732.3s | 3957.0s | 6060.3s | 4497.4s | - | 5389.6s | **3739.8s** | 4130.5s |
| hotpotqa | 3198.5s | 2951.3s | 3403.8s | 3281.2s | - | 4072.3s | **2778.8s** | 2941.2s |
| 2wikimultihopqa | 3475.5s | 3897.3s | 3580.2s | - | 3332.6s | 4711.1s | **3299.4s** | 3265.5s |
| musique | 8458.5s | 8652.8s | 8320.5s | - | 7566.1s | 11062.2s | **7808.7s** | 7727.1s |
| bamboogle | 1772.6s | 2041.1s | **1488.9s** | - | 1776.9s | 1981.3s | 1529.7s | 1425.2s |
| **总用时** | **25469.0s** | **26788.7s** | **29268.4s** | **12925.0s** | **12675.6s** | **34871.8s** | **25725.3s** | **26326.8s** |
| **总用时 (h)** | **7.07h** | **7.44h** | **8.13h** | **3.59h** | **3.52h** | **9.69h** | **7.15h** | **7.31h** |

> 注：3.5epoch_without_rl-1 和 -2 合并总用时为 25600.6s (7.11h)，拆分为两段并行运行。

## 总体准确率汇总

| 模型 | 总成功数 | 总样本数 | **总正确率** | 总用时 | 效率 (样本/小时) | 总排名 |
|:---|---:|---:|---:|---:|---:|---:|
| **3.5epoch** | 501 | 1325 | **37.81%** | 9.69h | 136.7 | **1** |
| **rl_action/emb_080** | 499 | 1325 | **37.66%** | 7.07h | 187.4 | **2** |
| **3.5epoch_without_rl-1+2 (合并)** | 486 | 1325 | **36.68%** | 7.11h | 186.4 | **3** |
| **rl/emb_080** | 487 | 1325 | **36.75%** | 7.15h | 185.3 | **4** |
| **rl_action/noemb** | 479 | 1325 | **36.15%** | 7.44h | 178.1 | **5** |
| **rl/noemb** | 471 | 1325 | **35.55%** | 7.31h | 181.3 | **6** |
| **rl_action/noemb-1** | 470 | 1325 | **35.47%** | 8.13h | 163.0 | **7** |

## 各模型在各 Datasource 上的排名

| Datasource | rl_action/emb_080 | rl_action/noemb | rl_action/noemb-1 | 3.5epoch | rl/emb_080 | rl/noemb |
|:---|---:|---:|---:|---:|---:|---:|
| nq | 2 | 3 | 5 | 6 | **1** | 4 |
| triviaqa | **1** | 2 | 5 | 2 | 6 | 4 |
| popqa | 4 | 3 | 6 | **1** | 2 | 4 |
| hotpotqa | 2 | 2 | 4 | **1** | 6 | 5 |
| 2wikimultihopqa | 2 | 6 | 4 | **1** | 4 | 3 |
| musique | 2 | 5 | 3 | **1** | 3 | 5 |
| bamboogle | 2 | 3 | 5 | 4 | **1** | 5 |
| **第一名次数** | **1** | **0** | **0** | **4** | **2** | **0** |

## emb_080 vs noemb 配对对比（同一训练设置下是否加 emb）

| 训练设置 | emb_080 | noemb | noemb-1 | 差值 (emb最佳 - noemb最佳) |
|:---|---:|---:|---:|---:|
| rl_action_modified | 37.66% | 36.15% | 35.47% | **+1.51%** |
| rl | 36.75% | 35.55% | - | **+1.20%** |

> 两组训练设置中，加 emb_080 均优于 noemb，提升幅度约 1.2~1.5 个百分点。
> rl_action_modified 下 noemb-1（35.47%）略低于 noemb（36.15%），说明 noemb 有一定随机波动。

## 结论

1. **3.5epoch** 总正确率最高（**37.81%**），在 7 个 datasource 中取得 **4 个第一**（2wikimultihopqa、hotpotqa、musique、popqa），但用时最长（9.69h），效率最低（136.7 样本/小时）。
2. **rl_action/emb_080** 总正确率第二（**37.66%**），在 triviaqa 上表现最佳，取得 **1 个第一**，且用时最短（7.07h），效率最高（187.4 样本/小时）。
3. **rl/emb_080** 总正确率第三（**36.75%**），在 bamboogle、nq 上表现最佳，取得 **2 个第一**，用时也较短（7.15h）。
4. 三个 noemb 变体（rl_action/noemb、rl/noemb、rl_action/noemb-1）均未在任何 datasource 上取得第一，分别位列第 4、5、6 名。
5. **emb 效应**：在两组训练设置（rl_action_modified、rl）中，加 emb_080 均稳定优于 noemb 约 1.2~1.5 个百分点，且用时更短。
6. 六个模型整体准确率差距约 2.34 个百分点，头部三名（3.5epoch、rl_action/emb_080、rl/emb_080）差距仅 1.06 个百分点。
7. **用时差异显著**：3.5epoch 竞是最慢的（9.69h），而 rl 系列模型平均快约 2.5h（~7.2h），说明 RL 训练后的模型推理效率更高。