# Seed21 测试结果汇总

## 模型路径

| 标签 | 模型路径 |
|------|----------|
| emb_3.5epoch_without_rl (SFT only, 无RL) | `/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-10500` |
| emb_080 (RL, emb b080) | `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs1/search_emb_b080_bs1/global_step_500/merged` |
| noemb (RL, 无emb) | `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs1/search_noemb_bs1/global_step_500/merged` |

## 逐数据集成功率比较

| 数据集 | emb_3.5epoch_without_rl | emb_080 | noemb |
|--------|:-----------------------:|:-------:|:-----:|
| 2wikimultihopqa_12576 | 71/200 = **35.50%** | 73/200 = **36.50%** | 74/200 = **37.00%** |
| bamboogle_125 | 37/125 = **29.60%** | 35/125 = **28.00%** | 35/125 = **28.00%** |
| hotpotqa_7405 | 72/200 = **36.00%** | 69/200 = **34.50%** | 75/200 = **37.50%** |
| musique_2417 | 22/200 = **11.00%** | 22/200 = **11.00%** | 22/200 = **11.00%** |
| nq_3610 | 75/200 = **37.50%** | 76/200 = **38.00%** | 82/200 = **41.00%** |
| popqa_14267 | 81/200 = **40.50%** | 84/200 = **42.00%** | 85/200 = **42.50%** |
| triviaqa_11313 | 116/200 = **58.00%** | 116/200 = **58.00%** | 117/200 = **58.50%** |

## 总成功率比较

| 模型 | 成功数 / 总数 | 成功率 |
|------|:------------:|:------:|
| emb_3.5epoch_without_rl | 474/1325 | **35.77%** |
| emb_080 | 475/1325 | **35.85%** |
| noemb | 490/1325 | **36.98%** |

## 结论

- **noemb** 模型在总成功率上表现最好（36.98%），在大多数数据集上与其它两个模型持平或更优（特别是 nq_3610 和 hotpotqa_7405）。
- **emb_080** 总成功率（35.85%）略高于 **emb_3.5epoch_without_rl**（35.77%），但差距很小（+0.08%）。
- 三个模型在 `musique_2417` 上表现完全一致（11.00%），在 `triviaqa_11313` 上也非常接近（58.00%–58.50%）。
- **emb_3.5epoch_without_rl** 仅在 `bamboogle_125` 上优于其它两个 RL 模型（29.60% vs 28.00%）。
- 整体来看，RL 训练（emb_080 和 noemb）相比纯 SFT（emb_3.5epoch_without_rl）有微小提升，其中 **noemb 变体提升最明显**。
