# 三组实验成功率与耗时对比

## 模型路径

| 实验 | 模型路径 |
|---|---|
| emb_3.5epoch_without_rl | `/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-10500` |
| emb_080 | `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs1/search_emb_b080_bs1/global_step_500/merged` |
| noemb | `/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs1/search_noemb_bs1/global_step_500/merged` |

## 逐 Data Source 对比

| Data Source | Total | emb_3.5epoch_without_rl | emb_080 | noemb |
|---|---|---|---|---|
| nq | 200 | 73 (36.50%) | 67 (33.50%) | 61 (30.50%) |
| triviaqa | 200 | 105 (52.50%) | 114 (57.00%) | 103 (51.50%) |
| popqa | 200 | 79 (39.50%) | 76 (38.00%) | 75 (37.50%) |
| hotpotqa | 200 | 66 (33.00%) | 73 (36.50%) | 65 (32.50%) |
| 2wikimultihopqa | 200 | 62 (31.00%) | 66 (33.00%) | 53 (26.50%) |
| musique | 200 | 24 (12.00%) | 26 (13.00%) | 31 (15.50%) |
| bamboogle | 125 | 43 (34.40%) | 29 (23.20%) | 38 (30.40%) |

## 总体对比

| 实验 | Total | Success | Rate | Elapsed Time |
|---|---|---|---|---|
| emb_3.5epoch_without_rl | 1325 | 452 | 34.11% | 36931.3s (10.26h) |
| emb_080 | 1325 | 451 | 34.04% | 35446.8s (9.85h) |
| noemb | 1325 | 426 | 32.15% | 37253.8s (10.35h) |

## 耗时对比

| Data Source | emb_3.5epoch_without_rl | emb_080 | noemb |
|---|---|---|---|
| nq | 4058.4s | 3916.3s | 4209.1s |
| triviaqa | 3351.3s | 3161.0s | 3860.0s |
| popqa | 6453.8s | 5929.7s | 5885.2s |
| hotpotqa | 4099.2s | 4453.3s | 4122.4s |
| 2wikimultihopqa | 5415.3s | 5143.9s | 5754.0s |
| musique | 11674.9s | 10339.7s | 11741.9s |
| bamboogle | 1878.4s | 2502.9s | 1681.2s |
| **总计** | **36931.3s (10.26h)** | **35446.8s (9.85h)** | **37253.8s (10.35h)** |
