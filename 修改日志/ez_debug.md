# grpo_alfworld_parallel_demo_ez.sh 调试修改总结

## 1. JSON 解析错误（`'list' object has no attribute 'values'`）

**文件**: `verl/trainer/main_ppo_alfworld.py` (line 66-70)

**原因**: `json.load()` 读取的是 JSON 数组（`[...]`），不是 dict，不能调用 `.values()`

**修改**: 判断数据类型，list 则按 `gamefile` 字段提取
```python
if isinstance(data, dict):
    gamefiles = list(data.values())
else:
    gamefiles = [item["gamefile"] for item in data]
```

---

## 2. 数据文件路径错误（`FileNotFoundError`）

**文件**: `grpo_alfworld_parallel_demo_ez.sh`

**修改 1**: `val_files` 路径缺少 `/alfworld/` 子目录
```
# 错误
data.val_files=.../verl_train_data/test.parquet
# 正确
data.val_files=.../verl_train_data/alfworld/test.parquet
```

**修改 2**: `train_files` 指向 `gamefiles/` 下新生成的文件（与 json 同目录）
```
# 原路径
data.train_files=.../verl_train_data/alfworld/parallel_train_data_demo_easy.parquet
# 新路径
data.train_files=.../gamefiles/parallel_train_data_demo_easy.parquet
```

---

## 3. TP 大小与 GPU 数不匹配（`AssertionError: world_size: 2 not divisible by infer_tp: 4`）

**文件**: `grpo_alfworld_parallel_demo_ez.sh`

**原因**: 2 块 GPU 但 `tensor_model_parallel_size=4`

**修改**: 改为 2
```
actor_rollout_ref.rollout.tensor_model_parallel_size=4  →  2
```

---

## 4. 缺失 `mlflow` 模块

**安装命令**:
```bash
conda run -n verl_train python3 -m pip install mlflow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 5. JSON 解析错误（`gamefile` 不是 JSON）

**文件**: `verl/trainer/ppo/ray_trainer.py` (line 1094)

**原因**: parquet 中 `gamefile` 列是普通字符串（如 `/.../game.tw-pddl`），但代码用 `json.loads()` 去解析

**修改**: 移除多余的 `json.loads()` 调用
```python
# 原代码
current_step_gamefiles = [json.loads(elem) for elem in batch.non_tensor_batch['gamefile']]
# 修改后
current_step_gamefiles = batch.non_tensor_batch['gamefile']
```

---

## 6. 缺失 `expert_actions` 字段（`KeyError: 'expert_actions'`）

**文件**: `agent_system/multi_turn_rollout/rollout_loop_parallel.py` (line 545)

**原因**: ALFWorld 的 parquet 数据不包含 `expert_actions` 字段

**修改**: 加保护判断
```python
if 'expert_actions' in gen_batch.non_tensor_batch:
    batch.non_tensor_batch['expert_actions'] = gen_batch.non_tensor_batch['expert_actions']
```

---

## 7. 验证集（test.parquet）没有 `gamefile` 字段（`KeyError: 'gamefile'`）

**文件**: `verl/trainer/ppo/ray_trainer.py` (line 734)

**原因**: test.parquet 列结构和 train.parquet 不同，没有 `gamefile` 列

**修改**: 检查 key 是否存在，不存在时跳过验证（`return {}`）
```python
if 'gamefile' in test_batch.non_tensor_batch:
    current_step_gamefiles = test_batch.non_tensor_batch['gamefile']
    self.val_envs = self.build_env_func(...)
else:
    current_step_gamefiles = None

if self.lazy_envs and current_step_gamefiles is None:
    print("Skipping validation: no gamefile in test batch")
    return {}
```

---

## 8. 验证集 `_validate()` 返回 None（`TypeError: 'NoneType' object is not iterable`）

**文件**: `verl/trainer/ppo/ray_trainer.py`

**原因**: `_validate()` 方法尾部的指标计算代码曾被注释掉，函数返回 `None`，导致 `metrics.update(None)` 报错

**修改**: 恢复并适配新的 `multi_turn_loop` 输出，从 `total_episode_rewards`、`total_batch_list` 等返回值中提取指标

---

## 9. 缺失配置项 `parallel_reward`（`ConfigAttributeError`）

**文件**: `verl/trainer/config/ppo_trainer.yaml` (line 201-202)

**原因**: `ray_trainer.py` 中访问 `self.config.reward_model.parallel_reward`，但 yaml 配置中没有此字段

**修改**: 添加
```yaml
parallel_reward: False
```

---

## 10. Prompt 超长问题

**文件**: `grpo_alfworld_parallel_demo_ez.sh`

**修改**: 
```
data.max_prompt_length=8192  →  16384
data.truncation='error'      →  'error'（保持，配合调大 max_length）
```

注：16384 在 Qwen2.5-0.5B 的 32K 上下文限制内，且 padding 开销可控。

---

## 11. 数据量少（7条）导致 `drop_last=True` 丢弃 3 条

**文件**: `verl/trainer/ppo/ray_trainer.py` (line 625) - 框架代码

**现状**: `StatefulDataLoader` 使用 `drop_last=True`，7 条数据 `batch_size=4` 时只取 4 条，遗弃 3 条

**解决方向**:
- 补数据到 8 条（能被 4 整除）
- 或改 `train_batch_size=7`
- 或改 `drop_last=False`

---

## 12. shell 脚本优化

**文件**: `grpo_alfworld_parallel_demo_ez.sh`

```bash
export PYTHONUNBUFFERED=1                              # 取消 Python 输出缓冲
LOG_FILE="webshop_checkpoint_para/test_alf_grpo_para.log"  # 日志文件
nohup ... > "$LOG_FILE" 2>&1 &                         # 后台运行
tail -f "$LOG_FILE"                                    # 实时查看日志
```
