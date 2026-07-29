# Penalty / Reward / Active_mask / 动作排除机制对比

> 对比日期：2026-07-22
> 对比文件：
> - `agent_system/multi_turn_rollout/rollout_loop_parallel.py`（ALFWorld，`TrajectoryCollectorParallel`）
> - `agent_system/multi_turn_rollout/rollout_loop_parallel_search.py`（Search，`TrajectoryCollectorParallelSearch`）

---

## 一、Penalty 计算作用域

### 核心结论

**三个 penalty 组件全部仅限当前 bs 下标内，不跨 bs。**

`calculate_penalties` 调用点（Search `:1022-1031` / ALFWorld `:569-577`）：

```python
for sample, his_acts in zip(dict_grouped_output, cur_history_actions):
    action_dict = sample['action_dict']       # ← 当前 bs 的 action_dict
    W = self.calculate_penalties(
        history_actions=his_acts,             # ← 当前 bs 的 history
        action_dict=action_dict,
    )
    sample['penalty_W'] = W
```

每个 bs 独立调用一次 `calculate_penalties`。其内部：

| 组件 | 数据来源 | 作用域 | 跨 bs？ |
|---|---|---|---|
| **depth_repeat** | `history_actions[key]` + `action_dict[key]` | 当前 bs 的 env `key` 的历史 | ❌ |
| **transition_repeat** | `history_actions[key]` 的相邻对 | 当前 bs 的 env `key` 的历史 | ❌ |
| **width_transition** | `history_actions[w_idx]` + `action_dict[w_idx]`，`w_idx` ∈ `action_keys - {key}` | 当前 bs 的**其他并行 env** | ❌ |
| **width_repeat** | `action_dict.values()` | 当前 bs 的所有并行 env | ❌ |

`history_actions` 来自 `get_history_info_group`，通过 `(gamefile, group_id)` 定位 worker（`env_manager_parallel_search.py:472-476`），每个 bs 对应一个独立的 `ParallelSearchWorker`（含独立的 `action_manager`），**bs 之间完全隔离**。

"非 active 环境的影响"只在**同一个 bs 内部的并行 env 之间**发生：bs `i` 的 env 3 是 null，会影响 bs `i` 的 env 1/2/4/5 的 width 计算（因为它们都在同一个 `action_dict` 里）。

---

## 二、三个 Penalty 组件的 null 排除机制

### 当前代码状态

| 组件 | ALFWorld | Search（emb OFF） | Search（emb ON） |
|---|---|---|---|
| **depth_repeat** | 无 null 概念，不过滤 | **不过滤**（null==null 精确匹配） | 不过滤（null==null -> 1.0） |
| **transition_repeat** | 无 null 概念，不过滤 | **不过滤**（`("null","null")` 被计数） | 不过滤（null 转移对贡献 1.0） |
| **width_repeat** | 过滤 `'look'` | **过滤 `'null'`**（已修改） | 过滤 `'null'` |

### 不排除 null 的有利影响（width 维度）

当某些并行 env 未工作（输出 null）时，多个 null 被计为重复，压低 `W_width_repeat`：

```
5 个 env，3 个输出 null：[null, null, null, A, B]
不排除 null：COUNT = 5 - len({null, A, B}) = 5 - 3 = 2  -> W = ω²（惩罚）
排除 null：  COUNT = 2 - len({A, B})   = 2 - 2 = 0  -> W = ω⁰ = 1.0（无惩罚）
```

**不排除 null -> 惩罚"并行 env 闲置" -> 鼓励模型充分利用所有并行 env -> 有利于并行探索效率。**

### 三个组件分别评估

| 组件 | 不排除 null 时的影响 | 是否合理 |
|---|---|---|
| **width_repeat** | `W_width_repeat` 压低 | ✅ **合理**：直接惩罚"多个 env 闲置 = 并行资源浪费" |
| **depth_repeat** | null 与历史 null 精确匹配，`W_depth_repeat` 压低 | ⚠️ **不合理**：null 不是真实动作，"重复 null"不是有意义的重复 |
| **transition_repeat** | `("null","null")` 转移对被计数，`W_depth_t_repeat` 压低 | ⚠️ **不合理**：null 转移不是有意义的转移重复 |

### 建议的混合策略

```
width 维度：不排除 null   <- 保留对并行资源浪费的惩罚（有利影响）
depth / transition：排除 null  <- null 不是真实动作，不应按重复动作惩罚
```

- **width（`_calculate_width_repeat_weighted` 非 emb 分支）**：改回不排除 null
  ```python
  actions_wo_look = [elem for elem in action_dict.values()]  # 不过滤
  return len(actions_wo_look) - len(set(actions_wo_look))
  ```
- **depth（`calculate_depth_repeat`）**：跳过 null 历史
  ```python
  for history_action in reversed(history_actions):
      if history_action == 'null':
          continue
      if action == history_action:
          repeat_count += 1
  ```
- **transition（`calculate_transition_repeat`）**：过滤含 null 的转移对
  ```python
  state_action_pair_a = [(a, b) for a, b in zip(full_action_list, full_action_list[1:])
                          if a != 'null' and b != 'null']
  ```

---

## 三、Active_mask 机制对比

### ALFWorld：全做 + 延迟过滤

```python
# 每步开始时计算 active_masks（步骤开始前的状态）
active_masks = np.logical_not(is_done)                    # :533

# step-level rewards 仍写入 batch（包括 done 后的）
batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, ...)
batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, ...)  # :606-607

# batch_list 仍 append（包括 done 后的，带 active_masks=False 标记）
for i in range(batch_size):
    total_batch_list[i].append(batch_list[i])              # :611

# 最终在 gather_rollout_data 中过滤
for data in total_batch_list[bs]:
    if data['active_masks']:                               # :376
        effective_batch.append(data)
```

**特点**：
- 推理：全做（done 样本也推理）
- action_dict：模型真实输出（不强制 null）
- 环境交互：仍交互（真实动作送入 env）
- penalty：仍计算（用真实动作）
- batch_list：仍 append（带 `active_masks=False` 标记）
- 最终过滤：`gather_rollout_data` 中 `if data['active_masks']`

### Search：active 推理 + 即时跳过

```python
# 每步开始时计算 active_masks
active_masks = np.logical_not(is_done)                    # :886

# 只对 active 样本推理
active_indices = np.where(active_masks)[0]                # :931
if num_active == batch_size:
    # 全部 active，走原始路径
else:
    batch_input_active = batch_input[active_indices]       # 只推理 active
    # done 样本的 responses 用 0 占位
    full_tensor = torch.full(full_shape, fill_value=0, ...)

# done 样本的 action_dict 强制设 null
for i in range(batch_size):
    if not active_masks[i]:
        action_dict[parallel_actions_dict[i].get('action_dict', {})] = "null"  # :1005-1011

# penalty 仍计算（用 null 动作）
for sample, his_acts in zip(dict_grouped_output, cur_history_actions):
    W = self.calculate_penalties(...)                     # :1022-1031
    sample['penalty_W'] = W

# batch_list 即时跳过
for i in range(batch_size):
    if not active_masks[i]:
        continue                                           # :1091-1093
    total_batch_list[i].append(batch_list[i])
```

**特点**：
- 推理：只对 active 推理（优化，省 GPU）
- action_dict：done 样本强制设 null
- 环境交互：仍交互（null 动作送入 env，**污染 history**）
- penalty：仍计算（用 null 动作，**语义错误但结果被丢弃**）
- batch_list：即时跳过（不 append done 后的步骤）
- 最终过滤：append 阶段已完成

### 对比表

| 维度 | ALFWorld | Search |
|---|---|---|
| 推理 | 全做（浪费 GPU） | ✅ 只对 active 推理 |
| done 样本 action_dict | 模型真实输出 | ❌ 强制设 null |
| 环境交互 | 真实动作 | null 动作（**污染 history**） |
| penalty 计算 | 用真实动作（语义正确） | 用 null（语义错误，但被丢弃） |
| batch_list | append + 延迟过滤 | 即时跳过 |
| history 是否被污染 | ❌ 不污染 | ⚠️ 污染（null 写入 `action_manager`） |
| 最终训练数据 | 等效（done 后步骤被过滤） | 等效（done 后步骤被跳过） |

---

## 四、Reward 处理对比

### 步级 reward（两者一致）

```python
# episode_rewards 只对 active 累加（done 后不累加）
episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
episode_lengths[active_masks] += 1

# 但 step-level rewards 仍写入 batch（包括 done 后的）
batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
```

两者**完全相同**：step-level `rewards` 写入 batch（含 done 后的），但 `episode_rewards` 累加用 `active_masks` 过滤（done 后不累加）。

### is_done 判定（两者不同）

| 维度 | ALFWorld | Search |
|---|---|---|
| 判定方式 | `is_done = np.logical_or(is_done, rewards)`（纯 reward） | `any_done = any(dones[bs])`（env 返回的 done）+ `null_count >= 2`（连续 null） |
| 组内联动 | ❌ 无 | ✅ 训练阶段一个 done 整组 done |
| null 提前退出 | ❌ 无 | ✅ 连续 2 步全 null 标记 done |
| 影响差 | reward=0 但 env 未 done 时继续 | env done 或连续 null 即退出 |

**对 penalty 的影响**：间接影响。is_done 时序不同 -> active_masks 不同 -> 进入训练数据的步骤不同 -> 最终 penalty 序列不同。但这是**设计差异**（Search 针对 Search 任务做的适配），不是 bug。

### process_reward（空列表保护差异）

| 维度 | ALFWorld | Search |
|---|---|---|
| expert_actions 为空时 | `round(... / len(expert_action_list), 5)` -> **ZeroDivisionError 崩溃** | `round(... / len(...), 5) if expert_action_list else 0.0` -> 返回 0.0 |

正常情况不触发。

---

## 五、Null 污染数据流详解

### Null 的两个来源

1. **模型未输出某个 `<env_k>`**：`extract_think_and_actions` 预填 null（`:92`），未解析到的 env 保留 null
2. **环境已 done 后的占位**：`:1005-1011` 强制覆盖为全 null

### Null 进入 history 的路径

```
step T:   env.step(action_T) -> done=True
          last_action_manager[idx] = action_T  (真实动作)
step T+1: active_masks[i] = False
          action_dict 被强制设为 null
          env.step(null) -> SearchEnv 的 done guard 生效，返回空 obs
          last_action_manager[idx] = null  ← 污染 history
step T+2: action_manager[idx].append(null)  ← null 进入 history
          ...
```

### Null 污染的影响范围

- **bs 之间隔离**：null 不影响其他 bs（worker 隔离）
- **bs 内部传播**：null 进入 `history_actions[key]`，影响同 bs 的 depth/transition 计算
- **width 维度**：null 在 `action_dict.values()` 中，影响 width_repeat 计数

### 训练数据是否被污染

| 场景 | 是否进入训练数据 | 原因 |
|---|---|---|
| `is_done=True` 之后的步骤 | ❌ 不进入 | batch_list 被 `continue` 跳过 |
| `is_done=True` 当步 | ✅ 进入（action_dict 是真实动作） | done 在 step_group 之后检测 |
| **`is_done=False` 但个别 env 输出 null** | ✅ **进入** | active_masks=True，batch_list 被保留 |

**唯一会污染训练数据的路径**：done 之前、sample 仍 active、个别 env 输出 null。此时含 null 的 step 的 `batch_list[i]` 会进入训练数据，penalty 直接影响 PPO 训练。

---

## 六、总结对比表

| 维度 | ALFWorld | Search（emb OFF） | Search（emb ON） | 是否影响结果 |
|---|---|---|---|---|
| **penalty 作用域** | per-bs | per-bs | per-bs | ❌ 一致 |
| **depth null 处理** | 无 null | 不过滤 | 不过滤（null==null->1.0） | ⚠️ null 时不同 |
| **transition null 处理** | 无 null | 不过滤 | 不过滤（null 转移->1.0） | ⚠️ null 时不同 |
| **width null 处理** | 过滤 `'look'` | 过滤 `'null'` | 过滤 `'null'` | ⚠️ null 时不同 |
| **width 不排除 null 的有利影响** | N/A | 丢失（已改为排除） | 丢失 | ✅ 可恢复 |
| **active_mask 推理** | 全做 | 只对 active | 只对 active | ❌ 不影响结果 |
| **done 样本 action_dict** | 真实输出 | 强制 null | 强制 null | ❌ 结果被丢弃 |
| **done 样本 history 污染** | ❌ 不污染 | ⚠️ 污染 | ⚠️ 污染 | ❌ 不影响训练数据 |
| **active 样本个别 env null** | N/A | ⚠️ 进入训练数据 | ⚠️ 进入训练数据 | ✅ **影响训练** |
| **is_done 判定** | 纯 reward | done + null_count | done + null_count | ⚠️ 间接影响 |
| **组内联动** | ❌ 无 | ✅ 训练阶段 | ✅ 训练阶段 | ⚠️ 间接影响 |
| **reward 累加** | active_masks 过滤 | active_masks 过滤 | active_masks 过滤 | ❌ 一致 |
| **reward 写入 batch** | 写入+延迟过滤 | 写入+即时跳过 | 写入+即时跳过 | ❌ 等效 |

---

## 七、建议的优化方向

### 1. width 维度：改回不排除 null（保留有利影响）

```python
# _calculate_width_repeat_weighted 非 emb 分支
actions_wo_look = [elem for elem in action_dict.values()]  # 不过滤 null
return len(actions_wo_look) - len(set(actions_wo_look))
```

**理由**：多个 env 闲置 = 并行资源浪费 = 应该惩罚。不排除 null 能保留这一惩罚信号。

### 2. depth / transition 维度：排除 null（null 不是真实动作）

```python
# calculate_depth_repeat
for history_action in reversed(history_actions):
    if history_action == 'null':
        continue
    if action == history_action:
        repeat_count += 1

# calculate_transition_repeat
state_action_pair_a = [(a, b) for a, b in zip(full_action_list, full_action_list[1:])
                        if a != 'null' and b != 'null']
```

**理由**：null 不是真实动作，"重复 null"不是有意义的重复。depth/transition 应只衡量真实动作的重复度。

### 3. env_manager 层：done guard（防止 history 污染）

```python
# ParallelSearchWorker.step
for action_index, action in action_dict.items():
    if action_index in self.env_pools:
        sub_env = self.env_pools[action_index]
        already_done = sub_env.is_done       # 记录调用前的 done 状态
        ob, reward, done, info = sub_env.step(action)
        if not already_done:                 # 已 done 的 env 不更新 history
            self.last_action_manager[action_index] = action
            self.last_obs_manager[action_index] = ob
            self.last_poa_manager[action_index] = info['admissible_commands']
```

**理由**：从根源避免 null 写入 history。step T done 时 `already_done=False`，真实动作 action_T 写入 history；step T+1 起 `already_done=True`，null 不写入。
