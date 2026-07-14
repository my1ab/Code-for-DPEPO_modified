# 奖励机制分析: rollout_loop_parallel vs rollout_loop_parallel_webshop

> 分析日期: 2026-07-14
> 涉及文件:
> - `rollout_loop_parallel.py` (ALFWorld, `TrajectoryCollectorParallel`)
> - `rollout_loop_parallel_webshop.py` (WebShop, `TrajectoryCollectorParallelWebShop`)
> - `rollout_loop_parallel_search.py` (Search, `TrajectoryCollectorParallelSearch`)

---

## 一、总体奖励架构

奖励机制分为 **三个层次**，从粗到细：

| 层次 | 名称 | 触发条件 | 计算时机 | 作用范围 |
|------|------|---------|---------|---------|
| 1 | 环境稀疏奖励 (Sparse Reward) | 每步 | rollout 循环内 | 轨迹级 |
| 2 | DPEPO 步级多样性惩罚 (Parallel Penalty) | `reward_model.parallel_reward=true` | 每步，step_group 之后 | 步级 (per step) |
| 3 | LCS 过程奖励 (Process Reward) | `reward_model.process_reward=true` | rollout 循环结束后 | 轨迹级 (仅失败时替代) |

---

## 二、层次 1: 环境稀疏奖励 (Sparse Reward)

### 代码位置

两个文件相同, 均在 `vanilla_multi_turn_loop` 的 step 循环内：

```python
np_rewards = np.array(single_dict_grouped_output['rewards'], dtype=object)
rewards = np.array([
    np.max(lst) if len(lst) > 0 else 0
    for lst in np_rewards
])
episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
```

### 逻辑

- **每步** 从环境管理器拿到所有并行子环境的奖励列表 `rewards`（List，长度 = 并行环境数 n）
- 取 **最大值** `np.max(lst)` 作为该样本该步的奖励
- 累加到 `episode_rewards[bs]`
- 这是一个 **二进制稀疏奖励**：成功 = 1，失败 = 0（对于 ALFWorld/WebShop 的 goal-conditioned 任务）

---

## 三、层次 2: DPEPO 步级多样性惩罚 (Parallel Penalty)

### 3.1 触发入口

**ALFWorld (`rollout_loop_parallel.py`):**
```python
if self.config.reward_model.parallel_reward:
    for sample, cur_history_actions in zip(dict_grouped_output, history_actions):
        action_dict = sample['action_dict']
        if len(action_dict) != 0:
            W = self.calculate_penalties(history_actions=cur_history_actions, action_dict=action_dict)
        else:
            W = self.config.reward_model.no_action_penalty
        sample['penalty_W'] = W
```

**WebShop (`rollout_loop_parallel_webshop.py`) 和 Search (`rollout_loop_parallel_search.py`):**
```python
if hasattr(self.config, 'reward_model') and self.config.reward_model.get('parallel_reward', False):
    cur_history_actions, _ = envs.get_history_info_group(non_tensor_batch)
    for sample, his_acts in zip(dict_grouped_output, cur_history_actions):
        action_dict = sample['action_dict']
        if action_dict and len(action_dict) != 0:
            W = self.calculate_penalties(history_actions=his_acts, action_dict=action_dict)
        else:
            W = self.config.reward_model.get('no_action_penalty', 1.0)
        sample['penalty_W'] = W
```

> **差异**: WebShop/Search 使用 `self.config.reward_model.get('parallel_reward', False)` （dict-like get），而 ALFWorld 使用 `self.config.reward_model.parallel_reward`（attribute access）。

### 3.2 核心函数 `calculate_penalties` 详解

对所有三个文件，**核心数学逻辑完全相同**。唯一差异是 WebShop/Search 多了一个空列表保护：

```python
# ALFWorld:
pooling_w_action = sum(action_penalty_per_env) / len(action_penalty_per_env)

# WebShop / Search:
pooling_w_action = sum(action_penalty_per_env) / len(action_penalty_per_env) if action_penalty_per_env else 0.0
```

#### 3.2.1 四个惩罚维度的定义

惩罚权重 `W` 由四个维度的指数折扣项组成：

##### (a) 深度重复惩罚 (Depth Repeat) — `W_depth_repeat`

```
COUNT_repeat_penalty = count of current_action in env_history
W_depth_repeat = depth_alpha ** COUNT_repeat_penalty
```

- 衡量：**当前动作在当前环境的纵向历史中重复出现的次数**
- 公式：`W_depth = α^C` （指数折扣，非论文加法 α·C）
- 超参：`depth_alpha` ∈ (0, 1]，α 越小，重复惩罚越大

##### (b) 深度转移重复惩罚 (Depth Transition Repeat) — `W_depth_t_repeat`

```
last_state_action = (prev_action, current_action)
COUNT_depth_transition_penalty = count of last_state_action in env_history transitions
W_depth_t_repeat = depth_t_gamma ** COUNT_depth_transition_penalty
```

- 衡量：**(上一动作 → 当前动作) 这个转移对在当前环境历史中重复出现的次数**
- 公式：`W_depth_t = γ^C`
- 超参：`depth_t_gamma` ∈ (0, 1]

##### (c) 宽度转移重复惩罚 (Width Transition Repeat) — `W_width_t_repeat`

```
for each OTHER env w_idx:
    width_history = history_actions[w_idx] + [action_dict[w_idx]]
    repeat_count = count of last_state_action in width_history transitions
COUNT_width_transition_repeat = sum(repeat_count for all other envs)
W_width_t_repeat = width_t_beta ** COUNT_width_transition_repeat
```

- 衡量：**(当前动作的转移对) 在其他并行环境中出现的总次数**
- 公式：`W_width_t = β^C`
- 超参：`width_t_beta` ∈ (0, 1]
- 本质：**跨环境转移多样性惩罚** — 如果多个并行环境在走相同的转移路径，则受到惩罚

##### (d) 宽度重复惩罚 (Width Repeat) — `W_width_repeat`

```
actions_wo_look = filter out 'look' actions from current step
COUNT_width_repeat = len(actions_wo_look) - len(set(actions_wo_look))
W_width_repeat = width_omega ** COUNT_width_repeat
```

- 衡量：**当前步中，各并行环境之间动作的重复程度**
- 本质：去除 `look` 动作后，如果多个环境在同一轮选择了相同动作，则宽度重复计数增加
- 公式：`W_width = ω^C`
- 超参：`width_omega` ∈ (0, 1]

#### 3.2.2 聚合方式

**阶段一：每个环境内的三种惩罚取平均**
```python
W_list = [W_depth_repeat, W_depth_t_repeat, W_width_t_repeat]
pooling_kind_weight = sum(W_list) / len(W_list)   # 三者的算术平均
```

**阶段二：所有有动作的环境取平均**
```python
action_penalty_per_env.append(pooling_kind_weight)
pooling_w_action = avg(action_penalty_per_env)     # 跨环境平均
```

**阶段三：宽度重复惩罚与跨环境平均加权平均**
```python
W = (W_width_repeat + pooling_w_action) / 2
```

最终 `W` ∈ (0, 1]，数值越小表示多样性越好（重复越少），在后续 advantage 计算中作为步级折扣因子。

#### 3.2.3 辅助函数

| 函数 | 作用 | 复杂度 |
|------|------|--------|
| `calculate_depth_repeat` | 当前动作在历史中的重复次数 | O(L) |
| `calculate_transition_repeat` | 转移对 (a,b) 在历史转移中的重复次数 | O(L) |
| `get_state_action_pair` | 将 action list 转为 transition pair list | O(L) |
| `calculate_width_repeat_rate` | 当前步去重后动作数量差 | O(n) |

### 3.3 ALFWorld 中的注释代码 (Bug 线索)

在 `rollout_loop_parallel.py` 的 `calculate_penalties` 中有两行被注释掉的旧代码：

```python
# width_omega = self.self.config.reward_model.width_omega    # [Bug] self.self 双重复
# W_width_repeat = width_omega ** (COUNT_width_transition_repeat - 1)  # [不同公式]
```

当前有效代码在下方重新定义了：
```python
width_omega = self.config.reward_model.width_omega
W_width_repeat = width_omega ** COUNT_width_repeat
```

说明：旧代码曾尝试在**宽度转移重复**维度上额外叠加 `width_omega` 折扣（公式为 `ω^(C-1)`），但已废弃。

---

## 四、层次 3: LCS 过程奖励 (Process Reward)

### 4.1 触发条件

两个文件一致：
```python
# ALFWorld:
if self.config.reward_model.process_reward:

# WebShop / Search:
if hasattr(self.config, 'reward_model') and self.config.reward_model.get('process_reward', False):
```

仅在 **环境稀疏奖励为 0**（即任务失败）时生效：
```python
episode_rewards = np.where(episode_rewards == 0, process_reward, episode_rewards)
```

### 4.2 核心函数 `calculate_process_reward`

#### 步骤 1: 轨迹重组

将多步 trajectory 数据按并行环境 (1 ~ num_parallel) 拆分为子轨迹：

```
For each trajectory sample:
    For env_idx = 1..num_parallel:
        parallel_action[env_idx] = [step1_action, step2_action, ...]
        parallel_obs[env_idx]     = [step1_obs, step2_obs, ...]
```

#### 步骤 2: LCS (最长有序子序列) 计算

使用 `count_longest_ordered_subsequence` 函数：

```python
def count_longest_ordered_subsequence(self, ground_truth, prediction):
    i = j = matched = 0
    while i < len(gt) and j < len(pred):
        if gt[i] == pred[j]:
            matched += 1
            i += 1     # 只有匹配成功才推进 ground_truth
        j += 1         # prediction 永远前进
    return matched
```

这是一个 **带有序约束的贪心匹配**：从 expert action 列表和预测 action 列表的开头开始，逐个匹配。匹配成功后 **expert 指针才前进**，而预测指针始终前进。

本质上是 **LCS (Longest Common Subsequence) 的线性简化版**，等价于计算 prediction 中有多少个 action 按顺序出现在 expert 列表中。

#### 步骤 3: 归一化与聚合

```python
# 每个并行子环境计算 LCS
for env_idx, action_list in parallel_actions.items():
    reward = count_longest_ordered_subsequence(expert_action_list, action_list)
    parallel_lcs_rewards.append(reward)

# 取所有并行子环境中的最大值
action_reward = max(parallel_lcs_rewards)

# 归一化
process_reward = action_reward / len(expert_action_list)
```

**公式**: `R_process = max_env(LCS(env_trajectory, expert)) / len(expert)`

### 4.3 两个文件的关键差异

| 方面 | ALFWorld | WebShop / Search |
|------|----------|-----------------|
| expert_actions 来源 | 始终从 `trajectory[0]['expert_actions']` 读取 | 有 fallback：若不存在则设为 `[]` |
| 除以零保护 | 无（假设 expert_actions 非空） | 有：`if expert_action_list else 0.0` |
| 配置访问方式 | `self.config.reward_model.process_reward` | `self.config.reward_model.get('process_reward', False)` |

---

## 五、三文件奖励机制对比总表

| 维度 | ALFWorld | WebShop | Search |
|------|----------|---------|--------|
| 类名 | `TrajectoryCollectorParallel` | `TrajectoryCollectorParallelWebShop` | `TrajectoryCollectorParallelSearch` |
| 环境奖励 | `np.max(lst)` | 同左 | 同左 |
| DPEPO 惩罚 | ✅ `calculate_penalties` | ✅ `calculate_penalties` | ✅ `calculate_penalties` |
| 空列表保护 | ❌ `pooling_w_action = avg(...)` → 可能除零 | ✅ `if action_penalty_per_env else 0.0` | ✅ 同左 |
| 注释 bug | ✅ 有 `self.self` 注释代码 | ❌ 无 | ❌ 无 |
| Process Reward | ✅ `calculate_process_reward` | ✅ 同左 | ✅ 同左 |
| expert 降级 | ❌ 无 fallback | ✅ `if 'expert_actions' in trajectory[0]` | ✅ 同左 |
| 除以零保护 | ❌ 无 | ✅ `if expert_action_list else 0.0` | ✅ 同左 |
| GLOBAL_TASK_COUNTER | ❌ 无 | ✅ 有 | ✅ 有 |
| process_reward 配置方式 | `self.config.reward_model.process_reward` | `.get('process_reward', False)` | 同左 |
| parallel_reward 配置方式 | `self.config.reward_model.parallel_reward` | `.get('parallel_reward', False)` | 同左 |

---

## 六、与论文公式的差异

| 论文公式 (DPEPO) | 代码实现 | 差异说明 |
|------------------|---------|---------|
| `R_action = α × C_repeat` | `W_depth = α ** C_repeat` | **指数折扣 vs 乘法折扣** |
| `R_transition = γ × C_transition` | `W_depth_t = γ ** C_transition` | 同上 |
| `R = R_action + R_transition` | `W = avg(W_depth, W_depth_t, W_width_t)` | **三因子平均** |
| 无宽度转移项 | `W_width_t = β ** C_width_transition` | **额外新增维度** |
| 无宽度重复项 | `W_width_repeat = ω ** C_width` | **额外新增维度** |
| 最终 R 分离 | 最终聚合成单权重 W | **统一折扣因子** |

核心区别：
1. **指数折扣** (`α^C`) 替代了论文的 **线性折扣** (`α·C`) — 指数折扣在重复次数多时会更快趋近于 0，惩罚更激进
2. **宽度维度** 的转移重复和动作重复是代码中额外增加的，论文未明确提及
3. 最终权重 W 通过两层平均聚合，而非加法聚合

---

## 七、Reward 流向

```
rollout 阶段（本文件）
    ↓
episode_rewards (轨迹级) + penalty_W (步级) 
    ↓
gather_rollout_data() → DataProto.non_tensor_batch
    ↓
ray_trainer.py: compute_advantage()
    ↓ 读取 penalty_W, 传入 add_step_penalty
core_algos.py: compute_grpo_outcome_advantage()
    ↓
Φ_traj = group_norm(episode_rewards)          # 公式7
Φ_step = R_step if Φ_traj>0 else 2-R_step     # 公式8
Φ = Φ_step · Φ_traj                            # 公式9
    ↓
GRPO 策略损失 (verl 框架)
```

其中 `R_step = penalty_W`，即 `calculate_penalties` 计算的多样性权重。
