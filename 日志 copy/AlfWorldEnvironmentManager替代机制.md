# AlfWorldEnvironmentManager 替代机制分析

## 概述

并行训练路径（`main_ppo_alfworld.py` / `main_ppo_webshop.py`）将传统路径（`verl/trainer/main_ppo.py`）中 `AlfWorldEnvironmentManager` 的职责**拆解并内联**到了三个模块中，不再需要该封装层。

---

## 两条训练路径对比

```
传统路径 (verl/trainer/main_ppo.py):
  make_envs(config)
    └── AlfWorldEnvironmentManager(envs, projection_f, config)   ← 封装层
          ├── reset()   → envs.reset() + build_text_obs()
          ├── step()    → projection_f() + envs.step() + build_text_obs()
          └── extract_task() / _process_batch()
    └── TrajectoryCollector(config, tokenizer, processor)
          └── env_manager.reset() / step()

并行路径 (main_ppo_alfworld.py / main_ppo_webshop.py):
  build_parallel_alfworld_envs(gamefiles, ...)
    └── ParallelAlfworldEnvs / ParallelWebShopEnvs (gym.Env)    ← 无封装层
          ├── get_start_info_group()
          ├── step_group()
          └── reset()
    └── TrajectoryCollectorParallel(config, tokenizer, processor)
          └── preprocess_single_sample()    ← prompt/历史构建移到这里
          └── envs.get_start_info_group()   ← 直接调用底层
          └── envs.step_group()             ← 直接调用底层
```

---

## 职责拆解对照

| AlfWorldEnvironmentManager 方法 | 并行路径中的替代者 | 所在模块 |
|---|---|---|
| `__init__(envs, projection_f, config)` | **无** — 不需要，直接持有 ParallelAlfworldEnvs | — |
| `reset()` → `envs.reset()` + `build_text_obs()` | 拆为两步：<br>1. `envs.get_start_info_group()` → 获取初始 obs<br>2. `preprocess_single_sample(step=0)` → 构建初始 prompt | `rollout_loop_parallel.py` |
| `step(text_actions)` → `projection_f()` + `envs.step()` + `build_text_obs()` | 拆为两步：<br>1. `envs.step_group()` → 环境交互<br>2. `preprocess_single_sample(step>0)` → 构建下一轮 prompt | `rollout_loop_parallel.py` |
| `extract_task(text_obs)` — 解析任务描述 | 内联到 `preprocess_single_sample()` 中，从 `start_obs` 拆解 | `rollout_loop_parallel.py` |
| `build_text_obs()` — prompt 模板格式化 | `preprocess_single_sample()` 中使用 `compressed_prompt_initial` / `compressed_prompt_process` | `rollout_loop_parallel.py` + `prompts.py` |
| `_process_batch()` / `_process_gamefile()` — 成功率统计 | `EpisodeRewardManager.__call__()` 中处理 `episode_rewards` | `reward_manager/episode.py` |
| 历史列表管理 (`memory.store/fetch`) | `ParallelAlfworldWorker`/`ParallelWebShopWorker` 内部维护 `action_manager` / `obs_manager` | `env_manager_parallel.py` / `env_manager_parallel_webshop.py` |

---

## 具体替换细节

### 1. prompt 构建：`build_text_obs()` → `preprocess_single_sample()`

```python
# AlfWorldEnvironmentManager 传统方式
full_text_obs = self.build_text_obs(text_obs, admissible_actions, init=True)
# 内部用 ALFWORLD_TEMPLATE / ALFWORLD_TEMPLATE_NO_HIS 格式化

# TrajectoryCollectorParallel 并行方式
prompt = compressed_prompt_initial.format(
    task_description=task,
    current_observation=start_obs,
    admissible_actions=admissible_actions
)
# 或
prompt = compressed_prompt_process.format(
    task_description=task,
    initial_observation=start_obs,
    history_info=history_prompt,
    last_history=last_action_obv,
)
```

### 2. 环境交互：`envs.step(actions)` → `envs.step_group(samples)`

```python
# AlfWorldEnvironmentManager
actions, valids = self.projection_f(text_actions, admissible_commands)
text_obs, ... = self.envs.step(actions)  # 传 flat action list

# TrajectoryCollectorParallel 直接调用
samples = envs.step_group(grouped_samples)  # 传 grouped dict
# 每个 sample 包含: {gamefile, group_id, action_dict, ...}
```

### 3. 模型推理

```python
# 传统路径: AlfWorldEnvironmentManager 不涉及
# 并行路径: 也不涉及，由 verl 的 Ray ActorWorkers 处理
# 两者最终都由 actor_rollout_wg.generate_sequences() 完成
```

---

## 为什么可以去掉 AlfWorldEnvironmentManager

- **`projection_f` 动作映射**：并行路径中 action 已是字符串 dict，不需要统一映射
- **单组 vs 多组**：并行路径支持 `group_n` 个副本，action 按 `{gamefile, group_id}` 路由，无法用 flat list 统一处理
- **prompt 格式不同**：并行路径使用 `compressed_prompt_*` 模板（含历史摘要），与传统路径的 `ALFWORLD_TEMPLATE` 不同
- **`Extract_task` 时机不同**：并行路径在第一次构造 prompt 时从 `start_obs` 提取，而非单独调用
