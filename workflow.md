main_ppo_alfworld.py
 │
 ├── [Hydra Config] config/ppo_trainer_parallel.yaml
 │
 ├── ❶ 环境导入
 │   └── from agent_system.environments import
 │         make_envs,                              ← agent_system/environments/env_manager.py
 │         build_parallel_alfworld_envs            ← agent_system/environments/__init__.py
 │                                                   (实际实现在 env_manager_parallel.py)
 │
 ├── ❷ 环境构建 (lazy_envs=False 时)
 │   └── build_parallel_alfworld_envs(
 │         gamefiles, group_n, resources, num_parallel)
 │       └── agent_system/environments/env_manager_parallel.py
 │             ├── class Env                     ← 底层单个 ALFWorld 环境封装
 │             │     └── build_env() → textworld.gym.make() + AlfredDemangler
 │             │         (from textworld + alfworld)
 │             ├── class ParallelAlfworldWorker  ← Ray Actor，管理多个 Env 实例
 │             │     └── step() / reset()
 │             └── class ParallelAlfworldEnvs    ← 顶层分组管理器 (gym.Env)
 │                   ├── step_group() → workers.step()
 │                   ├── step()  → workers.step()
 │                   ├── reset() → workers.reset()
 │                   ├── get_start_info_group()
 │                   ├── get_history_info_group()
 │                   └── get_last_actions_info_group()
 │
 ├── ❸ AlfWorldEnvironmentManager (继承 EnvironmentManagerBase)
 │   └── agent_system/environments/env_manager.py 中的同名类
 │         ├── reset()          → envs.reset() + build_text_obs()
 │         ├── step(actions)    → projection_f() + envs.step() + build_text_obs()
 │         ├── extract_task()   → 从 obs 解析 "Your task is to: " 部分
 │         ├── build_text_obs() → 用 ALFWORLD_TEMPLATE / ALFWORLD_TEMPLATE_NO_HIS 格式化
 │         │                      (from agent_system/environments/prompts/alfworld/)
 │         └── _process_batch() + _process_gamefile() → 按游戏类型统计成功率
 │
 ├── ❹ TrajectoryCollectorParallel
 │   └── agent_system/multi_turn_rollout/rollout_loop_parallel.py
 │         └── 调用 env_manager.reset() / step() 进行多轮 rollout
 │
 ├── ❺ EpisodeRewardManager
 │   └── agent_system/reward_manager/episode.py
 │         └── __call__() → 处理 reward_tensor
 │
 └── ❻ RayPPOTrainer (引入 traj_collector)
      └── verl/trainer/ppo/ray_trainer.py
            └── fit() → traj_collector 控制训练循环