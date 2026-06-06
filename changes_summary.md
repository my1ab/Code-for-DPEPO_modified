# 所有更改总结

> 日期: 2026-06-05

---

## 1. 环境问题修复

### 1.1 `pkg_resources` 模块缺失
- **问题**: `verl/__init__.py` 中 `import pkg_resources` 失败
- **原因**: conda 安装的 `setuptools==82.0.1` 损坏，缺少 `pkg_resources` 模块
- **修复**: `pip install --force-reinstall --no-deps setuptools==68.0.0`
- **注意**: `pkg_resources` 不是独立的 pip 包，不能 `pip install pkg_resources`

### 1.2 缺失依赖包安装
| 包名 | 环境 | 安装方式 |
|------|------|---------|
| `hydra-core` | verl_train | `pip install hydra-core` |
| `codetiming` | verl_train | `pip install codetiming` |
| `tensordict` | verl_train | `pip install tensordict` |
| `gigpo` (外部包) | verl_train | `pip install -e /diskpool/home/xuxz/verl-agent/gigpo` |
| `verl` (本项目) | verl_train | `pip install -e .` |

---

## 2. 配置文件修改

### 2.1 `prepare_data.sh`
- **问题**: `$train_data_size` 和 `$val_data_size` 未定义，导致参数缺失错误
- **修复**: 添加变量定义
  ```bash
  train_data_size=256
  val_data_size=256
  ```

### 2.2 `verl/trainer/config/ppo_trainer.yaml`
- **问题**: 代码访问 `self.config.reward_model.parallel_reward` 但配置中缺失
- **修复**: 在 `reward_model:` 下添加 `parallel_reward: False`

### 2.3 `agent_system/multi_turn_rollout/rollout_loop.py`
- **问题**: `multi_turn_loop` 在 `is_train=False`（验证模式）时始终返回单个 `DataProto`，但 `ray_trainer.py:_validate` 期望解包 5 个值
- **修复**: 根据 `is_train` 分支返回不同结果
  ```python
  if is_train:
      return gen_batch_output    # DataProto
  else:
      return total_batch_list, total_episode_rewards, total_episode_lengths, total_traj_uid, totoal_tool_callings  # 5值元组
  ```

### 2.4 `verl/trainer/ppo/ray_trainer.py`
- **问题**: `_validate()` 方法的指标计算代码被注释，返回 `None`，导致 `TypeError: 'NoneType' is not iterable`
- **修复**: 恢复并重写验证指标计算逻辑，使用 `multi_turn_loop` 的返回值构建 `metric_dict`

---

## 3. 数据路径修复

### 3.1 `test_alf.sh` 数据路径
- **问题**: 路径指向 `/data/home/zhangjs/...`（他人目录）且缺少 `/alfworld/` 子目录
- **修复**: 改为指向本地数据
  ```bash
  # 修改前
  data.train_files=.../verl_train_data/train.parquet
  data.val_files=.../verl_train_data/test.parquet

  # 修改后
  data.train_files=/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/verl_train_data/alfworld/train.parquet
  data.val_files=/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/verl_train_data/alfworld/test.parquet
  ```

---

## 4. 脚本功能增强

### 4.1 `test_alf.sh` - 超参数注释
- 在脚本开头添加完整的中文超参数解释注释块
- 说明每个参数的作用和约束关系

### 4.2 `test_alf.sh` - nohup + 日志
- 训练命令改为 `nohup` 后台运行
- 输出重定向到 `$LOG_FILE`
- 添加 `PYTHONUNBUFFERED=1` 确保日志实时输出
- 自动 `tail -f` 查看日志

---

## 5. 知识点总结

### Batch Size 层级关系
```
数据集
  │
  ▼
data.train_batch_size = 2       ← 每步取 2 条数据
  │
  ├── env.rollout.n = 2         ← 每条展开 2 条轨迹
  │
  ▼
总交互数 = 2 × 2 = 4
  │
  ▼
ppo_mini_batch_size = 4         ← PPO 更新用
  │
  ▼
ppo_micro_batch_size_per_gpu = 2 ← 每 GPU 微批次（控制显存）
```

### 总步数计算
```
每 epoch 步数 = train_data_size / train_batch_size
总步数       = 每 epoch 步数 × total_epochs
例: (16 / 2) × 150 = 1200 步
```
