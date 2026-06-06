# =============================================================================
# 超参数解释
# =============================================================================
# 数据层级:
#   data.train_batch_size=2       ← 每步从数据集取的样本数
#   env.rollout.n=2               ← 每条样本展开的并行轨迹数（探索+并行）
#   总模型交互数 = 2 × 2 = 4
#
# Batch Size 层级（遵循: 总交互数 ≥ mini_batch ≥ micro_batch × n_gpus）:
#   actor_rollout_ref.actor.ppo_mini_batch_size=4           ← PPO 每次更新的样本数
#   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2  ← 每 GPU 前向微批次（控制显存峰值）
#   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2  ← 对数概率计算的微批次
#   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2      ← 参考模型微批次
#
# 模型与推理:
#   actor_rollout_ref.rollout.tensor_model_parallel_size=2   ← 张量并行度（= GPU 数）
#   actor_rollout_ref.rollout.gpu_memory_utilization=0.6    ← VLLM 显存利用率
#   actor_rollout_ref.rollout.name=vllm                     ← 推理引擎
#
# 序列长度:
#   data.max_prompt_length=2048    ← prompt 最大长度
#   data.max_response_length=512   ← 生成最大长度
#
# GRPO 算法:
#   algorithm.adv_estimator=grpo   ← 使用 GRPO（无需 critic）
#   actor_rollout_ref.actor.use_kl_loss=True    ← 使用 KL 惩罚
#   actor_rollout_ref.actor.kl_loss_coef=0.01   ← KL 惩罚系数
#   actor_rollout_ref.actor.kl_loss_type=low_var_kl  ← 低方差 KL
#
# 无效动作惩罚:
#   actor_rollout_ref.actor.use_invalid_action_penalty=True
#   actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
#
# FSDP:
#   actor_rollout_ref.actor.fsdp_config.param_offload=False     ← 不卸载参数
#   actor_rollout_ref.actor.fsdp_config.optimizer_offload=False ← 不卸载优化器
#   actor_rollout_ref.ref.fsdp_config.param_offload=True        ← 参考模型卸载参数（省显存）
#
# 环境:
# train=3353; dataset len=16; eval len=140; test128
#   env.env_name=alfworld/AlfredTWEnv   ← 任务环境 
#   env.max_steps=50                    ← 每条轨迹最多 50 步
#   env.seed=0                          ← 随机种子
#
# 训练循环:
#   trainer.total_epochs=1              ← 总 epoch 数
#   trainer.test_freq=5                 ← 每 5 步验证一次
#   trainer.save_freq=-1                ← 不保存 checkpoint
#   trainer.val_before_train=False      ← 训练前不验证
#
# 每 epoch 步数 = train_data_size / train_batch_size
# 总步数       = 每 epoch 步数 × total_epochs
# =============================================================================

export CUDA_VISIBLE_DEVICES=3,5
echo GPU:$CUDA_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=XFORMERS 
export TMPDIR=/diskpool/home/xuxz/tmp 
export PYTHONUNBUFFERED=1
# /diskpool/home/xuxz/Code-for-DPEPO/my_scripts/train/alf-world/test_ver/test_alf.sh
# data.train_files=/data/home/zhangjs/disk/project/verl-agent/data_pipelines/verl_train_data/train.parquet \
#     data.val_files=/data/home/zhangjs/disk/project/verl-agent/data_pipelines/verl_train_data/test.parquet \
# LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"
LOG_FILE="webshop_checkpoint/test_alf_grpo.log"

nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/verl_train_data/alfworld/train.parquet \
    data.val_files=/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/verl_train_data/alfworld/test.parquet \
    actor_rollout_ref.model.path=/diskpool/home/xuxz/ms-swift/model/Qwen2.5-0.5B-Instruct \
    data.train_batch_size=2 \
    data.val_batch_size=128 \
    env.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.default_local_dir=./webshop_checkpoint \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    > "$LOG_FILE" 2>&1 &
# trainer.save_freq=100 \
# save_freq test_freq为全局步骤  即batch数
# 
echo "PID: $!"
echo "Log: $LOG_FILE"
tail -f "$LOG_FILE"

# 每 epoch 步数 = train_data_size / train_batch_size  （train_data_size 由 parquet 文件实际行数决定）
# 总步数       = 每 epoch 步数 × total_epochs