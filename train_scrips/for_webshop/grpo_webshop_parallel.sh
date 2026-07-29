# =============================================================================
# WebShop GRPO 并行训练启动脚本 - 修复版
# =============================================================================
# 训练入口: train_scrips/for_webshop/main_ppo_webshop.py
# Hydra 配置: train_scrips/for_webshop/config/ppo_trainer_webshop.yaml
#
# 使用说明:
#   单卡: batchsize=1, CUDA_VISIBLE_DEVICES=0, micro_para=4
#   双卡: batchsize=2, CUDA_VISIBLE_DEVICES=0,1, micro_para=2, tensor_model_parallel_size=2
# =============================================================================

# ========== 配置选择 - 根据需要修改 ==========
# 单卡模式配置
# batchsize=1
# export CUDA_VISIBLE_DEVICES=0
# micro_para=4
# tensor_model_parallel_size=1

# 双卡模式配置 - 低显存版本
batchsize=1
export CUDA_VISIBLE_DEVICES=7
micro_para=2 # 降低micro_batch_size，每张GPU每次处理1个样本，进一步减少显存占用
tensor_model_parallel_size=1
# ==========================================

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $batchsize"
export TMPDIR=/diskpool/home/xuxz/tmp
export PYTHONUNBUFFERED=1
# export VLLM_LOGGING_LEVEL=WARN
# export RAY_BACKEND_LOG_LEVEL=WARN
# 显式启用Flash Attention 2
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# 确保使用VLLM的FlashAttention2后端（新版本VLLM默认启用）
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ATTENTION_BACKEND=XFORMERS

LOG_FILE="webshop_para_full_result/grpo_webshop_parallel.log"

rank_alpha=16

# 清理之前的Ray会话，避免目录空间警告
# ray stop --force > /dev/null 2>&1
# sleep 2

nohup python3 train_scrips/for_webshop/main_ppo_webshop.py \
    algorithm.adv_estimator=grpo \
    env.env_path=/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/gamefiles/webshop/webshop_train_tasks_excluded.json \
    data.train_files=/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/verl_train_data/webshop/webshop_train_excluded.parquet \
    data.val_files=/diskpool/home/xuxz/Code-for-DPEPO/data_pipelines/verl_train_data/webshop/webshop_test_excluded.parquet \
    actor_rollout_ref.model.path=/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8/v0-20260602-201729/checkpoint-8800 \
    trainer.default_local_dir=/diskpool/home/xuxz/Code-for-DPEPO/webshop_para_full_result \
    trainer.experiment_name='grpo_1.5b_webshop_parallel' \
    data.train_batch_size=$batchsize \
    env.num_parallel=5 \
    env.add_limit_prompt=True \
    env.lazy_envs=True \
    env.rollout.n=5 \
    env.history_length=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.model.lora_rank=$rank_alpha \
    actor_rollout_ref.model.lora_alpha=$rank_alpha \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_para \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_para \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_para \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    data.max_prompt_length=28672 \
    data.max_response_length=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    algorithm.use_kl_in_reward=False \
    env.seed=0 \
    env.max_steps=20 \
    env.resources_per_worker.num_cpus=0.5 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name='parallel_verl_agent_webshop' \
    trainer.n_gpus_per_node=$batchsize \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=500 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Log: $LOG_FILE"
echo "To monitor: tail -f $LOG_FILE"
tail -F "$LOG_FILE"