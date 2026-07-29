# =============================================================================
# WebShop GRPO 并行训练启动脚本 (从 alfworld process 脚本修改)
# =============================================================================
# 训练入口: train_scrips/for_webshop/main_ppo_webshop.py
# Hydra 配置: train_scrips/for_webshop/config/ppo_trainer_webshop.yaml
# 冲突设置以 verl-webshop/for_webshop/grpo_webshop_parallel_1gpu.sh 为准
# =============================================================================

# =============================================================================
# [路径配置] 迁移到其他用户/机器时，只需修改下面这两个变量
# =============================================================================
export DPEPO_USER_HOME=/diskpool/home/xuxz
export DPEPO_PROJECT_NAME=Code-for-DPEPO
# =============================================================================
# 注意：设置此环境变量后，依赖它的 .py 文件（env_manager_*.py, coldstart_*.py, prepare_*.py 等）会自动读取。
# 以下 shell 变量均派生自 DPEPO_USER_HOME 和 DPEPO_PROJECT_NAME，无需逐个修改。
# =============================================================================
_CODE_BASE=${DPEPO_USER_HOME}/${DPEPO_PROJECT_NAME}
_TMP_DIR=${DPEPO_USER_HOME}/tmp
_MODEL_ROOT=${DPEPO_USER_HOME}/ms-swift

# ========== 训练超参数 - 根据需要修改 ==========
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:128
batchsize=2
export CUDA_VISIBLE_DEVICES=0,1
micro_para=1
# tensor_model_parallel_size=1
tensor_model_parallel_size=$batchsize
# 禁用 Ray Dashboard，避免 opentelemetry 版本冲突导致启动失败
# export RAY_DASHBOARD_DISABLE=true
# ==========================================

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $batchsize"
export TMPDIR=$_TMP_DIR
export PYTHONUNBUFFERED=1
# 选择 attention backend: FLASH_ATTN 或 XFORMERS
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export VLLM_ATTENTION_BACKEND=XFORMERS

ckpt_dir=${_CODE_BASE}/2gpu_webshop
# LOG_FILE="$ckpt_dir/2gpu_only_penalty.log"
# LOG_FILE="$SCRIPT_DIR/log/test_bert_webshop_$(date +%Y%m%d_%H%M%S).log"
# LOG_FILE="$ckpt_dir/log/only_penalty_$(date +%Y%m%d_%H%M%S).log"
LOG_FILE="$ckpt_dir/log/only_penalty_emb_$(date +%Y%m%d_%H%M%S).log"

# 确保 checkpoint 目录和日志目录存在，避免 nohup 写日志和 verl 保存 checkpoint 时失败
mkdir -p "$ckpt_dir/log"
# 同时确保 verl 的 default_local_dir 子目录存在（verl 不会自动创建）
mkdir -p "$ckpt_dir"



rank_alpha=16
max_steps=30
# save_freq=1
save_freq=25
# free_cache=False
free_cache=True
# 自定义选项
custom_print_debug=true
custom_save_traj=false

# ========== 限制训练任务数 ==========
# 完成固定个任务后停止训练（不遍历整个训练文件）
# -1: 不限制，跑完整个 parquet（默认行为）
# 正整数 N: 训练 N 个任务后停止（每个 step 处理 train_batch_size 个任务）
# 注意：实际处理的任务数为 total_training_steps * train_batch_size
#       所以 total_training_steps = ceil(max_tasks / train_batch_size)
# max_tasks=20
# if [ "$max_tasks" -gt 0 ]; then
#     # 向上取整：即使 max_tasks 不能被 batchsize 整除，也至少跑够 max_tasks 个任务
#     total_training_steps=$(( (max_tasks + batchsize - 1) / batchsize ))
# else
#     # 不限制：不传该参数，让 verl 使用默认值 len(dataloader)*total_epochs
#     total_training_steps=""
# fi
# echo "Max tasks: $max_tasks, Total training steps: ${total_training_steps:-unlimited} (batch_size=$batchsize)"
# ${total_training_steps:+trainer.total_training_steps=$total_training_steps} \

nohup python3 train_scrips/for_webshop/main_ppo_webshop.py \
    algorithm.adv_estimator=grpo \
    env.env_path=${_CODE_BASE}/data_pipelines/gamefiles/webshop/webshop_train_tasks_excluded.json \
    data.train_files=${_CODE_BASE}/data_pipelines/verl_train_data/webshop/webshop_train_excluded.parquet \
    data.val_files=${_CODE_BASE}/data_pipelines/verl_train_data/webshop/webshop_test_excluded.parquet \
    actor_rollout_ref.model.path=${_MODEL_ROOT}/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8/v0-20260602-201729/checkpoint-8800 \
    trainer.default_local_dir=$ckpt_dir \
    trainer.experiment_name='grpo_1.5b_webshop_parallel' \
    data.train_batch_size=$batchsize \
    env.num_parallel=5 \
    env.add_limit_prompt=True \
    env.lazy_envs=True \
    env.rollout.n=3 \
    env.history_length=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.model.lora_rank=$rank_alpha \
    actor_rollout_ref.model.lora_alpha=$rank_alpha \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_para \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_para \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_para \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
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
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=$free_cache \
    actor_rollout_ref.rollout.free_cache_engine=$free_cache \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    reward_model.process_reward=false \
    reward_model.parallel_reward=true \
    reward_model.no_action_penalty=0.5 \
    reward_model.depth_alpha=0.8 \
    reward_model.depth_t_gamma=0.95 \
    reward_model.width_t_beta=0.95 \
    reward_model.width_omega=0.80 \
    algorithm.use_kl_in_reward=False \
    env.seed=0 \
    env.max_steps=$max_steps \
    env.resources_per_worker.num_cpus=0.5 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name='parallel_verl_agent_webshop' \
    trainer.n_gpus_per_node=$batchsize \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    +custom.print_debug=$custom_print_debug \
    +custom.save_traj=$custom_save_traj \
    trainer.test_freq=500 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Log: $LOG_FILE"
echo "Checkpoint dir: $ckpt_dir"
echo "To monitor: tail -f $LOG_FILE"
tail -F "$LOG_FILE"