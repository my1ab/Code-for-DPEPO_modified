# =============================================================================
# 相似度方案对比训练 2/3：emb + threshold=0.8
#
# 用途：评估使用 bge-large-en 嵌入模型 + 阈值 0.8 时的 penalty 效果
# 预期：阈值偏低，惩罚力度强，区分度最优但 FP 风险较高
#
# 对比组：
#   - eval_emb_0.9.sh：emb + threshold=0.9（保守惩罚）
#   - 本脚本：emb + threshold=0.8（激进惩罚）
#   - eval_no_emb.sh：不使用 emb（精确匹配，penalty 基本失效）
#
# 训练完成后 checkpoint 保存到：2gpu_eval_emb_080/
# 日志保存到：2gpu_eval_emb_080/log/
# =============================================================================

# =============================================================================
# [路径配置] 迁移到其他用户/机器时，只需修改下面这两个变量
# =============================================================================
export DPEPO_USER_HOME=/diskpool/home/xuxz
export DPEPO_PROJECT_NAME=Code-for-DPEPO
# =============================================================================
_CODE_BASE=${DPEPO_USER_HOME}/${DPEPO_PROJECT_NAME}
_TMP_DIR=${DPEPO_USER_HOME}/tmp
_MODEL_ROOT=${DPEPO_USER_HOME}/ms-swift

# ========== 检索服务器端口 (与 retrieval_launch.sh 保持同步) ==========
# export SEARCH_PORT=8010
# export SEARCH_URL="http://127.0.0.1:${SEARCH_PORT}/retrieve"
# ===================================================================

# ========== 训练配置 - 根据需要修改 ==========
batchsize=4
export CUDA_VISIBLE_DEVICES=1,2
micro_para=1
# tensor_model_parallel_size=$batchsize
# tensor_model_parallel_size=2
tensor_model_parallel_size=2
n_gpus_per_node=$tensor_model_parallel_size
# ==========================================

echo "GPU: $CUDA_VISIBLE_DEVICES"
# echo "Number of GPUs: $batchsize"
echo "Number of GPUs: $tensor_model_parallel_size"
export TMPDIR=$_TMP_DIR
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


# ========== 输出路径（本脚本独占） ==========
ckpt_dir=${_CODE_BASE}/3emb_model_bs4_webshop/webshop_no_emb_bs4
LOG_FILE="$ckpt_dir/log/webshop_no_emb_bs4_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$ckpt_dir/log"
mkdir -p "$ckpt_dir"

rank_alpha=32
max_steps=30
save_freq=10
free_cache=True
gpu_memory_utilization=0.3
# 自定义选项（custom 系列参数，统一在此区域设置）
custom_print_debug=true
custom_save_traj=true
custom_use_embedding=false     # 启用嵌入相似度
custom_emb_threshold=0.8     # 阈值 0.8（激进）

# data_pipelines/verl_train_data/webshop/webshop_train_excluded.parquet
# data_pipelines/verl_train_data/webshop/webshop_test_excluded.parquet

# 1e-6  0.01
lr=1e-5
kl_loss_coef=0.001
nohup python3 train_scrips/for_webshop/main_ppo_webshop.py \
    algorithm.adv_estimator=grpo \
    env.env_path=${_CODE_BASE}/data_pipelines/gamefiles/webshop/webshop_train_tasks_excluded.json \
    data.train_files=${_CODE_BASE}/data_pipelines/verl_train_data/webshop/webshop_train_excluded.parquet \
    data.val_files=${_CODE_BASE}/data_pipelines/verl_train_data/webshop/webshop_test_excluded.parquet \
    actor_rollout_ref.model.path=${_MODEL_ROOT}/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8/v0-20260602-201729/checkpoint-8800 \
    trainer.default_local_dir=$ckpt_dir \
    trainer.experiment_name='grpo_1.5b_webshop_no_emb_bs4' \
    data.train_batch_size=$batchsize \
    env.num_parallel=5 \
    env.add_limit_prompt=True \
    env.lazy_envs=True \
    env.rollout.n=8 \
    env.history_length=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.model.lora_rank=$rank_alpha \
    actor_rollout_ref.model.lora_alpha=$rank_alpha \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_para \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_para \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_para \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    data.max_prompt_length=28672 \
    data.max_response_length=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
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
    trainer.project_name='parallel_verl_agent_search' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    +custom.print_debug=$custom_print_debug \
    +custom.save_traj=$custom_save_traj \
    +custom.use_embedding=$custom_use_embedding \
    +custom.emb_threshold=$custom_emb_threshold \
    trainer.test_freq=500 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Log: $LOG_FILE"
echo "Checkpoint dir: $ckpt_dir"
echo "Similarity config: emb=$custom_use_embedding, threshold=$custom_emb_threshold"
echo "To monitor: tail -f $LOG_FILE"
tail -F "$LOG_FILE"
