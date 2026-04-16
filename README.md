# README
## 🚀 Quick Start
 
To install the required dependencies, run the installation script:

```bash
bash install_sciworld.sh
```

> **Script Source**:  
> [install_sciworld.sh](https://github.com/LePanda026/Code-for-DPEPO/blob/main/my_scripts/install/install_sciworld.sh)

---

## Training the Parallel Agent with ColdStart SFT
We conducted SFT on Qwen2.5 series with OpenRLHF framework in turn-level.   
You can also finetune the LLM with other ms-swift, llama-factory, and etc.
The ColdStart datas and synthestic scripts can be found in [Ciallo～ (∠・ω< )⌒★](https://github.com/LePanda026/Code-for-DPEPO/blob/main/data_pipelines/ColdStart.zip)


## Training the Parallel Agent with DPEPO

Follow the steps below to start training the parallel agent with DPEPO on ScienceWorld.

### 1. Activate Conda Environment

```bash
conda activate verl-agent-sciworld
```

### 2. Set Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS
export TMPDIR=/root/autodl-tmp/tmp
```

### 3. Modify the Path in Configs into Your Local Path
For ALFWorld, You May need to modify the path in the following three files:
```bash
# https://github.com/LePanda026/Code-for-DPEPO/tree/main/data_pipelines/gamefiles/alfworld
1. gamefiles_eval.json
2. gamefiles_ood.json
3. gamefiles_train.json
```
The same as ScienceWorld.

### 4. Launch Training

```bash
python3 -m verl.trainer.main_ppo_sciworld \
    algorithm.adv_estimator=grpo \
    data.val_batch_size=128 \
    data.train_files=/root/autodl-tmp/verl-agent/data_pipelines/verl_train_data/sciworld/parallel_train_data_500.parquet \
    data.val_files=/root/autodl-tmp/verl-agent/data_pipelines/verl_train_data/alfworld/test.parquet \
    actor_rollout_ref.model.path=/root/autodl-tmp/Parallel-Qwen2.5-7B-Instruct-ColdStart-Epoch3-SciWorld \
    trainer.default_local_dir=/root/autodl-tmp/verl-agent/checkpoints/grpo_7b_coldstart_epoch3_outcome \
    trainer.experiment_name='grpo_7b_coldstart_epoch3_outcome' \
    env.max_steps=25 \
    trainer.max_actor_ckpt_to_keep=50 \
    reward_model.process_reward=false \
    reward_model.parallel_reward=false \
    reward_model.no_action_penalty=0.5 \
    reward_model.depth_alpha=0.8 \
    reward_model.depth_t_gamma=0.95 \
    reward_model.width_t_beta=0.95 \
    reward_model.width_omega=0.80 \
    reward_model.invalid_penalty=0.95 \
    data.train_batch_size=4 \
    env.num_parallel=4 \
    env.add_limit_prompt=True \
    env.rollout.n=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
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
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=false \
    env.env_name=sciworld/SciWorld \
    env.seed=0 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","mlflow"]' \
    trainer.project_name='parallel_verl_agent_sciworld' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=false
```

> 📝 **Note**:  
> - Ensure all paths (model, data, checkpoints, logs) exist or are writable.
