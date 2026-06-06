python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /data/home/zhangjs/disk/project/verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome/global_step_25/actor \
    --target_dir /data/home/zhangjs/disk/project/verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome/merged/global_step_25