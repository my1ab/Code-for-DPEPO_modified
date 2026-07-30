#!/bin/bash
# Coldstart Search Task - Data Generation Script
# 启动 coldstart_search.py 以生成 Search 任务的冷启动轨迹数据

set -e

# ================================================================
# 启动生成脚本
# ================================================================
# 前置依赖:
#   1. 检索服务: bash coldstart_genaration_search/retrieval_launch.sh
#   2. 预处理数据: python examples/data_preprocess/preprocess_search_r1_dataset.py
# ================================================================


BASE_PATH="$(dirname "$(realpath "$0")")"
export PYTHONPATH=$BASE_PATH:$PYTHONPATH
export PYTHONUNBUFFERED=1
# LOG_FILE="$BASE_PATH/log/search_test_multithread_5epoch_seed1.log"
LOG_FILE="$BASE_PATH/log/test_search_3.5epoch_without_rl_$(date +%Y%m%d_%H%M%S).log"
# 注意device设置
export CUDA_VISIBLE_DEVICES=2
echo "GPU: $CUDA_VISIBLE_DEVICES"

nohup python3 $BASE_PATH/test_search_local_3.5epoch_without_rl.py \
    --model "/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-10500" \
    --json_output_dir $BASE_PATH/test_emb_3.5epoch_without_rl \
    --seed 21 \
    > $LOG_FILE 2>&1 &

echo "Started. PID: $!"
echo "Log: $LOG_FILE"
echo "正在监控日志: tail -F $LOG_FILE"
echo '============================================='

# 自动开始查看日志
tail -F $LOG_FILE

# python3 $BASE_PATH/coldstart_genaration_search/coldstart_search.py
# python3 /diskpool/home/xuxz/verl-agent/coldstart_genaration_search/coldstart_search.py
# python3 /diskpool/home/xuxz/verl-agent/coldstart_genaration_search/test_model.py
# echo "Started. PID: $!"
