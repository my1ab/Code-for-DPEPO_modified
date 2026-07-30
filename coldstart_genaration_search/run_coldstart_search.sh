#!/bin/bash
# Coldstart Search Task - Data Generation Script
# 启动 coldstart_search.py 以生成 Search 任务的冷启动轨迹数据

set -e

# ================================================================
# 环境选择: remote(服务器) / local(本地)
# ================================================================
USE_REMOTE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote|-r)
            USE_REMOTE=1
            shift
            ;;
        --local|-l)
            USE_REMOTE=0
            shift
            ;;
        *)
            echo "Usage: $0 [--remote/-r | --local/-l]"
            exit 1
            ;;
    esac
done

if [ $USE_REMOTE -eq 1 ]; then
    BASE_PATH="/diskpool/home/xuxz/verl-agent"
else
    BASE_PATH="/home/dpepo/verl-agent"
fi

# ================================================================
# 启动生成脚本
# ================================================================
# 前置依赖:
#   1. 检索服务: bash coldstart_genaration_search/retrieval_launch.sh
#   2. 预处理数据: python examples/data_preprocess/preprocess_search_r1_dataset.py
# ================================================================


export PYTHONPATH=$BASE_PATH:$PYTHONPATH
export PYTHONUNBUFFERED=1

LOG_FILE="$BASE_PATH/coldstart_genaration_search/coldstart_search_gen.log"
# 注意device设置
export CUDA_VISIBLE_DEVICES=7
echo "GPU: $CUDA_VISIBLE_DEVICES"

# nohup python3 /diskpool/home/xuxz/verl-agent/coldstart_genaration_search/coldstart_search.py \
#     > $LOG_FILE 2>&1 &
nohup python3 /diskpool/home/xuxz/verl-agent/coldstart_genaration_search/coldstart_search_local.py \
    > $LOG_FILE 2>&1 &

echo "Started. PID: $!"
echo "Log: $LOG_FILE"
echo "正在监控日志: tail -f $LOG_FILE"
echo '============================================='

# 自动开始查看日志
tail -f $LOG_FILE

# python3 $BASE_PATH/coldstart_genaration_search/coldstart_search.py
# python3 /diskpool/home/xuxz/verl-agent/coldstart_genaration_search/coldstart_search.py
# python3 /diskpool/home/xuxz/verl-agent/coldstart_genaration_search/test_model.py
# echo "Started. PID: $!"
