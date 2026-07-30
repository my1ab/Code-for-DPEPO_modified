#!/bin/bash
# Coldstart WebShop Test Script

set -e

BASE_PATH="$(dirname "$(realpath "$0")")"
export PYTHONPATH=$BASE_PATH:$PYTHONPATH
export PYTHONUNBUFFERED=1

# 设定gpu编号
LOG_FILE="$BASE_PATH/log/Webshop_test_e3.5_hislen8_v1_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES=7
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"

nohup python3 $BASE_PATH/coldstart_para_his_test_1.5B_hislen8_epoch3.5.py \
    --model "/diskpool/home/xuxz/ms-swift/checkpoint/Qwen2.5-1.5B-Instruct-Parallel-Epoch5/v4-20260529-110357/checkpoint-6250" \
    --seed 42 \
    > $LOG_FILE 2>&1 &
echo "Test started. PID: $!"
tail -f $LOG_FILE

echo "Test ended."
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"