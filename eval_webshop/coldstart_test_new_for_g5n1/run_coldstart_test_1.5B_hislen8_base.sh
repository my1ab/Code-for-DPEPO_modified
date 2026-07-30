#!/bin/bash
# Coldstart WebShop Test Script

set -e

BASE_PATH="$(dirname "$(realpath "$0")")"
export PYTHONPATH=$BASE_PATH:$PYTHONPATH
export PYTHONUNBUFFERED=1

# 设定gpu编号
export CUDA_VISIBLE_DEVICES=7
echo "GPU: $CUDA_VISIBLE_DEVICES"
LOG_FILE="$BASE_PATH/log/Webshop_test_hislen8_base_$(date +%Y%m%d_%H%M%S).log"

nohup python3 $BASE_PATH/../coldstart_test/coldstart_para_his_test_1.5B_hislen_epoch3.5.py \
    --model "/diskpool/home/xuxz/ms-swift/model/Qwen2.5-1.5B-Instruct" \
    --seed 42 \
    --sequential true \
    > $LOG_FILE 2>&1 &
echo "Test started. PID: $!"
tail -f $LOG_FILE

echo "Test ended."
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"

