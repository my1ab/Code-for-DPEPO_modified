#!/bin/bash
# Coldstart WebShop Test Script

set -e

BASE_PATH="$(dirname "$(realpath "$0")")"
export PYTHONPATH=$BASE_PATH:$PYTHONPATH

# 设定gpu编号
export CUDA_VISIBLE_DEVICES=1
echo "GPU: $CUDA_VISIBLE_DEVICES"

python3 $BASE_PATH/coldstart_para_his_test_7B.py \
    --model "/diskpool/home/xuxz/ms-swift/model/Qwen2.5-7B-Instruct"
    --seed 42 \

# tail -f -n 10 Parallel-WebShop-1.5B-Epoch5_new.log

# echo "Test started. PID: $!"
echo "Test ended. PID: $!"
echo "GPU: $CUDA_VISIBLE_DEVICES"
# echo "Log: $BASE_PATH/coldstart_test/coldstart_test.log"
# tail -f -n 10 $BASE_PATH/coldstart_test/coldstart_test.log