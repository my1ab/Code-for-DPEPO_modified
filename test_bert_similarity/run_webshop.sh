#!/bin/bash
# 使用 GPU 卡7 运行 BERT 嵌入模型相似度测试
# 用法: bash run_gpu7.sh

export CUDA_VISIBLE_DEVICES=7

# 使用 verl_train conda 环境
PYTHON="/diskpool/home/xuxz/miniconda3/envs/verl_train/bin/python"

SCRIPT_DIR="$(dirname "$0")"
# LOG_FILE="$SCRIPT_DIR/run_$(date +%Y%m%d_%H%M%S).log"
# LOG_FILE="$SCRIPT_DIR/test_bert.log"
# LOG_FILE="$SCRIPT_DIR/test_bert_$(date +%Y%m%d_%H%M%S).log"
LOG_FILE="$SCRIPT_DIR/log/test_bert_webshop_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================="
echo "GPU 设备: $CUDA_VISIBLE_DEVICES"
echo "脚本路径: $SCRIPT_DIR/test_bert_similarity.py"
echo "日志文件: $LOG_FILE"
echo "=============================================="

"$PYTHON" "$SCRIPT_DIR/test_bert_similarity.py" 2>&1 | tee "$LOG_FILE"
