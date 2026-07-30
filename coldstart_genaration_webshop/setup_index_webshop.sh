#!/bin/bash

echo "=== 设置 Webshop 环境 ==="

# 定义路径
WEBSHOP_DIR="/diskpool/home/xuxz/verl-agent/agent_system/environments/env_package/webshop/webshop"
SEARCH_ENGINE_DIR="$WEBSHOP_DIR/search_engine"
DATA_DIR="$WEBSHOP_DIR/data"

echo "1. 创建数据目录（如果不存在）"
mkdir -p "$DATA_DIR"

echo "2. 创建搜索引擎资源目录"
cd "$SEARCH_ENGINE_DIR"
mkdir -p resources resources_100 resources_1k resources_100k

echo "3. 生成资源文件"
python convert_product_file_format.py

echo "4. 创建搜索引擎索引"
bash run_indexing.sh

echo "=== Webshop 环境设置完成 ==="