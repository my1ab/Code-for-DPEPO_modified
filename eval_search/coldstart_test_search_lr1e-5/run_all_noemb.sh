#!/bin/bash
# 启动所有 coldstart noemb 测试脚本
# 每个脚本内部 nohup 启动 Python 后即退出（tail -F 已注释）

set -e

BASE_PATH="$(dirname "$(realpath "$0")")"

echo "============================================="
echo "Starting all noemb coldstart search scripts..."
echo "============================================="

# ── 脚本列表 ──────────────────────────────────────
SCRIPTS=(
    "$BASE_PATH/coldstart_search_local_3.5epoch_noemb-1_seed1.sh"
    "$BASE_PATH/coldstart_search_local_3.5epoch_noemb-1_seed11.sh"
    "$BASE_PATH/coldstart_search_local_3.5epoch_noemb-1_seed21.sh"
    "$BASE_PATH/coldstart_search_local_3.5epoch_noemb-2_seed1.sh"
    "$BASE_PATH/coldstart_search_local_3.5epoch_noemb-2_seed11.sh"
    "$BASE_PATH/coldstart_search_local_3.5epoch_noemb-2_seed21.sh"
)

# ── 启动所有脚本 ──────────────────────────────────
for script in "${SCRIPTS[@]}"; do
    echo "Starting: $(basename "$script")"
    bash "$script" &
    # 错开启动时间，避免同时争抢资源
    sleep 2
done

echo ""
echo "============================================="
echo "All scripts launched in background."
echo "Check logs in: $BASE_PATH/log/"
echo "Use 'ps aux | grep test_search' to monitor."
echo "Use 'pkill -f test_search_local' to stop all."
echo "============================================="

# 等待所有后台任务完成
wait