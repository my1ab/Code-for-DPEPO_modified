#!/usr/bin/env bash
# =============================================================================
# 等待 PID=2233966 结束后，自动启动 eval_emb_0.6.sh
#
# 用法：
#   nohup bash train_scrips/for_search/eval_similarity_bs1/wait_and_run_emb0.6.sh \
#       > train_scrips/for_search/eval_similarity_bs1/wait_emb0.6.log 2>&1 &
#
# 取消等待：
#   kill <PID>   （PID 在脚本启动时打印）
# =============================================================================

WAIT_PID=2233966

PROJECT_ROOT=/diskpool/home/xuxz/Code-for-DPEPO
cd "$PROJECT_ROOT" || exit 1

TARGET_SCRIPT="$PROJECT_ROOT/train_scrips/for_search/eval_similarity_bs1/eval_emb_0.6.sh"

echo "等待进程 $WAIT_PID 结束...  (当前等待脚本 PID: $$)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "取消等待: kill $$"

tail --pid="$WAIT_PID" -f /dev/null

echo "进程 $WAIT_PID 已结束，开始执行: $TARGET_SCRIPT"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

bash "$TARGET_SCRIPT"
