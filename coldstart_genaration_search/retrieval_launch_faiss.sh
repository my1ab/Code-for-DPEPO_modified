
# 数据存储根目录 faiss向量索引文件 语料库文件 语料库文件路径 文本名称+路径
save_path=$HOME/data/searchR1
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2


# export HF_TOKEN=${HF_TOKEN:-your_hf_token_here}
export CUDA_VISIBLE_DEVICES=0
START_TIME=$(date +%s)
START_TIME_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

# Ctrl+C 时清理后台 tail 进程
cleanup() {
    echo ""
    echo "Cleaning up background processes ..."
    if [ -n "$TAIL_PID" ]; then
        kill $TAIL_PID 2>/dev/null
        wait $TAIL_PID 2>/dev/null
    fi
    exit 0
}
trap cleanup INT TERM

echo "========================================"
echo "  Retrieval Server Launcher"
echo "  启动时间: $START_TIME_HUMAN"
echo "========================================"
echo ""
echo "GPU: $CUDA_VISIBLE_DEVICES"

PORT=8010

# 杀掉旧进程
# OLD_PID=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+')
# if [ -n "$OLD_PID" ]; then
#     echo "Killing old server process(es) on port $PORT (PID: $OLD_PID) ..."
#     kill $OLD_PID 2>/dev/null
#     sleep 2
# fi

# 带时间戳的日志文件，避免多进程写入冲突
# TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
# LOG_FILE="coldstart_genaration_search/retrieval_server_faiss_${TIMESTAMP}.log"
# LOG_FILE="coldstart_genaration_search/retrieval_server_faiss.log"
LOG_FILE="coldstart_genaration_search/retrieval_server.log"
echo "Log file: $LOG_FILE"
# faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY 留在磁盘  使用时再调入
# 启动http服务器 加载索引+语料库 接收查询文本并返回最相似的文本片段
# --faiss_gpu \
nohup python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --port $PORT \
  --faiss_gpu \
  > $LOG_FILE 2>&1 &


LAUNCH_TIME=$(date +%s)
ELAPSED=$((LAUNCH_TIME - START_TIME))
echo "Process launched. PID: $!  (${ELAPSED}s elapsed)"
echo "logging in $LOG_FILE"

# 等待服务器就绪，同时输出日志进度
echo "Waiting for server to be ready..."
# touch "$LOG_FILE"
tail -F "$LOG_FILE" &
TAIL_PID=$!

# 轮询检测就绪信号
while true; do
    if grep -q "Uvicorn running on\|Application startup complete\|startup complete" "$LOG_FILE" 2>/dev/null; then
        READY_TIME=$(date +%s)
        TOTAL_ELAPSED=$((READY_TIME - START_TIME))
        echo ""
        echo "========================================"
        echo "Server is READY! Total startup time: ${TOTAL_ELAPSED}s"
        echo "========================================"
        break
    fi
    sleep 1
done

# 继续保持前台日志输出
wait $TAIL_PID

END_TIME=$(date +%s)
END_TIME_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_RUN=$((END_TIME - START_TIME))
echo ""
echo "========================================"
echo "  Server stopped."
echo "  结束时间: $END_TIME_HUMAN"
echo "  总运行时长: ${TOTAL_RUN}s"
echo "  Log file: $LOG_FILE"
echo "========================================"

echo "Server ended."
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"