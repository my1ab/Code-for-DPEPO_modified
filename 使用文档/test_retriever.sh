#!/bin/bash
# 测试 retrieval 服务是否可用
# 用法: bash coldstart_genaration_search/test_retriever.sh [端口号]
# 默认端口: 8000

# PORT=${1:-8000}
PORT=${1:-8010}
BASE_URL="http://localhost:$PORT"

# 初始化计时
overall_start=$(date +%s%3N)

echo "=========================================="
echo "  检索服务可用性测试"
echo "=========================================="

# 1. 检查端口是否在监听
step_start=$(date +%s%3N)
echo ""
echo "[1/4] 检查端口 $PORT 是否在监听 ..."
if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
    PID=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+')
    echo "  ✅ 端口 $PORT 已监听 (PID: $PID)"
else
    echo "  ❌ 端口 $PORT 未监听，服务可能未启动"
    exit 1
fi
step_end=$(date +%s%3N); echo "  ⏱ 用时: $(( step_end - step_start )) ms"

# 2. 检查 HTTP 服务是否响应
step_start=$(date +%s%3N)
echo ""
echo "[2/4] 检查 HTTP 服务是否响应 ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/retrieve" \
    -X POST -H "Content-Type: application/json" -d '{"query":"test","topk":1}' 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "  ✅ HTTP 响应正常 (200)"
else
    echo "  ❌ HTTP 响应异常 ($HTTP_CODE)"
    exit 1
fi
step_end=$(date +%s%3N); echo "  ⏱ 用时: $(( step_end - step_start )) ms"

# 3. 发送实际检索请求
# ques=machine learning
# ques="computer vision"
# ques="large language model"
ques="beijing"
# ques="Washington"
# 阶段时间: 0.2 6-9 6-9 0.1s
step_start=$(date +%s%3N)
echo ""
echo "[3/4] 发送检索查询 ..."
RESULT=$(curl -s -X POST "$BASE_URL/retrieve" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$ques\", \"topk\": 3}")
step_end=$(date +%s%3N); echo "  ⏱ 用时: $(( step_end - step_start )) ms"
echo "$RESULT" | python -m json.tool 2>/dev/null || echo "$RESULT"

# 4. 验证返回结果格式
step_start=$(date +%s%3N)
echo ""
echo "[4/4] 验证返回结果 ..."
DOC_COUNT=$(echo "$RESULT" | python3 -c "import json,sys; data=json.load(sys.stdin); print(len(data['result'][0]))" 2>/dev/null)
if [ -n "$DOC_COUNT" ] && [ "$DOC_COUNT" -gt 0 ]; then
    echo "  ✅ 返回 $DOC_COUNT 条结果，服务正常！"
else
    echo "  ⚠️  结果解析异常，请检查服务"
fi
step_end=$(date +%s%3N); echo "  ⏱ 用时: $(( step_end - step_start )) ms"

overall_end=$(date +%s%3N)
total_ms=$(( overall_end - overall_start ))

echo ""
echo "=========================================="
echo "  测试完成"
echo "=========================================="
echo ""
echo "📊 总用时: ${total_ms} ms ($(echo "scale=2; $total_ms/1000" | bc) s)"
