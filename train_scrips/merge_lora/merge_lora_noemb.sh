#!/bin/bash
# ============================================================
# 合并 LoRA adapter 到 base 模型, 输出标准 HF 格式目录
#
# 用途:
#   将 2gpu_only_penalty/global_step_8 中的 LoRA adapter 合并到 base 模型,
#   使其兼容 coldstart_test_search_seed1_1400sample/coldstart_search_local_3.5epoch_multithread.py
#   中 #sym:ckpt_path 处通过 load_local_model() -> AutoModelForCausalLM.from_pretrained()
#   的读取方式 (标准 HF 目录: 含 config.json / model.safetensors / tokenizer.json).
#
# 输出目录: 2gpu_only_penalty/global_step_8/merged
#
# 用法:
#   bash my_scripts/merge_lora_only_penalty_step8.sh
#   (或在此文件末尾按需修改 BASE_MODEL / ADAPTER_DIR / OUTPUT_DIR 后再执行)
# ============================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
echo GPU:$CUDA_VISIBLE_DEVICES
# ---- 路径配置 --------------------------------------------------
# base 模型: 取自 adapter_config.json 的 base_model_name_or_path
BASE_MODEL='/diskpool/home/xuxz/ms-swift/checkpoint_search/Qwen2.5-1.5B-Instruct-Parallel-Epoch5-hislen8_seed1_1000sample/v1-20260715-204318/checkpoint-10500'
ckpt_path='/diskpool/home/xuxz/Code-for-DPEPO/3emb_model_bs4/search_no_emb_bs4_lr1e-5/global_step_125'
# LoRA adapter 目录 (含 adapter_config.json + adapter_model.safetensors)
ADAPTER_DIR="$ckpt_path/actor/lora_adapter"

# 合并后输出目录 (兼容 load_local_model 的 AutoModelForCausalLM.from_pretrained 读取)
OUTPUT_DIR="$ckpt_path/merged"

# swift 可执行文件 (ms_swift conda env)
SWIFT_BIN="/diskpool/home/xuxz/miniconda3/envs/ms_swift/bin/swift"

# 日志文件: 输出文件夹上层目录下的 merge.log
LOG_FILE="$(dirname "$OUTPUT_DIR")/merge.log"

# 确保日志目录存在
mkdir -p "$(dirname "$OUTPUT_DIR")"

# ---- 启动 tail -F 实时监控 (主流程所有输出都写入 LOG_FILE) ----
# 先清空旧日志, 确保 tail 从头开始显示本次所有输出.
: > "$LOG_FILE"
tail -F "$LOG_FILE" &
TAIL_PID=$!

# 主流程结束后自动停止 tail 并退出
cleanup() {
    sleep 0.5
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---- 以下所有输出 (含 swift 进度条/警告/错误) 全部重定向到 LOG_FILE ----
exec > "$LOG_FILE" 2>&1

# ---- 前置检查 --------------------------------------------------
echo "=============================================="
echo "Base model  : $BASE_MODEL"
echo "LoRA adapter: $ADAPTER_DIR"
echo "Output dir  : $OUTPUT_DIR"
echo "Swift bin   : $SWIFT_BIN"
echo "Log file    : $LOG_FILE"
echo "=============================================="

# 若输出目录已存在, 提示并跳过 (避免覆盖)
if [ -e "$OUTPUT_DIR" ]; then
    echo "[WARN] OUTPUT_DIR already exists: $OUTPUT_DIR"
    echo "       如需重新生成, 请先删除该目录后再运行."
    # 不直接退出, 允许 swift 继续写入 (swift 会覆盖同名文件)
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

# ---- 执行合并 --------------------------------------------------
# swift export 会:
#   1. 加载 --model 指定的 base 模型
#   2. 加载 --adapters 指定的 LoRA adapter
#   3. --merge_lora true 将 LoRA 权重合并回 base
#   4. 将合并后的完整模型 + tokenizer 保存到 --output_dir
# 输出目录包含 config.json / model.safetensors / tokenizer.json 等,
# 可直接被 AutoModelForCausalLM.from_pretrained(local_files_only=True, use_safetensors=True) 读取.
echo "[INFO] Running swift export --merge_lora ..."
"$SWIFT_BIN" export \
    --model "$BASE_MODEL" \
    --adapters "$ADAPTER_DIR" \
    --merge_lora true \
    --output_dir "$OUTPUT_DIR"

# ---- 修复 tokenizer (swift 4.2.0 已知问题) --------------------
# swift export --merge_lora 写出的 tokenizer_config.json 使用了非标准字段
# `extra_special_tokens` (list 类型), 而 transformers 4.51.1 的
# `_set_model_specific_special_tokens` 会对该字段调用 .keys() (期望 dict),
# 导致 'list' object has no attribute 'keys' 报错, 且合并后的 tokenizer 缺失
# added_tokens_decoder / chat_template / additional_special_tokens 等字段.
# 由于 LoRA 训练不修改 tokenizer, 直接从 base 模型复制 tokenizer 相关文件覆盖.
echo "[INFO] Copying tokenizer files from base model (fix swift 4.2.0 incompatibility)..."
for f in config.json generation_config.json merges.txt tokenizer_config.json tokenizer.json vocab.json chat_template.jinja; do
    if [ -f "$BASE_MODEL/$f" ]; then
        cp "$BASE_MODEL/$f" "$OUTPUT_DIR/$f"
        echo "  copied: $f"
    fi
done
echo "merge done"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "=============================================="
# ---- 结果检查 --------------------------------------------------
# echo "=============================================="
# echo "[INFO] Merge finished. Checking output..."
# if [ -f "$OUTPUT_DIR/config.json" ] && \
#    { ls "$OUTPUT_DIR"/model*.safetensors >/dev/null 2>&1 || \
#      ls "$OUTPUT_DIR"/model*.bin >/dev/null 2>&1; }; then
#     echo "[OK] Merged model saved to: $OUTPUT_DIR"
#     echo
#     echo "在 coldstart_search_local_3.5epoch_multithread.py 的 #sym:ckpt_path 处设置:"
#     echo "    ckpt_path = \"$OUTPUT_DIR\""
# else
#     echo "[ERROR] Output check failed: config.json or model weights missing in $OUTPUT_DIR"
#     exit 1
# fi
# echo "=============================================="
